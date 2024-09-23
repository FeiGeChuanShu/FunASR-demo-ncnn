#include "sensevoice.h"
#include "custom_op.h"
#include <fstream>

static ncnn::Layer* Gather_layer_creator(void*) {
    return new Gather();
}
static ncnn::Layer* PositionalEncoding_layer_creator(void*) {
    return new PositionalEncoding();
}
static void apply_cmvn(const std::vector<float>& mean, 
    const std::vector<float>& vars, std::vector<float>& v) {
    int dim = 560;
    int num_frames = v.size() / dim;

    float* p = v.data();
    for (int i = 0; i < num_frames; ++i) {
        for (int k = 0; k != dim; ++k) {
            p[k] = (p[k] + mean[k]) * vars[k];
        }
        p += dim;
    }
}
static void decode_token(const ncnn::Mat& am_score,
    std::vector<int>& token_out, std::vector<int>& timestamps) {
    int prev_id = -1;
    for (int i = 0; i < am_score.h; ++i) {
        auto new_token = static_cast<int>(std::distance(am_score.row(i),
            std::max_element(am_score.row(i), am_score.row(i) + am_score.w)));
        if (new_token != 0 && new_token != prev_id) {
            token_out.emplace_back(new_token);
            timestamps.emplace_back(i);
            //fprintf(stderr, "%d\n", new_token);
        }
        prev_id = new_token;
    }
}

static std::vector<float> apply_lfr(std::vector<float>& in, std::vector<float>& lfr_out,
    int lfr_m, int lfr_n, int feat_len) {

    int in_feat_dim = feat_len;

    int in_feat_num = in.size() / in_feat_dim;
    int T_lfr = std::ceil(static_cast<float>(in_feat_num) / lfr_n);
    int left_padding = (lfr_m - 1) / 2;
    int out_feat_dim = in_feat_dim * lfr_m;
    int T = in_feat_num + left_padding;

    lfr_out.resize(T_lfr * out_feat_dim);

    for (int i = 0; i < left_padding; ++i) {
        in.insert(in.begin(), in.begin() + i * in_feat_dim, in.begin() + (i + 1) * in_feat_dim);
    }

    const float* in_ptr = in.data();
    float* out_ptr = lfr_out.data();
    for (int i = 0; i < T_lfr; ++i) {
        int start_idx = i * lfr_n;
        int end_idx = std::min(start_idx + lfr_m, in_feat_num);

        if (lfr_m <= T - i * lfr_n) {
            std::copy(in_ptr + start_idx * in_feat_dim, in_ptr + end_idx * in_feat_dim, out_ptr);
            out_ptr += out_feat_dim;
        }
        else {
            int num_padding = lfr_m - (T - start_idx);
            std::copy(in_ptr + start_idx * in_feat_dim, in_ptr + in_feat_num * in_feat_dim, out_ptr);
            out_ptr += (in_feat_num - start_idx) * in_feat_dim;
            
            for (int j = 0; j < num_padding; ++j) {
                std::copy(in_ptr + (in_feat_num - 1) * in_feat_dim, in_ptr + in_feat_num * in_feat_dim, out_ptr);
                out_ptr += in_feat_dim;
            }

        }
    }

    return lfr_out;
}


SenseVoice::~SenseVoice() {
    net_.clear();
    mean_.clear();
    vars_.clear();
}

int SenseVoice::init(const model_config_t& config) {
    std::ifstream f(config.token);
    if (!f.is_open()) {
        return -1;
    }
    token_json_ = json::parse(f);

    mean_.reserve(560);
    vars_.reserve(560);
    for (auto it = token_json_.begin(); it != token_json_.end(); ++it) {
        auto key = it.key();
        if(key == "mean"){
            auto value = it.value();
            if (value.is_array()) {
                for(const auto& v : value)
                    mean_.push_back(v);
            }
            continue;
        }
        if(key == "vars"){
            auto value = it.value();
            if (value.is_array()) {
                for(const auto& v : value)
                    vars_.push_back(v);
            }
            continue;
        }

        if (key.size() > 2) {
            const uint8_t* ptr = reinterpret_cast<const uint8_t*>(key.c_str());
            if (ptr[0] == 0xe2 && ptr[1] == 0x96 && ptr[2] == 0x81) {
                key = key.replace(0, 3, " ");
            }
        }
        tokens2str[it.value()] = key;
        tokens2id[key] = it.value();

    }
    
    int ret = 0;
    net_.opt.num_threads = 4;
    net_.opt.use_vulkan_compute = false;
    net_.opt.use_fp16_packed = false;
    net_.opt.use_fp16_storage = false;
    net_.opt.use_fp16_arithmetic = false;
    net_.register_custom_layer("Gather", Gather_layer_creator);
    net_.register_custom_layer("PositionalEncoding", PositionalEncoding_layer_creator);
    ret = net_.load_param((config.encoder + ".param").c_str());
    if (ret < 0)
        return -1;
    ret = net_.load_model((config.encoder + ".bin").c_str());
    if (ret < 0)
        return -1;

    return 0;
}
void SenseVoice::forward(const ncnn::Net& net, const std::vector<ncnn::Mat>& in, std::vector<ncnn::Mat>& out) {
    ncnn::Extractor ex = net.create_extractor();
    // ex.input("in0", in[0]);
    // ex.input("in3", in[1]);
    // ex.input("in1", in[2]);
    // ex.input("in2", in[3]);

    // out.resize(1);
    // ex.extract("out0", out[0]);

    for(size_t i = 0; i < in.size(); ++i){
        ex.input(("in" + std::to_string(i)).c_str(), in[i]);
    }

    for(size_t i = 0; i < out.size(); ++i){
        ex.extract(("out" + std::to_string(i)).c_str(), out[i]);
    }
}
int SenseVoice::recognize(std::vector<float>& samples, model_result_t& result) {

    std::vector<float> lfr_feat;
    apply_lfr(samples, lfr_feat, 7, 6, 80);
    apply_cmvn(mean_, vars_, lfr_feat);

    int frames_num = lfr_feat.size() / 560;
    ncnn::Mat feature = ncnn::Mat(lfr_feat.size(), (void*)lfr_feat.data()).reshape(560, frames_num).clone();
    ncnn::Mat text_norm = ncnn::Mat(1);
    ncnn::Mat lang_query = ncnn::Mat(1);
    text_norm.fill(0.f);//"woitn"
    lang_query.fill(15.f);//"auto"

    ncnn::Mat mask = ncnn::Mat(1, feature.h + 4);
    mask.fill(1.f);

    std::vector<ncnn::Mat> logits(1);
    //forward(net_, { feature, mask, text_norm, lang_query }, logits);
    forward(net_, { feature, text_norm, lang_query, mask}, logits);

    std::vector<int> token_out;
    std::vector<int> timestamps;

    decode_token(logits[0], token_out, timestamps);
    std::string text;
    for (size_t i = 4; i < token_out.size(); ++i) {
        text.append(tokens2str.at(token_out[i]));
    }
    result.tokens = std::move(token_out);
    result.text = std::move(text);


    return 0;
}


