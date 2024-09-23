#include "paraformer.h"
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

    for (int i = 0; i < am_score.h; ++i) {
        auto new_token = static_cast<int>(std::distance(am_score.row(i),
            std::max_element(am_score.row(i), am_score.row(i) + am_score.w)));
        if(new_token == 2) break;
        if (new_token != 0) {
            token_out.emplace_back(new_token);
            //fprintf(stderr, "%d\n", new_token);
        }
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

static void tail_process_fn(ncnn::Mat& hidden, ncnn::Mat& alphas, ncnn::Mat& mask,
    ncnn::Mat& pre_acoustic_embeds, int32_t& pre_token_length) {
    int t = hidden.h;
    int d = hidden.w;
    float tail_threshold = 0.45;
    ncnn::Mat mask_ = ncnn::Mat(t + 1, 1);
    ncnn::Mat alphas_ = ncnn::Mat(t + 1, 1);

    float token_num = 0;
    float* m_ptr = (float*)mask_.data;
    float* a_ptr = (float*)alphas_.data;
    const float* mask_ptr = (float*)mask.data;
    const float* alphas_ptr = (float*)alphas.data;
    m_ptr[0] = (1.f - mask_ptr[0]) * tail_threshold;
    m_ptr[t] = (mask_ptr[t - 1]) * tail_threshold;
    a_ptr[0] = m_ptr[0] + alphas_ptr[0];
    a_ptr[t] = m_ptr[t];
    token_num = a_ptr[0] + a_ptr[t];
    for (int32_t i = 1; i < t; ++i) {
        m_ptr[i] = (mask_ptr[i - 1] - mask_ptr[i]) * tail_threshold;
        a_ptr[i] = m_ptr[i] + alphas_ptr[i];
        token_num += a_ptr[i];
    }
    int32_t token_num_floor = std::floor(token_num);
    pre_token_length = token_num_floor;
    if (token_num_floor == 0)
        return;

    std::vector<float> hidden_data((t + 1) * d, 0);
    std::copy((float*)hidden.data, (float*)hidden.data + t * d, hidden_data.begin());

    double integrate = 0.f;
    std::vector<double> list_fires;
    std::vector<double> frame(d);
    std::vector<std::vector<double>> list_frames;
    for (int32_t i = 0; i < t + 1; ++i) {
        double alpha = alphas_[i];
        double distribution_completion = 1.f - integrate;
        integrate += alpha;
        list_fires.emplace_back(integrate);

        double cur = alpha;
        bool fire_place = integrate >= 1.0;
        if (fire_place) {
            integrate = (float)integrate - 1.f;
            cur = distribution_completion;
        }
        double remainds = alpha - cur;

        for (int32_t j = 0; j < d; ++j) {
            frame[j] += cur * hidden_data[i * d + j];
        }
        list_frames.emplace_back(frame);
        if (fire_place) {
            for (int32_t j = 0; j < d; ++j)
                frame[j] = hidden_data[i * d + j] * remainds;
        }
    }

    std::vector<std::vector<double>> frame_fires;
    for (int32_t i = 0; i < t + 1; ++i) {
        if (list_fires[i] >= 1.0) {
            frame_fires.emplace_back(list_frames[i]);
        }
    }

    int32_t max_label_len = static_cast<int32_t>(frame_fires.size());
    pre_acoustic_embeds.create(d, max_label_len);
    for (int32_t i = 0; i < max_label_len; ++i) {
        std::copy(frame_fires[i].begin(), frame_fires[i].end(), pre_acoustic_embeds.row(i));
    }
}
static std::string sentence_postprocess(const std::vector<int>& token_out, const std::unordered_map<int, std::string>& tokens2str){
    bool is_combining = false;
    std::string text;
    for (size_t i = 0; i < token_out.size(); ++i) {
        auto word = tokens2str.at(token_out[i]);
        if ((word.back() != '@') || (word.size() > 2 && *(word.end() - 1) != '@')) 
        {
            if (!(word[0] & 0x80)) {
                if (is_combining) {
                    is_combining = false;
                    text.append(word);
                }
                else {
                    text.append(" ");
                    text.append(word);
                }
            }
            else {
                if (i > 0) {
                    auto pre_word = tokens2str.at(token_out[i - 1]);
                    if (!(pre_word[0] & 0x80)) {
                        text.append(" ");
                    }
                }
                is_combining = false;
                text.append(word);
            }
        }
        else {
            word = word.erase(word.size() - 2);
            if (is_combining) {
                text.append(word);
            }
            else {
                text.append(" ");
                text.append(word);
                is_combining = true;
            }
        }
    }
    return text;
}

Paraformer::~Paraformer() {
    encoder_.clear();
    decoder_.clear();
    mean_.clear();
    vars_.clear();
}

int Paraformer::init(const model_config_t& config) {
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
    encoder_.opt.num_threads = 4;
    encoder_.opt.use_vulkan_compute = false;
    encoder_.opt.use_fp16_packed = false;
    encoder_.opt.use_fp16_storage = false;
    encoder_.opt.use_fp16_arithmetic = false;
    encoder_.register_custom_layer("Gather", Gather_layer_creator);
    encoder_.register_custom_layer("PositionalEncoding", PositionalEncoding_layer_creator);
    ret = encoder_.load_param((config.encoder + ".param").c_str());
    if (ret < 0)
        return -1;
    ret = encoder_.load_model((config.encoder + ".bin").c_str());
    if (ret < 0)
        return -1;

    decoder_.opt.num_threads = 4;
    decoder_.opt.use_vulkan_compute = false;
    decoder_.opt.use_fp16_packed = false;
    decoder_.opt.use_fp16_storage = false;
    decoder_.opt.use_fp16_arithmetic = false;
    ret = decoder_.load_param((config.decoder + ".param").c_str());
    if (ret < 0)
        return -1;
    ret = decoder_.load_model((config.decoder + ".bin").c_str());
    if (ret < 0)
        return -1;

    return 0;
}
void Paraformer::forward(const ncnn::Net& net, const std::vector<ncnn::Mat>& in, std::vector<ncnn::Mat>& out) {
    ncnn::Extractor ex = net.create_extractor();

	for(size_t i = 0; i < in.size(); ++i){
        ex.input(("in" + std::to_string(i)).c_str(), in[i]);
    }

    for(size_t i = 0; i < out.size(); ++i){
        ex.extract(("out" + std::to_string(i)).c_str(), out[i]);
    }
    
}
int Paraformer::recognize(std::vector<float>& samples, model_result_t& result) {

    std::vector<float> lfr_feat;
    apply_lfr(samples, lfr_feat, 7, 6, 80);
    apply_cmvn(mean_, vars_, lfr_feat);

    int frames_num = lfr_feat.size() / 560;
    ncnn::Mat feature = ncnn::Mat(lfr_feat.size(), (void*)lfr_feat.data()).reshape(560, frames_num).clone();
    ncnn::Mat enc_mask = ncnn::Mat(1, feature.h);
    enc_mask.fill(1.f);

    std::vector<ncnn::Mat> encoder_outs(2);
    forward(encoder_, { feature, enc_mask}, encoder_outs);

    int num_tokens = 0;
    ncnn::Mat acoustic_embeds;
    tail_process_fn(encoder_outs[0], encoder_outs[1], enc_mask, acoustic_embeds, num_tokens);
    if (num_tokens == 0)
        return 0;

    ncnn::Mat dec_mask = ncnn::Mat(1, num_tokens);
    dec_mask.fill(1.f);
    std::vector<ncnn::Mat> decoder_outs(1);
    forward(decoder_, {encoder_outs[0], acoustic_embeds, dec_mask}, decoder_outs);

    std::vector<int> token_out;
    std::vector<int> timestamps;
    decode_token(decoder_outs[0], token_out, timestamps);

    std::string text = sentence_postprocess(token_out, tokens2str);

    result.tokens = std::move(token_out);
    result.text = std::move(text);


    return 0;
}

