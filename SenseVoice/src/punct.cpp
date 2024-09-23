#include "punct.h"
#include "custom_op.h"
#include <fstream>
#include <sstream>
Punct::~Punct() {
    net_.clear();
}
static ncnn::Layer* Gather_layer_creator(void*) {
    return new Gather();
}
static ncnn::Layer* PositionalEncoding_layer_creator(void*) {
    return new PositionalEncoding();
}
static std::vector<std::string> split_string(const std::string& str) {
    std::vector<std::string> tokens;
    std::istringstream iss(str);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}
static std::vector<std::string> code_mix_split_words(const std::string& text) {

    std::vector<std::string> words_split;
    std::vector<std::string> segs = split_string(text);
    for (size_t i = 0; i < segs.size(); ++i) {
        std::string cur_word = "";

        int start_idx = 0;
        int end_idx = 0;
        for (size_t j = start_idx; j < segs[i].size(); ++j) {
            if ((segs[i][j] & 0x80)) {
                if (cur_word.size() > 0)
                {
                    words_split.push_back(cur_word);
                    cur_word = "";
                    start_idx = j;
                }
                else
                    end_idx = j;
            }
            else {
                if (end_idx > start_idx) {
                    words_split.push_back(segs[i].substr(start_idx, end_idx - start_idx + 1));
                    start_idx = end_idx;
                }
                cur_word += segs[i][j];
                start_idx++;
            }
        }
        if (cur_word.size() > 0)
            words_split.push_back(cur_word);
        if (end_idx > start_idx) {
            words_split.push_back(segs[i].substr(start_idx, end_idx - start_idx + 1));
            start_idx = end_idx;
        }
    }

    std::vector<std::string> words;
    for (size_t i = 0; i < words_split.size(); ++i) {
        if (words_split[i][0] & 0x80) {
            int ch_num = words_split[i].size() / 3;
            for (int j = 0; j < ch_num; ++j) {
                words.push_back(words_split[i].substr(j * 3, 3));
            }
        }
        else {
            words.push_back(words_split[i]);
        }
    }

    return words;
}

template<typename T>
std::vector<std::vector<T>> split_to_mini_sentence(const std::vector<T>& words, int word_limit) {
    if (words.size() <= word_limit)
        return { words };

    std::vector<std::vector<T>> sentences;
    int length = words.size();
    int sentence_len = std::ceil((length + word_limit - 1) / word_limit);
    sentences.resize(sentence_len);

    for (int i = 0; i < sentence_len; ++i) {
        int start_idx = i * word_limit;
        int end_idx = (i + 1) * word_limit;
        if (end_idx > length)
            end_idx = length;

        sentences[i].assign(words.begin() + start_idx, words.begin() + end_idx);

    }

    return sentences;

}


int Punct::init(const model_config_t& config) {
    std::ifstream f(config.token);
    if (!f.is_open()) {
        return -1;
    }
    token_json_ = json::parse(f);

    for (auto it = token_json_.begin(); it != token_json_.end(); ++it) {
        auto key = it.key();
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
void Punct::forward(const ncnn::Net& net, const std::vector<ncnn::Mat>& in, std::vector<ncnn::Mat>& out) {
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in[0]);
    ex.input("in1", in[1]);

    out.resize(1);
    ex.extract("out0", out[0]);
    
}
int Punct::add_punct(const std::string& text, const punct_config_t& config, model_result_t& result) {
    std::vector<std::string> words = code_mix_split_words(text);
    
    std::vector<int> split_text_id;
    split_text_id.reserve(words.size());

    for(auto& word : words){
        std::transform(word.begin(), word.end(), word.begin(),
            [](unsigned char c) { return std::tolower(c); });
        if (tokens2id.find(word) != tokens2id.end())
            split_text_id.push_back(tokens2id[word]);
        else
            split_text_id.push_back(config.punct_type.unk_id);
    }
    
    std::vector<std::vector<std::string>> mini_sentences;
    mini_sentences = split_to_mini_sentence(words, config.split_size);
    
    std::vector<std::vector<int>> mini_sentences_id;
    mini_sentences_id = split_to_mini_sentence(split_text_id, config.split_size);

    std::vector<int> cache_sent_id;
    std::vector<std::string> cache_sent;
    int cache_pop_trigger_limit = 200;
    std::vector<float> input_data;
    std::vector<std::string> new_mini_sentence;
    for (size_t i = 0; i < mini_sentences.size(); ++i) {
        auto mini_sentence = mini_sentences[i];
        auto mini_sentence_id = mini_sentences_id[i];
    
        mini_sentence.insert(mini_sentence.begin(), cache_sent.begin(), cache_sent.end());
        mini_sentence_id.insert(mini_sentence_id.begin(), cache_sent_id.begin(), cache_sent_id.end());
    
        int len = mini_sentence.size();
        ncnn::Mat mask1 = ncnn::Mat(1, len);
        mask1.fill(1.f);
    
        input_data.resize(len);
        std::transform(mini_sentence_id.begin(), mini_sentence_id.end(), input_data.begin(), [](float v) {return 1.f * v; });
        ncnn::Mat feat = ncnn::Mat(len, input_data.data()).clone();
    
        std::vector<ncnn::Mat> out;
        forward(net_, { feat, mask1 }, out);

        std::vector<int> punctuations;
        punctuations.reserve(len);
    
        const float* ptr = (float*)out[0].data;
        auto num_punct = out[0].w;
        for (int k = 0; k != len; ++k, ptr += num_punct) {
            auto idx = static_cast<int>(std::distance(ptr, std::max_element(ptr, ptr + num_punct)));
            punctuations.push_back(idx);
        }
    
        if (i < mini_sentences.size() - 1) {
            int sentence_end = -1;
            int last_comma_index = -1;
            for(int j = punctuations.size() - 2; j >= 1; --j){
                if (punctuations[j] == config.punct_type.dot_id || punctuations[j] == config.punct_type.quest_id) {
                    sentence_end = j;
                    break;
                }
    
                if (last_comma_index < 0 && punctuations[j] == config.punct_type.comma_id)
                    last_comma_index = j;
            }
            if (sentence_end < 0 && mini_sentence.size() > cache_pop_trigger_limit &&
                last_comma_index >= 0) {
                sentence_end = last_comma_index;
                punctuations[sentence_end] = config.punct_type.dot_id;
            }
            cache_sent.assign(mini_sentence.begin() + sentence_end + 1, mini_sentence.end());
            cache_sent_id.assign(mini_sentence_id.begin() + sentence_end + 1, mini_sentence_id.end());
            mini_sentence.assign(mini_sentence.begin(), mini_sentence.begin() + sentence_end + 1);
            punctuations.assign(punctuations.begin(), punctuations.begin() + sentence_end + 1);
        }
    
        std::vector<std::string> words_with_punc;
        for (size_t k = 0; k < mini_sentence.size(); ++k) {
            if (k > 0 && !(mini_sentence[k][0] & 0x80) &&
                !(mini_sentence[k - 1][0] & 0x80)) {
                words_with_punc.push_back(" ");
            }
            words_with_punc.push_back(mini_sentence[k]);
            if (punctuations[k] != config.punct_type.underline_id)
                words_with_punc.push_back(tokens2str[config.punct_type.unk_id + punctuations[k]]);
        }
        new_mini_sentence.insert(new_mini_sentence.end(), words_with_punc.begin(), words_with_punc.end());
    
        if (i == mini_sentences.size() - 1) {
            if (new_mini_sentence.back() == tokens2str[config.punct_type.unk_id + config.punct_type.pause_id] ||
                new_mini_sentence.back() == tokens2str[config.punct_type.unk_id + config.punct_type.comma_id])
                new_mini_sentence.back() = tokens2str[config.punct_type.unk_id + config.punct_type.dot_id];
            if (new_mini_sentence.back() != tokens2str[config.punct_type.unk_id + config.punct_type.dot_id] &&
                new_mini_sentence.back() != tokens2str[config.punct_type.unk_id + config.punct_type.quest_id])
                new_mini_sentence.push_back(tokens2str[config.punct_type.unk_id + config.punct_type.dot_id]);
        }
    }
    
    std::string text_with_punct;
    for(const auto& word : new_mini_sentence){
        text_with_punct.append(word);
    }
    
    result.text = std::move(text_with_punct);
    
    return 0;
}