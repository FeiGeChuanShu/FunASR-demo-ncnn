#include "vad.h"
#include <cmath>
#include <chrono>
#include "thread_pool.h"
Vad::~Vad() {
    net_.clear();
    context_.clear();
}

int Vad::init(const std::string& model_name, int model_version){

    net_.opt.use_vulkan_compute = false;
    net_.opt.use_fp16_packed = false;
    net_.opt.use_fp16_storage = false;
    net_.opt.use_fp16_arithmetic = false;

    if (net_.load_param((model_name + ".param").c_str()) < 0) {
        return -1;
    }
    if (net_.load_model((model_name + ".bin").c_str()) < 0) {
        return -1;
    }
    if (model_version == 0) {
        //v4
        h_ = ncnn::Mat(64, 1, 2);
        c_ = ncnn::Mat(64, 1, 2);
        h_.fill(0.f);
        c_.fill(0.f);
        forward_func = &Vad::forward_v4;
    }
    else {
        //v5
        state_ = ncnn::Mat(128, 1, 2);
        state_.fill(0.f);
        context_.resize(64, 0.f);
        forward_func = &Vad::forward_v5;
    }

    min_speech_samples_ = sampling_rate_ * min_speech_duration_ms_ / 1000.f;
    speech_pad_samples_ = sampling_rate_ * speech_pad_ms_ / 1000.f;
    max_speech_samples_ = sampling_rate_ * max_speech_duration_s_ - window_size_samples_ - 2 * speech_pad_samples_;
    min_silence_samples_ = sampling_rate_ * min_silence_duration_ms_ / 1000.f;
    min_silence_samples_at_max_speech_ = sampling_rate_ * 98 / 1000.f;

    return 0;
}

int Vad::segment(const std::vector<float>& samples, std::vector<speech_t>& speeches) {
    int current_start_sample = 0;
    int samples_len = samples.size();

    int n_samples = (samples_len + window_size_samples_ - 1) / window_size_samples_;
    std::vector<float> speech_probs;
    speech_probs.resize(n_samples);

    std::vector<float> chunk;
    chunk.resize(window_size_samples_);
    for (int i = 0; i < n_samples; ++i) {
        int start_idx = i * window_size_samples_;
        int end_idx = std::min(start_idx + window_size_samples_, samples_len);

        std::copy(samples.begin() + start_idx, samples.begin() + end_idx, chunk.begin());

        if(end_idx - start_idx < window_size_samples_)
            std::fill(chunk.begin()+(end_idx - start_idx), chunk.end(), 0.f);

        float speech_score = (this->*forward_func)(chunk);
        speech_probs[i] = speech_score;
    }

    bool triggered = false;
    float neg_threshold = threshold_ - 0.15;
    int temp_end = 0;
    int prev_end = 0;
    int next_start = 0;

    speech_t current_speech;
    for (size_t i = 0; i < speech_probs.size(); ++i) {
        if (speech_probs[i] >= threshold_ && temp_end > 0) {
            temp_end = 0;
            if (next_start < prev_end)
                next_start = window_size_samples_ * i;
        }

        if (speech_probs[i] >= threshold_ && !triggered) {
            triggered = true;
            current_speech.start = window_size_samples_ * i;
            continue;
        }

        if (triggered && (window_size_samples_ * i) - current_speech.start > max_speech_samples_) {
            if (prev_end > 0) {
                current_speech.end = prev_end;
                speeches.emplace_back(current_speech);
                current_speech = speech_t();
                if (next_start < prev_end)
                    triggered = false;
                else
                    current_speech.start = next_start;
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
            }
            else {
                current_speech.end = window_size_samples_ * i;

                speeches.emplace_back(current_speech);
                current_speech = speech_t();
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
                continue;
            }
        }

        if (speech_probs[i] < neg_threshold && triggered) {
            if (temp_end == 0)
                temp_end = window_size_samples_ * i;
            if ((window_size_samples_ * i) - temp_end > min_silence_samples_at_max_speech_)
                prev_end = temp_end;
            if ((window_size_samples_ * i) - temp_end < min_silence_samples_)
                continue;
            else {
                current_speech.end = temp_end;
                if (current_speech.end - current_speech.start > min_speech_samples_)
                    speeches.emplace_back(current_speech);
                current_speech = speech_t();
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
                continue;
            }
        }

    }

    if (current_speech.start > 0 && (samples_len - current_speech.start) > min_speech_samples_) {
        current_speech.end = samples_len;
        speeches.emplace_back(current_speech.start, current_speech.end);
    }

    for (size_t i = 0; i < speeches.size(); ++i) {
        if (i == 0)
            speeches[i].start = static_cast<int>(std::max(0.f, speeches[i].start - speech_pad_samples_));
        if (i != speeches.size() - 1) {
            float silence_duration = speeches[i + 1].start - speeches[i].end;
            if (silence_duration < 2 * speech_pad_samples_) {
                speeches[i].end += static_cast<int>(std::floor(silence_duration / 2));
                speeches[i + 1].start = static_cast<int>(std::max(0.f, speeches[i + 1].start - std::floor(silence_duration / 2)));
            }
            else {
                speeches[i].end = static_cast<int>(std::min((float)samples_len, speeches[i].end + speech_pad_samples_));
                speeches[i + 1].start = static_cast<int>(std::max(0.f, speeches[i + 1].start - speech_pad_samples_));
            }
        }
        else {
            speeches[i].end = static_cast<int>(std::min((float)samples_len, speeches[i].end + speech_pad_samples_));
        }
    }

    return 0;
}
float Vad::forward_v4(const std::vector<float>& samples) {
    
    ncnn::Mat in = ncnn::Mat(512, 1, 1, (float*)samples.data()).clone();
    
    ncnn::Extractor ex = net_.create_extractor();
    ex.input("/x.1", in);
    ex.input("/h.1", h_);
    ex.input("/c.1", c_);

    ncnn::Mat out;
    ex.extract("out", out);
    ex.extract("/h0", h_);
    ex.extract("/c0", c_);

    return out[0];
}
float Vad::forward_v5(const std::vector<float>& samples) {
    ncnn::Mat in(512 + 64, 1, 1);

    auto ptr = (float*)in.data;
    std::copy(samples.begin(), samples.end(), ptr + 64);
    std::copy(context_.begin(), context_.end(), ptr);

    ncnn::Extractor ex = net_.create_extractor();

    ncnn::Mat out;
    ex.input("in0", in);
    ex.input("in1", state_);

    ex.extract("out0", state_);
    ex.extract("out1", out);

    std::copy(samples.end() - 64, samples.end(), context_.begin());

    return out[0];
}