#ifndef VAD_H_
#define VAD_H_
#include <vector>
#include <string>
#include <stdint.h>
#include <limits>
#include "net.h"
typedef struct _speech {
    int start;
    int end;
    _speech(int start = -1, int end = -1) :start(start), end(end) {
    }
    _speech& operator=(const _speech& s) {
        start = s.start;
        end = s.end;
        return *this;
    }
}speech_t;


class Vad
{
public:
    Vad() = default;
    ~Vad();
    int init(const std::string& model_name, int model_version = 0);
    int segment(const std::vector<float>& samples, std::vector<speech_t>& speechs);

private:
    float forward_v5(const std::vector<float>& samples);
    float forward_v4(const std::vector<float>& samples);
    using ForwardFunc = float (Vad::*)(const std::vector<float>&);
    ForwardFunc forward_func;
private:
    float threshold_ = 0.5f;
    float sampling_rate_ = 16000.f;
    int min_speech_duration_ms_ = 250;
    float max_speech_duration_s_ = std::numeric_limits<float>::infinity();
    int min_silence_duration_ms_ = 100;
    int speech_pad_ms_ = 30;
    int window_size_samples_ = 512;

    float min_speech_samples_;
    float speech_pad_samples_;
    float max_speech_samples_;
    float min_silence_samples_;
    float min_silence_samples_at_max_speech_;
    
private:
    ncnn::Net net_;

    //v4
    ncnn::Mat h_;
    ncnn::Mat c_;
    
    //v5
    ncnn::Mat state_;
    std::vector<float> context_;
};


#endif
