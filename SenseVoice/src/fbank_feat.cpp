#include "fbank_feat.h"
#include <algorithm>

FbankFeature::FbankFeature(const feature_config_t& config):feat_config_(config){
    opts_.frame_opts.samp_freq = config.sampling_rate;
    opts_.mel_opts.num_bins = config.feat_dim;
    //fprintf(stderr, "%s\n", opts_.ToString().c_str());

    fbank_ = std::make_shared<knf::OnlineFbank>(opts_);
}

int FbankFeature::extract_feat(const std::vector<float>& wav_data, std::vector<float>& fbank_data) {

    float sampling_rate = feat_config_.sampling_rate;
    if (!feat_config_.normalize) {
        std::vector<float> buf(wav_data.size());
        std::transform(wav_data.begin(), wav_data.end(), buf.begin(),
            [](float v) {return v * 32768.; });
        fbank_->AcceptWaveform(sampling_rate, buf.data(), buf.size());
    }
    else
        fbank_->AcceptWaveform(sampling_rate, wav_data.data(), wav_data.size());
    
    int32_t feat_len = fbank_->NumFramesReady();

    fbank_data.resize(feat_len * opts_.mel_opts.num_bins);
    float* ptr = fbank_data.data();
    for (int i = 0; i < feat_len; ++i) {
        auto frame = fbank_->GetFrame(i);
        std::copy(frame, frame + opts_.mel_opts.num_bins, ptr);
        ptr += opts_.mel_opts.num_bins;
    }
    
    return 0;
}
int FbankFeature::extract_feat(const float* wav_data, int start_idx, int end_idx, std::vector<float>& fbank_data) {
    
    float sampling_rate = feat_config_.sampling_rate;
    int data_len = end_idx - start_idx;
    if (!feat_config_.normalize) {
        std::vector<float> buf(data_len);
        std::transform(wav_data + start_idx, wav_data + end_idx, buf.begin(),
            [](float v) {return v * 32768.; });
        fbank_->AcceptWaveform(sampling_rate, buf.data(), buf.size());
    }
    else
        fbank_->AcceptWaveform(sampling_rate, wav_data + start_idx, data_len);

    int32_t feat_len = fbank_->NumFramesReady();

    fbank_data.resize(feat_len * opts_.mel_opts.num_bins);
    float* ptr = fbank_data.data();
    for (int i = 0; i < feat_len; ++i) {
        auto frame = fbank_->GetFrame(i);
        std::copy(frame, frame + opts_.mel_opts.num_bins, ptr);
        ptr += opts_.mel_opts.num_bins;
    }
    
    return 0;
}