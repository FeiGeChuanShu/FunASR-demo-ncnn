#ifndef FBANK_FEAT_H_
#define FBANK_FEAT_H_
#include "kaldi-native-fbank/csrc/online-feature.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"

typedef struct _feature_config {
    int sampling_rate = 16000;
    int feat_dim = 80;
    bool normalize = false;
}feature_config_t;

class FbankFeature
{
public:
    FbankFeature(const feature_config_t& config);
    
    int extract_feat(const std::vector<float>& wav_data, std::vector<float>& fbank_data);
    int extract_feat(const float* wav_data, int start_idx, int data_len, std::vector<float>& fbank_data);
private:
    knf::FbankOptions opts_;
    std::shared_ptr<knf::OnlineFbank> fbank_;
    feature_config_t feat_config_;
};



#endif

