#ifndef SENSEVOICE_H_
#define SENSEVOICE_H_
#include "net.h"
#include"nlohmann/json.hpp"
#include "model.h"

using json = nlohmann::json;
class SenseVoice : public ASRModel
{
public:
    SenseVoice() = default;
    ~SenseVoice();
    int init(const model_config_t& config) override;
    void forward(const ncnn::Net& net, const std::vector<ncnn::Mat>& in, std::vector<ncnn::Mat>& out) override;

    int recognize(std::vector<float>& samples, model_result_t& result);
private:
    ncnn::Net net_;
    json token_json_;
    std::vector<float> mean_;
    std::vector<float> vars_;
};




#endif
