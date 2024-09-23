#ifndef MODEL_H_
#define MODEL_H_
#include <string>
#include <vector>
#include <unordered_map>
typedef struct _model_config {
    std::string encoder;
    std::string decoder;
    std::string token;
}model_config_t;

typedef struct _model_result {
    std::string text;
    std::vector<int> tokens;
}model_result_t;
class ASRModel
{
public:
    ASRModel() = default;
    virtual ~ASRModel() {};
    virtual int init(const model_config_t& config) = 0;
    virtual void forward(const ncnn::Net& net, const std::vector<ncnn::Mat>& in, std::vector<ncnn::Mat>& out) = 0;
protected:
    std::unordered_map<int, std::string> tokens2str;
    std::unordered_map<std::string, int> tokens2id;
};


#endif
