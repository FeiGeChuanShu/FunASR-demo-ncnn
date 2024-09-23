#ifndef PUNCT_H_
#define PUNCT_H_
#include <net.h>
#include "model.h"
#include"nlohmann/json.hpp"

using json = nlohmann::json;

typedef struct _punct_type {
    int unk_id = 272726;
    int underline_id = 1;
    int comma_id = 2;
    int dot_id = 3;
    int quest_id = 4;
    int pause_id = 5;
    int num_punct = 6;
}punct_type_t;

typedef struct _punct_config {
    int split_size = 20;
    int cache_pop_trigger_limit = 200;
    punct_type_t punct_type;
}punct_config_t;
class Punct : public ASRModel
{
public:
    Punct() = default;
    ~Punct();
    int init(const model_config_t& config) override;
    void forward(const ncnn::Net& net, const std::vector<ncnn::Mat>& in, std::vector<ncnn::Mat>& out) override;
    int add_punct(const std::string& text, const punct_config_t& config, model_result_t& result);
private:
    ncnn::Net net_;
    json token_json_;
};


#endif
