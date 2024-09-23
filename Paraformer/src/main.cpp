#include "paraformer.h"
#include "punct.h"
#include "wav_read.h"
#include "vad.h"
#include "fbank_feat.h"
#include "thread_pool.h"

#define N_THREADS 4


typedef struct _recognizer {
    std::shared_ptr<FbankFeature> feat_extractor;
    std::shared_ptr<Paraformer> asr;
    std::shared_ptr<Punct> punct;
    _recognizer(std::shared_ptr<FbankFeature> feat_extractor,
        std::shared_ptr<Paraformer> asr,
        std::shared_ptr<Punct> punct) :feat_extractor(feat_extractor),
        asr(asr), punct(punct) {}
}recognizer_t;

typedef struct _asr_result {
    float start;
    float end;
    std::string text;
}asr_result_t;


int asr_worker_thread(const std::vector<recognizer_t>& recognizers,
    const std::vector<speech_t>& speeches, int n_samples, int n_threads,
    int start,int end, int sampling_rate, const std::vector<float>& samples, 
    punct_config_t& punct_config, std::vector<asr_result_t>& asr_results) {

    for (int i = start; i < end; ++i) {
        auto recognizer = recognizers[i];
        auto speech = speeches[i];
        auto fbank_feat = recognizer.feat_extractor;

        std::vector<float> feat;
        fbank_feat->extract_feat(samples.data(), speech.start, speech.end, feat);

        //rec
        model_result_t model_result;
        recognizer.asr->recognize(feat, model_result);

        //add punct
        recognizer.punct->add_punct(model_result.text, punct_config, model_result);

        asr_result_t asr_result;
        asr_result.start = static_cast<float>(speech.start) / sampling_rate;
        asr_result.end = static_cast<float>(speech.end) / sampling_rate;
        asr_result.text = std::move(model_result.text);
        asr_results[i] = std::move(asr_result);

        fprintf(stderr, "%d in [%d:%d] done\n", i, start, end);
    }
    
    return 0;
}


int main(int argc, char** argv) {
    if (argc < 3){
        fprintf(stderr, "./sensevoice_demo xxx.wav ../models/ \n");
        return -1;
    }

    Vad vad_segmentor;
    if (vad_segmentor.init(std::string(argv[2]) + "/vad5", 1) < 0) {
        fprintf(stderr, "vad init failed\n");
        return -1;
    }
    fprintf(stderr, "1.vad init done\n");

    std::shared_ptr<Paraformer> sense_voice(new Paraformer());
    model_config_t asr_model_config;
    asr_model_config.encoder = std::string(argv[2]) + "/paraformer-small-encoder";
    asr_model_config.decoder = std::string(argv[2]) + "/paraformer-small-decoder";
    asr_model_config.token = std::string(argv[2]) + "/paraformer-small.json";
    if (sense_voice->init(asr_model_config) < 0) {
        fprintf(stderr, "asr init failed\n");
        return -1;
    }
    fprintf(stderr, "2.asr init done\n");

    std::shared_ptr<Punct> punct(new Punct());
    model_config_t punct_model_config;
    punct_model_config.encoder = std::string(argv[2]) + "/punct";
    punct_model_config.token = std::string(argv[2]) + "/punct.json";
    if (punct->init(punct_model_config) < 0) {
        fprintf(stderr, "punct init failed\n");
        return -1;
    }
    fprintf(stderr, "3.punct init done\n");

    //load 
    std::vector<float> wav_data;
    if (load_wav(argv[1], wav_data) < 0) {
        fprintf(stderr, "load wav failed\n");
        return -1;
    }
    fprintf(stderr, "4.load wav done\n");

    //do vad
    std::vector<speech_t> speeches;
    vad_segmentor.segment(wav_data, speeches);

    fprintf(stderr, "5.vad segment done\n");

    int speeches_num = speeches.size();
    fprintf(stderr, "speech segment count: %d\n", speeches_num);

    //create feat extractor
    feature_config_t feat_config;
    feat_config.feat_dim = 80;
    feat_config.normalize = false;
    feat_config.sampling_rate = 16000;

    std::vector<recognizer_t> recognizers;
    for (int i = 0; i < speeches_num; ++i) {
        recognizers.emplace_back(std::make_shared<FbankFeature>(feat_config), sense_voice, punct);
    }
    
    fprintf(stderr, "starting recognize\n");
    ThreadPool pool(N_THREADS);

    std::vector<asr_result_t> asr_results;
    asr_results.resize(speeches_num);
    int per_thread_sample = (speeches_num + N_THREADS - 1) / N_THREADS;

    punct_config_t punct_config;
    punct_config.cache_pop_trigger_limit = 200;
    punct_config.split_size = 20;

    std::queue<std::future<int>> futures;
    
    for (int i = 0; i < N_THREADS; ++i) {
        int start = i * per_thread_sample;
        int end = std::min(start + per_thread_sample, speeches_num);
        futures.push(pool.enqueue(asr_worker_thread, std::cref(recognizers), 
            std::cref(speeches), speeches_num, N_THREADS, start, end, feat_config.sampling_rate,
            std::ref(wav_data), std::ref(punct_config), std::ref(asr_results)));
    }
    

    while (!futures.empty()){
        auto ans = futures.front().get();
        futures.pop();
    }
    fprintf(stderr, "6.recognize done\n");

    fprintf(stderr, "output result: \n");
    for (const auto& asr_res : asr_results) {
        fprintf(stderr, "[%.3f : %.3f]: %s\n", asr_res.start, asr_res.end, asr_res.text.c_str());
    }

    return 0;
}
