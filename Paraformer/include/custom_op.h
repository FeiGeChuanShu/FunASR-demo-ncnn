#ifndef CUSTOM_H_
#define CUSTIM_H_
#include "net.h"
#include <cmath>
class PositionalEncoding : public ncnn::Layer {
public:
    PositionalEncoding() {
        one_blob_only = true;
        support_inplace = false;
    }
    virtual int load_param(const ncnn::ParamDict& pd) {
        feat_dim = pd.get(0, 0);
        if (feat_dim == 0) {
            return -100;
        }
        scale = sqrtf((float)feat_dim);
        return 0;
    }

    virtual int forward(const ncnn::Mat& bottom_blob,
        ncnn::Mat& top_blob,
        const ncnn::Option& opt) const {
        int outw = bottom_blob.w;
        int outh = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;

        top_blob.create(outw, outh, elemsize, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        float log_timescale_increment = -std::log(10000.f) / (256 / 2.f - 1.f);
        int outw_half = outw / 2;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < outh; ++y) {
            float* out_ptr = top_blob.row(y);
            const float* ptr = bottom_blob.row(y);
            int offset = y + 1;
            for (int x = 0; x < outw_half; ++x) {
                float inv_timescale = offset * expf(x * log_timescale_increment);

                float sin_d = sinf(inv_timescale);
                float cos_d = cosf(inv_timescale);
                out_ptr[x] = ptr[x] * scale + sin_d;
                out_ptr[x + outw_half] = ptr[x + outw_half] * scale + cos_d;
            }
        }
        return 0;
    }
private:
    float scale;
    int feat_dim;
};


class Gather : public ncnn::Layer {
public:
    Gather() {
        one_blob_only = false;
        support_inplace = false;
    }
    virtual int load_param(const ncnn::ParamDict& pd) {
        axis = pd.get(0, 0);

        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs,
        const ncnn::Option& opt) const {
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const ncnn::Mat& indices = bottom_blobs[1];
        int dims = bottom_blob.dims;
        int indices_dims = indices.dims;
        size_t elemsize = bottom_blob.elemsize;
        int positive_axis = axis < 0 ? dims + axis : axis;
        ncnn::Mat& top_blob = top_blobs[0];
        if (indices.h > 1) {
            fprintf(stderr, "ncnn Gather layer indices dims not eual 1\n");
            return -1;
        }

        const float* indices_ptr = indices;

        if (dims == 2 && positive_axis == 0 && indices_dims == 1) {
            int w = bottom_blob.w;
            top_blob.create(w, indices.w, elemsize, opt.blob_allocator);
            if (top_blob.empty()) {
                return -100;
            }
            for (int i = 0; i < indices.w; i++) {
                const int selected = (int)(indices_ptr[i] + 0.5);
                memcpy(top_blob.row(i), bottom_blob.row(selected), w * elemsize);
            }

            return 0;
        }


        return 0;
    }

private:
    int axis;
};

#endif
