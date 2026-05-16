/*
 * @Filename: gesture_classifier.cpp
 * @Description: 手势分类器实现
 */

#include <assert.h>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm> 
#include "common.hpp"
#include "utils.hpp"

static void ApplyClippedHistogramEqualization(uint8_t* img, int width, int height) {
    int total_pixels = width * height;
    int hist[256] = {0};

    for (int i = 0; i < total_pixels; i++) {
        hist[img[i]]++;
    }

    int clip_limit = std::max(1, total_pixels / 256 * 3); 
    int excess = 0;
    
    for (int i = 0; i < 256; i++) {
        if (hist[i] > clip_limit) {
            excess += (hist[i] - clip_limit);
            hist[i] = clip_limit;
        }
    }

    int bin_inc = excess / 256;
    int upper = excess % 256;
    for (int i = 0; i < 256; i++) {
        hist[i] += bin_inc;
        if (i < upper) hist[i]++;
    }

    int lut[256] = {0};
    int sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += hist[i];
        lut[i] = std::min(255, std::max(0, (sum * 255) / total_pixels));
    }

    for (int i = 0; i < total_pixels; i++) {
        img[i] = lut[img[i]];
    }
}


void HANDGESTURECLASSIFIER::RunSingleFrameInference(ssne_tensor_t* img, float out_probs[6], bool use_clahe) {
    uint8_t* curr = static_cast<uint8_t*>(get_data(*img));

    if (use_clahe) {
        ApplyClippedHistogramEqualization(curr, img_shape[0], img_shape[1]);
    }

    int ret = RunAiPreprocessPipe(pipe_offline, *img, inputs[0]);
    if (ret != 0) {
        for (int i = 0; i < 6; i++) out_probs[i] = 1.0f / 6.0f;
        return;
    }

    if (ssne_inference(model_id, 1, inputs)) {
        fprintf(stderr, "[ERROR] GESTURE ssne_inference failed!\n");
        for (int i = 0; i < 6; i++) out_probs[i] = 1.0f / 6.0f;
        return;
    }

    int getout_ret = ssne_getoutput(model_id, 1, outputs);
    if (getout_ret != 0 || get_data(outputs[0]) == nullptr) {
        fprintf(stderr, "[ERROR] GESTURE ssne_getoutput failed! ret=%d\n", getout_ret);
        for (int i = 0; i < 6; i++) out_probs[i] = 1.0f / 6.0f;
        return;
    }
    const float* raw = static_cast<const float*>(get_data(outputs[0]));

    float max_v = raw[0];
    for (int i = 1; i < 6; i++) {
        if (raw[i] > max_v) max_v = raw[i];
    }

    float sum = 0.f;
    for (int i = 0; i < 6; i++) {
        out_probs[i] = std::exp(raw[i] - max_v);
        sum += out_probs[i];
    }
    for (int i = 0; i < 6; i++) {
        out_probs[i] /= sum;
    }
}

void HANDGESTURECLASSIFIER::Initialize(std::string& model_path, std::array<int, 2>* in_img_shape, std::array<int, 2>* in_model_shape) {
    img_shape = *in_img_shape;
    model_shape = *in_model_shape;

    char* model_path_cstr = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(model_path_cstr, SSNE_STATIC_ALLOC);
    printf("[HANDGESTURECLASSIFIER] Model loaded, id=%d\n", model_id);

    uint32_t mw = static_cast<uint32_t>(model_shape[0]);
    uint32_t mh = static_cast<uint32_t>(model_shape[1]);
    inputs[0] = create_tensor(mw, mh, SSNE_Y_8, SSNE_BUF_AI);
    memset(&outputs[0], 0, sizeof(ssne_tensor_t));
    printf("[HANDGESTURECLASSIFIER] Initialized: crop=%dx%d -> model=%dx%d\n",
        img_shape[0], img_shape[1], model_shape[0], model_shape[1]);
}

void HANDGESTURECLASSIFIER::Predict(ssne_tensor_t* img_in, HandGestureResult* result, bool use_clahe) {
    float probs[6] = { 0.f };
    RunSingleFrameInference(img_in, probs, use_clahe);

    for (int i = 0; i < 6; i++) {
        result->all_probs[i] = probs[i];
    }

    float conf = 0.f;
    HandGestureClass gesture = utils::ArgmaxProbsHandGesture(probs, &conf);

    temporal_buffer.Push(gesture, conf);
    float smoothed_conf = 0.f;
    gesture = temporal_buffer.GetWeightedVote(&smoothed_conf);

    result->gesture = gesture;
    result->confidence = smoothed_conf;
}

void HANDGESTURECLASSIFIER::Release() {
    release_tensor(inputs[0]);

    // NOTE: outputs[0] 由 ssne_getoutput 填充，其 data 指向模型内部 buffer，
    // 不应由 release_tensor 释放。ssne_release() 会统一释放模型资源。
    outputs[0].data = nullptr;

    ReleaseAIPreprocessPipe(pipe_offline);
    printf("[HANDGESTURECLASSIFIER] Resources released.\n");
}