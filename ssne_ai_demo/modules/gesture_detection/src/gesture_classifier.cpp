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
    int hist[256] = { 0 };

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

    int lut[256] = { 0 };
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
    if (!model_ready || model_id < 0 || img == nullptr ||
        get_data(*img) == nullptr ||
        get_mem_size(*img) < static_cast<size_t>(img_shape[0] * img_shape[1])) {
        for (int i = 0; i < 6; i++) out_probs[i] = 1.0f / 6.0f;
        return;
    }

    uint8_t* curr = static_cast<uint8_t*>(get_data(*img));

    if (use_clahe) {
        ApplyClippedHistogramEqualization(curr, img_shape[0], img_shape[1]);
    }

    int ret = RunAiPreprocessPipe(pipe_offline, *img, inputs[0]);
    if (ret != 0) {
        fprintf(stderr, "[ERROR] GESTURE RunAiPreprocessPipe failed! ret=%d\n", ret);
        for (int i = 0; i < 6; i++) out_probs[i] = 1.0f / 6.0f;
        return;
    }

    if (ssne_inference(static_cast<uint16_t>(model_id), 1, inputs)) {
        fprintf(stderr, "[ERROR] GESTURE ssne_inference failed!\n");
        for (int i = 0; i < 6; i++) out_probs[i] = 1.0f / 6.0f;
        return;
    }

    int getout_ret = ssne_getoutput(
        static_cast<uint16_t>(model_id), 1, outputs);
    if (getout_ret != 0 || get_data(outputs[0]) == nullptr ||
        get_data_type(outputs[0]) != SSNE_FLOAT32 ||
        get_mem_size(outputs[0]) < 6 * sizeof(float)) {
        fprintf(stderr, "[ERROR] GESTURE ssne_getoutput failed! ret=%d\n", getout_ret);
        for (int i = 0; i < 6; i++) out_probs[i] = 1.0f / 6.0f;
        return;
    }
    const float* raw = static_cast<const float*>(get_data(outputs[0]));

    if (!std::isfinite(raw[0])) {
        for (int i = 0; i < 6; i++) out_probs[i] = 1.0f / 6.0f;
        return;
    }
    float max_v = raw[0];
    for (int i = 1; i < 6; i++) {
        if (!std::isfinite(raw[i])) {
            for (int j = 0; j < 6; j++) out_probs[j] = 1.0f / 6.0f;
            return;
        }
        if (raw[i] > max_v) max_v = raw[i];
    }

    float sum = 0.f;
    for (int i = 0; i < 6; i++) {
        out_probs[i] = std::exp(raw[i] - max_v);
        sum += out_probs[i];
    }
    if (!std::isfinite(sum) || sum <= 0.0f) {
        for (int i = 0; i < 6; i++) out_probs[i] = 1.0f / 6.0f;
        return;
    }
    for (int i = 0; i < 6; i++) {
        out_probs[i] /= sum;
    }
}

void HANDGESTURECLASSIFIER::Initialize(std::string& model_path, std::array<int, 2>* in_img_shape, std::array<int, 2>* in_model_shape) {
    model_ready = false;
    img_shape = *in_img_shape;
    model_shape = *in_model_shape;

    char* model_path_cstr = const_cast<char*>(model_path.c_str());
    const int loaded_model_id = ssne_loadmodel(model_path_cstr, SSNE_STATIC_ALLOC);
    if (loaded_model_id < 0) {
        fprintf(stderr, "[HANDGESTURECLASSIFIER] ERROR: Failed to load model from %s, error code %d\n", model_path_cstr, loaded_model_id);
        model_id = -1;
        return;
    }
    model_id = loaded_model_id;
    printf("[HANDGESTURECLASSIFIER] Model loaded, id=%d\n", model_id);

    uint32_t mw = static_cast<uint32_t>(model_shape[0]);
    uint32_t mh = static_cast<uint32_t>(model_shape[1]);
    inputs[0] = create_tensor(mw, mh, SSNE_Y_8, SSNE_BUF_AI);
    memset(&outputs[0], 0, sizeof(ssne_tensor_t));
    model_ready = pipe_offline != nullptr && get_data(inputs[0]) != nullptr &&
                  get_mem_size(inputs[0]) >= static_cast<size_t>(mw * mh);
    if (!model_ready) {
        fprintf(stderr,
                "[HANDGESTURECLASSIFIER] ERROR: input tensor/pipe initialization failed.\n");
    }

    // pipe_offline 已在 common.hpp 中通过 GetAIPreprocessPipe() 默认初始化，
    // 此处无需手动创建，参考 scrfd_gray.cpp / rps_classifier.cpp 风格

    printf("[HANDGESTURECLASSIFIER] Initialized: crop=%dx%d -> model=%dx%d\n",
        img_shape[0], img_shape[1], model_shape[0], model_shape[1]);
}

void HANDGESTURECLASSIFIER::Predict(ssne_tensor_t* img_in, HandGestureResult* result, bool use_clahe) {
    if (result == nullptr) return;
    float probs[6] = {0.f};
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
    if (get_data(inputs[0]) != nullptr) {
        release_tensor(inputs[0]);
        inputs[0].data = nullptr;
    }
    outputs[0].data = nullptr;
    if (pipe_offline != nullptr) {
        ReleaseAIPreprocessPipe(pipe_offline);
        pipe_offline = nullptr;
    }
    model_ready = false;
    model_id = -1;
    printf("[HANDGESTURECLASSIFIER] Resources released.\n");
}
