/*
 * @Filename: emotion_classifier.cpp
 * @Description: 情感分类器实现（4类，输入 224x224 灰度图，无归一化）
 */

#include <assert.h>
#include <iostream>
#include <cstdio>
#include <cmath>

#include "common.hpp"
#include "utils.hpp"

void EMOTIONCLASSIFIER::RunSingleFrameInference(ssne_tensor_t* img, float out_probs[4]) {
    int ret = RunAiPreprocessPipe(pipe_offline, *img, inputs[0]);
    if (ret != 0) {
        for (int i = 0; i < 4; i++) out_probs[i] = 1.0f / 4.0f;
        return;
    }

    if (ssne_inference(model_id, 1, inputs)) {
        fprintf(stderr, "[ERROR] EMOTION ssne_inference failed!\n");
        for (int i = 0; i < 4; i++) out_probs[i] = 1.0f / 4.0f;
        return;
    }

    int getout_ret = ssne_getoutput(model_id, 1, outputs);
    if (getout_ret != 0 || get_data(outputs[0]) == nullptr) {
        fprintf(stderr, "[ERROR] EMOTION ssne_getoutput failed! ret=%d\n", getout_ret);
        for (int i = 0; i < 4; i++) out_probs[i] = 1.0f / 4.0f;
        return;
    }
    const float* raw = static_cast<const float*>(get_data(outputs[0]));

    float max_v = raw[0];
    for (int i = 1; i < 4; i++) {
        if (raw[i] > max_v) max_v = raw[i];
    }

    float sum = 0.f;
    for (int i = 0; i < 4; i++) {
        out_probs[i] = std::exp(raw[i] - max_v);
        sum += out_probs[i];
    }
    for (int i = 0; i < 4; i++) {
        out_probs[i] /= sum;
    }
}

void EMOTIONCLASSIFIER::Initialize(std::string& model_path, std::array<int, 2>* in_img_shape, std::array<int, 2>* in_model_shape) {
    img_shape = *in_img_shape;  
    model_shape = *in_model_shape;  

    frame_pixels = img_shape[0] * img_shape[1];
    prev_frame_buf = new uint8_t[frame_pixels]();
    diff_buf = new uint8_t[frame_pixels]();
    has_prev_frame = false;

    char* model_path_cstr = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(model_path_cstr, SSNE_STATIC_ALLOC);
    printf("[EMOTIONCLASSIFIER] Model loaded, id=%d\n", model_id);

    uint32_t mw = static_cast<uint32_t>(model_shape[0]);
    uint32_t mh = static_cast<uint32_t>(model_shape[1]);
    inputs[0] = create_tensor(mw, mh, SSNE_Y_8, SSNE_BUF_AI);
    memset(&outputs[0], 0, sizeof(ssne_tensor_t));
    printf("[EMOTIONCLASSIFIER] Initialized: crop=%dx%d -> model=%dx%d\n",
        img_shape[0], img_shape[1], model_shape[0], model_shape[1]);
}

void EMOTIONCLASSIFIER::Predict(ssne_tensor_t* img_in, EmotionResult* result) {
    float probs[4] = { 0.f };
    RunSingleFrameInference(img_in, probs);

    float conf = 0.f;
    EmotionClass emotion = utils::ArgmaxProbsEmotion(probs, &conf);

    const float CONF_THRESHOLD = 0.0f;
    if (conf < CONF_THRESHOLD) {
        emotion = EmotionClass::NEUTRAL;
        conf = probs[static_cast<int>(EmotionClass::NEUTRAL)];
    }

    temporal_buffer.Push(emotion, conf);
    result->emotion = temporal_buffer.GetWeightedVote(&result->confidence);
}

void EMOTIONCLASSIFIER::Release() {
    if (prev_frame_buf) { delete[] prev_frame_buf; prev_frame_buf = nullptr; }
    if (diff_buf) { delete[] diff_buf;       diff_buf = nullptr; }

    release_tensor(inputs[0]);

    // NOTE: outputs[0] 由 ssne_getoutput 填充，其 data 指向模型内部 buffer，
    // 不应由 release_tensor 释放。ssne_release() 会统一释放模型资源。
    outputs[0].data = nullptr;

    ReleaseAIPreprocessPipe(pipe_offline);
    printf("[EMOTIONCLASSIFIER] Resources released.\n");
}