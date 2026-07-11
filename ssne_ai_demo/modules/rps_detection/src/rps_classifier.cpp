/*
 * @Filename: rps_classifier.cpp
 * @Description: 剪刀石头布手势分类器实现
 */

#include <assert.h>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <cmath>
#include "common.hpp"
#include "utils.hpp"

namespace utils {

/**
 * @brief 对 4 维 softmax 概率数组取 argmax
 * @param probs    已经过 Softmax 的 4 维概率
 * @param out_conf 输出最高类别的概率值
 */
GestureClass ArgmaxProbs(const float probs[4], float* out_conf) {
    int   best_idx  = 0;
    float best_prob = probs[0];
    for (int i = 1; i < 4; i++) {
        if (probs[i] > best_prob) {
            best_prob = probs[i];
            best_idx  = i;
        }
    }
    if (out_conf) *out_conf = best_prob;
    return static_cast<GestureClass>(best_idx);
}

} // namespace utils


void RPSCLASSIFIER::RunSingleFrameInference(ssne_tensor_t* img, float out_probs[4]) {
    uint8_t* curr = static_cast<uint8_t*>(get_data(*img));
    if (has_prev_frame) {
        for (int i = 0; i < frame_pixels; i++) {
            int d = (int)curr[i] - (int)prev_frame_buf[i];
            curr[i] = (uint8_t)(d < 0 ? -d : d);
        }
    } else {
        memset(curr, 0, frame_pixels);
    }
    memcpy(prev_frame_buf, curr, frame_pixels);
    has_prev_frame = true;

    int ret = RunAiPreprocessPipe(pipe_offline, *img, inputs[0]);
    if (ret != 0) {
        for (int i = 0; i < 4; i++) out_probs[i] = 0.25f;
        return;
    }
    if (ssne_inference(model_id, 1, inputs)) {
        fprintf(stderr, "[ERROR] RPS ssne_inference failed!\n");
        for (int i = 0; i < 4; i++) out_probs[i] = 0.25f;
        return;
    }
    int getout_ret = ssne_getoutput(model_id, 1, outputs);
    if (getout_ret != 0 || get_data(outputs[0]) == nullptr) {
        fprintf(stderr, "[ERROR] RPS ssne_getoutput failed! ret=%d\n", getout_ret);
        for (int i = 0; i < 4; i++) out_probs[i] = 0.25f;
        return;
    }
    const float* raw = static_cast<const float*>(get_data(outputs[0]));
    float max_v = raw[0];
    for (int i = 1; i < 4; i++) if (raw[i] > max_v) max_v = raw[i];
    float sum = 0.f;
    for (int i = 0; i < 4; i++) { out_probs[i] = expf(raw[i] - max_v); sum += out_probs[i]; }
    for (int i = 0; i < 4; i++) out_probs[i] /= sum;
}

/**
 * @brief 根据人类手势返回 AI 获胜的反制手势
 */
GestureClass RPSCLASSIFIER::GetCounterMove(GestureClass human_gesture) {
    switch (human_gesture) {
        case GestureClass::ROCK:     return GestureClass::PAPER;
        case GestureClass::PAPER:    return GestureClass::SCISSORS;
        case GestureClass::SCISSORS: return GestureClass::ROCK;
        default:                     return GestureClass::IDLE;
    }
}

void RPSCLASSIFIER::UpdateStateMachine(GestureClass current_gesture,
                                       float        current_confidence,
                                       RpsResult*   result) {
    state_frame_count++;

    switch (game_state) {

    case GameState::IDLE: {
        int non_idle_streak = temporal_buffer.ConsecutiveNonIdle(WIND_UP_FRAMES);
        if (non_idle_streak >= WIND_UP_FRAMES) {
            if (RuntimeLogAtLeast(RuntimeLogMode::VERIFY)) {
                printf("[RPS] Wind-up detected! Switching to WIND_UP.\n");
            }
            game_state        = GameState::WIND_UP;
            state_frame_count = 0;
        }
        result->game_state    = GameState::IDLE;
        result->human_gesture = GestureClass::IDLE;
        result->ai_counter    = GestureClass::IDLE;
        result->confidence    = 0.f;
        result->is_locked     = false;
        break;
    }

    case GameState::WIND_UP: {
        float voted_conf = 0.f;
        GestureClass voted = temporal_buffer.GetWeightedVote(WIND_UP_FRAMES, &voted_conf);
        if (voted == GestureClass::IDLE) {
            voted      = current_gesture;
            voted_conf = current_confidence;
        }
        locked_prediction = voted;
        if (RuntimeLogAtLeast(RuntimeLogMode::VERIFY)) {
            printf("[RPS] WIND_UP -> PREDICTED: human=%s (conf=%.2f)\n",
                   GESTURE_NAMES[static_cast<int>(voted)], voted_conf);
        }
        game_state        = GameState::PREDICTED;
        state_frame_count = 0;

        result->human_gesture = locked_prediction;
        result->ai_counter    = GetCounterMove(locked_prediction);
        result->confidence    = voted_conf;
        result->game_state    = GameState::PREDICTED;
        result->is_locked     = true;
        break;
    }

    case GameState::PREDICTED: {
        result->human_gesture = locked_prediction;
        result->ai_counter    = GetCounterMove(locked_prediction);
        result->confidence    = current_confidence;
        result->game_state    = GameState::PREDICTED;
        result->is_locked     = true;

        if (state_frame_count >= DISPLAY_HOLD_FRAMES) {
            if (RuntimeLogAtLeast(RuntimeLogMode::VERIFY)) {
                printf("[RPS] PREDICTED -> DISPLAY\n");
            }
            game_state        = GameState::DISPLAY;
            state_frame_count = 0;
        }
        break;
    }

    case GameState::DISPLAY: {
        result->human_gesture = locked_prediction;
        result->ai_counter    = GetCounterMove(locked_prediction);
        result->confidence    = current_confidence;
        result->game_state    = GameState::DISPLAY;
        result->is_locked     = true;

        int idle_streak = temporal_buffer.ConsecutiveIdle(IDLE_RESET_FRAMES);
        if (idle_streak >= IDLE_RESET_FRAMES) {
            if (RuntimeLogAtLeast(RuntimeLogMode::VERIFY)) {
                printf("[RPS] DISPLAY -> IDLE (player reset hand)\n");
            }
            game_state        = GameState::IDLE;
            state_frame_count = 0;
            locked_prediction = GestureClass::IDLE;
        }
        break;
    }

    default:
        break;
    }
}


void RPSCLASSIFIER::Initialize(std::string&        model_path,
                                std::array<int, 2>* in_img_shape,
                                std::array<int, 2>* in_model_shape) {
    img_shape   = *in_img_shape;    // [640, 640]
    model_shape = *in_model_shape;  // [640, 480]

    game_state        = GameState::IDLE;
    state_frame_count = 0;
    locked_prediction = GestureClass::IDLE;

    frame_pixels   = img_shape[0] * img_shape[1];
    prev_frame_buf = new uint8_t[frame_pixels]();  
    diff_buf       = new uint8_t[frame_pixels]();
    has_prev_frame = false;
    printf("[RPSCLASSIFIER] Frame diff buffers allocated (%d pixels)\n", frame_pixels);

    char* model_path_cstr = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(model_path_cstr, SSNE_STATIC_ALLOC);
    printf("[RPSCLASSIFIER] Model loaded, id=%d\n", model_id);

    uint32_t mw = static_cast<uint32_t>(model_shape[0]);  
    uint32_t mh = static_cast<uint32_t>(model_shape[1]);  
    inputs[0] = create_tensor(mw, mh, SSNE_Y_8, SSNE_BUF_AI);

    memset(&outputs[0], 0, sizeof(ssne_tensor_t));
    printf("[RPSCLASSIFIER] Initialized: crop=%dx%d -> model=%dx%d\n",
           img_shape[0], img_shape[1], model_shape[0], model_shape[1]);
}

void RPSCLASSIFIER::Predict(ssne_tensor_t* img_in, RpsResult* result) {
    float probs[4] = {0.f, 0.f, 0.f, 0.f};
    RunSingleFrameInference(img_in, probs);

    float        frame_conf    = 0.f;
    GestureClass frame_gesture = utils::ArgmaxProbs(probs, &frame_conf);

    const float CONF_THRESHOLD = 0.40f;
    if (frame_conf < CONF_THRESHOLD) {
        frame_gesture = GestureClass::IDLE;
        frame_conf    = probs[static_cast<int>(GestureClass::IDLE)];
    }

    temporal_buffer.Push(frame_gesture, frame_conf);
    UpdateStateMachine(frame_gesture, frame_conf, result);
}

void RPSCLASSIFIER::Release() {
    if (prev_frame_buf) { delete[] prev_frame_buf; prev_frame_buf = nullptr; }
    if (diff_buf)       { delete[] diff_buf;       diff_buf       = nullptr; }

    release_tensor(inputs[0]);

    // NOTE: outputs[0] 由 ssne_getoutput 填充，其 data 指向模型内部 buffer，
    // 不应由 release_tensor 释放。ssne_release() 会统一释放模型资源。
    outputs[0].data = nullptr;

    ReleaseAIPreprocessPipe(pipe_offline);
    printf("[RPSCLASSIFIER] Resources released.\n");
}
