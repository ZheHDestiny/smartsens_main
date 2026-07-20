/*
 * @Filename: focus_tracker.cpp
 * @Description: Shared focus tracking implementation.
 */

#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

FocusTrackingConfig::FocusTrackingConfig()
    : width(360),
      height(240),
      target_w(72),
      target_h(72),
      search_radius(28),
      search_step(3),
      sample_step(4),
      lost_frame_limit(8),
      npu_interval_frames(15),
      template_lr(0.08f),
      response_threshold(0.58f),
      motion_alpha(0.35f),
      position_smooth_alpha(0.65f),
      template_update_threshold(0.68f),
      min_focus_score_for_update(3.0f),
      mode(FocusTrackingMode::NO_NPU_TRACKER),
      npu_model_path("./app_assets/models/focus_mobilenet.m1model") {}

FocusTargetState::FocusTargetState() {
    Clear();
}

void FocusTargetState::Clear() {
    locked = false;
    x = 0;
    y = 0;
    w = 0;
    h = 0;
    cx = 0.0f;
    cy = 0.0f;
    vx = 0.0f;
    vy = 0.0f;
    confidence = 0.0f;
    focus_score = 0.0f;
    age = 0;
    lost_frames = 0;
}

FocusTracker::FocusTracker() : initialized_(false) {}

void FocusTracker::Initialize(const FocusTrackingConfig& config) {
    cfg_ = config;
    cfg_.target_w = std::max(16, std::min(cfg_.target_w, cfg_.width));
    cfg_.target_h = std::max(16, std::min(cfg_.target_h, cfg_.height));
    cfg_.search_step = std::max(1, cfg_.search_step);
    cfg_.sample_step = std::max(1, cfg_.sample_step);
    cfg_.position_smooth_alpha = std::max(0.0f, std::min(1.0f, cfg_.position_smooth_alpha));
    cfg_.template_update_threshold = std::max(0.0f, std::min(1.0f, cfg_.template_update_threshold));
    cfg_.min_focus_score_for_update = std::max(0.0f, cfg_.min_focus_score_for_update);
    templ_.assign((size_t)cfg_.target_w * (size_t)cfg_.target_h, 0);
    state_.Clear();
    initialized_ = true;
}

void FocusTracker::Reset() {
    state_.Clear();
    std::fill(templ_.begin(), templ_.end(), 0);
}

bool FocusTracker::SetTarget(const uint8_t* frame, const FocusTargetState& target) {
    if (!initialized_ || frame == nullptr || target.w <= 0 || target.h <= 0) return false;

    state_ = target;
    state_.w = cfg_.target_w;
    state_.h = cfg_.target_h;
    state_.x = target.x;
    state_.y = target.y;
    ClampRoi(&state_.x, &state_.y);
    state_.cx = (float)state_.x + 0.5f * (float)state_.w;
    state_.cy = (float)state_.y + 0.5f * (float)state_.h;
    state_.vx = 0.0f;
    state_.vy = 0.0f;
    state_.locked = true;
    state_.lost_frames = 0;
    state_.age = 1;
    if (state_.confidence <= 0.0f) state_.confidence = 0.75f;
    state_.focus_score = FocusScore(frame, state_.x, state_.y, state_.w, state_.h);
    ExtractTemplate(frame, state_.x, state_.y);
    return true;
}

bool FocusTracker::HasTarget() const {
    return initialized_ && state_.locked;
}

bool FocusTracker::Update(const uint8_t* frame, FocusTargetState* out_state) {
    if (!initialized_ || frame == nullptr) return false;

    if (!state_.locked) {
        if (!SelectTarget(frame)) {
            if (out_state) *out_state = state_;
            return false;
        }
    } else {
        int pred_x = state_.x + (int)std::round(state_.vx);
        int pred_y = state_.y + (int)std::round(state_.vy);
        ClampRoi(&pred_x, &pred_y);

        int best_x = pred_x;
        int best_y = pred_y;
        float best_score = -1.0f;

        for (int dy = -cfg_.search_radius; dy <= cfg_.search_radius; dy += cfg_.search_step) {
            for (int dx = -cfg_.search_radius; dx <= cfg_.search_radius; dx += cfg_.search_step) {
                int x = pred_x + dx;
                int y = pred_y + dy;
                ClampRoi(&x, &y);
                float score = MatchTemplateSad(frame, x, y);
                if (score > best_score) {
                    best_score = score;
                    best_x = x;
                    best_y = y;
                }
            }
        }

        if (best_score < cfg_.response_threshold) {
            state_.lost_frames++;
            state_.confidence = std::max(0.0f, state_.confidence * 0.82f);
            if (state_.lost_frames > cfg_.lost_frame_limit) {
                Reset();
                SelectTarget(frame);
            }
        } else {
            float old_cx = state_.cx;
            float old_cy = state_.cy;
            if (state_.age > 1) {
                float keep = 1.0f - cfg_.position_smooth_alpha;
                state_.x = (int)std::round(keep * (float)state_.x +
                                           cfg_.position_smooth_alpha * (float)best_x);
                state_.y = (int)std::round(keep * (float)state_.y +
                                           cfg_.position_smooth_alpha * (float)best_y);
                ClampRoi(&state_.x, &state_.y);
            } else {
                state_.x = best_x;
                state_.y = best_y;
            }
            state_.w = cfg_.target_w;
            state_.h = cfg_.target_h;
            state_.cx = (float)state_.x + 0.5f * (float)cfg_.target_w;
            state_.cy = (float)state_.y + 0.5f * (float)cfg_.target_h;
            state_.vx = cfg_.motion_alpha * (state_.cx - old_cx) + (1.0f - cfg_.motion_alpha) * state_.vx;
            state_.vy = cfg_.motion_alpha * (state_.cy - old_cy) + (1.0f - cfg_.motion_alpha) * state_.vy;
            state_.confidence = best_score;
            state_.focus_score = FocusScore(frame, state_.x, state_.y, state_.w, state_.h);
            state_.age++;
            state_.lost_frames = 0;
            if (best_score >= cfg_.template_update_threshold &&
                state_.focus_score >= cfg_.min_focus_score_for_update) {
                UpdateTemplate(frame, state_.x, state_.y);
            }
        }
    }

    if (out_state) *out_state = state_;
    return state_.locked;
}

bool FocusTracker::SelectTarget(const uint8_t* frame) {
    const int stride_x = std::max(8, cfg_.target_w / 4);
    const int stride_y = std::max(8, cfg_.target_h / 4);
    const float center_x = 0.5f * (float)cfg_.width;
    const float center_y = 0.5f * (float)cfg_.height;
    const float norm = center_x * center_x + center_y * center_y;

    int best_x = 0;
    int best_y = 0;
    float best_score = -1.0f;

    for (int y = 0; y <= cfg_.height - cfg_.target_h; y += stride_y) {
        for (int x = 0; x <= cfg_.width - cfg_.target_w; x += stride_x) {
            float texture = TextureScore(frame, x, y, cfg_.target_w, cfg_.target_h);
            float cx = (float)x + 0.5f * (float)cfg_.target_w;
            float cy = (float)y + 0.5f * (float)cfg_.target_h;
            float dx = cx - center_x;
            float dy = cy - center_y;
            float center_prior = 1.0f - 0.35f * ((dx * dx + dy * dy) / std::max(1.0f, norm));
            float score = texture * center_prior;
            if (score > best_score) {
                best_score = score;
                best_x = x;
                best_y = y;
            }
        }
    }

    if (best_score <= 1.5f) return false;

    state_.locked = true;
    state_.x = best_x;
    state_.y = best_y;
    state_.w = cfg_.target_w;
    state_.h = cfg_.target_h;
    state_.cx = (float)best_x + 0.5f * (float)cfg_.target_w;
    state_.cy = (float)best_y + 0.5f * (float)cfg_.target_h;
    state_.vx = 0.0f;
    state_.vy = 0.0f;
    state_.confidence = 0.72f;
    state_.focus_score = FocusScore(frame, state_.x, state_.y, state_.w, state_.h);
    state_.age = 1;
    state_.lost_frames = 0;
    ExtractTemplate(frame, state_.x, state_.y);
    return true;
}

void FocusTracker::ExtractTemplate(const uint8_t* frame, int x, int y) {
    ClampRoi(&x, &y);
    for (int row = 0; row < cfg_.target_h; row++) {
        const uint8_t* src = frame + (y + row) * cfg_.width + x;
        uint8_t* dst = templ_.data() + row * cfg_.target_w;
        std::memcpy(dst, src, (size_t)cfg_.target_w);
    }
}

void FocusTracker::UpdateTemplate(const uint8_t* frame, int x, int y) {
    ClampRoi(&x, &y);
    const float keep = 1.0f - cfg_.template_lr;
    for (int row = 0; row < cfg_.target_h; row++) {
        const uint8_t* src = frame + (y + row) * cfg_.width + x;
        uint8_t* dst = templ_.data() + row * cfg_.target_w;
        for (int col = 0; col < cfg_.target_w; col++) {
            float v = keep * (float)dst[col] + cfg_.template_lr * (float)src[col];
            dst[col] = (uint8_t)std::max(0, std::min(255, (int)(v + 0.5f)));
        }
    }
}

float FocusTracker::MatchTemplateSad(const uint8_t* frame, int x, int y) const {
    int count = 0;
    int sad = 0;
    for (int row = 0; row < cfg_.target_h; row += cfg_.sample_step) {
        const uint8_t* src = frame + (y + row) * cfg_.width + x;
        const uint8_t* ref = templ_.data() + row * cfg_.target_w;
        for (int col = 0; col < cfg_.target_w; col += cfg_.sample_step) {
            sad += std::abs((int)src[col] - (int)ref[col]);
            count++;
        }
    }
    if (count == 0) return 0.0f;
    float mean_abs_diff = (float)sad / (255.0f * (float)count);
    return std::max(0.0f, 1.0f - mean_abs_diff);
}

float FocusTracker::TextureScore(const uint8_t* frame, int x, int y, int w, int h) const {
    int count = 0;
    int grad_sum = 0;
    for (int row = y + 1; row < y + h - 1; row += cfg_.sample_step) {
        for (int col = x + 1; col < x + w - 1; col += cfg_.sample_step) {
            int gx = std::abs((int)frame[row * cfg_.width + col + 1] -
                              (int)frame[row * cfg_.width + col - 1]);
            int gy = std::abs((int)frame[(row + 1) * cfg_.width + col] -
                              (int)frame[(row - 1) * cfg_.width + col]);
            grad_sum += gx + gy;
            count++;
        }
    }
    return count > 0 ? (float)grad_sum / (float)count : 0.0f;
}

float FocusTracker::FocusScore(const uint8_t* frame, int x, int y, int w, int h) const {
    int count = 0;
    int lap_sum = 0;
    for (int row = y + 1; row < y + h - 1; row += cfg_.sample_step) {
        for (int col = x + 1; col < x + w - 1; col += cfg_.sample_step) {
            int center = (int)frame[row * cfg_.width + col] * 4;
            int around = (int)frame[row * cfg_.width + col - 1] +
                         (int)frame[row * cfg_.width + col + 1] +
                         (int)frame[(row - 1) * cfg_.width + col] +
                         (int)frame[(row + 1) * cfg_.width + col];
            lap_sum += std::abs(center - around);
            count++;
        }
    }
    return count > 0 ? (float)lap_sum / (float)count : 0.0f;
}

void FocusTracker::ClampRoi(int* x, int* y) const {
    if (*x < 0) *x = 0;
    if (*y < 0) *y = 0;
    int max_x = std::max(0, cfg_.width - cfg_.target_w);
    int max_y = std::max(0, cfg_.height - cfg_.target_h);
    if (*x > max_x) *x = max_x;
    if (*y > max_y) *y = max_y;
}

MobileNetFocusSelector::MobileNetFocusSelector() : ready_(false) {
    std::memset(inputs, 0, sizeof(inputs));
    std::memset(outputs, 0, sizeof(outputs));
}

void MobileNetFocusSelector::Initialize(const std::string& model_path,
                                        std::array<int, 2>* in_img_shape,
                                        std::array<int, 2>* in_model_shape) {
    if (in_img_shape) img_shape = *in_img_shape;
    if (in_model_shape) model_shape = *in_model_shape;

    pipe_offline = GetAIPreprocessPipe();

    if (model_path.empty() || model_shape[0] <= 0 || model_shape[1] <= 0) {
        ready_ = false;
        if (RuntimeLogEnabled()) {
            printf("[FOCUS_NPU] MobileNet selector reserved, invalid model path or shape.\n");
        }
        return;
    }

    model_id = ssne_loadmodel(const_cast<char*>(model_path.c_str()), SSNE_STATIC_ALLOC);
    if (model_id == 0) {
        ready_ = false;
        if (RuntimeLogEnabled()) {
            printf("[FOCUS_NPU] MobileNet model load failed: %s\n", model_path.c_str());
        }
        return;
    }

    SetNormalize(pipe_offline, model_id);
    inputs[0] = create_tensor(model_shape[0], model_shape[1], SSNE_Y_8, SSNE_BUF_AI);
    std::memset(outputs, 0, sizeof(outputs));
    ready_ = true;
    if (RuntimeLogEnabled()) {
        printf("[FOCUS_NPU] MobileNet focus selector loaded, id=%d\n", model_id);
    }
}

bool MobileNetFocusSelector::Predict(ssne_tensor_t* img_in, FocusTargetState* target) {
    if (target) target->Clear();
    if (!ready_ || img_in == nullptr || target == nullptr) return false;

    int ret = RunAiPreprocessPipe(pipe_offline, *img_in, inputs[0]);
    if (ret != 0) {
        if (RuntimeLogAtLeast(RuntimeLogMode::VERIFY)) {
            printf("[FOCUS_NPU] preprocess failed ret=%d\n", ret);
        }
        return false;
    }

    ret = ssne_inference(model_id, 1, inputs);
    if (ret != 0) {
        if (RuntimeLogAtLeast(RuntimeLogMode::VERIFY)) {
            printf("[FOCUS_NPU] inference failed ret=%d\n", ret);
        }
        return false;
    }

    /*
     * Reserved output contract for the planned MobileNet focus model:
     *   outputs[0]: 4 floats, normalized xywh in [0, 1]
     *   outputs[1]: 1 float, target confidence
     * The exact tensor count/shape should be adjusted after exporting the
     * final m1model, so the current interface intentionally returns false.
     */
    return false;
}

void MobileNetFocusSelector::Release() {
    if (inputs[0].data != nullptr) {
        release_tensor(inputs[0]);
    }
    memset(&inputs[0], 0, sizeof(inputs[0]));
    outputs[0].data = nullptr;
    outputs[1].data = nullptr;
    if (pipe_offline != nullptr) {
        ReleaseAIPreprocessPipe(pipe_offline);
        pipe_offline = nullptr;
    }
    model_id = 0;
    ready_ = false;
}

bool MobileNetFocusSelector::IsReady() const {
    return ready_;
}
