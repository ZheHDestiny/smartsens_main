/*
 * @Filename: emotion_classifier.cpp
 * @Description: 情感分类器实现（4类，Y8摄像头 -> RGB 3x112x112，无归一化）
 */

#include <assert.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cstdio>
#include <cmath>

#include "common.hpp"
#include "utils.hpp"

constexpr float EmotionTemporalBuffer::EMA_TAU_MS;
constexpr float EMOTIONCLASSIFIER::CONF_ON;
constexpr float EMOTIONCLASSIFIER::MARGIN_ON;
constexpr float EMOTIONCLASSIFIER::SWITCH_MARGIN;

static_assert(static_cast<int>(EmotionClass::SURPRISE) == 0,
              "emotion model class 0 must be SURPRISE");
static_assert(static_cast<int>(EmotionClass::HAPPY) == 1,
              "emotion model class 1 must be HAPPY");
static_assert(static_cast<int>(EmotionClass::SAD) == 2,
              "emotion model class 2 must be SAD");
static_assert(static_cast<int>(EmotionClass::NEUTRAL) == 3,
              "emotion model class 3 must be NEUTRAL");

namespace {

uint64_t MonotonicMs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool IsFiniteProbability(const float probs[4]) {
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        if (!std::isfinite(probs[i]) || probs[i] < 0.0f || probs[i] > 1.0f) {
            return false;
        }
        sum += probs[i];
    }
    return std::isfinite(sum) && std::fabs(sum - 1.0f) < 1.0e-3f;
}

void FindTop2(const float probs[4], int* top1, int* top2) {
    *top1 = 0;
    *top2 = 1;
    if (probs[*top2] > probs[*top1]) std::swap(*top1, *top2);
    for (int i = 2; i < 4; i++) {
        if (probs[i] > probs[*top1]) {
            *top2 = *top1;
            *top1 = i;
        } else if (probs[i] > probs[*top2]) {
            *top2 = i;
        }
    }
}

struct PixelDistribution {
    bool valid;
    int dtype;
    float min_value;
    float max_value;
    float mean;
    float stddev;
    float low_edge_ratio;
    float high_edge_ratio;
    uint32_t histogram[16];
    uint32_t count;

    PixelDistribution()
        : valid(false), dtype(-1), min_value(0.0f), max_value(0.0f),
          mean(0.0f), stddev(0.0f), low_edge_ratio(0.0f),
          high_edge_ratio(0.0f), count(0) {
        for (int i = 0; i < 16; i++) histogram[i] = 0;
    }
};

PixelDistribution MeasureTensorDistribution(
        const ssne_tensor_t* tensor, uint32_t expected_pixels) {
    PixelDistribution stats;
    if (tensor == nullptr || get_data(*tensor) == nullptr ||
        expected_pixels == 0) {
        return stats;
    }

    stats.dtype = get_data_type(*tensor);
    const size_t bytes = get_mem_size(*tensor);
    size_t element_size = 0;
    if (stats.dtype == SSNE_UINT8 || stats.dtype == SSNE_INT8) {
        element_size = 1;
    } else if (stats.dtype == SSNE_FLOAT32) {
        element_size = sizeof(float);
    } else {
        return stats;
    }
    if (bytes < static_cast<size_t>(expected_pixels) * element_size) {
        return stats;
    }

    const uint8_t* raw_bytes =
        static_cast<const uint8_t*>(get_data(*tensor));
    const int8_t* raw_int8 =
        static_cast<const int8_t*>(get_data(*tensor));
    const float* raw_float =
        static_cast<const float*>(get_data(*tensor));
    double sum = 0.0;
    double sum_sq = 0.0;
    uint32_t low_edges = 0;
    uint32_t high_edges = 0;
    stats.min_value = std::numeric_limits<float>::max();
    stats.max_value = -std::numeric_limits<float>::max();

    for (uint32_t i = 0; i < expected_pixels; i++) {
        float value = 0.0f;
        int histogram_index = 0;
        if (stats.dtype == SSNE_UINT8) {
            value = raw_bytes[i];
            histogram_index = raw_bytes[i] >> 4;
            if (raw_bytes[i] == 0) low_edges++;
            if (raw_bytes[i] == 255) high_edges++;
        } else if (stats.dtype == SSNE_INT8) {
            value = raw_int8[i];
            histogram_index =
                (static_cast<int>(raw_int8[i]) + 128) >> 4;
            if (raw_int8[i] == -128) low_edges++;
            if (raw_int8[i] == 127) high_edges++;
        } else {
            value = raw_float[i];
            if (!std::isfinite(value)) return PixelDistribution();
            histogram_index =
                static_cast<int>(std::max(0.0f, std::min(255.0f, value))) >> 4;
            if (value <= 0.0f) low_edges++;
            if (value >= 255.0f) high_edges++;
        }
        histogram_index = std::max(0, std::min(15, histogram_index));
        stats.histogram[histogram_index]++;
        stats.min_value = std::min(stats.min_value, value);
        stats.max_value = std::max(stats.max_value, value);
        sum += value;
        sum_sq += static_cast<double>(value) * value;
    }

    stats.count = expected_pixels;
    stats.mean = static_cast<float>(sum / expected_pixels);
    const double variance = std::max(
        0.0, sum_sq / expected_pixels -
             static_cast<double>(stats.mean) * stats.mean);
    stats.stddev = static_cast<float>(std::sqrt(variance));
    stats.low_edge_ratio =
        static_cast<float>(low_edges) / expected_pixels;
    stats.high_edge_ratio =
        static_cast<float>(high_edges) / expected_pixels;
    stats.valid = true;
    return stats;
}

void PrintDistribution(const char* name, const PixelDistribution& stats) {
    if (!stats.valid) {
        printf("[EMOTION_INPUT] %s distribution unavailable\n", name);
        return;
    }
    printf("[EMOTION_INPUT] %s dtype=%d n=%u min=%.1f max=%.1f "
           "mean=%.2f std=%.2f edge_low=%.3f edge_high=%.3f\n",
           name, stats.dtype, stats.count, stats.min_value, stats.max_value,
           stats.mean, stats.stddev, stats.low_edge_ratio,
           stats.high_edge_ratio);
    printf("[EMOTION_HIST] %s 16bins=[", name);
    for (int i = 0; i < 16; i++) {
        const float ratio = stats.count > 0
            ? 100.0f * stats.histogram[i] / stats.count : 0.0f;
        printf("%s%.1f", i == 0 ? "" : " ", ratio);
    }
    printf("]%%\n");
}

}  // namespace

bool EMOTIONCLASSIFIER::RunSingleFrameInference(ssne_tensor_t* img, float out_probs[4]) {
    last_inference_failure_reason = "none";
    if (!model_ready) {
        last_inference_failure_reason = "model_not_ready";
        return false;
    }
    if (img == nullptr || get_data(*img) == nullptr) {
        last_inference_failure_reason = "invalid_input_image";
        return false;
    }
    if (out_probs == nullptr) {
        last_inference_failure_reason = "invalid_probability_buffer";
        return false;
    }

    int ret = RunAiPreprocessPipe(pipe_offline, *img, inputs[0]);
    if (ret != 0) {
        last_inference_failure_reason = "preprocess_failed";
        fprintf(stderr, "[ERROR] EMOTION preprocessing failed! ret=%d\n", ret);
        return false;
    }
    const PixelDistribution camera_distribution =
        MeasureTensorDistribution(
            img, static_cast<uint32_t>(frame_pixels));
    const PixelDistribution model_distribution =
        MeasureTensorDistribution(
            &inputs[0],
            static_cast<uint32_t>(3 * model_shape[0] * model_shape[1]));
    if (camera_distribution.valid && model_distribution.valid &&
        camera_distribution.max_value > 32.0f &&
        model_distribution.max_value <= 1.0f) {
        last_inference_failure_reason = "preprocess_range_collapsed";
        if (!preprocess_contract_logged) {
            fprintf(stderr,
                    "[ERROR] EMOTION preprocess range collapsed: "
                    "camera max=%.1f mean=%.2f -> model max=%.1f mean=%.2f\n",
                    camera_distribution.max_value, camera_distribution.mean,
                    model_distribution.max_value, model_distribution.mean);
            preprocess_contract_logged = true;
        }
        return false;
    }
    if (!preprocess_contract_logged) {
        printf("[EMOTIONCLASSIFIER] Preprocess output: %ux%u format=%u "
               "storage_dtype=%u bytes=%zu (gray replicated to RGB; "
               "model input is NCHW-compatible)\n",
               get_width(inputs[0]), get_height(inputs[0]),
               get_data_format(inputs[0]), get_data_type(inputs[0]),
               get_mem_size(inputs[0]));
        fflush(stdout);
        preprocess_contract_logged = true;
    }

    ret = ssne_inference(model_id, 1, inputs);
    if (ret != 0) {
        last_inference_failure_reason = "npu_inference_failed";
        fprintf(stderr, "[ERROR] EMOTION ssne_inference failed! ret=%d\n", ret);
        return false;
    }

    int getout_ret = ssne_getoutput(model_id, 1, outputs);
    if (getout_ret != 0 || get_data(outputs[0]) == nullptr) {
        last_inference_failure_reason = "get_output_failed";
        fprintf(stderr, "[ERROR] EMOTION ssne_getoutput failed! ret=%d\n", getout_ret);
        return false;
    }
    const float* raw = static_cast<const float*>(get_data(outputs[0]));

    for (int i = 0; i < 4; i++) {
        if (!std::isfinite(raw[i])) {
            last_inference_failure_reason = "non_finite_logit";
            fprintf(stderr, "[ERROR] EMOTION output contains NaN/Inf.\n");
            return false;
        }
        last_logits[i] = raw[i];
    }

    float max_v = raw[0];
    for (int i = 1; i < 4; i++) {
        if (raw[i] > max_v) max_v = raw[i];
    }

    float sum = 0.f;
    for (int i = 0; i < 4; i++) {
        out_probs[i] = std::exp(raw[i] - max_v);
        sum += out_probs[i];
    }
    if (!std::isfinite(sum) || sum <= 0.0f) {
        last_inference_failure_reason = "invalid_softmax_sum";
        fprintf(stderr, "[ERROR] EMOTION softmax sum is invalid.\n");
        return false;
    }
    for (int i = 0; i < 4; i++) {
        out_probs[i] /= sum;
    }
    if (!IsFiniteProbability(out_probs)) {
        last_inference_failure_reason = "invalid_probability";
        return false;
    }
    return true;
}

void EMOTIONCLASSIFIER::ResetTemporalState(uint64_t now_ms) {
    temporal_buffer.Reset();
    stable_emotion = EmotionClass::NEUTRAL;
    stable_confidence = 0.0f;
    candidate_emotion = EmotionClass::NEUTRAL;
    candidate_count = 0;
    candidate_since_ms = 0;
    last_switch_ms = now_ms;
    last_valid_ms = now_ms;
    has_prev_frame = false;
    motion_mean = 0.0f;
    motion_m2 = 0.0f;
    motion_samples = 0;
    last_debug_ms = 0;
    stats_start_ms = now_ms;
    total_frames = 0;
    accepted_frames = 0;
    inference_failures = 0;
    quality_rejections = 0;
    dark_rejections = 0;
    bright_rejections = 0;
    contrast_rejections = 0;
    motion_rejections = 0;
    uncertain_frames = 0;
    class_switches = 0;
    switch_delay_sum_ms = 0.0f;
    preprocess_contract_logged = false;
    for (int i = 0; i < 4; i++) last_logits[i] = 0.0f;
}

void EMOTIONCLASSIFIER::KeepLastStableResult(EmotionResult* result) const {
    if (result == nullptr) return;
    result->emotion = stable_emotion;
    result->confidence = stable_confidence;
}

bool EMOTIONCLASSIFIER::ImageQualityAccepted(
        const ssne_tensor_t* img, float* brightness, float* contrast,
        float* motion_score, const char** reject_reason) {
    if (brightness) *brightness = 0.0f;
    if (contrast) *contrast = 0.0f;
    if (motion_score) *motion_score = 0.0f;
    if (reject_reason) *reject_reason = "accepted";

    if (img == nullptr || get_data(*img) == nullptr || frame_pixels <= 0) {
        if (reject_reason) *reject_reason = "invalid_image";
        return false;
    }

    const uint8_t* current = static_cast<const uint8_t*>(get_data(*img));
    double sum = 0.0;
    double sum_sq = 0.0;
    uint64_t diff_sum = 0;
    for (int i = 0; i < frame_pixels; i++) {
        const uint8_t value = current[i];
        sum += value;
        sum_sq += static_cast<double>(value) * value;
        if (has_prev_frame) {
            const uint8_t diff = static_cast<uint8_t>(
                std::abs(static_cast<int>(value) -
                         static_cast<int>(prev_frame_buf[i])));
            diff_buf[i] = diff;
            diff_sum += diff;
        }
    }

    const float mean = static_cast<float>(sum / frame_pixels);
    const double variance = std::max(0.0, sum_sq / frame_pixels - mean * mean);
    const float stddev = static_cast<float>(std::sqrt(variance));
    const float motion = has_prev_frame
                       ? static_cast<float>(diff_sum) / frame_pixels : 0.0f;

    // Always advance the frame reference. A rejected frame must not make the
    // next good frame look like motion accumulated over several frame periods.
    std::memcpy(prev_frame_buf, current, static_cast<size_t>(frame_pixels));
    const bool had_previous = has_prev_frame;
    has_prev_frame = true;

    if (brightness) *brightness = mean;
    if (contrast) *contrast = stddev;
    if (motion_score) *motion_score = motion;

    if (mean < 20.0f) {
        if (reject_reason) *reject_reason = "too_dark";
        return false;
    }
    if (mean > 235.0f) {
        if (reject_reason) *reject_reason = "too_bright";
        return false;
    }
    if (stddev < 12.0f) {
        if (reject_reason) *reject_reason = "low_contrast";
        return false;
    }

    // Learn the motion limit from this board/camera stream instead of baking
    // in an uncalibrated score. The first 15 deltas form the baseline.
    bool motion_rejected = false;
    if (had_previous && motion_samples >= 15) {
        const float variance_motion =
            motion_samples > 1 ? motion_m2 / (motion_samples - 1) : 0.0f;
        const float learned_limit = std::max(
            motion_mean * 2.5f,
            motion_mean + 3.0f *
                std::sqrt(std::max(0.0f, variance_motion)));
        motion_rejected = motion > learned_limit;
    }

    if (had_previous && !motion_rejected) {
        motion_samples++;
        const float delta = motion - motion_mean;
        motion_mean += delta / motion_samples;
        motion_m2 += delta * (motion - motion_mean);
    }

    if (motion_rejected) {
        if (reject_reason) *reject_reason = "excessive_motion";
        return false;
    }
    return true;
}

void EMOTIONCLASSIFIER::MaybePrintDebug(
        uint64_t now_ms, const ssne_tensor_t* camera_img,
        const float raw_probs[4], float brightness, float contrast,
        float motion_score, bool accepted, const char* reason) {
    if (!RuntimeLogAtLeast(RuntimeLogMode::VERIFY) ||
        (last_debug_ms != 0 && now_ms - last_debug_ms < 500)) {
        return;
    }

    const float* smooth = temporal_buffer.GetProbs();
    int top1 = 0;
    int top2 = 1;
    FindTop2(smooth, &top1, &top2);
    printf("[EMOTION_DEBUG] probs=[%.3f %.3f %.3f %.3f] "
           "smooth=[%.3f %.3f %.3f %.3f] top=%d/%d margin=%.3f "
           "stable=%s candidate=%s/%d luma=%.1f contrast=%.1f motion=%.2f "
           "frame=%s(%s)\n",
           raw_probs[0], raw_probs[1], raw_probs[2], raw_probs[3],
           smooth[0], smooth[1], smooth[2], smooth[3],
           top1, top2, smooth[top1] - smooth[top2],
           EMOTION_NAMES[static_cast<int>(stable_emotion)],
           EMOTION_NAMES[static_cast<int>(candidate_emotion)], candidate_count,
           brightness, contrast, motion_score,
           accepted ? "accepted" : "rejected", reason);

    if (std::strcmp(last_inference_failure_reason, "none") == 0) {
        const float logit_min = *std::min_element(
            last_logits, last_logits + 4);
        const float logit_max = *std::max_element(
            last_logits, last_logits + 4);
        printf("[EMOTION_LOGITS] [SURPRISE=%.6f HAPPY=%.6f SAD=%.6f "
               "NEUTRAL=%.6f] spread=%.6f\n",
               last_logits[0], last_logits[1], last_logits[2],
               last_logits[3], logit_max - logit_min);

        const PixelDistribution roi_distribution =
            MeasureTensorDistribution(camera_img,
                                      static_cast<uint32_t>(frame_pixels));
        const PixelDistribution model_distribution =
            MeasureTensorDistribution(
                &inputs[0],
            static_cast<uint32_t>(3 * model_shape[0] * model_shape[1]));
        PrintDistribution("roi640_gray", roi_distribution);
        PrintDistribution("model_rgb3_112", model_distribution);
    }

    if (now_ms - stats_start_ms >= 60000) {
        const float uncertain_ratio = accepted_frames > 0
            ? 100.0f * uncertain_frames / accepted_frames : 0.0f;
        const float reject_ratio = total_frames > 0
            ? 100.0f * quality_rejections / total_frames : 0.0f;
        const float average_switch_delay = class_switches > 0
            ? switch_delay_sum_ms / class_switches : 0.0f;
        printf("[EMOTION_STATS] frames=%u accepted=%u inference_fail=%u "
               "quality_reject=%u(%.1f%% dark=%u bright=%u contrast=%u "
               "motion=%u) uncertain=%u(%.1f%%) switches=%u "
               "avg_switch_delay=%.0fms\n",
               total_frames, accepted_frames, inference_failures,
               quality_rejections, reject_ratio, dark_rejections,
               bright_rejections, contrast_rejections, motion_rejections,
               uncertain_frames, uncertain_ratio, class_switches,
               average_switch_delay);
        stats_start_ms = now_ms;
        total_frames = 0;
        accepted_frames = 0;
        inference_failures = 0;
        quality_rejections = 0;
        dark_rejections = 0;
        bright_rejections = 0;
        contrast_rejections = 0;
        motion_rejections = 0;
        uncertain_frames = 0;
        class_switches = 0;
        switch_delay_sum_ms = 0.0f;
    }
    last_debug_ms = now_ms;
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

    int mean[3] = { 0, 0, 0 };
    int std[3] = { 0, 0, 0 };
    int is_uint8 = 0;
    int dtype = -1;
    const int input_num = ssne_get_model_input_num(model_id);
    const int norm_ret =
        ssne_get_model_normalize_params(model_id, mean, std, &is_uint8);
    const int dtype_ret = ssne_get_model_input_dtype(model_id, &dtype);

    // This model was trained and calibrated with direct grayscale values in
    // [0, 255]. SetNormalize() interprets the model's fixed-point std metadata
    // as a hardware scale and collapses this particular Y8 input to 0/1.
    // A fresh/cleared pipe therefore performs resize and Y8->RGB conversion,
    // but no range normalization.
    if (pipe_offline != nullptr) {
        Clear(pipe_offline);
    }

    uint32_t mw = static_cast<uint32_t>(model_shape[0]);
    uint32_t mh = static_cast<uint32_t>(model_shape[1]);
    // 输入源仍为Y8；RGB输出tensor让SDK预处理完成resize和灰度复制三通道。
    inputs[0] = create_tensor(mw, mh, SSNE_RGB, SSNE_BUF_AI);
    memset(&outputs[0], 0, sizeof(ssne_tensor_t));

    const bool tensor_valid =
        get_data(inputs[0]) != nullptr &&
        get_width(inputs[0]) == mw &&
        get_height(inputs[0]) == mh &&
        get_data_format(inputs[0]) == SSNE_RGB &&
        get_mem_size(inputs[0]) >= static_cast<size_t>(3u * mw * mh);
    model_ready =
        pipe_offline != nullptr &&
        input_num == 1 &&
        model_shape[0] == 112 && model_shape[1] == 112 &&
        img_shape[0] == 640 && img_shape[1] == 640 &&
        tensor_valid;

    printf("[EMOTIONCLASSIFIER] Model metadata: inputs=%d "
           "compiled_dtype=%d mean=[%d,%d,%d] std=[%d,%d,%d] "
           "is_uint8=%d norm_query_ret=%d dtype_query_ret=%d "
           "manual_normalize=disabled\n",
           input_num, dtype, mean[0], mean[1], mean[2],
           std[0], std[1], std[2], is_uint8,
           norm_ret, dtype_ret);

    if (!model_ready) {
        fprintf(stderr,
                "[ERROR] EMOTION model contract mismatch: inputs=%d "
                "crop=%dx%d shape=%dx%d dtype=%d format=%u "
                "tensor_valid=%d pipe=%p\n",
                input_num, img_shape[0], img_shape[1],
                model_shape[0], model_shape[1], dtype,
                get_data_format(inputs[0]),
                tensor_valid ? 1 : 0, static_cast<void*>(pipe_offline));
    } else {
        printf("[EMOTIONCLASSIFIER] Model contract verified: "
               "camera ROI=640x640 Y8 -> resize=112x112 -> RGB(3 planes), "
               "source semantics=0..255 without manual normalization, "
               "classes=[SURPRISE,HAPPY,SAD,NEUTRAL]\n");
    }
    fflush(stdout);

    ResetTemporalState(MonotonicMs());
    printf("[EMOTIONCLASSIFIER] Initialized: crop=%dx%d -> model=%dx%d\n",
        img_shape[0], img_shape[1], model_shape[0], model_shape[1]);
}

void EMOTIONCLASSIFIER::Predict(ssne_tensor_t* img_in, EmotionResult* result) {
    if (result == nullptr) return;
    const uint64_t now_ms = MonotonicMs();
    total_frames++;

    float probs[4] = { 0.f };
    if (!RunSingleFrameInference(img_in, probs) ||
        !IsFiniteProbability(probs)) {
        inference_failures++;
        KeepLastStableResult(result);
        MaybePrintDebug(now_ms, img_in, probs, 0.0f, 0.0f, 0.0f, false,
                        last_inference_failure_reason);
        return;
    }

    float brightness = 0.0f;
    float contrast = 0.0f;
    float motion_score = 0.0f;
    const char* quality_reason = "accepted";
    if (!ImageQualityAccepted(img_in, &brightness, &contrast, &motion_score,
                              &quality_reason)) {
        quality_rejections++;
        if (std::strcmp(quality_reason, "too_dark") == 0) {
            dark_rejections++;
        } else if (std::strcmp(quality_reason, "too_bright") == 0) {
            bright_rejections++;
        } else if (std::strcmp(quality_reason, "low_contrast") == 0) {
            contrast_rejections++;
        } else if (std::strcmp(quality_reason, "excessive_motion") == 0) {
            motion_rejections++;
        }
        KeepLastStableResult(result);
        MaybePrintDebug(now_ms, img_in, probs, brightness, contrast, motion_score,
                        false, quality_reason);
        return;
    }

    accepted_frames++;
    temporal_buffer.Push(probs, now_ms);
    const float* smooth = temporal_buffer.GetProbs();

    if (temporal_buffer.valid_count < WARMUP_VALID_FRAMES) {
        stable_emotion = EmotionClass::NEUTRAL;
        stable_confidence =
            smooth[static_cast<int>(EmotionClass::NEUTRAL)];
        KeepLastStableResult(result);
        MaybePrintDebug(now_ms, img_in, probs, brightness, contrast, motion_score,
                        true, "warmup");
        return;
    }

    int top1 = 0;
    int top2 = 1;
    FindTop2(smooth, &top1, &top2);
    const float confidence = smooth[top1];
    const float margin = confidence - smooth[top2];

    if (confidence < CONF_ON || margin < MARGIN_ON) {
        uncertain_frames++;
        candidate_count = 0;
        candidate_since_ms = 0;
        if (now_ms - last_valid_ms > UNCERTAIN_TIMEOUT_MS) {
            stable_emotion = EmotionClass::NEUTRAL;
            stable_confidence =
                smooth[static_cast<int>(EmotionClass::NEUTRAL)];
        }
    } else if (top1 == static_cast<int>(stable_emotion)) {
        candidate_count = 0;
        candidate_since_ms = 0;
        stable_confidence = confidence;
        last_valid_ms = now_ms;
    } else {
        const EmotionClass top_emotion = static_cast<EmotionClass>(top1);
        if (candidate_count == 0 || candidate_emotion != top_emotion) {
            candidate_emotion = top_emotion;
            candidate_count = 1;
            candidate_since_ms = now_ms;
        } else {
            candidate_count++;
        }

        if (candidate_count >= SWITCH_CONFIRM_FRAMES &&
            now_ms - last_switch_ms >= MIN_HOLD_MS &&
            smooth[top1] >=
                smooth[static_cast<int>(stable_emotion)] + SWITCH_MARGIN) {
            stable_emotion = top_emotion;
            stable_confidence = confidence;
            last_valid_ms = now_ms;
            last_switch_ms = now_ms;
            switch_delay_sum_ms +=
                static_cast<float>(now_ms - candidate_since_ms);
            class_switches++;
            candidate_count = 0;
            candidate_since_ms = 0;
        }
    }

    KeepLastStableResult(result);
    MaybePrintDebug(now_ms, img_in, probs, brightness, contrast, motion_score,
                    true, "accepted");
}

void EMOTIONCLASSIFIER::Release() {
    if (prev_frame_buf) { delete[] prev_frame_buf; prev_frame_buf = nullptr; }
    if (diff_buf) { delete[] diff_buf;       diff_buf = nullptr; }

    if (get_data(inputs[0]) != nullptr) release_tensor(inputs[0]);
    memset(&inputs[0], 0, sizeof(inputs[0]));

    // NOTE: outputs[0] 由 ssne_getoutput 填充，其 data 指向模型内部 buffer，
    // 不应由 release_tensor 释放。ssne_release() 会统一释放模型资源。
    outputs[0].data = nullptr;

    if (pipe_offline != nullptr) {
        ReleaseAIPreprocessPipe(pipe_offline);
        pipe_offline = nullptr;
    }
    temporal_buffer.Reset();
    has_prev_frame = false;
    model_ready = false;
    preprocess_contract_logged = false;
    last_inference_failure_reason = "released";
    printf("[EMOTIONCLASSIFIER] Resources released.\n");
}
