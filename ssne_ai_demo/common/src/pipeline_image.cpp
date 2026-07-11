/*
 * @Filename: pipeline_image.cpp
 * @Description: 统一的图像采集管道实现
 */

#include "common.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <unistd.h>

const char* IMAGEPROCESSOR::HealthStateName(ImagePipelineHealthState state) {
    switch (state) {
        case ImagePipelineHealthState::OK:
            return "OK";
        case ImagePipelineHealthState::DEGRADED:
            return "DEGRADED";
        case ImagePipelineHealthState::RECOVERING:
            return "RECOVERING";
        case ImagePipelineHealthState::FAILED:
            return "FAILED";
        default:
            return "UNKNOWN";
    }
}

const char* IMAGEPROCESSOR::QualityStateName(ImageQualityState state) {
    switch (state) {
        case ImageQualityState::UNKNOWN:
            return "UNKNOWN";
        case ImageQualityState::NORMAL:
            return "NORMAL";
        case ImageQualityState::TOO_DARK:
            return "TOO_DARK";
        case ImageQualityState::TOO_BRIGHT:
            return "TOO_BRIGHT";
        case ImageQualityState::LOW_TEXTURE:
            return "LOW_TEXTURE";
        default:
            return "UNKNOWN";
    }
}

void IMAGEPROCESSOR::SetHealthState(ImagePipelineHealthState next, const char* reason) {
    if (health_state == next) return;

    if (RuntimeLogEnabled()) {
        printf("[IMAGEPROCESSOR][HEALTH] %s -> %s",
               HealthStateName(health_state),
               HealthStateName(next));
        if (reason != nullptr && reason[0] != '\0') {
            printf(" | %s", reason);
        }
        printf("\n");
    }
    health_state = next;
}

void IMAGEPROCESSOR::SetQualityState(ImageQualityState next, const char* reason) {
    if (quality_state == next) return;

    if (RuntimeLogAtLeast(RuntimeLogMode::VERIFY)) {
        printf("[IMAGEPROCESSOR][QUALITY] %s -> %s",
               QualityStateName(quality_state),
               QualityStateName(next));
        if (reason != nullptr && reason[0] != '\0') {
            printf(" | %s", reason);
        }
        printf("\n");
    }
    quality_state = next;
}

void IMAGEPROCESSOR::ResetHealth() {
    health_state = ImagePipelineHealthState::OK;
    quality_state = ImageQualityState::UNKNOWN;
    valid_frames = 0;
    invalid_frame_streak = 0;
    invalid_frame_total = 0;
    max_invalid_frame_streak = 0;
    recover_attempts = 0;
    recover_successes = 0;
    recover_failures = 0;
    consecutive_recover_failures = 0;
    poor_quality_streak = 0;
    last_invalid_warn_streak = 0;
    invalid_log_count = 0;
    mean_luma = 0.0f;
    dark_ratio = 0.0f;
    bright_ratio = 0.0f;
    texture_score = 0.0f;
    avg_fps = 0.0f;
    p95_frame_ms = 0.0f;
    frame_period_index = 0;
    frame_period_count = 0;
    has_last_valid_frame_time = false;
    last_recover_attempt = std::chrono::steady_clock::now() - std::chrono::seconds(10);
    last_valid_frame_time = std::chrono::steady_clock::now();
    last_summary_time = std::chrono::steady_clock::now();
}

void IMAGEPROCESSOR::ConfigureRecovery(uint32_t in_warn_frames,
                                       uint32_t in_recover_frames,
                                       uint32_t in_max_recover_failures,
                                       int in_backoff_base_ms,
                                       int in_backoff_max_ms) {
    warn_frames = std::max<uint32_t>(1, in_warn_frames);
    recover_frames = std::max(warn_frames, in_recover_frames);
    max_recover_failures = std::max<uint32_t>(1, in_max_recover_failures);
    backoff_base_ms = std::max(1, in_backoff_base_ms);
    backoff_max_ms = std::max(backoff_base_ms, in_backoff_max_ms);
}

void IMAGEPROCESSOR::ConfigureQualityCheck(uint32_t sample_interval,
                                           uint32_t warn_samples,
                                           float in_dark_mean_threshold,
                                           float in_bright_mean_threshold,
                                           float in_low_texture_threshold) {
    quality_sample_interval = std::max<uint32_t>(1, sample_interval);
    quality_warn_samples = std::max<uint32_t>(1, warn_samples);
    dark_mean_threshold = std::max(0.0f, in_dark_mean_threshold);
    bright_mean_threshold = std::min(255.0f, std::max(dark_mean_threshold, in_bright_mean_threshold));
    low_texture_threshold = std::max(0.0f, in_low_texture_threshold);
}

ImagePipelineHealthSnapshot IMAGEPROCESSOR::GetHealthSnapshot() const {
    ImagePipelineHealthSnapshot snapshot;
    snapshot.state = health_state;
    snapshot.quality_state = quality_state;
    snapshot.valid_frames = valid_frames;
    snapshot.invalid_frame_streak = invalid_frame_streak;
    snapshot.invalid_frame_total = invalid_frame_total;
    snapshot.max_invalid_frame_streak = max_invalid_frame_streak;
    snapshot.recover_attempts = recover_attempts;
    snapshot.recover_successes = recover_successes;
    snapshot.recover_failures = recover_failures;
    snapshot.consecutive_recover_failures = consecutive_recover_failures;
    snapshot.poor_quality_streak = poor_quality_streak;
    snapshot.mean_luma = mean_luma;
    snapshot.dark_ratio = dark_ratio;
    snapshot.bright_ratio = bright_ratio;
    snapshot.texture_score = texture_score;
    snapshot.avg_fps = avg_fps;
    snapshot.p95_frame_ms = p95_frame_ms;
    return snapshot;
}

void IMAGEPROCESSOR::RecordFrameTiming() {
    auto now = std::chrono::steady_clock::now();
    if (has_last_valid_frame_time) {
        std::chrono::duration<float, std::milli> diff = now - last_valid_frame_time;
        float ms = diff.count();
        if (ms > 0.0f && ms < 1000.0f) {
            frame_period_ms[frame_period_index] = ms;
            frame_period_index = (frame_period_index + 1) % 120;
            if (frame_period_count < 120) frame_period_count++;
        }
    } else {
        has_last_valid_frame_time = true;
    }
    last_valid_frame_time = now;
}

void IMAGEPROCESSOR::AnalyzeFrameQuality(const ssne_tensor_t* img_sensor) {
    if (img_sensor == nullptr || img_sensor->data == nullptr) return;
    if (valid_frames % quality_sample_interval != 0) return;

    const uint8_t* data = (const uint8_t*)get_data(*img_sensor);
    if (data == nullptr || out_w_ == 0 || out_h_ == 0) return;

    const int stride = 16;
    uint32_t count = 0;
    uint32_t dark_count = 0;
    uint32_t bright_count = 0;
    uint32_t grad_count = 0;
    uint32_t sum = 0;
    uint32_t grad_sum = 0;

    for (uint16_t y = 0; y < out_h_; y += stride) {
        const uint8_t* row = data + (size_t)y * out_w_;
        for (uint16_t x = 0; x < out_w_; x += stride) {
            uint8_t v = row[x];
            sum += v;
            if (v < 24) dark_count++;
            if (v > 235) bright_count++;
            count++;
            if ((uint32_t)x + stride < out_w_) {
                grad_sum += std::abs((int)row[x + stride] - (int)v);
                grad_count++;
            }
        }
    }

    if (count == 0) return;

    mean_luma = (float)sum / (float)count;
    dark_ratio = (float)dark_count / (float)count;
    bright_ratio = (float)bright_count / (float)count;
    texture_score = grad_count > 0 ? (float)grad_sum / (float)grad_count : 0.0f;

    ImageQualityState next = ImageQualityState::NORMAL;
    const char* reason = "image quality normal";
    if (mean_luma < dark_mean_threshold || dark_ratio > 0.65f) {
        next = ImageQualityState::TOO_DARK;
        reason = "image too dark";
    } else if (mean_luma > bright_mean_threshold || bright_ratio > 0.65f) {
        next = ImageQualityState::TOO_BRIGHT;
        reason = "image too bright";
    } else if (texture_score < low_texture_threshold) {
        next = ImageQualityState::LOW_TEXTURE;
        reason = "image texture too weak";
    }

    if (next == ImageQualityState::NORMAL) {
        poor_quality_streak = 0;
        SetQualityState(next, reason);
        if (health_state == ImagePipelineHealthState::DEGRADED &&
            invalid_frame_streak == 0 &&
            consecutive_recover_failures == 0) {
            SetHealthState(ImagePipelineHealthState::OK, "image quality recovered");
        }
    } else {
        poor_quality_streak++;
        SetQualityState(next, reason);
        if (poor_quality_streak >= quality_warn_samples &&
            health_state == ImagePipelineHealthState::OK) {
            SetHealthState(ImagePipelineHealthState::DEGRADED, reason);
        }
    }
}

void IMAGEPROCESSOR::MaybePrintRuntimeSummary() {
    RuntimeLogMode mode = GetRuntimeLogMode();
    if (mode == RuntimeLogMode::SILENT) return;

    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<float> diff = now - last_summary_time;
    if (diff.count() < 1.0f) return;
    last_summary_time = now;

    if (frame_period_count > 0) {
        float sum = 0.0f;
        float periods[120];
        uint32_t n = frame_period_count;
        for (uint32_t i = 0; i < n; i++) {
            periods[i] = frame_period_ms[i];
            sum += periods[i];
        }
        std::sort(periods, periods + n);
        uint32_t idx = (n * 95) / 100;
        if (idx >= n) idx = n - 1;
        p95_frame_ms = periods[idx];
        float mean_ms = sum / (float)n;
        avg_fps = mean_ms > 0.001f ? 1000.0f / mean_ms : 0.0f;
    }

    if (mode == RuntimeLogMode::SUMMARY) {
        printf("[PIPE_SUM] health=%s quality=%s fps=%.1f invalid=%u recover=%u/%u\n",
               HealthStateName(health_state),
               QualityStateName(quality_state),
               avg_fps,
               invalid_frame_total,
               recover_successes,
               recover_attempts);
    } else {
        printf("[PIPE_VERIFY] health=%s quality=%s fps=%.1f p95=%.2fms "
               "luma=%.1f dark=%.2f bright=%.2f texture=%.1f invalid=%u max_invalid=%u recover=%u/%u fail=%u\n",
               HealthStateName(health_state),
               QualityStateName(quality_state),
               avg_fps,
               p95_frame_ms,
               mean_luma,
               dark_ratio,
               bright_ratio,
               texture_score,
               invalid_frame_total,
               max_invalid_frame_streak,
               recover_successes,
               recover_attempts,
               recover_failures);
    }
}

bool IMAGEPROCESSOR::OpenConfiguredPipeline() {
    if (!config_valid) return false;

    format_online = SSNE_Y_8;
    OnlineSetCrop(kPipeline0, crop_x1_, crop_x2_, crop_y1_, crop_y2_);
    OnlineSetOutputImage(kPipeline0, format_online, out_w_, out_h_);
    int res0 = OpenOnlinePipeline(kPipeline0);
    if (res0 != 0) {
        if (RuntimeLogEnabled()) {
            printf("[ERROR] Failed to open online pipeline! ret: %d\n", res0);
        }
        is_opened = false;
        return false;
    }

    is_opened = true;
    return true;
}

bool IMAGEPROCESSOR::RecoverPipeline() {
    auto now = std::chrono::steady_clock::now();
    uint32_t shift = std::min<uint32_t>(consecutive_recover_failures, 3);
    int backoff_ms = std::min(backoff_max_ms, backoff_base_ms * (1 << shift));
    std::chrono::duration<float, std::milli> since_last = now - last_recover_attempt;

    if (since_last.count() < (float)backoff_ms) {
        return false;
    }

    last_recover_attempt = now;
    recover_attempts++;
    SetHealthState(ImagePipelineHealthState::RECOVERING, "reopening camera pipeline");

    {
        SigintBlocker sig_blocker;
        if (is_opened) {
            CloseOnlinePipeline(kPipeline0);
            is_opened = false;
        }
        usleep(60000);
        bool reopened = OpenConfiguredPipeline();
        if (reopened) {
            recover_successes++;
            consecutive_recover_failures = 0;
            invalid_frame_streak = 0;
            last_invalid_warn_streak = 0;
            invalid_log_count = 0;
            SetHealthState(ImagePipelineHealthState::OK, "camera pipeline recovered");
            return true;
        }
    }

    recover_failures++;
    consecutive_recover_failures++;
    if (consecutive_recover_failures >= max_recover_failures) {
        SetHealthState(ImagePipelineHealthState::FAILED,
                       "camera pipeline recovery failed repeatedly");
    } else {
        SetHealthState(ImagePipelineHealthState::DEGRADED,
                       "camera pipeline recovery failed, will retry");
    }

    return false;
}

/**
 * @brief 图像处理器初始化函数
 * @param in_img_shape 输入原始图像尺寸 [宽度, 高度]
 * @param crop_x1 裁剪区域左边界
 * @param crop_x2 裁剪区域右边界
 * @param crop_y1 裁剪区域上边界
 * @param crop_y2 裁剪区域下边界
 * @param out_w 输出图像宽度
 * @param out_h 输出图像高度
 */
void IMAGEPROCESSOR::Initialize(std::array<int, 2>* in_img_shape, 
                                uint16_t crop_x1, uint16_t crop_x2, 
                                uint16_t crop_y1, uint16_t crop_y2,
                                uint16_t out_w, uint16_t out_h) {
    SigintBlocker sig_blocker;
    img_shape = *in_img_shape;
    crop_x1_ = crop_x1;
    crop_x2_ = crop_x2;
    crop_y1_ = crop_y1;
    crop_y2_ = crop_y2;
    out_w_ = out_w;
    out_h_ = out_h;
    config_valid = true;
    is_opened = false;
    ResetHealth();

    if (!OpenConfiguredPipeline()) {
        SetHealthState(ImagePipelineHealthState::FAILED,
                       "camera pipeline open failed at startup");
    }
}

/**
 * @brief 从 pipeline 获取图像数据
 * @param img_sensor 输出参数：存储从 pipe0 获取的裁剪图像
 */
void IMAGEPROCESSOR::GetImage(ssne_tensor_t* img_sensor) {
    if (!is_opened) {
        if (img_sensor) {
            img_sensor->data = nullptr;
        }
        invalid_frame_streak++;
        invalid_frame_total++;
        max_invalid_frame_streak = std::max(max_invalid_frame_streak, invalid_frame_streak);
        if (invalid_frame_streak >= recover_frames) {
            RecoverPipeline();
        }
        return;
    }

    int capture_code = GetImageData(img_sensor, kPipeline0, kSensor0, 0);
    if (capture_code != 0) {
        if (img_sensor) {
            img_sensor->data = nullptr;
        }

        invalid_frame_streak++;
        invalid_frame_total++;
        max_invalid_frame_streak = std::max(max_invalid_frame_streak, invalid_frame_streak);
        invalid_log_count++;
        if (RuntimeLogAtLeast(RuntimeLogMode::VERIFY) &&
            (invalid_log_count == 1 || invalid_log_count % 60 == 0)) {
            printf("[IMAGEPROCESSOR] Get Invalid Image from kPipeline0! count=%u\n",
                   invalid_log_count);
        }

        if (invalid_frame_streak >= warn_frames &&
            last_invalid_warn_streak < warn_frames) {
            last_invalid_warn_streak = invalid_frame_streak;
            SetHealthState(ImagePipelineHealthState::DEGRADED,
                           "camera frame invalid streak exceeded warning threshold");
        }

        if (invalid_frame_streak >= recover_frames) {
            RecoverPipeline();
        }
    } else {
        valid_frames++;
        RecordFrameTiming();
        AnalyzeFrameQuality(img_sensor);
        if (invalid_frame_streak > 0) {
            if (RuntimeLogEnabled()) {
                printf("[IMAGEPROCESSOR][HEALTH] camera frame stream recovered after %u invalid frames.\n",
                       invalid_frame_streak);
            }
            invalid_frame_streak = 0;
            last_invalid_warn_streak = 0;
            consecutive_recover_failures = 0;
            SetHealthState(ImagePipelineHealthState::OK, "camera frame stream valid again");
        }
        invalid_log_count = 0;
        MaybePrintRuntimeSummary();
    }
}

/**
 * @brief 释放图像处理器资源，关闭 pipeline
 */
void IMAGEPROCESSOR::Release() {
    SigintBlocker sig_blocker;
    if (is_opened) {
        CloseOnlinePipeline(kPipeline0);
        is_opened = false;
    }
}
