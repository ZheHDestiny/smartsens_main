/*
 * @Filename: common.hpp
 * @Description: SSNE AI Demo 统一数据结构与类声明
 */
#pragma once

#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <array>
#include <string>
#include <math.h>
#include <cmath>
#include <cstring>
#include <atomic>
#include <csignal>
#include <sys/select.h>
#include <unistd.h>
#include <iostream>
#include <limits>
#include <termios.h>
#include <chrono>
#include "smartsoc/ssne_api.h"

enum class ImagePipelineHealthState {
    OK = 0,
    DEGRADED,
    RECOVERING,
    FAILED
};

enum class ImageQualityState {
    UNKNOWN = 0,
    NORMAL,
    TOO_DARK,
    TOO_BRIGHT,
    LOW_TEXTURE
};

enum class RuntimeLogMode : int {
    SILENT = 0,
    SUMMARY = 1,
    VERIFY = 2
};

struct ImagePipelineHealthSnapshot {
    ImagePipelineHealthState state;
    ImageQualityState quality_state;
    uint32_t valid_frames;
    uint32_t invalid_frame_streak;
    uint32_t invalid_frame_total;
    uint32_t max_invalid_frame_streak;
    uint32_t recover_attempts;
    uint32_t recover_successes;
    uint32_t recover_failures;
    uint32_t consecutive_recover_failures;
    uint32_t poor_quality_streak;
    float mean_luma;
    float dark_ratio;
    float bright_ratio;
    float texture_score;
    float avg_fps;
    float p95_frame_ms;
};

extern std::atomic<int> g_runtime_log_mode;

inline RuntimeLogMode GetRuntimeLogMode() {
    return static_cast<RuntimeLogMode>(g_runtime_log_mode.load());
}

inline void SetRuntimeLogMode(RuntimeLogMode mode) {
    g_runtime_log_mode.store(static_cast<int>(mode));
}

inline bool RuntimeLogAtLeast(RuntimeLogMode mode) {
    return g_runtime_log_mode.load() >= static_cast<int>(mode);
}

inline bool RuntimeLogEnabled() {
    return RuntimeLogAtLeast(RuntimeLogMode::SUMMARY);
}

class IMAGEPROCESSOR {
public:
    void Initialize(std::array<int, 2>* in_img_shape, 
                    uint16_t crop_x1, uint16_t crop_x2, 
                    uint16_t crop_y1, uint16_t crop_y2,
                    uint16_t out_w, uint16_t out_h);
                    
    void GetImage(ssne_tensor_t* img_sensor);
    void Release();
    bool IsOpened() const { return is_opened; }
    void ConfigureRecovery(uint32_t warn_frames,
                           uint32_t recover_frames,
                           uint32_t max_recover_failures,
                           int backoff_base_ms,
                           int backoff_max_ms);
    void ConfigureQualityCheck(uint32_t sample_interval,
                               uint32_t warn_samples,
                               float dark_mean_threshold,
                               float bright_mean_threshold,
                               float low_texture_threshold);
    ImagePipelineHealthSnapshot GetHealthSnapshot() const;
    static const char* HealthStateName(ImagePipelineHealthState state);
    static const char* QualityStateName(ImageQualityState state);
    
    std::array<int, 2> img_shape;
private:
    uint8_t format_online;
    bool is_opened = false;
    bool config_valid = false;
    uint16_t crop_x1_ = 0;
    uint16_t crop_x2_ = 0;
    uint16_t crop_y1_ = 0;
    uint16_t crop_y2_ = 0;
    uint16_t out_w_ = 0;
    uint16_t out_h_ = 0;

    ImagePipelineHealthState health_state = ImagePipelineHealthState::OK;
    ImageQualityState quality_state = ImageQualityState::UNKNOWN;
    uint32_t valid_frames = 0;
    uint32_t invalid_frame_streak = 0;
    uint32_t invalid_frame_total = 0;
    uint32_t max_invalid_frame_streak = 0;
    uint32_t recover_attempts = 0;
    uint32_t recover_successes = 0;
    uint32_t recover_failures = 0;
    uint32_t consecutive_recover_failures = 0;
    uint32_t poor_quality_streak = 0;
    uint32_t last_invalid_warn_streak = 0;
    uint32_t invalid_log_count = 0;
    uint32_t quality_sample_interval = 5;
    uint32_t quality_warn_samples = 6;
    uint32_t warn_frames = 15;
    uint32_t recover_frames = 30;
    uint32_t max_recover_failures = 5;
    float mean_luma = 0.0f;
    float dark_ratio = 0.0f;
    float bright_ratio = 0.0f;
    float texture_score = 0.0f;
    float avg_fps = 0.0f;
    float p95_frame_ms = 0.0f;
    float dark_mean_threshold = 28.0f;
    float bright_mean_threshold = 235.0f;
    float low_texture_threshold = 2.0f;
    float frame_period_ms[120] = {0.0f};
    uint32_t frame_period_index = 0;
    uint32_t frame_period_count = 0;
    bool has_last_valid_frame_time = false;
    int backoff_base_ms = 300;
    int backoff_max_ms = 3000;
    std::chrono::steady_clock::time_point last_recover_attempt =
        std::chrono::steady_clock::now() - std::chrono::seconds(10);
    std::chrono::steady_clock::time_point last_valid_frame_time =
        std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_summary_time =
        std::chrono::steady_clock::now();

    bool OpenConfiguredPipeline();
    bool RecoverPipeline();
    void SetHealthState(ImagePipelineHealthState next, const char* reason);
    void SetQualityState(ImageQualityState next, const char* reason);
    void ResetHealth();
    void AnalyzeFrameQuality(const ssne_tensor_t* img_sensor);
    void RecordFrameTiming();
    void MaybePrintRuntimeSummary();
};
struct ObjectDetectionResult {
    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    void Clear() {
        boxes.clear();
        scores.clear();
        class_ids.clear();
    }
    void Reserve(int size) {
        boxes.reserve(size);
        scores.reserve(size);
        class_ids.reserve(size);
    }
};



struct FaceDetectionResult {
    std::vector<std::array<float, 4>> boxes;
    std::vector<std::array<float, 2>> landmarks;
    std::vector<float> scores;
    int landmarks_per_face;

    FaceDetectionResult() { landmarks_per_face = 0; }
    FaceDetectionResult(const FaceDetectionResult& res);
    void Clear();
    void Free();
    void Reserve(int size);
    void Resize(int size);
};

class SCRFDGRAY {
public:
    std::string ModelName() const { return "scrfd_gray"; }
    void Predict(ssne_tensor_t* img_in, FaceDetectionResult* result, float conf_threshold = 0.25f);
    void Initialize(std::string& model_path, std::array<int, 2>* in_img_shape, 
                    std::array<int, 2>* in_det_shape, bool in_use_kps, int in_box_len);
    void Release();
    
    void saveImageBin(const void* data, int w, int h, const char* filename);
    void saveFloatBin(const float* data, int length, const char* filename);

    float nms_threshold;
    int keep_top_k;
    int top_k;
    std::array<int, 2> img_shape;
    std::array<int, 2> det_shape;
    int box_len;
    float w_scale;
    float h_scale;
    bool use_kps;
    std::vector<std::array<int, 2>> min_sizes;
    std::vector<int> steps;
    std::vector<float> variance;
    bool clip = false;
    std::vector<float> ratios;

private:
    uint16_t model_id = 0;
    ssne_tensor_t inputs[1];
    ssne_tensor_t outputs[6];
    AiPreprocessPipe pipe_offline = GetAIPreprocessPipe();
    std::vector<std::array<float, 4>> anchors;

    void GenerateBoxes();
    void DecodeBoxes(std::vector<std::array<float, 4>>& boxes);
    void Postprocess(std::vector<std::array<float, 4>>* boxes, std::vector<float>* scores, 
                     FaceDetectionResult* result, float* conf_threshold);
};



class YOLOV8_OBJECT {
public:
    void Initialize(const std::string& model_path, std::array<int, 2>* in_img_shape, std::array<int, 2>* in_det_shape);
    void Release();
    void Predict(ssne_tensor_t* img_in, ObjectDetectionResult* result, float conf_threshold = 0.25f);

private:
    uint16_t model_id = 0;
    AiPreprocessPipe pipe_offline; 
    ssne_tensor_t inputs[1];
    ssne_tensor_t outputs[6]; 

    std::array<int, 2> img_shape;
    std::array<int, 2> det_shape;
    float w_scale;
    float h_scale;
    float nms_threshold = 0.45f;
    int top_k = 300;
    int keep_top_k = 100;

    const int kNumClasses = 20;
    const int kRegBins = 16;
    const int kRegChannels = 64; 
    const int kNumScales = 3;
    const int kStrides[3] = {8, 16, 32};

    void DecodeHeadOutputs(const float* cls_head, const float* reg_head,
                           int height, int width, int stride, float conf_threshold,
                           std::vector<std::array<float, 4>>& boxes,
                           std::vector<float>& scores, std::vector<int>& class_ids);
                           
    void Postprocess(std::vector<std::array<float, 4>>* boxes,
                     std::vector<float>* scores, std::vector<int>* class_ids,
                     ObjectDetectionResult* result);
};



class YOLOV8_SPEED {
public:
    void Initialize(const std::string& model_path, std::array<int, 2>* in_img_shape, std::array<int, 2>* in_det_shape);
    void Predict(ssne_tensor_t* img, ObjectDetectionResult* result, float conf_threshold);
    void Release();

private:
    void DecodeHeadOutputs(const float* cls_head, const float* reg_head, int height, int width, int stride, float conf_threshold, std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores, std::vector<int>& class_ids);
    void Postprocess(std::vector<std::array<float, 4>>* boxes, std::vector<float>* scores, std::vector<int>* class_ids, ObjectDetectionResult* result);

    std::array<int, 2> img_shape;
    std::array<int, 2> det_shape;
    float w_scale, h_scale;
    
    uint16_t model_id = 0;
    AiPreprocessPipe pipe_offline; 
    ssne_tensor_t inputs[1];
    ssne_tensor_t outputs[6];

    const int kNumClasses = 3; 
    const int kRegBins = 16;
    const float nms_threshold = 0.45f;
    const int top_k = 300;
    const int keep_top_k = 100;
};



struct FeaturePoint {
    float x, y;           
    float dx, dy;         
    float score;          
    bool tracked;         

    FeaturePoint() : x(0), y(0), dx(0), dy(0), score(0), tracked(false) {}
    FeaturePoint(float x_, float y_) : x(x_), y(y_), dx(0), dy(0), score(0), tracked(false) {}
    FeaturePoint(float x_, float y_, float s) : x(x_), y(y_), dx(0), dy(0), score(s), tracked(false) {}
};

struct ObstacleInfo {
    enum Region { LEFT = 0, CENTER = 1, RIGHT = 2, REGION_COUNT = 3 };
    enum Priority { EMERGENCY = 0, CAUTION = 1, CLEAR = 4 };
    float danger_level[REGION_COUNT];
    float ttc_seconds[REGION_COUNT];
    int support_count[REGION_COUNT];
    bool has_obstacle[REGION_COUNT];
    int most_dangerous_region;
    int safest_region;
    int priority;
    float global_dx;
    float global_dy;
    float tracking_quality;

    ObstacleInfo() {
        for (int i = 0; i < REGION_COUNT; i++) {
            danger_level[i] = 0.0f;
            ttc_seconds[i] = -1.0f;
            support_count[i] = 0;
            has_obstacle[i] = false;
        }
        most_dangerous_region = CENTER;
        safest_region = CENTER;
        priority = CLEAR;
        global_dx = 0.0f;
        global_dy = 0.0f;
        tracking_quality = 0.0f;
    }
};

class OPTICALFLOW {
public:
    void Initialize(int width, int height);
    void ComputeFlow(const uint8_t* prev_frame, const uint8_t* curr_frame, std::vector<FeaturePoint>& features);
    void DetectFeatures(const uint8_t* frame, std::vector<FeaturePoint>& features);
    void ResetHistory();
    void Release();

    int max_features;      
    int fast_threshold;    
    int feature_scan_step;
    int nms_radius;        
    int grid_size;         
    int grid_max_per_cell; 
    int pyramid_levels;    
    int lk_win_size;       
    int lk_max_iter;       
    float lk_epsilon;      
    float lk_min_eig;      

private:
    int width_, height_;
    std::vector<std::vector<uint8_t>> pyramid_prev_;
    std::vector<std::vector<uint8_t>> pyramid_curr_;
    std::vector<int> pyr_widths_;
    std::vector<int> pyr_heights_;
    bool pyramid_cache_valid_ = false;

    void BuildPyramid(const uint8_t* frame, std::vector<std::vector<uint8_t>>& pyramid);
    static float Interp(const uint8_t* img, int w, int h, float x, float y);
    bool TrackPointSingleLevel(const uint8_t* prev, const uint8_t* curr,
                               int w, int h, float px, float py,
                               float& cx, float& cy,
                               float* photometric_error = nullptr);
};

class OBSTACLE_DETECTOR {
public:
    void Initialize(int width, int height);
    void DetectObstacles(const std::vector<FeaturePoint>& features, ObstacleInfo& obstacle_info);
    void SetFrameInterval(float seconds);
    void Release();

    float ttc_threshold;   
    float divergence_threshold;  

private:
    int width_, height_;
    float frame_interval_seconds_ = 1.0f / 75.0f;
    float smoothed_danger_[ObstacleInfo::REGION_COUNT] = {0.0f, 0.0f, 0.0f};
    bool latched_obstacle_[ObstacleInfo::REGION_COUNT] = {false, false, false};
    float ComputeTTC(float x, float y, float dx, float dy) const;
};

enum class FocusTrackingMode : int {
    NO_NPU_TRACKER = 0,
    NPU_MOBILENET = 1
};

struct FocusTrackingConfig {
    int width;
    int height;
    int target_w;
    int target_h;
    int search_radius;
    int search_step;
    int sample_step;
    int lost_frame_limit;
    int npu_interval_frames;
    float template_lr;
    float response_threshold;
    float motion_alpha;
    float position_smooth_alpha;
    float template_update_threshold;
    float min_focus_score_for_update;
    FocusTrackingMode mode;
    std::string npu_model_path;

    FocusTrackingConfig();
};

struct FocusTargetState {
    bool locked;
    int x;
    int y;
    int w;
    int h;
    float cx;
    float cy;
    float vx;
    float vy;
    float confidence;
    float focus_score;
    int age;
    int lost_frames;

    FocusTargetState();
    void Clear();
};

class FocusTracker {
public:
    FocusTracker();

    void Initialize(const FocusTrackingConfig& config);
    void Reset();
    bool SetTarget(const uint8_t* frame, const FocusTargetState& target);
    bool Update(const uint8_t* frame, FocusTargetState* state);
    bool HasTarget() const;

private:
    FocusTrackingConfig cfg_;
    FocusTargetState state_;
    std::vector<uint8_t> templ_;
    bool initialized_;

    bool SelectTarget(const uint8_t* frame);
    void ExtractTemplate(const uint8_t* frame, int x, int y);
    void UpdateTemplate(const uint8_t* frame, int x, int y);
    float MatchTemplateSad(const uint8_t* frame, int x, int y) const;
    float TextureScore(const uint8_t* frame, int x, int y, int w, int h) const;
    float FocusScore(const uint8_t* frame, int x, int y, int w, int h) const;
    void ClampRoi(int* x, int* y) const;
};

class MobileNetFocusSelector {
public:
    MobileNetFocusSelector();

    void Initialize(const std::string& model_path,
                    std::array<int, 2>* in_img_shape,
                    std::array<int, 2>* in_model_shape);
    bool Predict(ssne_tensor_t* img_in, FocusTargetState* target);
    void Release();
    bool IsReady() const;

private:
    uint16_t model_id = 0;
    ssne_tensor_t inputs[1];
    ssne_tensor_t outputs[2];
    AiPreprocessPipe pipe_offline = GetAIPreprocessPipe();
    std::array<int, 2> img_shape;
    std::array<int, 2> model_shape;
    bool ready_;
};

struct EyeBox {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;

    float Width() const { return x2 - x1; }
    float Height() const { return y2 - y1; }
    float Cx() const { return 0.5f * (x1 + x2); }
    float Cy() const { return 0.5f * (y1 + y2); }
};

struct EyePair {
    EyeBox left;
    EyeBox right;
    float score = 0.0f;
    uint64_t track_id = 0;

    float Cx() const { return 0.5f * (left.Cx() + right.Cx()); }
    float Cy() const { return 0.5f * (left.Cy() + right.Cy()); }
    float EyeDistance() const {
        const float dx = right.Cx() - left.Cx();
        const float dy = right.Cy() - left.Cy();
        return std::sqrt(dx * dx + dy * dy);
    }
};

struct EyeDetResult {
    std::vector<EyeBox> eyes;
    std::vector<EyePair> pairs;
    int selected_index = -1;
    int lost_frames = 0;
    float preprocess_ms = 0.0f;
    float npu_ms = 0.0f;
    float postprocess_ms = 0.0f;
    float total_ms = 0.0f;
    float max_class_score = 0.0f;
    int candidates_before_nms = 0;

    void Clear() {
        eyes.clear();
        pairs.clear();
        selected_index = -1;
        lost_frames = 0;
        preprocess_ms = 0.0f;
        npu_ms = 0.0f;
        postprocess_ms = 0.0f;
        total_ms = 0.0f;
        max_class_score = 0.0f;
        candidates_before_nms = 0;
    }
};

struct IdentityResult {
    bool valid = false;
    bool enrolled = false;
    bool expired = false;
    float similarity = 0.0f;
    float npu_ms = 0.0f;
    int enrolled_count = 0;
    std::string label = "unknown";

    void Clear() { *this = IdentityResult(); }
};

class EyeDetFaceIdEngine {
public:
    EyeDetFaceIdEngine();
    ~EyeDetFaceIdEngine();

    bool Initialize(const std::string& eyedet_path,
                    const std::string& faceid_path,
                    int capture_w,
                    int capture_h);
    bool DetectEyes(ssne_tensor_t* capture_y8, EyeDetResult* out);
    bool Identify(const uint8_t* capture_y8,
                  int width,
                  int height,
                  const EyePair& pair,
                  IdentityResult* out);

    bool BeginEnroll();
    void ClearEnrollment();
    void ResetSession();
    void Release();

    bool IsReady() const { return eyedet_ready_; }
    bool FaceIdReady() const { return faceid_ready_; }
    bool HasEnrollment() const { return prototype_valid_; }
    bool IsEnrolling() const { return enrolling_; }
    int EnrollmentCount() const { return static_cast<int>(enroll_samples_.size()); }
    uint64_t SelectedTrackId() const { return selected_track_id_; }

private:
    static const int kDetSize = 640;
    static const int kFaceSize = 112;
    static const int kEmbeddingSize = 128;

    uint16_t eyedet_model_id_ = 0;
    uint16_t faceid_model_id_ = 0;
    AiPreprocessPipe eyedet_pipe_ = nullptr;
    AiPreprocessPipe faceid_pipe_ = nullptr;
    ssne_tensor_t det_canvas_;
    ssne_tensor_t det_input_;
    ssne_tensor_t det_outputs_[6];
    ssne_tensor_t face_roi_;
    ssne_tensor_t face_input_;
    ssne_tensor_t face_output_[1];

    bool initialized_ = false;
    bool eyedet_ready_ = false;
    bool faceid_ready_ = false;
    bool det_contract_logged_ = false;
    bool det_preprocess_logged_ = false;
    bool face_contract_logged_ = false;
    int capture_w_ = 0;
    int capture_h_ = 0;
    float letterbox_scale_ = 1.0f;
    int letterbox_w_ = 0;
    int letterbox_h_ = 0;
    std::vector<int> resize_x0_;
    std::vector<int> resize_x1_;
    std::vector<int> resize_y0_;
    std::vector<int> resize_y1_;
    std::vector<uint16_t> resize_wx_;
    std::vector<uint16_t> resize_wy_;

    bool selected_valid_ = false;
    uint64_t selected_track_id_ = 0;
    uint64_t next_track_id_ = 1;
    float selected_cx_ = 0.0f;
    float selected_cy_ = 0.0f;
    float selected_eye_distance_ = 0.0f;
    int selected_lost_frames_ = 0;

    bool enrolling_ = false;
    uint64_t enroll_track_id_ = 0;
    std::vector<std::array<float, kEmbeddingSize>> enroll_samples_;
    std::array<float, kEmbeddingSize> prototype_;
    bool prototype_valid_ = false;

    uint64_t det_runs_ = 0;
    uint64_t det_failures_ = 0;
    uint64_t face_runs_ = 0;
    uint64_t face_failures_ = 0;

    bool PrepareEyeDetInput(const uint8_t* src);
    bool ValidateEyeDetOutputs();
    void DecodeHead(const float* cls,
                    const float* reg,
                    int width,
                    int height,
                    int stride,
                    std::vector<EyeBox>* candidates,
                    float* max_class_score) const;
    void NmsAndMap(const std::vector<EyeBox>& candidates,
                   std::vector<EyeBox>* eyes) const;
    void PairEyes(const std::vector<EyeBox>& eyes, std::vector<EyePair>* pairs) const;
    void SelectPair(EyeDetResult* result);

    bool BuildFaceRoi(const uint8_t* src,
                      int width,
                      int height,
                      const EyePair& pair);
    bool ReadEmbedding(std::array<float, kEmbeddingSize>* embedding);
    void ConsumeEnrollment(const std::array<float, kEmbeddingSize>& embedding);

    static float Sigmoid(float x);
    static float IoU(const EyeBox& a, const EyeBox& b);
    static float IntersectionOverMinArea(const EyeBox& a, const EyeBox& b);
    static float Cosine(const std::array<float, kEmbeddingSize>& a,
                        const std::array<float, kEmbeddingSize>& b);
    static bool Normalize(std::array<float, kEmbeddingSize>* value);
};



enum class GestureClass : int {
    IDLE     = 0,
    ROCK     = 1,
    PAPER    = 2,
    SCISSORS = 3,
    NUM_CLASSES = 4
};

static const char* GESTURE_NAMES[4] = { "IDLE", "ROCK", "PAPER", "SCISSORS" };

enum class GameState {
    IDLE,
    WIND_UP,
    PREDICTED,
    DISPLAY
};

struct RpsResult {
    GestureClass human_gesture;
    GestureClass ai_counter;
    float        confidence;
    GameState    game_state;
    bool         is_locked;

    RpsResult()
        : human_gesture(GestureClass::IDLE),
          ai_counter(GestureClass::IDLE),
          confidence(0.f),
          game_state(GameState::IDLE),
          is_locked(false) {}

    void Clear() {
        human_gesture = GestureClass::IDLE;
        ai_counter    = GestureClass::IDLE;
        confidence    = 0.f;
        game_state    = GameState::IDLE;
        is_locked     = false;
    }
};

struct TemporalBuffer {
    static const int CAPACITY = 90;

    GestureClass gestures[CAPACITY];
    float        confidences[CAPACITY];
    int          head;
    int          count;

    TemporalBuffer() : head(0), count(0) {
        for (int i = 0; i < CAPACITY; i++) {
            gestures[i]    = GestureClass::IDLE;
            confidences[i] = 0.f;
        }
    }

    void Push(GestureClass gesture, float confidence) {
        gestures[head]    = gesture;
        confidences[head] = confidence;
        head = (head + 1) % CAPACITY;
        if (count < CAPACITY) count++;
    }

    GestureClass GetWeightedVote(int most_recent_n, float* out_confidence) const {
        float class_score[4] = {0.f, 0.f, 0.f, 0.f};
        int look = (count < most_recent_n) ? count : most_recent_n;
        for (int i = 0; i < look; i++) {
            int idx = ((head - 1 - i) + CAPACITY) % CAPACITY;
            float weight = static_cast<float>(look - i) / static_cast<float>(look);
            int g = static_cast<int>(gestures[idx]);
            class_score[g] += weight * confidences[idx];
        }
        int best = 0;
        for (int c = 1; c < 4; c++) {
            if (class_score[c] > class_score[best]) best = c;
        }
        if (out_confidence) *out_confidence = class_score[best];
        return static_cast<GestureClass>(best);
    }

    int ConsecutiveNonIdle(int max_look_back) const {
        int look = (count < max_look_back) ? count : max_look_back;
        int cnt  = 0;
        for (int i = 0; i < look; i++) {
            int idx = ((head - 1 - i) + CAPACITY) % CAPACITY;
            if (gestures[idx] != GestureClass::IDLE) cnt++;
            else break;
        }
        return cnt;
    }

    int ConsecutiveIdle(int max_look_back) const {
        int look = (count < max_look_back) ? count : max_look_back;
        int cnt  = 0;
        for (int i = 0; i < look; i++) {
            int idx = ((head - 1 - i) + CAPACITY) % CAPACITY;
            if (gestures[idx] == GestureClass::IDLE) cnt++;
            else break;
        }
        return cnt;
    }
};

class RPSCLASSIFIER {
public:
    std::string ModelName() const { return "rps_gesture_classifier"; }

    void Initialize(std::string& model_path, std::array<int, 2>* in_img_shape, std::array<int, 2>* in_model_shape);
    void Predict(ssne_tensor_t* img_in, RpsResult* result);
    void Release();

    std::array<int, 2> img_shape;    
    std::array<int, 2> model_shape;  

    static const int WIND_UP_FRAMES      = 15;  
    static const int IDLE_RESET_FRAMES   = 45;  
    static const int DISPLAY_HOLD_FRAMES = 90;  

private:
    uint16_t      model_id = 0;
    ssne_tensor_t inputs[1];    
    ssne_tensor_t outputs[1];   
    AiPreprocessPipe pipe_offline = GetAIPreprocessPipe();

    uint8_t* prev_frame_buf = nullptr;   
    uint8_t* diff_buf       = nullptr;   
    bool     has_prev_frame = false;     
    int      frame_pixels   = 0;         

    TemporalBuffer temporal_buffer;
    GameState      game_state;
    int            state_frame_count;
    GestureClass   locked_prediction;

    void RunSingleFrameInference(ssne_tensor_t* img, float out_probs[4]);
    GestureClass GetCounterMove(GestureClass human_gesture);
    void UpdateStateMachine(GestureClass current_gesture, float current_confidence, RpsResult* result);
};

enum class EmotionClass : int {
    SURPRISE = 0, HAPPY = 1, SAD = 2, NEUTRAL = 3, NUM_CLASSES = 4
};
static const char* EMOTION_NAMES[] = { "SURPRISE", "HAPPY", "SAD", "NEUTRAL" };

struct EmotionResult {
    EmotionClass emotion;
    float confidence;
    EmotionResult() : emotion(EmotionClass::NEUTRAL), confidence(0.f) {}
    void Clear() { emotion = EmotionClass::NEUTRAL; confidence = 0.f; }
};

struct EmotionTemporalBuffer {
    static constexpr float EMA_TAU_MS = 400.0f;
    float ema_probs[4];
    uint64_t last_timestamp_ms;
    int valid_count;

    EmotionTemporalBuffer() { Reset(); }

    void Reset() {
        for (int i = 0; i < 4; i++) ema_probs[i] = 0.0f;
        last_timestamp_ms = 0;
        valid_count = 0;
    }

    void Push(const float probs[4], uint64_t timestamp_ms) {
        if (valid_count == 0) {
            for (int i = 0; i < 4; i++) ema_probs[i] = probs[i];
        } else {
            uint64_t dt_ms = timestamp_ms > last_timestamp_ms
                           ? timestamp_ms - last_timestamp_ms : 0;
            float alpha = 1.0f - expf(-static_cast<float>(dt_ms) / EMA_TAU_MS);
            for (int i = 0; i < 4; i++) {
                ema_probs[i] = alpha * probs[i] + (1.0f - alpha) * ema_probs[i];
            }
        }
        last_timestamp_ms = timestamp_ms;
        valid_count++;
    }

    const float* GetProbs() const {
        return ema_probs;
    }
};

class EMOTIONCLASSIFIER {
public:
    void Initialize(std::string& model_path, std::array<int, 2>* in_img_shape, std::array<int, 2>* in_model_shape);
    void Predict(ssne_tensor_t* img_in, EmotionResult* result);
    bool RunSingleFrameInference(ssne_tensor_t* img, float out_probs[4]);
    void Release();
    std::array<int, 2> img_shape; std::array<int, 2> model_shape;
private:
    static const int WARMUP_VALID_FRAMES = 3;
    static const int SWITCH_CONFIRM_FRAMES = 3;
    static constexpr float CONF_ON = 0.55f;
    static constexpr float MARGIN_ON = 0.12f;
    static constexpr float SWITCH_MARGIN = 0.08f;
    static const uint64_t MIN_HOLD_MS = 400;
    static const uint64_t UNCERTAIN_TIMEOUT_MS = 1000;

    uint16_t model_id = 0;
    ssne_tensor_t inputs[1]; ssne_tensor_t outputs[1];
    AiPreprocessPipe pipe_offline = GetAIPreprocessPipe();
    uint8_t* prev_frame_buf = nullptr; uint8_t* diff_buf = nullptr;
    bool has_prev_frame = false; int frame_pixels = 0;
    bool model_ready = false;
    bool preprocess_contract_logged = false;
    const char* last_inference_failure_reason = "not_started";
    float last_logits[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    EmotionTemporalBuffer temporal_buffer;

    EmotionClass stable_emotion = EmotionClass::NEUTRAL;
    float stable_confidence = 0.0f;
    EmotionClass candidate_emotion = EmotionClass::NEUTRAL;
    int candidate_count = 0;
    uint64_t candidate_since_ms = 0;
    uint64_t last_switch_ms = 0;
    uint64_t last_valid_ms = 0;

    float motion_mean = 0.0f;
    float motion_m2 = 0.0f;
    uint32_t motion_samples = 0;
    uint64_t last_debug_ms = 0;
    uint64_t stats_start_ms = 0;
    uint32_t total_frames = 0;
    uint32_t accepted_frames = 0;
    uint32_t inference_failures = 0;
    uint32_t quality_rejections = 0;
    uint32_t dark_rejections = 0;
    uint32_t bright_rejections = 0;
    uint32_t contrast_rejections = 0;
    uint32_t motion_rejections = 0;
    uint32_t uncertain_frames = 0;
    uint32_t class_switches = 0;
    float switch_delay_sum_ms = 0.0f;

    bool ImageQualityAccepted(const ssne_tensor_t* img, float* brightness,
                              float* contrast, float* motion_score,
                              const char** reject_reason);
    void KeepLastStableResult(EmotionResult* result) const;
    void ResetTemporalState(uint64_t now_ms);
    void MaybePrintDebug(uint64_t now_ms, const ssne_tensor_t* camera_img,
                         const float raw_probs[4],
                         float brightness, float contrast, float motion_score,
                         bool accepted, const char* reason);
};

enum class HandGestureClass : int {
    GESTURE_0 = 0, GESTURE_1 = 1, GESTURE_2 = 2, GESTURE_3 = 3, GESTURE_4 = 4, GESTURE_5 = 5, NUM_CLASSES = 6
};
static const char* HAND_GESTURE_NAMES[] = { "0", "1", "2", "3", "4", "5" };

struct HandGestureResult {
    HandGestureClass gesture;
    float confidence; float all_probs[6];
    HandGestureResult() : gesture(HandGestureClass::GESTURE_0), confidence(0.f) {
        for (int i = 0; i < 6; i++) all_probs[i] = 0.f;
    }
    void Clear() { 
        gesture = HandGestureClass::GESTURE_0; confidence = 0.f;
        for(int i = 0; i < 6; i++) all_probs[i] = 0.f; 
    }
};

struct HandGestureTemporalBuffer {
    static const int CAPACITY = 5;
    HandGestureClass gestures[CAPACITY];
    float confidences[CAPACITY];
    int head; int count;
    HandGestureTemporalBuffer() : head(0), count(0) {
        for (int i = 0; i < CAPACITY; i++) { gestures[i] = HandGestureClass::GESTURE_0; confidences[i] = 0.f; }
    }
    void Push(HandGestureClass gesture, float confidence) {
        gestures[head] = gesture; confidences[head] = confidence;
        head = (head + 1) % CAPACITY;
        if (count < CAPACITY) count++;
    }
    HandGestureClass GetWeightedVote(float* out_confidence) const {
        float class_score[6] = { 0.f };
        int look = count;
        for (int i = 0; i < look; i++) {
            int idx = ((head - 1 - i) + CAPACITY) % CAPACITY;
            class_score[static_cast<int>(gestures[idx])] += 1.f * confidences[idx];
        }
        int best = 0;
        for (int c = 1; c < 6; c++) { if (class_score[c] > class_score[best]) best = c; }
        if (out_confidence) *out_confidence = (look > 0) ? (class_score[best] / look) : 0.f;
        return static_cast<HandGestureClass>(best);
    }
};

class HANDGESTURECLASSIFIER {
public:
    void Initialize(std::string& model_path, std::array<int, 2>* in_img_shape, std::array<int, 2>* in_model_shape);
    void Predict(ssne_tensor_t* img_in, HandGestureResult* result, bool use_clahe = false);
    void Release();
    std::array<int, 2> img_shape; std::array<int, 2> model_shape;
private:
    uint16_t model_id = 0;
    ssne_tensor_t inputs[1]; ssne_tensor_t outputs[1];
    AiPreprocessPipe pipe_offline = GetAIPreprocessPipe();
    uint8_t* prev_frame_buf = nullptr; uint8_t* diff_buf = nullptr;
    bool has_prev_frame = false; int frame_pixels = 0;
    HandGestureTemporalBuffer temporal_buffer;
    void RunSingleFrameInference(ssne_tensor_t* img, float out_probs[6], bool use_clahe);
};

extern std::atomic<bool> g_signal_received;

inline void setup_signal_handlers() {
    auto handler = [](int) { g_signal_received.store(true); };
    std::signal(SIGINT, handler);
    std::signal(SIGTERM, handler);
}

inline bool nonblocking_getline(std::string& out, int timeout_ms) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    struct timeval tv = {timeout_ms / 1000, (timeout_ms % 1000) * 1000};

    int ret = select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv);
    if (ret > 0) {
        if (std::getline(std::cin, out)) {
            if (!out.empty() && out.back() == '\r') out.pop_back();
            return true;
        }
    }
    return false;
}

inline void clear_stdin_residual() {
    std::cin.clear();
    tcflush(STDIN_FILENO, TCIFLUSH);
}

class SigintBlocker {
    sigset_t old_mask_;
public:
    SigintBlocker() {
        sigset_t new_mask;
        sigemptyset(&new_mask);
        sigaddset(&new_mask, SIGINT);
        sigaddset(&new_mask, SIGTERM);
        pthread_sigmask(SIG_BLOCK, &new_mask, &old_mask_);
    }
    ~SigintBlocker() {
        pthread_sigmask(SIG_SETMASK, &old_mask_, nullptr);
    }
};
