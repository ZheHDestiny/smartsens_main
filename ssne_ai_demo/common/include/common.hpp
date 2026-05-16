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
#include <cstring>
#include <atomic>
#include <csignal>
#include <sys/select.h>
#include <unistd.h>
#include <iostream>
#include <limits>
#include <termios.h>
#include "smartsoc/ssne_api.h"

class IMAGEPROCESSOR {
public:
    void Initialize(std::array<int, 2>* in_img_shape, 
                    uint16_t crop_x1, uint16_t crop_x2, 
                    uint16_t crop_y1, uint16_t crop_y2,
                    uint16_t out_w, uint16_t out_h);
                    
    void GetImage(ssne_tensor_t* img_sensor);
    void Release();
    
    std::array<int, 2> img_shape;
private:
    uint8_t format_online;
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
    float danger_level[REGION_COUNT];
    bool has_obstacle[REGION_COUNT];
    int most_dangerous_region;
    int priority;

    ObstacleInfo() {
        for (int i = 0; i < REGION_COUNT; i++) {
            danger_level[i] = 0.0f;
            has_obstacle[i] = false;
        }
        most_dangerous_region = CENTER;
        priority = 4;
    }
};

class OPTICALFLOW {
public:
    void Initialize(int width, int height);
    void ComputeFlow(const uint8_t* prev_frame, const uint8_t* curr_frame, std::vector<FeaturePoint>& features);
    void DetectFeatures(const uint8_t* frame, std::vector<FeaturePoint>& features);
    void Release();

    int max_features;      
    int fast_threshold;    
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

    void BuildPyramid(const uint8_t* frame, std::vector<std::vector<uint8_t>>& pyramid);
    static float Interp(const uint8_t* img, int w, int h, float x, float y);
    bool TrackPointSingleLevel(const uint8_t* prev, const uint8_t* curr, int w, int h, float px, float py, float& cx, float& cy);
};

class OBSTACLE_DETECTOR {
public:
    void Initialize(int width, int height);
    void DetectObstacles(const std::vector<FeaturePoint>& features, ObstacleInfo& obstacle_info);
    void Release();

    float ttc_threshold;   
    float divergence_threshold;  

private:
    int width_, height_;
    float ComputeTTC(float x, float y, float dx, float dy);
    float ComputeDivergence(const std::vector<FeaturePoint>& features);
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
    static const int CAPACITY = 5;
    EmotionClass emotions[CAPACITY];
    float confidences[CAPACITY];
    int head; int count;
    EmotionTemporalBuffer() : head(0), count(0) {
        for (int i = 0; i < CAPACITY; i++) { emotions[i] = EmotionClass::NEUTRAL; confidences[i] = 0.f; }
    }
    void Push(EmotionClass emotion, float confidence) {
        emotions[head] = emotion; confidences[head] = confidence;
        head = (head + 1) % CAPACITY;
        if (count < CAPACITY) count++;
    }
    EmotionClass GetWeightedVote(float* out_confidence) const {
        float class_score[4] = { 0.f };
        int look = count;
        for (int i = 0; i < look; i++) {
            int idx = ((head - 1 - i) + CAPACITY) % CAPACITY;
            class_score[static_cast<int>(emotions[idx])] += 1.f * confidences[idx];
        }
        int best = 0;
        for (int c = 1; c < 4; c++) { if (class_score[c] > class_score[best]) best = c; }
        if (out_confidence) *out_confidence = (look > 0) ? (class_score[best] / look) : 0.f;
        return static_cast<EmotionClass>(best);
    }
};

class EMOTIONCLASSIFIER {
public:
    void Initialize(std::string& model_path, std::array<int, 2>* in_img_shape, std::array<int, 2>* in_model_shape);
    void Predict(ssne_tensor_t* img_in, EmotionResult* result);
    void RunSingleFrameInference(ssne_tensor_t* img, float out_probs[4]);
    void Release();
    std::array<int, 2> img_shape; std::array<int, 2> model_shape;
private:
    uint16_t model_id = 0;
    ssne_tensor_t inputs[1]; ssne_tensor_t outputs[1];
    AiPreprocessPipe pipe_offline = GetAIPreprocessPipe();
    uint8_t* prev_frame_buf = nullptr; uint8_t* diff_buf = nullptr;
    bool has_prev_frame = false; int frame_pixels = 0;
    EmotionTemporalBuffer temporal_buffer;
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