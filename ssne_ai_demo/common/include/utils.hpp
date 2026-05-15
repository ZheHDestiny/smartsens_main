/*
 * @Filename: utils.hpp
 * @Description: SSNE AI Demo 统一工具函数与可视化器声明
 */
#pragma once

#include "osd-device.hpp"
#include "common.hpp"
#include <algorithm>
#include <vector>
#include <array>
#include <string>

namespace utils {
    void Merge(FaceDetectionResult* result, size_t low, size_t mid, size_t high);
    void MergeSort(FaceDetectionResult* result, size_t low, size_t high);
    void SortDetectionResult(FaceDetectionResult* result);
    void NMS(FaceDetectionResult* result, float iou_threshold, int top_k);

    void Merge(ObjectDetectionResult* result, size_t low, size_t mid, size_t high);
    void MergeSort(ObjectDetectionResult* result, size_t low, size_t high);
    void SortDetectionResult(ObjectDetectionResult* result);
    void NMS(ObjectDetectionResult* result, float iou_threshold, int top_k);

    GestureClass ArgmaxProbs(const float probs[4], float* out_conf);
    EmotionClass ArgmaxProbsEmotion(const float probs[4], float* out_conf);
    HandGestureClass ArgmaxProbsHandGesture(const float probs[6], float* out_conf);
}

class VISUALIZER {
public:
    void Initialize(std::array<int, 2>& in_img_shape, const std::string& bitmap_lut_path = "");
    void Release();
    void Clear();

    static const int DETECTION_LAYER_ID = 0;

    void Draw(); 
    void Draw(const std::vector<std::array<float, 4>>& boxes); 
    void Draw(const std::vector<std::array<float, 4>>& boxes,
              const std::vector<float>& scores,
              const std::vector<int>& class_ids);
    void DrawSpeed(const std::vector<std::array<float, 4>>& boxes,
                   const std::vector<float>& scores,
                   const std::vector<int>& class_ids,
                   const std::vector<float>& speeds,
                   const std::vector<int>& directions,
                   int crop_offset_y = 0,
                   int crop_height = 0);
    void DrawAll(const std::vector<FeaturePoint>& features,
                 const ObstacleInfo& obstacle_info,
                 int crop_offset_y,
                uint32_t frame_count = 0);
    void Draw(const RpsResult& result, const std::array<float, 4>& hand_roi);

    void DrawFixedSquare(int x_min, int y_min, int x_max, int y_max, int layer_id = 1);
    void DrawBitmap(const std::string& bitmap_path, const std::string& lut_path = "", int pos_x = 0, int pos_y = 0, int layer_id = 2);
    void DrawSimple(const EmotionResult& result, const std::array<float, 4>& hand_roi);
    void Draw(const HandGestureResult& result, const std::array<float, 4>& hand_roi);
private:
    sst::device::osd::OsdDevice osd_device;
    int m_width;   
    int m_height;  
    std::string m_bitmap_lut_path_full;  
    bool enabled_ = false;

    static const int LAYER_FEATURES  = 0;
    static const int LAYER_OBSTACLES = 1;
    static const int LAYER_BITMAP    = 2;
    static const int LAYER_SAFEDIR   = 3;
    static const int LAYER_MASK      = 4; 

    static const int MAX_FEATURE_MARKERS = 24;
    static const int MAX_FLOW_ARROWS     = 10;
    
    void DrawFeaturePoints(const std::vector<FeaturePoint>& features, int crop_offset_y);
    void DrawFlowArrows(const std::vector<FeaturePoint>& features, int crop_offset_y);
    void DrawObstacleRegions(const ObstacleInfo& obstacle_info, int crop_offset_y,uint32_t frame_count);
    void DrawSafeDirection(const ObstacleInfo& obstacle_info, int crop_offset_y);
    void DrawMask(int crop_offset_y, int crop_height); 
    
    static const int C_ROCK     = 0;
    static const int C_PAPER    = 1;
    static const int C_SCISSORS = 2;
    static const int C_IDLE     = 3;
    static const int C_WINDUP   = 4;
    static const int HUMAN_CX   = 180;
    static const int AI_CX      = 540;
    static const int ICON_Y0    = 110;
    static const int ICON_H     = 420;
    static const int STATUS_X1 = 5, STATUS_X2 = 715, STATUS_Y1 = 5, STATUS_Y2 = 85;
    static const int CONF_X0 = 10, CONF_Y1 = 1200, CONF_Y2 = 1260, CONF_MAXW = 700;

    int GestureColor(GestureClass g);
    void DrawGestureIcon(int layer_id, int cx, int y0, GestureClass gesture, bool active);
    void DrawRock(int layer_id, int cx, int y0, int color, bool active);
    void DrawPaper(int layer_id, int cx, int y0, int color, bool active);
    void DrawScissors(int layer_id, int cx, int y0, int color, bool active);
    void DrawIdle(int layer_id, int cx, int y0);
    void DrawStatusBar(int layer_id, GameState state, GestureClass locked);
    void DrawConfBar(int layer_id, float confidence, int color);
    void CommitBoxes(int layer_id, int border, int color, std::vector<std::array<float,4>>& boxes);
    static const int C_SURPRISE = 2;   
    static const int C_HAPPY    = 3;
    static const int C_SAD      = 0;   
    static const int C_NEUTRAL  = 6;   
    int EmotionColor(EmotionClass e);
    void DrawEmotionIcon(int layer_id, EmotionClass emotion);
    EmotionClass last_drawn_emotion = EmotionClass::NUM_CLASSES; 
    EmotionClass candidate_emotion = EmotionClass::NUM_CLASSES; 
    int emotion_hold_count = 0; 
    static const int C_GESTURE0 = 0; 
    static const int C_GESTURE1 = 1;
    int HandGestureColor(HandGestureClass g);
    void DrawResultLines(int layer_id_primary, int layer_id_secondary, HandGestureClass gesture, const std::array<float, 4>& hand_roi);
};