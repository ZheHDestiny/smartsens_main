/*
 * @Filename: yolo.cpp
 * @Description: YOLOv8 速度检测底层实现
 */
#include "common.hpp"
#include "utils.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>

static inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static inline float IoU(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    float x1 = std::max(a[0], b[0]);
    float y1 = std::max(a[1], b[1]);
    float x2 = std::min(a[2], b[2]);
    float y2 = std::min(a[3], b[3]);
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / (area_a + area_b - inter);
}

void YOLOV8_SPEED::Initialize(const std::string& model_path, std::array<int, 2>* in_img_shape, std::array<int, 2>* in_det_shape) {
    img_shape = *in_img_shape; 
    det_shape = *in_det_shape; 
    w_scale = static_cast<float>(img_shape[0]) / det_shape[0]; 
    h_scale = static_cast<float>(img_shape[1]) / det_shape[1];

    pipe_offline = GetAIPreprocessPipe();
    model_id = ssne_loadmodel(const_cast<char*>(model_path.c_str()), SSNE_STATIC_ALLOC); 
    
    SetNormalize(pipe_offline, model_id); 
    
    inputs[0] = create_tensor(det_shape[0], det_shape[1], SSNE_Y_8, SSNE_BUF_AI);
    memset(outputs, 0, sizeof(outputs));
    
    if (!get_data(inputs[0])) {
        fprintf(stderr, "[FATAL] YOLOV8_SPEED::Initialize - OCM Memory is FULL! Failed to allocate inputs[0].\n");
        exit(EXIT_FAILURE);
    }
}

void YOLOV8_SPEED::DecodeHeadOutputs(const float* cls_head, const float* reg_head,
                              int height, int width, int stride, float conf_threshold,
                              std::vector<std::array<float, 4>>& boxes,
                              std::vector<float>& scores, std::vector<int>& class_ids) {
    std::array<float, 16> softmax_buf = {}; 
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int spatial_idx = y * width + x;
            int best_class = 0;
            float best_score = Sigmoid(cls_head[spatial_idx * kNumClasses + 0]);
            
            for (int cls_id = 1; cls_id < kNumClasses; ++cls_id) {
                const float score = Sigmoid(cls_head[spatial_idx * kNumClasses + cls_id]);
                if (score > best_score) {
                    best_score = score;
                    best_class = cls_id;
                }
            }

            if (best_score > 1.0f) {
                best_score = 1.0f;
            }

            if (best_score < conf_threshold) continue;

            std::array<float, 4> dist = {};
            for (int side = 0; side < 4; ++side) {
                float max_logit = -1e9f;
                for (int bin = 0; bin < kRegBins; ++bin) {
                    const int c = side * kRegBins + bin;
                    max_logit = std::max(max_logit, reg_head[spatial_idx * (4 * kRegBins) + c]);
                }
                float sum = 0.0f;
                for (int bin = 0; bin < kRegBins; ++bin) {
                    const int c = side * kRegBins + bin;
                    float diff = reg_head[spatial_idx * (4 * kRegBins) + c] - max_logit;
                    if (diff < -20.0f) diff = -20.0f; 
                    softmax_buf[bin] = std::exp(diff);
                    sum += softmax_buf[bin];
                }
                float expectation = 0.0f;
                for (int bin = 0; bin < kRegBins; ++bin) {
                    expectation += (softmax_buf[bin] / sum) * static_cast<float>(bin);
                }
                dist[side] = expectation;
            }

            const float anchor_x = static_cast<float>(x) + 0.5f;
            const float anchor_y = static_cast<float>(y) + 0.5f;
            float x1 = (anchor_x - dist[0]) * stride;
            float y1 = (anchor_y - dist[1]) * stride;
            float x2 = (anchor_x + dist[2]) * stride;
            float y2 = (anchor_y + dist[3]) * stride;

            x1 = std::max(0.0f, std::min(x1, static_cast<float>(det_shape[0])));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(det_shape[1])));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(det_shape[0])));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(det_shape[1])));

            boxes.push_back({x1, y1, x2, y2});
            scores.push_back(best_score);
            class_ids.push_back(best_class);
        }
    }
}

void YOLOV8_SPEED::Postprocess(std::vector<std::array<float, 4>>* boxes,
                        std::vector<float>* scores, std::vector<int>* class_ids,
                        ObjectDetectionResult* result) {
    std::vector<int> order(boxes->size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);

    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return scores->at(lhs) > scores->at(rhs);
    });
    if (static_cast<int>(order.size()) > top_k) order.resize(top_k);

    std::vector<int> keep;
    keep.reserve(order.size());
    for (int idx : order) {
        bool suppressed = false;
        for (int kept_idx : keep) {
            
            if (IoU(boxes->at(idx), boxes->at(kept_idx)) > nms_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) keep.push_back(idx);
        if (static_cast<int>(keep.size()) >= keep_top_k) break;
    }

    result->Clear();
    result->Reserve(static_cast<int>(keep.size()));
    for (int idx : keep) {
        std::array<float, 4> box = boxes->at(idx);
        float x1 = box[0] * w_scale;
        float y1 = box[1] * h_scale;
        float x2 = box[2] * w_scale;
        float y2 = box[3] * h_scale;
        box[0] = std::max(0.0f, std::min(x1, static_cast<float>(img_shape[0])));
        box[1] = std::max(0.0f, std::min(y1, static_cast<float>(img_shape[1])));
        box[2] = std::max(0.0f, std::min(x2, static_cast<float>(img_shape[0])));
        box[3] = std::max(0.0f, std::min(y2, static_cast<float>(img_shape[1])));
        
        result->boxes.emplace_back(box);
        result->scores.emplace_back(scores->at(idx));
        result->class_ids.emplace_back(class_ids->at(idx));
    }
}

void YOLOV8_SPEED::Predict(ssne_tensor_t* img, ObjectDetectionResult* result, float conf_threshold) {
    uint8_t* src_data = (uint8_t*)get_data(*img); 
    if (!src_data) return;
    int ret = RunAiPreprocessPipe(pipe_offline, *img, inputs[0]);
    if (ret != 0) {
        result->Clear();
        return;
    }

    ret = ssne_inference(model_id, 1, inputs);
    if (ret) {
        result->Clear();
        return;
    }

    int getout_ret = ssne_getoutput(model_id, 6, outputs);
    if (getout_ret != 0) {
        fprintf(stderr, "[ERROR] ssne_getoutput failed! ret=%d\n", getout_ret);
        result->Clear();
        return;
    }

    for (int i = 0; i < 6; i++) {
        if (get_data(outputs[i]) == nullptr) {
            fprintf(stderr, "[ERROR] ssne_getoutput returned null data for output %d\n", i);
            result->Clear();
            return;
        }
    }

    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    for (int i = 0; i < 3; ++i) {
        int cls_idx = i;
        int reg_idx = i + 3; 
        
        int feat_w = get_width(outputs[cls_idx]);
        int feat_h = get_height(outputs[cls_idx]);
        
        int stride = 320 / feat_w; 
        
        DecodeHeadOutputs((float*)get_data(outputs[cls_idx]), (float*)get_data(outputs[reg_idx]), 
                          feat_h, feat_w, stride, conf_threshold, boxes, scores, class_ids);
    }

    Postprocess(&boxes, &scores, &class_ids, result);
}

void YOLOV8_SPEED::Release() {
    release_tensor(inputs[0]);
    // NOTE: outputs[i] 由 ssne_getoutput 填充，其 data 指向模型内部 buffer，
    // 不应由 release_tensor 释放。ssne_release() 会统一释放模型资源。
    for(int i=0; i<6; i++) {
        outputs[i].data = nullptr;
    }
    ReleaseAIPreprocessPipe(pipe_offline);
}