/*
 * @Filename: yolov8_det.cpp
 * @Description: YOLOv8 目标检测算法实现
 */
#include "common.hpp"
#include "utils.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float IoU(const std::array<float, 4>& a, const std::array<float, 4>& b) {
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

// Letterbox 参数（供 Postprocess 做正确逆映射）
static float g_letterbox_pad_x = 0.0f;
static float g_letterbox_pad_y = 0.0f;
static float g_letterbox_scale = 1.0f;

void YOLOV8_OBJECT::Initialize(const std::string& model_path, std::array<int, 2>* in_img_shape, std::array<int, 2>* in_det_shape) {
    img_shape = *in_img_shape;
    det_shape = *in_det_shape;
    w_scale = static_cast<float>(img_shape[0]) / det_shape[0];
    h_scale = static_cast<float>(img_shape[1]) / det_shape[1];
    pipe_offline = GetAIPreprocessPipe();

    model_id = ssne_loadmodel(const_cast<char*>(model_path.c_str()), SSNE_STATIC_ALLOC);
    SetNormalize(pipe_offline, model_id);
    inputs[0] = create_tensor(det_shape[0], det_shape[1], SSNE_Y_8, SSNE_BUF_AI);
    memset(outputs, 0, sizeof(outputs));
}

void YOLOV8_OBJECT::DecodeHeadOutputs(const float* cls_head, const float* reg_head,
    int height, int width, int stride, float conf_threshold,
    std::vector<std::array<float, 4>>& boxes,
    std::vector<float>& scores, std::vector<int>& class_ids) {
    std::array<float, 16> softmax_buf = {};
    const int spatial_size = height * width;
    static bool printed_first_pixel = true;

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

            boxes.push_back({ x1, y1, x2, y2 });
            scores.push_back(best_score);
            class_ids.push_back(best_class);
        }
    }
}

void YOLOV8_OBJECT::Postprocess(std::vector<std::array<float, 4>>* boxes,
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
            if (class_ids->at(idx) != class_ids->at(kept_idx)) continue;
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

        // 正确逆映射：从 letterbox 坐标系映射到原始图像坐标系
        float x1 = (box[0] - g_letterbox_pad_x) / g_letterbox_scale;
        float y1 = (box[1] - g_letterbox_pad_y) / g_letterbox_scale;
        float x2 = (box[2] - g_letterbox_pad_x) / g_letterbox_scale;
        float y2 = (box[3] - g_letterbox_pad_y) / g_letterbox_scale;

        // 裁剪到图像边界
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_shape[0])));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_shape[1])));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(img_shape[0])));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(img_shape[1])));

        result->boxes.emplace_back(std::array<float, 4>{x1, y1, x2, y2});
        result->scores.emplace_back(scores->at(idx));
        result->class_ids.emplace_back(class_ids->at(idx));
    }
}

static ssne_tensor_t g_letterbox_tensor = {};
static bool g_lb_init = false;

void YOLOV8_OBJECT::Predict(ssne_tensor_t* img, ObjectDetectionResult* result, float conf_threshold) {
    // ---------- 输入校验 ----------
    if (img == nullptr) {
        fprintf(stderr, "[ERROR] Predict received null img pointer\n");
        result->Clear();
        return;
    }

    uint8_t* sensor_data = (uint8_t*)get_data(*img);
    if (sensor_data == nullptr) {
        fprintf(stderr, "[ERROR] Input image data is null\n");
        result->Clear();
        return;
    }

    // 尝试从 tensor 读取实际尺寸，失败则回退到 img_shape
    int src_w = get_width(*img);
    int src_h = get_height(*img);
    if (src_w <= 0 || src_h <= 0) {
        fprintf(stderr, "[WARN] get_width/get_height invalid (%dx%d), fallback to img_shape\n", src_w, src_h);
        src_w = img_shape[0];
        src_h = img_shape[1];
    }

    if (src_w != img_shape[0] || src_h != img_shape[1]) {
        fprintf(stderr, "[WARN] Input size mismatch: expected %dx%d, got %dx%d\n",
            img_shape[0], img_shape[1], src_w, src_h);
    }

    // ---------- 准备 letterbox tensor ----------
    if (!g_lb_init) {
        g_letterbox_tensor = create_tensor(det_shape[0], det_shape[1], SSNE_Y_8, SSNE_BUF_AI);
        g_lb_init = get_data(g_letterbox_tensor) != nullptr &&
                    get_mem_size(g_letterbox_tensor) >=
                        static_cast<size_t>(det_shape[0] * det_shape[1]);
    }
    uint8_t* lb_data = (uint8_t*)get_data(g_letterbox_tensor);
    if (!g_lb_init || lb_data == nullptr) {
        fprintf(stderr, "[ERROR] Failed to allocate object letterbox tensor\n");
        result->Clear();
        return;
    }
    // 每次调用前清零整个 tensor，防止 padding 区域被上次推理残留污染
    memset(lb_data, 0, det_shape[0] * det_shape[1]);

    // ---------- 计算保持 aspect ratio 的 letterbox 参数 ----------
    float scale_w = static_cast<float>(det_shape[0]) / src_w;
    float scale_h = static_cast<float>(det_shape[1]) / src_h;
    float scale = std::min(scale_w, scale_h);

    int target_w = static_cast<int>(src_w * scale);
    int target_h = static_cast<int>(src_h * scale);
    int pad_x = (det_shape[0] - target_w) / 2;
    int pad_y = (det_shape[1] - target_h) / 2;

    // 保存供 Postprocess 做逆映射
    g_letterbox_scale = scale;
    g_letterbox_pad_x = static_cast<float>(pad_x);
    g_letterbox_pad_y = static_cast<float>(pad_y);

    // ---------- 浮点 nearest-neighbor resize ----------
    // 核心修复：用浮点坐标 + round，彻底避免整数除法的 temporal aliasing
    for (int y = 0; y < target_h; y++) {
        float src_y_f = (y + 0.5f) / scale - 0.5f;
        int src_y = static_cast<int>(roundf(src_y_f));
        src_y = std::max(0, std::min(src_y, src_h - 1));

        uint8_t* dst_row = lb_data + ((y + pad_y) * det_shape[0]) + pad_x;
        uint8_t* src_row = sensor_data + (src_y * src_w);  // 假设 stride == width

        for (int x = 0; x < target_w; x++) {
            float src_x_f = (x + 0.5f) / scale - 0.5f;
            int src_x = static_cast<int>(roundf(src_x_f));
            src_x = std::max(0, std::min(src_x, src_w - 1));
            dst_row[x] = src_row[src_x];
        }
    }

    // ---------- 推理 ----------
    int ret = RunAiPreprocessPipe(pipe_offline, g_letterbox_tensor, inputs[0]);
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

    // ---------- 解码 ----------
    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    int cls_80_idx = 0; int cls_40_idx = 1; int cls_20_idx = 2;
    int reg_80_idx = 3; int reg_40_idx = 4; int reg_20_idx = 5;

    DecodeHeadOutputs((float*)get_data(outputs[cls_80_idx]), (float*)get_data(outputs[reg_80_idx]),
        get_height(outputs[cls_80_idx]), get_width(outputs[cls_80_idx]), 8,
        conf_threshold, boxes, scores, class_ids);

    DecodeHeadOutputs((float*)get_data(outputs[cls_40_idx]), (float*)get_data(outputs[reg_40_idx]),
        get_height(outputs[cls_40_idx]), get_width(outputs[cls_40_idx]), 16,
        conf_threshold, boxes, scores, class_ids);

    DecodeHeadOutputs((float*)get_data(outputs[cls_20_idx]), (float*)get_data(outputs[reg_20_idx]),
        get_height(outputs[cls_20_idx]), get_width(outputs[cls_20_idx]), 32,
        conf_threshold, boxes, scores, class_ids);
    Postprocess(&boxes, &scores, &class_ids, result);
}

void YOLOV8_OBJECT::Release() {
    if (get_data(inputs[0]) != nullptr) {
        release_tensor(inputs[0]);
        inputs[0].data = nullptr;
    }
    if (g_lb_init) {
        release_tensor(g_letterbox_tensor);
        g_letterbox_tensor.data = nullptr;
        g_lb_init = false;
    }
    // NOTE: outputs[i] 由 ssne_getoutput 填充，其 data 指向模型内部 buffer，
    // 不应由 release_tensor 释放。ssne_release() 会统一释放模型资源。
    for (int i = 0; i < 6; i++) {
        outputs[i].data = nullptr;
    }
    if (pipe_offline != nullptr) {
        ReleaseAIPreprocessPipe(pipe_offline);
        pipe_offline = nullptr;
    }
}
