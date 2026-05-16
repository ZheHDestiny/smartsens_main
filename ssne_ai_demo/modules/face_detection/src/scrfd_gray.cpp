/*
 * @Filename: scrfd_gray.cpp
 * @Author: Hongying He
 * @Email: hongying.he@smartsenstech.com
 * @Date: 2025-12-30 14-57-47
 * @Copyright (c) 2025 SmartSens
 * @Description: SCRFD灰度图人脸检测实现文件
 */
#include <assert.h>
#include "utils.hpp"
#include "common.hpp"
#include <iostream>
#include <cstdio>

namespace utils {

/**
 * @brief 归并排序的合并操作
 * @param result 人脸检测结果结构体指针
 * @param low 合并区间的起始索引
 * @param mid 合并区间的中间索引
 * @param high 合并区间的结束索引
 * @description 将两个已排序的子数组合并成一个有序数组，按照分数从高到低排序
 */
void Merge(FaceDetectionResult* result, size_t low, size_t mid, size_t high) {
  std::vector<std::array<float, 4>>& boxes = result->boxes;
  std::vector<float>& scores = result->scores;
  std::vector<std::array<float, 4>> temp_boxes(boxes);
  std::vector<float> temp_scores(scores);
  size_t i = low;      // 左半部分的索引
  size_t j = mid + 1;  // 右半部分的索引
  size_t k = i;        // 合并结果的索引
  for (; i <= mid && j <= high; k++) {
    if (temp_scores[i] >= temp_scores[j]) {
      scores[k] = temp_scores[i];
      boxes[k] = temp_boxes[i];
      i++;
    } else {
      scores[k] = temp_scores[j];
      boxes[k] = temp_boxes[j];
      j++;
    }
  }
  while (i <= mid) {
    scores[k] = temp_scores[i];
    boxes[k] = temp_boxes[i];
    k++;
    i++;
  }
  while (j <= high) {
    scores[k] = temp_scores[j];
    boxes[k] = temp_boxes[j];
    k++;
    j++;
  }
}

/**
 * @brief 归并排序递归函数
 * @param result 人脸检测结果结构体指针
 * @param low 排序区间的起始索引
 * @param high 排序区间的结束索引
 * @description 使用归并排序算法对检测结果按分数从高到低排序
 */
void MergeSort(FaceDetectionResult* result, size_t low, size_t high) {
  if (low < high) {
    size_t mid = (high - low) / 2 + low;  // 计算中间索引
    MergeSort(result, low, mid);          // 递归排序左半部分
    MergeSort(result, mid + 1, high);     // 递归排序右半部分
    Merge(result, low, mid, high);        // 合并两个有序子数组
  }
}

/**
 * @brief 对检测结果进行排序
 * @param result 人脸检测结果结构体指针
 * @description 按照检测分数从高到低对检测结果进行排序
 */
void SortDetectionResult(FaceDetectionResult* result) {
  size_t low = 0;
  size_t high = result->scores.size();
  if (high == 0) {
    return;  // 如果没有检测结果，直接返回
  }
  high = high - 1;  // 转换为最大索引
  MergeSort(result, low, high);
}

/**
 * @brief 非极大值抑制（NMS）算法
 * @param result 人脸检测结果结构体指针
 * @param iou_threshold IoU阈值，超过此阈值的重叠框将被抑制
 * @param top_k 保留前k个检测结果
 * @description 去除重叠的检测框，保留分数最高的检测结果
 */
void NMS(FaceDetectionResult* result, float iou_threshold, int top_k) {
  SortDetectionResult(result);

  int res_count = static_cast<int>(result->boxes.size());
  result->Resize(std::min(res_count, top_k));
  
  std::vector<float> area_of_boxes(result->boxes.size());
  std::vector<int> suppressed(result->boxes.size(), 0);  // 标记被抑制的框
  for (size_t i = 0; i < result->boxes.size(); ++i) {
    area_of_boxes[i] = (result->boxes[i][2] - result->boxes[i][0] + 1) *
                       (result->boxes[i][3] - result->boxes[i][1] + 1);
  }
  
  for (size_t i = 0; i < result->boxes.size(); ++i) {
    if (suppressed[i] == 1) {
      continue;  // 跳过已被抑制的框
    }
    for (size_t j = i + 1; j < result->boxes.size(); ++j) {
      if (suppressed[j] == 1) {
        continue;  // 跳过已被抑制的框
      }
      float xmin = std::max(result->boxes[i][0], result->boxes[j][0]);
      float ymin = std::max(result->boxes[i][1], result->boxes[j][1]);
      float xmax = std::min(result->boxes[i][2], result->boxes[j][2]);
      float ymax = std::min(result->boxes[i][3], result->boxes[j][3]);
      float overlap_w = std::max(0.0f, xmax - xmin + 1);
      float overlap_h = std::max(0.0f, ymax - ymin + 1);
      float overlap_area = overlap_w * overlap_h;
      float overlap_ratio =
          overlap_area / (area_of_boxes[i] + area_of_boxes[j] - overlap_area);
      if (overlap_ratio > iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }
  FaceDetectionResult backup(*result);
  int landmarks_per_face = result->landmarks_per_face;

  result->Clear();
  result->landmarks_per_face = landmarks_per_face;
  result->Reserve(suppressed.size());
  for (size_t i = 0; i < suppressed.size(); ++i) {
    if (suppressed[i] == 1) {
      continue;  // 跳过被抑制的框
    }
    result->boxes.emplace_back(backup.boxes[i]);
    result->scores.push_back(backup.scores[i]);
    if (result->landmarks_per_face > 0) {
      for (size_t j = 0; j < result->landmarks_per_face; ++j) {
        result->landmarks.emplace_back(
            backup.landmarks[i * result->landmarks_per_face + j]);
      }
    }
  }
}
}  // namespace utils


/**
 * @brief 释放FaceDetectionResult的内存
 * @description 使用swap技巧释放vector占用的内存
 */
void FaceDetectionResult::Free() {
  std::vector<std::array<float, 4>>().swap(boxes);
  std::vector<float>().swap(scores);
  std::vector<std::array<float, 2>>().swap(landmarks);
  landmarks_per_face = 0;
}

/**
 * @brief 清空FaceDetectionResult的内容
 * @description 清空所有检测框、分数和关键点，但保留内存分配
 */
void FaceDetectionResult::Clear() {
  boxes.clear();
  scores.clear();
  landmarks.clear();
  landmarks_per_face = 0;
}

/**
 * @brief 预分配内存空间
 * @param size 要保留的元素数量
 * @description 为检测框、分数和关键点预分配内存，提高性能
 */
void FaceDetectionResult::Reserve(int size) {
  boxes.reserve(size);
  scores.reserve(size);
  if (landmarks_per_face > 0) {
    landmarks.reserve(size * landmarks_per_face);
  }
}

/**
 * @brief 调整FaceDetectionResult的大小
 * @param size 新的元素数量
 * @description 调整检测框、分数和关键点的数量
 */
void FaceDetectionResult::Resize(int size) {
  boxes.resize(size);
  scores.resize(size);
  if (landmarks_per_face > 0) {
    landmarks.resize(size * landmarks_per_face);
  }
}

/**
 * @brief FaceDetectionResult的拷贝构造函数
 * @param res 要拷贝的FaceDetectionResult对象
 * @description 深拷贝检测结果的所有数据
 */
FaceDetectionResult::FaceDetectionResult(const FaceDetectionResult& res) {
  boxes.assign(res.boxes.begin(), res.boxes.end());
  landmarks.assign(res.landmarks.begin(), res.landmarks.end());
  scores.assign(res.scores.begin(), res.scores.end());
  landmarks_per_face = res.landmarks_per_face;
}

/**
 * @brief 生成anchor boxes（锚框）
 * @description 为不同尺度的特征图生成所有可能的anchor boxes，用于目标检测
 */
void SCRFDGRAY::GenerateBoxes() {
    std::array<int, 2> tmp_img_shape = {det_shape[1], det_shape[0]};
    std::vector<std::array<int, 2>> feature_maps;
    for (int step_idx = 0; step_idx < steps.size(); step_idx++) {
        int ix = int(ceil(tmp_img_shape[0] / steps[step_idx]));
        int iy = int(ceil(tmp_img_shape[1] / steps[step_idx]));
        feature_maps.push_back({ix, iy});
    }

    for (int k = 0; k < feature_maps.size(); k++) {
        for (int i = 0; i < feature_maps[k][0]; i++) {
            for (int j = 0; j < feature_maps[k][1]; j++) {
                for (auto ratio : ratios) {
                    float tmp_ratio = sqrt(ratio);
                    for (auto min_size : min_sizes[k]) {
                        float s_kx = static_cast<float>(min_size);
                        float s_ky = static_cast<float>(min_size);
                        s_kx /= tmp_ratio;
                        s_ky *= tmp_ratio;
                        float dense_cx = static_cast<float>(j * steps[k]);
                        float dense_cy = static_cast<float>(i * steps[k]);
                        anchors.push_back({dense_cx, dense_cy, s_kx, s_ky});
                    }
                }
            }
        }
    }
}

/**
 * @brief 解码检测框坐标
 * @param boxes 检测框数组，输入为相对坐标，输出为绝对坐标
 * @description 将网络输出的相对坐标转换为图像上的绝对坐标，并进行边界裁剪
 */
void SCRFDGRAY::DecodeBoxes(std::vector<std::array<float, 4>>& boxes) {
    size_t numBoxes = boxes.size();
    for (unsigned int i = 0; i < numBoxes; i++) {
        boxes[i][0] = fmax(0.0f, anchors[i][0] - boxes[i][0]);
        boxes[i][1] = fmax(0.0f, anchors[i][1] - boxes[i][1]);
        boxes[i][2] = fmin(det_shape[0], anchors[i][0] + boxes[i][2]);
        boxes[i][3] = fmin(det_shape[1], anchors[i][1] + boxes[i][3]);
    }
}

/**
 * @brief 后处理函数
 * @param boxes 检测框数组指针
 * @param scores 检测分数数组指针
 * @param result 输出的人脸检测结果
 * @param conf_threshold 置信度阈值指针
 * @description 对网络推理结果进行解码、过滤、NMS和尺度恢复
 */
void SCRFDGRAY::Postprocess(std::vector<std::array<float, 4>>* boxes, 
    std::vector<float>* scores, FaceDetectionResult* result, 
    float* conf_threshold) {
    DecodeBoxes(*boxes);

    size_t num_res = boxes->size();

    result->Clear();
    result->Reserve(num_res);
    int res_count = 0;
    for (unsigned int i = 0; i < num_res; i++) {
        float score = scores->at(i);
        if (score <= *conf_threshold) {
            continue;  // 跳过低分检测框
        }
        result->boxes.emplace_back(boxes->at(i));
        result->scores.push_back(scores->at(i));
        res_count += 1;
    }
    result->Resize(res_count);
    
    utils::NMS(result, nms_threshold, top_k);
    
    res_count = static_cast<int>(result->boxes.size());
    result->Resize(std::min(res_count, keep_top_k));

    for (unsigned int i = 0; i < result->boxes.size(); i++) {
        result->boxes[i][0] = result->boxes[i][0] * w_scale;  // x1
        result->boxes[i][1] = result->boxes[i][1] * h_scale;  // y1
        result->boxes[i][2] = result->boxes[i][2] * w_scale;  // x2
        result->boxes[i][3] = result->boxes[i][3] * h_scale;  // y2
    }
}

/**
 * @brief 初始化SCRFD检测器
 * @param model_path 模型文件路径
 * @param in_img_shape 裁剪后图像尺寸 [宽度, 高度]
 * @param in_det_shape 检测输入尺寸 [宽度, 高度]
 * @param in_use_kps 是否使用关键点检测
 * @param in_box_len 检测框长度
 * @description 初始化检测器的各种参数，加载模型，生成anchor boxes
 */
void SCRFDGRAY::Initialize(std::string& model_path, std::array<int, 2>* in_img_shape, 
                           std::array<int, 2>* in_det_shape, bool in_use_kps,
                           int in_box_len) {
                            
    nms_threshold = 0.2;   // NMS的IoU阈值
    keep_top_k = 30;       // 最终保留的检测框数量
    top_k = 150;           // NMS前保留的检测框数量
    img_shape = *in_img_shape;  // 原始图像尺寸
    det_shape = *in_det_shape;  // 检测输入尺寸
    use_kps = in_use_kps;       // 是否使用关键点
    box_len = in_box_len;

    w_scale = static_cast<float>(img_shape[0]) / static_cast<float>(det_shape[0]);
    h_scale = static_cast<float>(img_shape[1]) / static_cast<float>(det_shape[1]);

    min_sizes = {{16, 32}, {64, 128}, {256, 512}};  // 每层的最小尺寸
    steps = {8, 16, 32};                             // 每层的步长
    variance = {0.1, 0.2};                           // 方差参数
    clip = false;                                     // 是否裁剪到图像边界
    ratios = {1.0};                                  // 宽高比

    GenerateBoxes();
    
    char* model_path_char = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(model_path_char, SSNE_STATIC_ALLOC);

    uint32_t det_width = static_cast<uint32_t>(det_shape[0]);
    uint32_t det_height = static_cast<uint32_t>(det_shape[1]);
    inputs[0] = create_tensor(det_width, det_height, SSNE_Y_8, SSNE_BUF_AI);
    memset(outputs, 0, sizeof(outputs));
}



/**
 * @brief 执行人脸检测预测
 * @param img 输入图像tensor
 * @param result 输出的人脸检测结果
 * @param conf_threshold 置信度阈值
 * @description 完整的检测流程：预处理、推理、后处理
 */
void SCRFDGRAY::Predict(ssne_tensor_t* img, FaceDetectionResult* result, float conf_threshold) {
    int ret = RunAiPreprocessPipe(pipe_offline, *img, inputs[0]);
    if (ret != 0) {
        result->Clear();
        return;
    }

    if (ssne_inference(model_id, 1, inputs)) {
        fprintf(stderr, "[ERROR] SCRFD ssne_inference failed!\n");
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

    std::vector<std::array<float, 4>> bboxes;
    std::vector<float> scores;
    std::array<float, 4> tmp_bbox;

    float *out_scores0 = (float*)get_data(outputs[0]);  // 第一层分数输出
    float *out_scores1 = (float*)get_data(outputs[1]);  // 第二层分数输出
    float *out_scores2 = (float*)get_data(outputs[2]);  // 第三层分数输出
    float *out_bboxes0 = (float*)get_data(outputs[3]);  // 第一层检测框输出
    float *out_bboxes1 = (float*)get_data(outputs[4]);  // 第二层检测框输出
    float *out_bboxes2 = (float*)get_data(outputs[5]);  // 第三层检测框输出
    
    int idx_s = 0;  // 分数索引
    int idx_b = 0;  // 检测框索引
    int num_bbox = det_shape[0] * det_shape[1] / 1024;  // 每层的检测框数量
    for (int i = 0; i < num_bbox * 16; i++) {
        for (int j = 0; j < 2; j++) {
            scores.push_back(out_scores0[idx_s]);
            idx_s += 1;
            
            for (int k = 0; k < 4; k++) {
                tmp_bbox[k] = out_bboxes0[idx_b+k];                
            }
            idx_b+=4;
            bboxes.push_back(tmp_bbox);            
        }
    }

    idx_s = 0;
    idx_b = 0;
    for (int i = 0; i < num_bbox * 4; i++) {
        for (int j = 0; j < 2; j++) {
            scores.push_back(out_scores1[idx_s]);
            idx_s += 1;
            
            for (int k = 0; k < 4; k++) {
                tmp_bbox[k] = out_bboxes1[idx_b+k];                
            }
            idx_b+=4;
            bboxes.push_back(tmp_bbox);            
        }
    }

    idx_s = 0;
    idx_b = 0;    
    for (int i = 0; i < num_bbox; i++) {
        for (int j = 0; j < 2; j++) {
            scores.push_back(out_scores2[idx_s]);
            idx_s += 1;
            
            for (int k = 0; k < 4; k++) {
                tmp_bbox[k] = out_bboxes2[idx_b+k];                
            }
            idx_b+=4;
            bboxes.push_back(tmp_bbox);            
        }
    }

    Postprocess(&bboxes, &scores, result, &conf_threshold);
}

/**
 * @brief 释放资源
 * @description 释放所有tensor、预处理管道和计时器资源
 */
void SCRFDGRAY::Release() {
    release_tensor(inputs[0]);
    for (int i = 0; i < 6; i++) {
        outputs[i].data = nullptr;
    }
    ReleaseAIPreprocessPipe(pipe_offline);
}