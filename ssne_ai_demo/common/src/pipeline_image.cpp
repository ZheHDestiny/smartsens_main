/*
 * @Filename: pipeline_image.cpp
 * @Description: 统一的图像采集管道实现
 */

#include "common.hpp"
#include <iostream>
#include <unistd.h>

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
    is_opened = false;
    format_online = SSNE_Y_8; 
    OnlineSetCrop(kPipeline0, crop_x1, crop_x2, crop_y1, crop_y2);
    OnlineSetOutputImage(kPipeline0, format_online, out_w, out_h);
    int res0 = OpenOnlinePipeline(kPipeline0);
    if (res0 != 0) {
        printf("[ERROR] Failed to open online pipeline! ret: %d\n", res0);
        return;
    }
    is_opened = true;
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
        return;
    }
    int capture_code = GetImageData(img_sensor, kPipeline0, kSensor0, 0);
    if (capture_code != 0) {
        printf("[IMAGEPROCESSOR] Get Invalid Image from kPipeline0!\n");
        if (img_sensor) {
            img_sensor->data = nullptr;
        }
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
