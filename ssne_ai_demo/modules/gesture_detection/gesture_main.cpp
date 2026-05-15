/*
 * @Filename: gesture_main.cpp
 * @Description: 手势识别演示主程序（带 CLAHE 增强实时开关，接入主控菜单版）
 */

#include <fstream>
#include <iostream>
#include <cstring>
#include <thread>
#include <mutex>
#include <unistd.h>

#include "common.hpp"
#include "utils.hpp"

using namespace std;

static bool g_exit_flag = false;
static bool g_clahe_enabled = false; 
static std::mutex g_mtx;

static void keyboard_listener() {
    std::string input;
    std::cout << "\n========================================" << std::endl;
    std::cout << " 手势识别控制面板已启动..." << std::endl;
    std::cout << "  [q/Q] 退出手势识别并返回主菜单" << std::endl;
    std::cout << "  [c/C] 开启/关闭 CLAHE 对比度增强" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    while (true) {
        std::cin >> input;
        std::lock_guard<std::mutex> lock(g_mtx);
        if (input == "q" || input == "Q") {
            g_exit_flag = true;
            std::cout << "检测到退出指令，通知业务线程退出..." << std::endl;
            break;
        } else if (input == "c" || input == "C") {
            g_clahe_enabled = !g_clahe_enabled;
            std::cout << ">> CLAHE 状态已切换为: " << (g_clahe_enabled ? "ON (开启)" : "OFF (关闭)") << std::endl;
        } else {
            std::cout << "无效输入。请按 'q' 退出，或按 'c' 切换 CLAHE。" << std::endl;
        }
    }
}

static bool check_exit_flag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}

static bool check_clahe_flag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_clahe_enabled;
}

/**
 * @brief 手势识别演示程序入口
 * 【适配修改】：将 main 改为 run_gesture_detection
 */
int run_gesture_detection() {
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        g_exit_flag = false;
        g_clahe_enabled = false; // 默认关闭 CLAHE
    }

    int img_width  = 720;
    int img_height = 1280;

    array<int, 2> model_shape = {224, 224}; 
    string path_model = "./app_assets/models/gesture.m1model";
    array<int, 2> crop_shape = {640, 640};

    const std::array<float, 4> hand_roi = {40.f, 320.f, 680.f, 960.f};

    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
        return -1;
    }

    array<int, 2> img_shape = {img_width, img_height};
    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape, 40, 680, 320, 960, 640, 640);

    HANDGESTURECLASSIFIER classifier;
    classifier.Initialize(path_model, &crop_shape, &model_shape);

    HandGestureResult gesture_result;
    VISUALIZER visualizer;
    visualizer.Initialize(img_shape);

    cout << "sleep for 0.2 second!" << endl;
    sleep(0.2);

    ssne_tensor_t img_sensor;
    std::thread listener_thread(keyboard_listener);

    while (!check_exit_flag()) {
        processor.GetImage(&img_sensor);
        if (img_sensor.data == nullptr) {
            usleep(10000);
            continue;
        }

        bool use_clahe = check_clahe_flag();

        classifier.Predict(&img_sensor, &gesture_result, use_clahe);

        visualizer.Draw(gesture_result, hand_roi);

        const char* gesture_name = HAND_GESTURE_NAMES[static_cast<int>(gesture_result.gesture)];
        printf("[GESTURE] Final: %-2s (CLAHE:%s, conf=%.2f) | Raw: [0]:%.2f [1]:%.2f [2]:%.2f [3]:%.2f [4]:%.2f [5]:%.2f\n", 
               gesture_name, 
               use_clahe ? "ON " : "OFF",
               gesture_result.confidence,
               gesture_result.all_probs[0],
               gesture_result.all_probs[1],
               gesture_result.all_probs[2],
               gesture_result.all_probs[3],
               gesture_result.all_probs[4],
               gesture_result.all_probs[5]);
    }

    if (listener_thread.joinable())
        listener_thread.join();

    classifier.Release();
    processor.Release();
    visualizer.Release();

    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }

    return 0;
}