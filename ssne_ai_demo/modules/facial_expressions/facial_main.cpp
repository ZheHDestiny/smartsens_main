/*
 * @Filename: facial_main.cpp
 * @Description: 情感识别演示主程序（接入主控菜单版）
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
static std::mutex g_mtx;

static void keyboard_listener() {
    std::string input;
    std::cout << "键盘监听线程已启动，输入 'q' 退出表情识别并返回主菜单..." << std::endl;
    while (true) {
        std::cin >> input;
        std::lock_guard<std::mutex> lock(g_mtx);
        if (input == "q" || input == "Q") {
            g_exit_flag = true;
            std::cout << "检测到退出指令，通知业务线程退出..." << std::endl;
            break;
        }
        else {
            std::cout << "输入无效（仅 'q' 有效），请重新输入：" << std::endl;
        }
    }
}

static bool check_exit_flag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}

/**
 * @brief 表情识别演示程序入口
 * 【适配修改】：将 main() 改为 run_facial_expressions() 供顶级 main.cpp 调用
 */
int run_facial_expressions() {
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        g_exit_flag = false;
    }

    /******************************************************************************************
     * 1. 参数配置
     ******************************************************************************************/

    int img_width = 720;
    int img_height = 1280;

    array<int, 2> model_shape = { 224, 224 };
    
    string path_model = "./app_assets/models/emotion_4class.m1model";

    array<int, 2> crop_shape = { 640, 640 };

    const std::array<float, 4> face_roi = { 40.f, 320.f, 680.f, 960.f };

    /******************************************************************************************
     * 2. 系统初始化
     ******************************************************************************************/

    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
        return -1;
    }

    array<int, 2> img_shape = { img_width, img_height };
    IMAGEPROCESSOR processor;
    
    processor.Initialize(&img_shape, 40, 680, 320, 960, 640, 640);

    EMOTIONCLASSIFIER classifier;
    classifier.Initialize(path_model, &crop_shape, &model_shape);

    EmotionResult emotion_result;
    VISUALIZER visualizer;
    visualizer.Initialize(img_shape,"shared_colorLUT.sscl");

    cout << "sleep for 0.2 second!" << endl;
    sleep(0.2);

    ssne_tensor_t img_sensor;
    std::thread listener_thread(keyboard_listener);

    /******************************************************************************************
     * 3. 主处理循环
     ******************************************************************************************/
    while (!check_exit_flag()) {
        processor.GetImage(&img_sensor);
        
        if (img_sensor.data == nullptr) {
            usleep(10000);
            continue;
        }

        classifier.Predict(&img_sensor, &emotion_result);

        visualizer.DrawSimple(emotion_result, face_roi);

        const char* emotion_name = EMOTION_NAMES[static_cast<int>(emotion_result.emotion)];
        printf("[EMOTION] %-10s (conf=%.2f)\n", emotion_name, emotion_result.confidence);
    }

    if (listener_thread.joinable())
        listener_thread.join();

    /******************************************************************************************
     * 4. 资源释放
     ******************************************************************************************/
    classifier.Release();
    processor.Release();
    visualizer.Release();

    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }

    return 0;
}