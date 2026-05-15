/*
 * @Filename: rps_detection_main.cpp
 * @Description: 剪刀石头布前摇预测演示主程序 (接入主控菜单版)
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
    std::cout << "键盘监听线程已启动，输入 'q' 退出剪刀石头布并返回主菜单..." << std::endl;
    while (true) {
        std::cin >> input;
        std::lock_guard<std::mutex> lock(g_mtx);
        if (input == "q" || input == "Q") {
            g_exit_flag = true;
            std::cout << "检测到退出指令，通知业务线程退出..." << std::endl;
            break;
        } else {
            std::cout << "输入无效（仅 'q' 有效），请重新输入：" << std::endl;
        }
    }
}

static bool check_exit_flag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}

/**
 * @brief 剪刀石头布演示程序入口
 * 【适配修改】：将 main() 改为 run_rps_detection() 供顶级 main.cpp 调用
 */
int run_rps_detection() {
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        g_exit_flag = false;
    }

    /******************************************************************************************
     * 1. 参数配置
     ******************************************************************************************/

    int img_width  = 720;
    int img_height = 1280;

    array<int, 2> model_shape = {640, 480};   // [W, H]，对应训练的 IMG_W × IMG_H
    string path_model = "./app_assets/models/paper.m1model";

    array<int, 2> crop_shape = {640, 640};   // [W, H]

    const std::array<float, 4> hand_roi = {40.f, 320.f, 680.f, 960.f};

    /******************************************************************************************
     * 2. 系统初始化
     ******************************************************************************************/

    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
        return -1;
    }

    array<int, 2> img_shape = {img_width, img_height};
    IMAGEPROCESSOR processor;
    
    processor.Initialize(&img_shape, 40, 680, 320, 960, 640, 640);

    RPSCLASSIFIER classifier;
    classifier.Initialize(path_model, &crop_shape, &model_shape);

    RpsResult* rps_result = new RpsResult;

    VISUALIZER visualizer;
    visualizer.Initialize(img_shape);

    cout << "sleep for 0.2 second!" << endl;
    sleep(0.2);

    ssne_tensor_t img_sensor;
    std::thread listener_thread(keyboard_listener);

    /******************************************************************************************
     * 3. 主处理循环（90fps 高帧率实时推理）
     ******************************************************************************************/
    while (!check_exit_flag()) {

        processor.GetImage(&img_sensor);
        
        if (img_sensor.data == nullptr) {
            usleep(10000); // 等待 10ms
            continue;
        }

        classifier.Predict(&img_sensor, rps_result);

        visualizer.Draw(*rps_result, hand_roi);

        if (rps_result->is_locked) {
            const char* human_name = GESTURE_NAMES[static_cast<int>(rps_result->human_gesture)];
            const char* ai_name   = GESTURE_NAMES[static_cast<int>(rps_result->ai_counter)];
            printf("[GAME] Human: %-8s  →  AI plays: %-8s  (conf=%.2f, state=%d)\n",
                   human_name, ai_name,
                   rps_result->confidence,
                   static_cast<int>(rps_result->game_state));
        }
    }

    if (listener_thread.joinable())
        listener_thread.join();

    /******************************************************************************************
     * 4. 资源释放
     ******************************************************************************************/
    delete rps_result;
    classifier.Release();
    processor.Release();
    visualizer.Release();

    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }
    

    return 0;
}