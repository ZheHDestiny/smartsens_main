/*
 * @Filename: facial_main.cpp
 * @Description: 表情识别演示主程序
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
    std::cout << "按 'q' 退出表情识别" << std::endl;
    while (true) {
        {
            std::lock_guard<std::mutex> lock(g_mtx);
            if (g_exit_flag || g_signal_received.load()) break;
        }
        if (nonblocking_getline(input, 100)) {
            std::lock_guard<std::mutex> lock(g_mtx);
            if (g_exit_flag || g_signal_received.load()) break;
            if (input == "q" || input == "Q") {
                g_exit_flag = true;
                break;
            }
        }
    }
}

static bool check_exit_flag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}

int run_facial_expressions() {
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        g_exit_flag = false;
    }

    clear_stdin_residual();

    int img_width = 720;
    int img_height = 1280;

    array<int, 2> model_shape = { 224, 224 };
    string path_model = "./app_assets/models/emotion_4class.m1model";
    array<int, 2> crop_shape = { 640, 640 };
    const std::array<float, 4> face_roi = { 40.f, 320.f, 680.f, 960.f };

    {
        SigintBlocker blocker;
        if (ssne_initial()) {
            fprintf(stderr, "SSNE initialization failed!\n");
            return -1;
        }
    }

    array<int, 2> img_shape = { img_width, img_height };
    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape, 40, 680, 320, 960, 640, 640);

    EMOTIONCLASSIFIER classifier;
    classifier.Initialize(path_model, &crop_shape, &model_shape);

    EmotionResult emotion_result;
    VISUALIZER visualizer;
    visualizer.Initialize(img_shape,"shared_colorLUT.sscl");

    usleep(200000);

    ssne_tensor_t img_sensor;
    memset(&img_sensor, 0, sizeof(img_sensor));
    std::thread listener_thread(keyboard_listener);

    {
        SigintBlocker blocker;
        while (!check_exit_flag()) {
        if (g_signal_received.load()) break;

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

    }

    if (listener_thread.joinable())
        listener_thread.join();

    classifier.Release();
    processor.Release();
    visualizer.Release();

    {
        SigintBlocker blocker;
        if (ssne_release()) {
            fprintf(stderr, "SSNE release failed!\n");
            return -1;
        }
    }
    usleep(500000);

    return 0;
}
