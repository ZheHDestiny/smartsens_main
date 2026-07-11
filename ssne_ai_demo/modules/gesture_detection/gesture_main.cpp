/*
 * @Filename: gesture_main.cpp
 * @Description: 手势识别演示主程序 (带背景减除)
 */

#include <fstream>
#include <iostream>
#include <cstring>
#include <thread>
#include <mutex>
#include <unistd.h>
#include <chrono>

#include "common.hpp"
#include "utils.hpp"

using namespace std;

static bool g_exit_flag = false;
static bool g_clahe_enabled = false;
static std::mutex g_mtx;

// ========== 背景减除辅助函数（原地修改） ==========
static void subtract_background(const uint8_t* curr, const uint8_t* bg,
    uint8_t* dst, int pixel_count, int threshold = 30) {
    for (int i = 0; i < pixel_count; ++i) {
        int diff = abs((int)curr[i] - (int)bg[i]);
        if (diff < threshold) {
            dst[i] = 0;
        } else {
            dst[i] = (uint8_t)diff;
        }
    }
}

static void keyboard_listener() {
    std::string input;
    std::cout << "按 'q' 退出，按 'c' 切换 CLAHE" << std::endl;

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
            } else if (input == "c" || input == "C") {
                g_clahe_enabled = !g_clahe_enabled;
                std::cout << "CLAHE: " << (g_clahe_enabled ? "ON" : "OFF") << std::endl;
            }
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

int run_gesture_detection() {
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        g_exit_flag = false;
        g_clahe_enabled = false;
    }

    clear_stdin_residual();

    int img_width  = 720;
    int img_height = 1280;

    array<int, 2> model_shape = {224, 224};
    string path_model = "./app_assets/models/gesture.m1model";
    array<int, 2> crop_shape = {640, 640};
    const std::array<float, 4> hand_roi = {40.f, 320.f, 680.f, 960.f};

    {
        SigintBlocker blocker;
        if (ssne_initial()) {
            fprintf(stderr, "SSNE initialization failed!\n");
            return -1;
        }
    }

    array<int, 2> img_shape = {img_width, img_height};
    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape, 40, 680, 320, 960, 640, 640);

    HANDGESTURECLASSIFIER classifier;
    classifier.Initialize(path_model, &crop_shape, &model_shape);

    HandGestureResult gesture_result;
    VISUALIZER visualizer;
    visualizer.Initialize(img_shape);

    usleep(200000);

    ssne_tensor_t img_sensor;
    memset(&img_sensor, 0, sizeof(img_sensor));

    // 背景减除相关变量
    const int crop_pixels = crop_shape[0] * crop_shape[1];
    uint8_t* background = new uint8_t[crop_pixels];
    bool bg_captured = false;

    std::thread listener_thread(keyboard_listener);
    HandGestureClass last_reported_gesture = HandGestureClass::NUM_CLASSES;
    bool last_reported_clahe = false;
    auto last_result_report = std::chrono::steady_clock::now()
                            - std::chrono::seconds(1);

    // ========== 捕获背景帧 ==========
    {
        std::cout << "请将手移出画面，3秒后自动捕获背景..." << std::endl;
        for (int i = 3; i > 0 && !check_exit_flag() && !g_signal_received.load(); --i) {
            sleep(1);
        }

        if (check_exit_flag() || g_signal_received.load()) {
            std::cout << "捕获背景前收到退出信号，跳过捕获。" << std::endl;
            goto cleanup;
        }

        processor.GetImage(&img_sensor);
        // 使用 get_data() 获取 CPU 可访问的映射地址
        void* img_data = get_data(img_sensor);
        if (img_data != nullptr) {
            memcpy(background, img_data, crop_pixels);
            bg_captured = true;
            std::cout << "背景已捕获。" << std::endl;
        } else {
            std::cerr << "背景捕获失败！将使用原始图像识别。" << std::endl;
            bg_captured = false;
        }
    }

    {
        SigintBlocker blocker;
        while (!check_exit_flag()) {
            if (g_signal_received.load()) break;

            processor.GetImage(&img_sensor);
            // 使用 get_data() 获取 CPU 可访问的映射地址
            void* img_data = get_data(img_sensor);
            if (img_data == nullptr) {
                usleep(10000);
                continue;
            }

            // 使用 get_data() 返回的指针进行原地背景减除
            if (bg_captured) {
                uint8_t* curr = static_cast<uint8_t*>(img_data);
                subtract_background(curr, background, curr, crop_pixels, 30);
            }

            bool use_clahe = check_clahe_flag();
            classifier.Predict(&img_sensor, &gesture_result, use_clahe);
            visualizer.Draw(gesture_result, hand_roi);

            const auto report_now = std::chrono::steady_clock::now();
            const bool result_changed =
                gesture_result.gesture != last_reported_gesture ||
                use_clahe != last_reported_clahe;
            const bool report_due =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    report_now - last_result_report).count() >= 1000;
            if (result_changed || report_due) {
                const char* gesture_name = HAND_GESTURE_NAMES[static_cast<int>(gesture_result.gesture)];
                printf("[GESTURE_RESULT] Final: %-2s (CLAHE:%s, conf=%.2f)\n",
                       gesture_name,
                       use_clahe ? "ON " : "OFF",
                       gesture_result.confidence);
                last_reported_gesture = gesture_result.gesture;
                last_reported_clahe = use_clahe;
                last_result_report = report_now;
            }
        }
    }

cleanup:
    if (listener_thread.joinable())
        listener_thread.join();

    delete[] background;

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
