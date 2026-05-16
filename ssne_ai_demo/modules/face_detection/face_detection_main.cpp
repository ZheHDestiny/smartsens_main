/*
 * @Filename: face_detection_main.cpp
 * @Description: 人脸检测业务主入口
 */
#include <fstream>
#include <iostream>
#include <cstring>
#include <thread>
#include <mutex>
#include <fcntl.h>
#include <regex>
#include <dirent.h>
#include <unistd.h>
#include "utils.hpp"
#include "common.hpp"

using namespace std;

bool g_exit_flag = false;
std::mutex g_mtx;

void keyboard_listener() {
    std::string input;
    std::cout << "按 'q' 退出人脸检测" << std::endl;

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

bool check_exit_flag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}

int run_face_detection() {
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        g_exit_flag = false;
    }

    clear_stdin_residual();

    int img_width = 720;
    int img_height = 1280;
    array<int, 2> det_shape = {640, 480};
    string path_det = "./app_assets/models/face_640x480.m1model";

    {
        SigintBlocker blocker;
        if (ssne_initial()) {
            fprintf(stderr, "SSNE initialization failed!\n");
            return -1;
        }
    }

    array<int, 2> img_shape = {img_width, img_height};
    array<int, 2> crop_shape = {720, 540};
    const int crop_offset_y = 370;

    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape, 0, 720, 370, 910, 720, 540);

    SCRFDGRAY detector;
    int box_len = det_shape[0] * det_shape[1] / 512 * 21;
    detector.Initialize(path_det, &crop_shape, &det_shape, false, box_len);

    FaceDetectionResult* det_result = new FaceDetectionResult;

    VISUALIZER visualizer;
    visualizer.Initialize(img_shape, "shared_colorLUT.sscl");

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

        detector.Predict(&img_sensor, det_result, 0.4f);

        if (det_result->boxes.size() > 0) {
            std::vector<std::array<float, 4>> boxes_original_coord;
            for (size_t i = 0; i < det_result->boxes.size(); i++) {
                float x1_crop = det_result->boxes[i][0];
                float y1_crop = det_result->boxes[i][1];
                float x2_crop = det_result->boxes[i][2];
                float y2_crop = det_result->boxes[i][3];

                float x1_orig = x1_crop;
                float y1_orig = y1_crop + crop_offset_y;
                float x2_orig = x2_crop;
                float y2_orig = y2_crop + crop_offset_y;

                if (x1_orig < 0 || y1_orig < 0 || x2_orig > img_width || y2_orig > img_height ||
                    x1_orig >= x2_orig || y1_orig >= y2_orig ||
                    std::isnan(x1_orig) || std::isnan(y1_orig) || std::isnan(x2_orig) || std::isnan(y2_orig)) {
                    continue;
                }
                boxes_original_coord.push_back({x1_orig, y1_orig, x2_orig, y2_orig});
                printf("[FACE DETECTED] id=%zu, coords:(%.1f, %.1f) to (%.1f, %.1f), score: %.2f\n",
                       i, x1_orig, y1_orig, x2_orig, y2_orig, det_result->scores[i]);
            }
            visualizer.Draw(boxes_original_coord);
        }
        else {
            std::vector<std::array<float, 4>> empty_boxes;
            visualizer.Draw(empty_boxes);
        }
    }

    }

    if (listener_thread.joinable()) {
        listener_thread.join();
    }

    delete det_result;
    detector.Release();
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
