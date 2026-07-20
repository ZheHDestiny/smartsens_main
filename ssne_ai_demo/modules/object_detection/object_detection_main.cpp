/*
 * @Filename: object_detection_main.cpp
 * @Description: 目标检测业务主入口
 */
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <string>
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
    std::cout << "按 'q' 退出目标检测" << std::endl;
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

static std::map<int, std::string> LoadClassNames(const std::string& path) {
    std::map<int, std::string> class_names;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[WARNING] Failed to open cls.yaml at: " << path << std::endl;
        return class_names;
    }

    std::string line;
    while (std::getline(file, line)) {
        auto colon_pos = line.find(':');
        if (colon_pos != std::string::npos && line.find("names:") == std::string::npos) {
            int id = std::stoi(line.substr(0, colon_pos));
            std::string name = line.substr(colon_pos + 1);
            name.erase(0, name.find_first_not_of(" \t\r\n"));
            name.erase(name.find_last_not_of(" \t\r\n") + 1);
            class_names[id] = name;
        }
    }
    return class_names;
}

int run_object_detection() {
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        g_exit_flag = false;
    }

    clear_stdin_residual();

    int img_width = 720;
    int img_height = 1280;

    array<int, 2> det_shape = { 640, 480 };
    string path_det = "./app_assets/models/yolov8n_object_detection.m1model";

    {
        SigintBlocker blocker;
        if (ssne_initial()) {
            fprintf(stderr, "SSNE initialization failed!\n");
            return -1;
        }
    }

    std::map<int, std::string> class_names = LoadClassNames("./app_assets/cls.yaml");

    array<int, 2> img_shape = { img_width, img_height };

    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape, 0, 720, 0, 1280, 720, 1280);

    YOLOV8_OBJECT detector;
    // 传原始 sensor 尺寸，detector 内部自行处理 letterbox 和坐标映射
    detector.Initialize(path_det, &img_shape, &det_shape);

    ObjectDetectionResult det_result;
    VISUALIZER visualizer;
    visualizer.Initialize(img_shape, "shared_colorLUT.sscl");

    std::thread listener_thread(keyboard_listener);
    ssne_tensor_t img_sensor;
    memset(&img_sensor, 0, sizeof(img_sensor));
    auto last_result_report = std::chrono::steady_clock::now()
        - std::chrono::seconds(1);

    {
        SigintBlocker blocker;
        while (true) {
            if (g_signal_received.load()) break;

            bool current_exit = false;
            {
                std::lock_guard<std::mutex> lock(g_mtx);
                current_exit = g_exit_flag;
            }
            if (current_exit) break;

            processor.GetImage(&img_sensor);

            if (img_sensor.data == nullptr) {
                usleep(10000);
                continue;
            }

            detector.Predict(&img_sensor, &det_result, 0.30f);
            const auto report_now = std::chrono::steady_clock::now();
            const bool report_result =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    report_now - last_result_report).count() >= 1000;

            if (det_result.boxes.size() > 0) {
                std::vector<std::array<float, 4>> boxes_original_coord;
                std::vector<float> display_scores;
                std::vector<int> display_class_ids;

                for (size_t i = 0; i < det_result.boxes.size(); i++) {
                    // detector.Postprocess 已经做了完整的 letterbox 逆映射，
                    // 坐标直接就是原始图像 (720x1280) 坐标系，无需再转换
                    float x1_orig = det_result.boxes[i][0];
                    float y1_orig = det_result.boxes[i][1];
                    float x2_orig = det_result.boxes[i][2];
                    float y2_orig = det_result.boxes[i][3];

                    // 仅保留边界和 NaN 校验
                    if (x1_orig < 0 || y1_orig < 0 || x2_orig > img_width || y2_orig > img_height ||
                        x1_orig >= x2_orig || y1_orig >= y2_orig ||
                        std::isnan(x1_orig) || std::isnan(y1_orig) || std::isnan(x2_orig) || std::isnan(y2_orig)) {
                        continue;
                    }
                    boxes_original_coord.push_back({ x1_orig, y1_orig, x2_orig, y2_orig });
                    display_scores.push_back(det_result.scores[i]);
                    display_class_ids.push_back(det_result.class_ids[i]);

                    int id = det_result.class_ids[i];
                    std::string name = class_names.count(id) ? class_names[id] : "Unknown";
                    if (report_result) {
                        std::cout << "[DET] Class: " << name << " Score: " << det_result.scores[i] << std::endl;
                    }
                }
                if (report_result) last_result_report = report_now;
                visualizer.Draw(boxes_original_coord, display_scores, display_class_ids);
            }
            else {
                std::vector<std::array<float, 4>> empty_boxes;
                std::vector<float> empty_scores;
                std::vector<int> empty_ids;
                visualizer.Draw(empty_boxes, empty_scores, empty_ids);
            }
        }

    }

    if (listener_thread.joinable()) listener_thread.join();

    visualizer.Clear();
    visualizer.Release();
    detector.Release();
    processor.Release();

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
