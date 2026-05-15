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
    std::cout << "键盘监听线程已启动，输入 'q' 退出目标检测并返回主菜单..." << std::endl;
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

/**
 * @brief 目标检测程序入口
 * 【适配修改】：将 main 改为 run_object_detection 供顶级 main.cpp 调用
 */
int run_object_detection() {
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        g_exit_flag = false;
    }

    int img_width = 720;    
    int img_height = 1280;  

    array<int, 2> det_shape = {640, 480};  // YOLOv8 标准输入
    string path_det = "./app_assets/models/yolov8n_object_detection.m1model"; 

    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
        return -1;
    }

    std::map<int, std::string> class_names = LoadClassNames("./app_assets/cls.yaml");

    array<int, 2> img_shape = {img_width, img_height};  
    array<int, 2> crop_shape = {720, 540}; // 裁剪为正方形
    const int crop_offset_y = 370;  // Y轴裁剪偏移：(1280-720)/2

    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape, 0, 720, 0, 1280, 720, 1280);

    YOLOV8_OBJECT detector;
    detector.Initialize(path_det, &crop_shape, &det_shape);

    ObjectDetectionResult det_result;
    VISUALIZER visualizer;
    visualizer.Initialize(img_shape, "shared_colorLUT.sscl");

    std::thread listener_thread(keyboard_listener);
    ssne_tensor_t img_sensor;

    while (true) {
        bool current_exit = false;
        {
            std::lock_guard<std::mutex> lock(g_mtx);
            current_exit = g_exit_flag;
        }
        if(current_exit) break;

        processor.GetImage(&img_sensor);
        
        if (img_sensor.data == nullptr) {
            usleep(10000);
            continue;
        }

        detector.Predict(&img_sensor, &det_result, 0.40f);

        if (det_result.boxes.size() > 0) {
            std::vector<std::array<float, 4>> boxes_original_coord;
            std::vector<float> display_scores;
            std::vector<int> display_class_ids;

            for (size_t i = 0; i < det_result.boxes.size(); i++) {
                
                float pad_x = 185.0f; 
                float scale = 1280.0f / 480.0f; // 缩放倍数 2.6666...

                float x1_orig = (det_result.boxes[i][0] - pad_x) * scale;
                float y1_orig = det_result.boxes[i][1] * scale;
                float x2_orig = (det_result.boxes[i][2] - pad_x) * scale;
                float y2_orig = det_result.boxes[i][3] * scale;

                x1_orig = std::max(0.0f, std::min(x1_orig, 720.0f));
                y1_orig = std::max(0.0f, std::min(y1_orig, 1280.0f));
                x2_orig = std::max(0.0f, std::min(x2_orig, 720.0f));
                y2_orig = std::max(0.0f, std::min(y2_orig, 1280.0f));

                boxes_original_coord.push_back({x1_orig, y1_orig, x2_orig, y2_orig});
                display_scores.push_back(det_result.scores[i]);
                display_class_ids.push_back(det_result.class_ids[i]);

                int id = det_result.class_ids[i];
                std::string name = class_names.count(id) ? class_names[id] : "Unknown";
                std::cout << "[DET] Class: " << name << " Score: " << det_result.scores[i] << std::endl;
            }
            visualizer.Draw(boxes_original_coord, display_scores, display_class_ids);
        } else {
            std::vector<std::array<float, 4>> empty_boxes;
            std::vector<float> empty_scores;
            std::vector<int> empty_ids;
            visualizer.Draw(empty_boxes, empty_scores, empty_ids);
        }
    }

    if (listener_thread.joinable()) listener_thread.join();

    detector.Release();
    processor.Release();
    visualizer.Release();
    
    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }

    return 0; // 成功退出，返回主菜单
}