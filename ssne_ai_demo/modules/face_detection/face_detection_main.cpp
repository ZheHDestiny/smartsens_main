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

struct osdInfo {
    std::string filename; 
    uint16_t x;           
    uint16_t y;           
};

void keyboard_listener() {
    std::string input;
    std::cout << "键盘监听线程已启动，输入 'q' 退出人脸检测并返回主菜单..." << std::endl;

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

bool check_exit_flag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}

int run_face_detection() {
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        g_exit_flag = false;
    }

    int img_width = 720;    
    int img_height = 1280;  
    array<int, 2> det_shape = {640, 480};  
    
    string path_det = "./app_assets/models/face_640x480.m1model";  

    static osdInfo osds[3] = {
        {"si.ssbmp", 10, 10},
        {"te.ssbmp", 90, 10},
        {"wei.ssbmp", 170, 10}
    };

    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
        return -1;
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

    cout << "sleep for 0.2 second!" << endl;
    sleep(0.2);  

    visualizer.DrawBitmap(osds[0].filename, "shared_colorLUT.sscl", osds[0].x, osds[0].y, 2);

    uint16_t num_frames = 0;  
    uint8_t osd_index = 0; 
    ssne_tensor_t img_sensor;  

    std::thread listener_thread(keyboard_listener);

    while (!check_exit_flag()) {

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

        num_frames += 1;  
    }

    if (listener_thread.joinable()) {
        listener_thread.join();
    }

    delete det_result;  
    detector.Release();  
    processor.Release();  
    visualizer.Release();  

    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }

    return 0;
}