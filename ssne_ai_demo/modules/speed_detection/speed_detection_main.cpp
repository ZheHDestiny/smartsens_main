/*
 * @Filename: speed_detection_main.cpp
 * @Description: 速度检测业务主入口
 */

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <deque>
#include <set>
#include <unistd.h>

#include "common.hpp"
#include "utils.hpp"

using namespace std;

static bool g_exit_flag = false;
static std::mutex g_mtx;

static void keyboard_listener() {
    std::string input;
    std::cout << "按 'q' 退出速度检测" << std::endl;
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

static const double fx = 915.0745539;
static const double fy = 917.0690011;
static const double cx = 393.0308165;
static const double cy = 669.2393082;
static const double k1 = -0.4758479;
static const double k2 = 0.2961891;
static const double p1 = 0.0003155;
static const double p2 = -0.0016621;
static const double k3 = -0.1058404;

static const std::map<int, double> REAL_LENGTH_CM = {
    {0, 7.1},
    {1, 8.6},
    {2, 7.8}
};

static const char* CLASS_NAMES[]={"CAR","TRUCK","BUS"};

struct TrackedVehicle {
    double smoothed_x = 0.0;
    double last_calc_x = 0.0;
    std::chrono::steady_clock::time_point last_calc_time;
    float current_speed = 0.0f;
    float smoothed_speed = 0.0f;
    int direction = 0;
    bool initialized = false;
    int missed_frames = 0;
};

static void UndistortPoint(double u_d, double v_d, double& u_u, double& v_u) {
    double x_d = (u_d - cx) / fx;
    double y_d = (v_d - cy) / fy;
    double x = x_d, y = y_d;
    for (int i = 0; i < 5; ++i) {
        double r2 = x*x + y*y;
        double k_radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2;
        double delta_x = 2*p1*x*y + p2*(r2 + 2*x*x);
        double delta_y = p1*(r2 + 2*y*y) + 2*p2*x*y;
        x = (x_d - delta_x) / k_radial;
        y = (y_d - delta_y) / k_radial;
    }
    u_u = x * fx + cx;
    v_u = y * fy + cy;
}

int run_speed_detection() {
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        g_exit_flag = false;
    }

    clear_stdin_residual();

    array<int, 2> img_shape = {720, 720};
    array<int, 2> det_shape = {320, 320};

    string path_det = "./app_assets/models/yolov8n_speed_detection.m1model";

    {
        SigintBlocker blocker;
        if (ssne_initial()) {
            fprintf(stderr, "SSNE initialization failed!\n");
            return -1;
        }
    }

    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape, 0, 720, 560, 1280, 720, 720);

    YOLOV8_SPEED detector;
    detector.Initialize(path_det, &img_shape, &det_shape);

    VISUALIZER visualizer;
    array<int, 2> osd_shape = {720, 1280};
    visualizer.Initialize(osd_shape, "shared_colorLUT.sscl");

    ObjectDetectionResult det_result;
    std::thread listener_thread(keyboard_listener);
    ssne_tensor_t img_sensor;
    memset(&img_sensor, 0, sizeof(img_sensor));

    std::map<int, TrackedVehicle> trackers;
    int next_tracker_id = 0;

    {
        SigintBlocker blocker;
        while (true) {
        if (g_signal_received.load()) break;

        {
            std::lock_guard<std::mutex> lock(g_mtx);
            if(g_exit_flag) break;
        }

        processor.GetImage(&img_sensor);

        if (img_sensor.data == nullptr) {
            usleep(10000);
            continue;
        }

        detector.Predict(&img_sensor, &det_result, 0.35f);

        std::vector<std::array<float, 4>> boxes_draw;
        std::vector<float> scores_draw;
        std::vector<int> ids_draw;
        std::vector<float> speeds_draw;
        std::vector<int> directions_draw;

        auto now = std::chrono::steady_clock::now();

        for (auto it = trackers.begin(); it != trackers.end(); ) {
            it->second.missed_frames++;
            if (it->second.missed_frames > 30) {
                it = trackers.erase(it);
            } else {
                ++it;
            }
        }

        std::set<int> matched_trackers_this_frame;

        for (size_t i = 0; i < det_result.boxes.size(); i++) {
            int cid = det_result.class_ids[i];

            float x1_orig = det_result.boxes[i][0];
            float y1_orig = det_result.boxes[i][1] + 560.0f;
            float x2_orig = det_result.boxes[i][2];
            float y2_orig = det_result.boxes[i][3] + 560.0f;

            double u1_u, v1_u, u2_u, v2_u, center_u, center_v;
            UndistortPoint(x1_orig, (y1_orig+y2_orig)/2, u1_u, v1_u);
            UndistortPoint(x2_orig, (y1_orig+y2_orig)/2, u2_u, v2_u);
            UndistortPoint((x1_orig+x2_orig)/2, (y1_orig+y2_orig)/2, center_u, center_v);

            double pixel_width = std::abs(u2_u - u1_u);
            double real_len = REAL_LENGTH_CM.count(cid) ? REAL_LENGTH_CM.at(cid) : 7.0;
            double depth_z = (fx * real_len) / pixel_width;
            double real_x = (center_u - cx) * depth_z / fx;

            int best_tracker_id = -1;
            double min_dist = 15.0;

            for (auto& kv : trackers) {
                if (matched_trackers_this_frame.count(kv.first)) {
                    continue;
                }

                double dist = std::abs(real_x - kv.second.last_calc_x);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_tracker_id = kv.first;
                }
            }

            if (best_tracker_id == -1) {
                best_tracker_id = next_tracker_id++;
                trackers[best_tracker_id].initialized = false;
            }

            matched_trackers_this_frame.insert(best_tracker_id);
            trackers[best_tracker_id].missed_frames = 0;

            if (!trackers[best_tracker_id].initialized) {
                trackers[best_tracker_id].smoothed_x = real_x;
                trackers[best_tracker_id].last_calc_x = real_x;
                trackers[best_tracker_id].last_calc_time = now;
                trackers[best_tracker_id].initialized = true;
                trackers[best_tracker_id].smoothed_speed = 0.0f;
                trackers[best_tracker_id].current_speed = 0.0f;
            } else {
                trackers[best_tracker_id].smoothed_x = 0.6 * real_x + 0.4 * trackers[best_tracker_id].smoothed_x;
                double dt_calc = std::chrono::duration<double>(now - trackers[best_tracker_id].last_calc_time).count();

                if (dt_calc >= 0.15) {
                    double dx = trackers[best_tracker_id].smoothed_x - trackers[best_tracker_id].last_calc_x;
                    float raw_speed = std::abs(dx) / dt_calc;

                    if (std::abs(dx) < 1.5) {
                        raw_speed = 0.0f;
                    } else {
                        trackers[best_tracker_id].last_calc_x = trackers[best_tracker_id].smoothed_x;
                    }

                    if (raw_speed < 150.0) {
                        if (raw_speed == 0.0f) {
                            trackers[best_tracker_id].smoothed_speed *= 0.2f;
                        } else {
                            trackers[best_tracker_id].smoothed_speed = 0.4f * raw_speed + 0.6f * trackers[best_tracker_id].smoothed_speed;
                        }

                        if (trackers[best_tracker_id].smoothed_speed < 1.5f) {
                            trackers[best_tracker_id].smoothed_speed = 0.0f;
                        }

                        trackers[best_tracker_id].current_speed = trackers[best_tracker_id].smoothed_speed;
                        if (trackers[best_tracker_id].smoothed_speed > 0.0f && std::abs(dx) >= 1.5) {
                            trackers[best_tracker_id].direction = (dx > 0) ? 1 : -1;
                        }
                    }
                    trackers[best_tracker_id].last_calc_time = now;
                }
            }

            if (x1_orig < 0 || y1_orig < 0 || x2_orig > osd_shape[0] || y2_orig > osd_shape[1] ||
                x1_orig >= x2_orig || y1_orig >= y2_orig ||
                std::isnan(x1_orig) || std::isnan(y1_orig) || std::isnan(x2_orig) || std::isnan(y2_orig)) {
                continue;
            }
            boxes_draw.push_back({x1_orig, y1_orig, x2_orig, y2_orig});
            scores_draw.push_back(det_result.scores[i]);
            ids_draw.push_back(cid);
            speeds_draw.push_back(trackers[best_tracker_id].current_speed);
            directions_draw.push_back(trackers[best_tracker_id].direction);
        }

        if (det_result.boxes.size() > 0) {
            std::string frame_log = "";
            char string_buf[256];
            snprintf(string_buf, sizeof(string_buf), "[SPEED_LOG] DETS: ");
            frame_log += string_buf;
            for (size_t i = 0; i < boxes_draw.size(); i++) {
                const char* class_str = (ids_draw[i] >= 0 && ids_draw[i] <= 2) ? CLASS_NAMES[ids_draw[i]] : "UNK";
                snprintf(string_buf, sizeof(string_buf), "[ID:%zu %s %s %.1fcm/s] ",
                         i, class_str, (directions_draw[i] > 0 ? "R" : "L"), speeds_draw[i]);
                frame_log += string_buf;
            }
            printf("%s\n", frame_log.c_str());
        }

        visualizer.DrawSpeed(boxes_draw, scores_draw, ids_draw, speeds_draw, directions_draw,560,720);
    }

    }

    if (listener_thread.joinable()) listener_thread.join();

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
