/*
 * @Filename: optical_flow_main.cpp
 * @Description: 上板实时光流障碍物检测程序 (接入主控菜单版)
 */

#include <iostream>
#include <cstring>
#include <thread>
#include <mutex>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <sys/stat.h>
#include <fstream>
#include <string>

#include "common.hpp"
#include "utils.hpp"

using namespace std;

static bool g_exit_flag = false;            
static bool g_pause_flag = false;           
static std::mutex g_mtx;

static bool dev_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static bool module_loaded(const char* mod) {
    std::ifstream f("/proc/modules");
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind(mod, 0) == 0) return true;
    }
    return false;
}

static void keyboard_listener() {
    string input;
    printf("[键盘面板] ┌─────────────────────────────────┐\n");
    printf("[键盘面板] │ 控制指令:                         │\n");
    printf("[键盘面板] │  - P/p: 暂停/继续                 │\n");
    printf("[键盘面板] │  - q/Q: 退出光流调试并返回主菜单  │\n");
    printf("[键盘面板] └─────────────────────────────────┘\n");
    printf("[键盘监听] 线程已启动，等待用户输入...\n\n");

    while (true) {
        {
            lock_guard<mutex> lock(g_mtx);
            if (g_exit_flag || g_signal_received.load()) break;
        }
        if (nonblocking_getline(input, 100)) {
            lock_guard<mutex> lock(g_mtx);
            if (g_exit_flag || g_signal_received.load()) break;
            if (input == "q" || input == "Q") {
                g_exit_flag = true;
                printf("\n[键盘监听] ✓ 检测到退出指令 (q)，通知业务线程退出...\n");
                break;
            }
            else if (input == "p" || input == "P") {
                g_pause_flag = !g_pause_flag;
            }
            else {
                printf("[键盘监听]  无效指令: '%s' (仅支持 p/P 或 q/Q)\n", input.c_str());
            }
        }
    }
}

static bool check_exit_flag() {
    lock_guard<mutex> lock(g_mtx);
    return g_exit_flag;
}

static bool check_pause_flag() {
    lock_guard<mutex> lock(g_mtx);
    return g_pause_flag;
}

/**
 * @brief 光流演示程序主函数
 * 【适配修改】：将 main() 改为 run_optical_flow_debug() 供顶级 main.cpp 调用
 */
int run_optical_flow_debug() {
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        g_exit_flag = false;
        g_pause_flag = false;
    }

    clear_stdin_residual();

    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("     SmartSens 光流障碍物避障系统\n");
    printf("     Optical Flow Obstacle Avoidance\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    int img_width = 720;    
    int img_height = 1280;  
    array<int, 2> img_shape = {img_width, img_height};
    array<int, 2> proc_shape = {720, 540};  

    printf("[配置] 图像参数\n");
    printf("  • 处理分辨率: %dx%d\n\n", proc_shape[0], proc_shape[1]);

    bool ssne_ok = false;
    IMAGEPROCESSOR image_processor;
    
    {
        SigintBlocker blocker;
        if (ssne_initial() != 0) {
            fprintf(stderr, "[WARN] ssne_initial() failed!\n");
        } else {
            ssne_ok = true;
        }
    }

    image_processor.Initialize(&img_shape, 0, 720, 370, 910, 720, 540);
    printf("  ✓ 图像管道已打开\n\n");

    OPTICALFLOW optical_flow;
    optical_flow.Initialize(proc_shape[0], proc_shape[1]);
    
    optical_flow.max_features = 80;         
    optical_flow.fast_threshold = 25;       
    optical_flow.pyramid_levels = 2;        
    optical_flow.lk_max_iter = 6;           
    optical_flow.lk_win_size = 3;           

    OBSTACLE_DETECTOR obstacle_detector;
    obstacle_detector.Initialize(proc_shape[0], proc_shape[1]);
    obstacle_detector.ttc_threshold = 1.0f;
    obstacle_detector.divergence_threshold = 0.5f;

    VISUALIZER visualizer;
    bool osd_present = module_loaded("osd_kmod") || dev_exists("/dev/osddev") || dev_exists("/dev/osd0");
    if (osd_present) {
        visualizer.Initialize(img_shape); // 默认无位图参数
        printf("  ✓ OSD 可视化器初始化完成\n\n");
    }

    cout << "[INFO] 系统稳定等待 0.2 秒..." << endl;
    usleep(200000);

    printf("[处理] 开始实时处理\n");
    printf("        P/p键: 暂停/继续 | Q/q键: 退出\n");
    printf("────────────────────────────────────────────────────────────\n\n");

    vector<FeaturePoint> features;
    ObstacleInfo obstacle_info;
    memset(&obstacle_info, 0, sizeof(obstacle_info));
    obstacle_info.priority = 4;  

    ssne_tensor_t curr_frame;
    memset(&curr_frame, 0, sizeof(ssne_tensor_t));

    std::vector<uint8_t> prev_buf;
    uint8_t* prev_ptr = nullptr;

    uint32_t frame_count = 0;
    uint32_t obstacle_frame_count = 0;
    uint32_t paused_frame_count = 0;
    bool first_frame = true;

    int last_log_priority = 4;
    int last_danger_region = -1;
    uint32_t last_log_frame = 0;
    
    int last_osd_priority = -1;
    int last_osd_region = -1;
    float last_osd_danger[3] = {-1.0f, -1.0f, -1.0f};
    bool is_osd_safe_cleared = false;

    auto program_start_time = std::chrono::steady_clock::now();
    std::chrono::duration<float> total_paused_time(0); 
    std::chrono::steady_clock::time_point frame_times[10];
    for (int i = 0; i < 10; i++) {
        frame_times[i] = program_start_time;
    }

    thread listener_thread(keyboard_listener);

    {
        SigintBlocker blocker;
        while (!check_exit_flag()) {
        if (g_signal_received.load()) break;

        if (check_pause_flag()) {
            auto pause_enter_time = std::chrono::steady_clock::now();
            
            int calc_frames = (frame_count < 10) ? frame_count : 10;
            auto last_valid_time = (frame_count > 0) ? frame_times[(frame_count - 1) % 10] : pause_enter_time;
            auto time_10_ago = frame_times[frame_count % 10]; 
            
            std::chrono::duration<float> inst_diff = last_valid_time - time_10_ago;
            float inst_fps = (inst_diff.count() > 0.001f && calc_frames > 0) ? (calc_frames / inst_diff.count()) : 0.0f;

            std::chrono::duration<float> active_time = (last_valid_time - program_start_time) - total_paused_time;
            float avg_fps = (active_time.count() > 0.001f) ? (frame_count / active_time.count()) : 0.0f;

            printf("\n[系统暂停] 性能快照 (按 P 继续) =================\n");
            printf("  • 运行总帧数: %d\n", frame_count);
            printf("  • 平均帧率:   %4.1f FPS\n", avg_fps);
            printf("  • 瞬时帧率:   %4.1f FPS (近10帧)\n", inst_fps);
            printf("=================================================\n\n");

            auto pause_start = std::chrono::steady_clock::now();
            while (check_pause_flag() && !check_exit_flag()) {
                usleep(100000);  
                paused_frame_count++;
            }
            auto pause_end = std::chrono::steady_clock::now();
            
            total_paused_time += (pause_end - pause_start);
            if (!check_exit_flag()) {
                printf("[系统继续] 恢复处理...\n\n");
            }
        }
        
        if (check_exit_flag()) break;  

        uint8_t* curr_ptr = nullptr;
        image_processor.GetImage(&curr_frame);
        
        if (curr_frame.data == nullptr) {
            usleep(10000);
            continue;
        }
        curr_ptr = (uint8_t*)get_data(curr_frame);

        auto now = std::chrono::steady_clock::now();
        frame_times[frame_count % 10] = now;
        frame_count++;

        if (first_frame) {
            optical_flow.DetectFeatures(curr_ptr, features);
            printf("[系统] 首帧准备就绪, 提取 %zu 个特征点...\n", features.size());
            first_frame = false;
        } else {
            int tracked_count = 0;

            if (prev_ptr != nullptr && curr_ptr != nullptr) {
                optical_flow.ComputeFlow(prev_ptr, curr_ptr, features);
                for (const auto& f : features) {
                    if (f.tracked) tracked_count++;
                }
            }

            obstacle_detector.DetectObstacles(features, obstacle_info);

            int curr_priority = obstacle_info.priority;
            int curr_region = obstacle_info.most_dangerous_region;

            if (curr_priority == 0) {
                bool is_new_alert = (last_log_priority != 0); 
                bool region_changed = (curr_region != last_danger_region); 
                bool low_freq_tick = (frame_count - last_log_frame >= 15); 

                if (is_new_alert || region_changed || low_freq_tick) {
                    float danger_val = obstacle_info.danger_level[curr_region];
                    char reg_char = (curr_region == 0) ? 'L' : ((curr_region == 1) ? 'C' : 'R');
                    
                    printf("[ALERT] f=%d reg=%c danger=%.2f tracked=%d/%zu\n",
                           frame_count, reg_char, danger_val, tracked_count, features.size());
                    
                    last_log_frame = frame_count;
                }
            } else if (last_log_priority == 0 && curr_priority != 0) {
                printf("[RECOVER] f=%d\n", frame_count);
            }
            
            last_log_priority = curr_priority;
            last_danger_region = curr_region;

            if (curr_priority < 4) obstacle_frame_count++;

            visualizer.DrawAll(features, obstacle_info,370,frame_count);

            if (tracked_count < 15) {
                optical_flow.DetectFeatures(curr_ptr, features);
            }
        }

        size_t frame_bytes = (size_t)proc_shape[0] * (size_t)proc_shape[1];
        if (curr_ptr != nullptr) {
            prev_buf.resize(frame_bytes);
            memcpy(prev_buf.data(), curr_ptr, frame_bytes);
            prev_ptr = prev_buf.data();
        } else {
            prev_ptr = nullptr;
            prev_buf.clear();
        }
    }

    int calc_frames = (frame_count < 10) ? frame_count : 10;
    auto time_10_ago = frame_times[frame_count % 10];
    auto true_exit_time = (frame_count > 0) ? frame_times[(frame_count - 1) % 10] : std::chrono::steady_clock::now();
    
    std::chrono::duration<float> final_inst_diff = true_exit_time - time_10_ago;
    float final_inst_fps = (final_inst_diff.count() > 0.001f && calc_frames > 0) ? (calc_frames / final_inst_diff.count()) : 0.0f;

    std::chrono::duration<float> final_active_time = (true_exit_time - program_start_time) - total_paused_time;
    float final_avg_fps = (final_active_time.count() > 0.001f) ? (frame_count / final_active_time.count()) : 0.0f;

    printf("\n────────────────────────────────────────────────────────────\n");
    printf("\n[统计] 处理完成\n");
    printf("  • 处理总帧数: %d\n", frame_count);
    printf("  • 最终平均帧率: %4.1f FPS\n", final_avg_fps);
    printf("  • 退出瞬时帧率: %4.1f FPS (近10帧)\n", final_inst_fps);
    printf("  • 暂停期间检查次数: %d\n", paused_frame_count);
    printf("  • 有障碍物的帧: %d (%.1f%%)\n",
           obstacle_frame_count,
           frame_count > 0 ? 100.0f * obstacle_frame_count / frame_count : 0);

    printf("\n[释放] 清理资源\n");

    }

    if (listener_thread.joinable()) {
        listener_thread.join();
    }
    
    optical_flow.Release();
    obstacle_detector.Release();
    image_processor.Release();
    
    visualizer.Release();

    {
        SigintBlocker blocker;
        if (ssne_ok) {
            ssne_release();
        }
    }
    usleep(500000);

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("     光流检测结束，返回主菜单\n");
    printf("════════════════════════════════════════════════════════════\n\n");

    return 0;
}