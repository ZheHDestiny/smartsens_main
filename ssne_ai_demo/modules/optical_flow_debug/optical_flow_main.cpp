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
#include <limits>
#include <string>

#include "common.hpp"
#include "utils.hpp"
#include "arduino_bridge.hpp"

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
            else if (input.empty()) {
                // Serial terminals may send CR/LF as two events. Ignore the
                // empty event instead of reporting a false invalid command.
                continue;
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

static const char* region_name(int region) {
    switch (region) {
        case ObstacleInfo::LEFT: return "左侧";
        case ObstacleInfo::RIGHT: return "右侧";
        default: return "中间";
    }
}

static const char* clear_corridor_name(int region) {
    switch (region) {
        case ObstacleInfo::LEFT: return "左侧通道";
        case ObstacleInfo::RIGHT: return "右侧通道";
        default: return "中间通道";
    }
}

static const char* optical_state_name(int priority) {
    switch (priority) {
        case ObstacleInfo::EMERGENCY: return "紧急制动";
        case ObstacleInfo::CAUTION: return "注意避障";
        default: return "通道安全";
    }
}

enum class OpticalFlowOutputMode {
    LOCAL_ONLY = 1,
    ARDUINO_RELAY = 2
};

static OpticalFlowOutputMode choose_optical_flow_output_mode() {
    while (!g_signal_received.load()) {
        int choice = 0;
        printf("\n");
        printf("═══════════════════════════════════════════════════════════\n");
        printf("     光流避障输出模式\n");
        printf("═══════════════════════════════════════════════════════════\n");
        printf("  1. 标准模式：仅运行原光流避障，不启用 Arduino 通信\n");
        printf("  2. Arduino 模式：通过电脑中继驱动三色指示灯\n");
        printf("───────────────────────────────────────────────────────────\n");
        printf("请选择输出模式 (1-2) 并按回车: ");

        std::cin >> choice;
        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            printf("\n[错误] 输入无效，请输入数字 1 或 2。\n");
            continue;
        }
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        if (choice == 1) return OpticalFlowOutputMode::LOCAL_ONLY;
        if (choice == 2) return OpticalFlowOutputMode::ARDUINO_RELAY;
        printf("\n[提示] 无效的输出模式选项 (%d)，请重新选择。\n", choice);
    }

    return OpticalFlowOutputMode::LOCAL_ONLY;
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

    const OpticalFlowOutputMode output_mode = choose_optical_flow_output_mode();
    if (g_signal_received.load()) return 0;
    const bool arduino_enabled = output_mode == OpticalFlowOutputMode::ARDUINO_RELAY;
    if (arduino_enabled) {
        // uart0 is also the Type-C debug console. The Windows relay filters
        // @OF packets from ttyS0 and forwards them to the Arduino USB port.
        if (setenv("A1_ARDUINO_PORT", "/dev/ttyS0", 1) != 0) {
            std::perror("[Arduino] 无法设置 A1_ARDUINO_PORT");
            return -1;
        }
        printf("[输出模式] Arduino 电脑中继模式\n");
        printf("  • 已在程序内设置 A1_ARDUINO_PORT=/dev/ttyS0\n");
        printf("  • 请保持 Windows 中继脚本运行并占用 A1/Arduino 两个串口\n\n");
    } else {
        printf("[输出模式] 标准光流模式（不打开 Arduino 串口）\n\n");
    }
    
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

        // The previous module owns shutdown of its image pipeline.  Do not
        // unconditionally close pipe0 here: on A1, closing an already closed
        // or concurrently released pipe can race the next acquisition and
        // cause a fault during module switching.  IMAGEPROCESSOR owns the
        // pipe from this point onward.
        printf("[图像] 由统一图像管线接管 pipe0...\n");
    }

    image_processor.Initialize(&img_shape, 0, 720, 370, 910, 720, 540);
    printf("  ✓ 图像管道已打开\n\n");

    OPTICALFLOW optical_flow;
    optical_flow.Initialize(proc_shape[0], proc_shape[1]);
    
    // Cortex-A7 real-time profile. Robustness is retained in the regional
    // median/TTC/hysteresis stage, so LK itself can remain compact.
    optical_flow.max_features = 80;
    optical_flow.fast_threshold = 24;
    optical_flow.feature_scan_step = 2;
    optical_flow.grid_size = 8;
    optical_flow.grid_max_per_cell = 2;
    optical_flow.pyramid_levels = 2;
    optical_flow.lk_max_iter = 6;
    optical_flow.lk_win_size = 3;

    OBSTACLE_DETECTOR obstacle_detector;
    obstacle_detector.Initialize(proc_shape[0], proc_shape[1]);
    obstacle_detector.ttc_threshold = 2.0f;
    obstacle_detector.divergence_threshold = 0.45f;

    VISUALIZER visualizer;
    bool osd_present = module_loaded("osd_kmod") || dev_exists("/dev/osddev") || dev_exists("/dev/osd0");
    if (osd_present) {
        // This mode uses only a compact status bitmap, not a full-screen RLE
        // texture. Smaller image buffers avoid OSD CMA fragmentation.
        visualizer.Initialize(img_shape, "", 0x20000);
        printf("  ✓ OSD 可视化器初始化完成\n\n");
    }

    cout << "[INFO] 系统稳定等待 0.2 秒..." << endl;
    usleep(200000);

    printf("[处理] 开始实时处理\n");
    printf("        P/p键: 暂停/继续 | Q/q键: 退出\n");
    printf("────────────────────────────────────────────────────────────\n\n");

    vector<FeaturePoint> features;
    ObstacleInfo obstacle_info;
    ArduinoBridge arduino_bridge;
    if (arduino_enabled) arduino_bridge.Start();

    ssne_tensor_t curr_frame;
    memset(&curr_frame, 0, sizeof(ssne_tensor_t));

    std::vector<uint8_t> prev_buf;
    uint8_t* prev_ptr = nullptr;

    uint32_t frame_count = 0;
    uint32_t obstacle_frame_count = 0;
    uint32_t paused_frame_count = 0;
    bool first_frame = true;
    uint32_t last_feature_detect_frame = 0;

    int last_log_priority = 4;
    auto last_status_report = std::chrono::steady_clock::now();
    auto last_event_log = last_status_report - std::chrono::seconds(1);
    
    auto program_start_time = std::chrono::steady_clock::now();
    auto last_frame_time = program_start_time;
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
            prev_ptr = nullptr;
            prev_buf.clear();
            features.clear();
            first_frame = true;
            optical_flow.ResetHistory();
            usleep(10000);
            continue;
        }
        curr_ptr = (uint8_t*)get_data(curr_frame);
        const size_t expected_frame_bytes =
            static_cast<size_t>(proc_shape[0]) * static_cast<size_t>(proc_shape[1]);
        if (curr_ptr == nullptr ||
            get_width(curr_frame) != proc_shape[0] ||
            get_height(curr_frame) != proc_shape[1] ||
            get_mem_size(curr_frame) < expected_frame_bytes) {
            prev_ptr = nullptr;
            prev_buf.clear();
            features.clear();
            first_frame = true;
            optical_flow.ResetHistory();
            if (RuntimeLogAtLeast(RuntimeLogMode::VERIFY)) {
                printf("[图像] 丢弃无效帧 descriptor=%dx%d bytes=%zu expected=%zu\n",
                       get_width(curr_frame), get_height(curr_frame),
                       get_mem_size(curr_frame), expected_frame_bytes);
            }
            usleep(10000);
            continue;
        }

        auto now = std::chrono::steady_clock::now();
        obstacle_detector.SetFrameInterval(
            std::chrono::duration<float>(now - last_frame_time).count());
        last_frame_time = now;
        frame_times[frame_count % 10] = now;
        frame_count++;

        if (first_frame) {
            optical_flow.DetectFeatures(curr_ptr, features);
            optical_flow.ResetHistory();
            last_feature_detect_frame = frame_count;
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
            if (arduino_enabled) {
                arduino_bridge.Update(MakeArduinoFeedback(obstacle_info));
            }

            int curr_priority = obstacle_info.priority;
            int curr_region = obstacle_info.most_dangerous_region;

            const bool state_changed = curr_priority != last_log_priority;
            const bool emergency_entered = curr_priority == ObstacleInfo::EMERGENCY &&
                                           last_log_priority != ObstacleInfo::EMERGENCY;
            const bool event_log_due =
                std::chrono::duration<float>(now - last_event_log).count() >= 0.5f;
            if (state_changed && (emergency_entered || event_log_due)) {
                if (curr_priority == ObstacleInfo::EMERGENCY) {
                    printf("[避障][紧急] %s快速接近；建议优先保持%s，TTC≈%.1fs，可靠点=%d。\n",
                           region_name(curr_region),
                           clear_corridor_name(obstacle_info.safest_region),
                           obstacle_info.ttc_seconds[curr_region],
                           obstacle_info.support_count[curr_region]);
                } else if (curr_priority == ObstacleInfo::CAUTION) {
                    printf("[避障][提示] %s存在接近趋势；建议关注%s，风险=%.0f%%。\n",
                           region_name(curr_region),
                           clear_corridor_name(obstacle_info.safest_region),
                           100.0f * obstacle_info.danger_level[curr_region]);
                } else if (last_log_priority != ObstacleInfo::CLEAR) {
                    printf("[避障][恢复] 风险已解除；推荐通行%s。\n",
                           clear_corridor_name(obstacle_info.safest_region));
                }
                last_event_log = now;
            }

            if (RuntimeLogAtLeast(RuntimeLogMode::VERIFY) &&
                std::chrono::duration<float>(now - last_status_report).count() >= 2.0f) {
                const int sample_count = std::min(frame_count, static_cast<uint32_t>(10));
                const auto oldest = frame_times[(frame_count + 10 - sample_count) % 10];
                const float elapsed = std::chrono::duration<float>(now - oldest).count();
                const float fps = elapsed > 0.001f ? sample_count / elapsed : 0.0f;
                printf("[避障] 状态=%s FPS=%.1f 跟踪=%d/%zu 推荐=%s 质量=%.0f%%\n",
                       optical_state_name(curr_priority), fps, tracked_count, features.size(),
                       clear_corridor_name(obstacle_info.safest_region),
                       100.0f * obstacle_info.tracking_quality);
                last_status_report = now;
            }
            
            last_log_priority = curr_priority;

            if (curr_priority < 4) obstacle_frame_count++;

            // 15-20 Hz remains visually smooth; state transitions still
            // refresh immediately.
            if ((frame_count % 5) == 0 || emergency_entered) {
                visualizer.DrawAll(features, obstacle_info,370,frame_count);
            }

            const uint32_t frames_since_detect = frame_count - last_feature_detect_frame;
            const bool low_feature_refresh = tracked_count < 20 && frames_since_detect >= 6;
            const bool periodic_refresh = frames_since_detect >= 120;
            if (low_feature_refresh || periodic_refresh) {
                optical_flow.DetectFeatures(curr_ptr, features);
                optical_flow.ResetHistory();
                last_feature_detect_frame = frame_count;
            }
        }

        prev_buf.resize(expected_frame_bytes);
        memcpy(prev_buf.data(), curr_ptr, expected_frame_bytes);
        prev_ptr = prev_buf.data();
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

    // Tell the external controller to enter its fail-safe state before the
    // image pipeline and serial port are released.
    if (arduino_enabled) {
        arduino_bridge.SendStop();
        arduino_bridge.Close();
    }
    
    visualizer.Clear();
    visualizer.Release();
    optical_flow.Release();
    obstacle_detector.Release();
    image_processor.Release();

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
