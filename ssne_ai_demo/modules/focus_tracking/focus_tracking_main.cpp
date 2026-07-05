/*
 * @Filename: focus_tracking_main.cpp
 * @Description: Monocular focus tracking demo for grayscale SmartSens camera.
 */

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#include "common.hpp"
#include "utils.hpp"

using namespace std;

static bool g_focus_exit = false;
static bool g_focus_pause = false;
static bool g_focus_reset = false;
static std::mutex g_focus_mtx;

static const char* focus_mode_name(FocusTrackingMode mode) {
    switch (mode) {
        case FocusTrackingMode::NO_NPU_TRACKER:
            return "传统视觉追焦";
        case FocusTrackingMode::NPU_MOBILENET:
            return "MobileNet NPU追焦";
        default:
            return "未知追焦模式";
    }
}

static void print_focus_mode_menu() {
    cout << "\n======================================================\n";
    cout << "          追焦功能子菜单 / Focus Tracking            \n";
    cout << "======================================================\n";
    cout << "  1. 传统视觉追焦 (No NPU - 高帧率模板追踪)\n";
    cout << "  2. MobileNet NPU追焦 (模型选目标 + 传统追踪接管)\n";
    cout << "  0. 返回主菜单\n";
    cout << "======================================================\n";
    cout << "请输入追焦子功能编号 (0-2) 并按回车: ";
}

static bool choose_focus_mode(FocusTrackingMode* mode) {
    if (mode == nullptr) return false;

    while (!g_signal_received.load()) {
        int choice = -1;
        clear_stdin_residual();
        print_focus_mode_menu();
        cin >> choice;

        if (cin.fail()) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "\n[错误] 输入无效，请输入数字编号！\n";
            continue;
        }

        if (choice == 0) {
            cout << "\n>> 返回主菜单。\n";
            return false;
        }
        if (choice == 1) {
            *mode = FocusTrackingMode::NO_NPU_TRACKER;
            return true;
        }
        if (choice == 2) {
            *mode = FocusTrackingMode::NPU_MOBILENET;
            return true;
        }

        cout << "\n[提示] 无效的追焦子功能选项 (" << choice << ")，请重新选择。\n";
    }

    return false;
}

static bool dev_exists_focus(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static bool module_loaded_focus(const char* mod) {
    std::ifstream f("/proc/modules");
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind(mod, 0) == 0) return true;
    }
    return false;
}

static void focus_keyboard_listener() {
    string input;
    printf("[追焦键盘] ┌─────────────────────────────────┐\n");
    printf("[追焦键盘] │ P/p: 暂停/继续                  │\n");
    printf("[追焦键盘] │ R/r: 重置锁定目标                │\n");
    printf("[追焦键盘] │ Q/q: 退出当前子功能并返回子菜单  │\n");
    printf("[追焦键盘] └─────────────────────────────────┘\n\n");

    while (true) {
        {
            lock_guard<mutex> lock(g_focus_mtx);
            if (g_focus_exit || g_signal_received.load()) break;
        }
        if (nonblocking_getline(input, 100)) {
            lock_guard<mutex> lock(g_focus_mtx);
            if (input == "q" || input == "Q") {
                g_focus_exit = true;
                break;
            }
            if (input == "p" || input == "P") {
                g_focus_pause = !g_focus_pause;
            }
            if (input == "r" || input == "R") {
                g_focus_pause = false;
                g_focus_reset = true;
            }
        }
    }
}

static bool focus_should_exit() {
    lock_guard<mutex> lock(g_focus_mtx);
    return g_focus_exit || g_signal_received.load();
}

static bool focus_is_paused() {
    lock_guard<mutex> lock(g_focus_mtx);
    return g_focus_pause;
}

static bool focus_take_reset_request() {
    lock_guard<mutex> lock(g_focus_mtx);
    bool reset = g_focus_reset;
    g_focus_reset = false;
    return reset;
}

static void resize_gray_nearest(const uint8_t* src,
                                int src_w,
                                int src_h,
                                uint8_t* dst,
                                int dst_w,
                                int dst_h) {
    if (src == nullptr || dst == nullptr) return;
    for (int y = 0; y < dst_h; y++) {
        int src_y = y * src_h / dst_h;
        const uint8_t* src_row = src + src_y * src_w;
        uint8_t* dst_row = dst + y * dst_w;
        for (int x = 0; x < dst_w; x++) {
            int src_x = x * src_w / dst_w;
            dst_row[x] = src_row[src_x];
        }
    }
}

static FocusTargetState scale_target_to_proc(const FocusTargetState& src,
                                             int src_w,
                                             int src_h,
                                             int dst_w,
                                             int dst_h) {
    FocusTargetState dst = src;
    dst.x = src.x * dst_w / src_w;
    dst.y = src.y * dst_h / src_h;
    dst.w = std::max(1, src.w * dst_w / src_w);
    dst.h = std::max(1, src.h * dst_h / src_h);
    dst.cx = src.cx * (float)dst_w / (float)src_w;
    dst.cy = src.cy * (float)dst_h / (float)src_h;
    return dst;
}

static int run_focus_tracking_mode(FocusTrackingMode selected_mode) {
    {
        lock_guard<mutex> lock(g_focus_mtx);
        g_focus_exit = false;
        g_focus_pause = false;
        g_focus_reset = false;
    }

    clear_stdin_residual();

    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("     SmartSens 单目追焦系统\n");
    printf("     Monocular Focus Tracking\n");
    printf("     当前子模式: %s\n", focus_mode_name(selected_mode));
    printf("═══════════════════════════════════════════════════════════\n\n");

    const int sensor_w = 720;
    const int sensor_h = 1280;
    const int crop_x1 = 0;
    const int crop_x2 = 720;
    const int crop_y1 = 370;
    const int crop_y2 = 910;
    const int capture_w = crop_x2 - crop_x1;
    const int capture_h = crop_y2 - crop_y1;
    const int proc_w = 360;
    const int proc_h = 270;

    array<int, 2> img_shape = {sensor_w, sensor_h};
    IMAGEPROCESSOR image_processor;

    bool ssne_ok = false;
    {
        SigintBlocker blocker;
        if (ssne_initial() != 0) {
            fprintf(stderr, "[WARN] ssne_initial() failed!\n");
        } else {
            ssne_ok = true;
        }
    }

    image_processor.Initialize(&img_shape, crop_x1, crop_x2, crop_y1, crop_y2, capture_w, capture_h);
    if (!image_processor.IsOpened()) {
        fprintf(stderr, "[ERROR] 追焦图像管线打开失败，请检查 camera/pipeline 状态。\n");
        if (ssne_ok) {
            SigintBlocker blocker;
            ssne_release();
        }
        return -1;
    }

    FocusTrackingConfig tracker_config;
    tracker_config.width = proc_w;
    tracker_config.height = proc_h;
    tracker_config.target_w = 72;
    tracker_config.target_h = 72;
    tracker_config.search_radius = 30;
    tracker_config.search_step = 3;
    tracker_config.sample_step = 4;
    tracker_config.mode = selected_mode;
    tracker_config.npu_model_path = "./app_assets/models/focus_mobilenet.m1model";
    tracker_config.npu_interval_frames = 15;

    FocusTracker tracker;
    tracker.Initialize(tracker_config);

    MobileNetFocusSelector npu_selector;
    array<int, 2> npu_model_shape = {160, 160};
    array<int, 2> capture_shape = {capture_w, capture_h};
    if (tracker_config.mode == FocusTrackingMode::NPU_MOBILENET) {
        npu_selector.Initialize(tracker_config.npu_model_path, &capture_shape, &npu_model_shape);
        if (!npu_selector.IsReady()) {
            fprintf(stderr,
                    "[WARN] MobileNet NPU 模式已选择，但模型/接口尚不可用；"
                    "当前帧会继续由传统追踪器兜底。\n");
        }
    }

    VISUALIZER visualizer;
    bool osd_present = module_loaded_focus("osd_kmod") ||
                       dev_exists_focus("/dev/osddev") ||
                       dev_exists_focus("/dev/osd0");
    if (osd_present) {
        visualizer.Initialize(img_shape, "shared_colorLUT.sscl");
    }

    printf("[配置] sensor=%dx%d crop=(%d,%d)-(%d,%d) capture=%dx%d proc=%dx%d\n",
           sensor_w, sensor_h, crop_x1, crop_y1, crop_x2, crop_y2, capture_w, capture_h, proc_w, proc_h);
    printf("[模式] %s\n", focus_mode_name(tracker_config.mode));
    printf("[处理] P/p暂停 | R/r重置 | Q/q返回追焦子菜单\n\n");

    thread listener_thread(focus_keyboard_listener);

    ssne_tensor_t curr_frame;
    memset(&curr_frame, 0, sizeof(ssne_tensor_t));
    vector<uint8_t> proc_frame((size_t)proc_w * (size_t)proc_h);
    FocusTargetState target;

    uint32_t frame_count = 0;
    uint32_t locked_count = 0;
    uint32_t last_log_frame = 0;
    bool last_osd_locked = false;
    auto start_time = chrono::steady_clock::now();
    chrono::steady_clock::time_point frame_times[10];
    for (int i = 0; i < 10; i++) frame_times[i] = start_time;

    {
        SigintBlocker blocker;
        while (!focus_should_exit()) {
            if (focus_is_paused()) {
                usleep(50000);
                continue;
            }

            image_processor.GetImage(&curr_frame);
            if (curr_frame.data == nullptr) {
                usleep(5000);
                continue;
            }

            uint8_t* capture_ptr = (uint8_t*)get_data(curr_frame);
            if (capture_ptr == nullptr) {
                usleep(5000);
                continue;
            }
            resize_gray_nearest(capture_ptr, capture_w, capture_h, proc_frame.data(), proc_w, proc_h);
            uint8_t* frame_ptr = proc_frame.data();

            if (focus_take_reset_request()) {
                tracker.Reset();
                target.Clear();
                printf("[追焦] 已重置锁定目标，下一帧重新选择。\n");
            }

            frame_times[frame_count % 10] = chrono::steady_clock::now();
            frame_count++;

            if (tracker_config.mode == FocusTrackingMode::NPU_MOBILENET &&
                npu_selector.IsReady() &&
                (frame_count == 1 || frame_count % tracker_config.npu_interval_frames == 0 ||
                !tracker.HasTarget())) {
                FocusTargetState npu_target;
                if (npu_selector.Predict(&curr_frame, &npu_target)) {
                    FocusTargetState proc_target =
                        scale_target_to_proc(npu_target, capture_w, capture_h, proc_w, proc_h);
                    tracker.SetTarget(frame_ptr, proc_target);
                }
            }

            bool locked = tracker.Update(frame_ptr, &target);
            if (locked) locked_count++;

            if (locked && osd_present) {
                int sx1 = crop_x1 + target.x * (crop_x2 - crop_x1) / proc_w;
                int sy1 = crop_y1 + target.y * (crop_y2 - crop_y1) / proc_h;
                int sx2 = crop_x1 + (target.x + target.w) * (crop_x2 - crop_x1) / proc_w;
                int sy2 = crop_y1 + (target.y + target.h) * (crop_y2 - crop_y1) / proc_h;
                vector<array<float, 4>> focus_boxes;
                vector<float> focus_scores;
                vector<int> focus_class_ids;
                focus_boxes.push_back({(float)sx1, (float)sy1, (float)sx2, (float)sy2});
                focus_scores.push_back(target.confidence);
                focus_class_ids.push_back(0);
                visualizer.Draw(focus_boxes, focus_scores, focus_class_ids);
                last_osd_locked = true;
            } else if (!locked && osd_present && last_osd_locked) {
                vector<array<float, 4>> empty_boxes;
                vector<float> empty_scores;
                vector<int> empty_ids;
                visualizer.Draw(empty_boxes, empty_scores, empty_ids);
                last_osd_locked = false;
            }

            if (frame_count - last_log_frame >= 30) {
                int calc_frames = frame_count < 10 ? frame_count : 10;
                auto now = frame_times[(frame_count - 1) % 10];
                auto old = frame_times[frame_count % 10];
                chrono::duration<float> diff = now - old;
                float fps = (diff.count() > 0.001f && calc_frames > 0) ?
                            ((float)calc_frames / diff.count()) : 0.0f;
                float dx = locked ? (target.cx - 0.5f * (float)proc_w) : 0.0f;
                float dy = locked ? (target.cy - 0.5f * (float)proc_h) : 0.0f;
                printf("[追焦] f=%u fps=%4.1f lock=%d conf=%.2f focus=%.1f dx=%.1f dy=%.1f lost=%d\n",
                       frame_count, fps, locked ? 1 : 0, target.confidence,
                       target.focus_score, dx, dy, target.lost_frames);
                last_log_frame = frame_count;
            }
        }
    }

    {
        lock_guard<mutex> lock(g_focus_mtx);
        g_focus_exit = true;
    }
    if (listener_thread.joinable()) listener_thread.join();

    auto end_time = chrono::steady_clock::now();
    chrono::duration<float> active_time = end_time - start_time;
    float avg_fps = active_time.count() > 0.001f ? (float)frame_count / active_time.count() : 0.0f;

    printf("\n[统计] 追焦结束\n");
    printf("  • 总帧数: %u\n", frame_count);
    printf("  • 平均帧率: %4.1f FPS\n", avg_fps);
    printf("  • 锁定帧占比: %.1f%%\n", frame_count > 0 ? 100.0f * locked_count / frame_count : 0.0f);

    tracker.Reset();
    npu_selector.Release();
    visualizer.Release();
    image_processor.Release();

    {
        SigintBlocker blocker;
        if (ssne_ok) ssne_release();
    }
    usleep(300000);

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("     当前追焦子功能结束，返回追焦子菜单\n");
    printf("════════════════════════════════════════════════════════════\n\n");

    return 0;
}

int run_focus_tracking() {
    while (!g_signal_received.load()) {
        FocusTrackingMode selected_mode = FocusTrackingMode::NO_NPU_TRACKER;
        if (!choose_focus_mode(&selected_mode)) {
            return 0;
        }

        cout << "\n>> 正在启动 [" << focus_mode_name(selected_mode) << "] 子功能...\n";
        int ret = run_focus_tracking_mode(selected_mode);
        if (g_signal_received.load()) {
            return ret;
        }
    }

    return 0;
}
