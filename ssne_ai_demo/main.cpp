/*
 * @Filename: main.cpp
 * @Description: SSNE AI Demo 多合一统摄主程序 (带全屏彩色 UI 叠加版)
 */

#include <iostream>
#include <string>
#include <limits>
#include "utils.hpp" // 【新增】引入通用可视化器，用于主菜单的 UI 叠加
#include <csignal>
#include <cstdio>
#include <cstring>
#include <malloc.h>
#include <cerrno>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern int run_face_detection();
extern int run_object_detection();
extern int run_speed_detection();
extern int run_rps_detection();
extern int run_optical_flow_debug();
extern int run_facial_expressions();
extern int run_gesture_detection();
extern int run_focus_tracking();

namespace {

long read_process_status_kb(const char* key) {
    FILE* fp = std::fopen("/proc/self/status", "r");
    if (fp == nullptr) return -1;

    char line[160];
    long value = -1;
    const size_t key_len = std::strlen(key);
    while (std::fgets(line, sizeof(line), fp) != nullptr) {
        if (std::strncmp(line, key, key_len) == 0) {
            if (std::sscanf(line + key_len, "%ld", &value) != 1) value = -1;
            break;
        }
    }
    std::fclose(fp);
    return value;
}

void reclaim_module_heap(const char* phase) {
    const int reclaimed = malloc_trim(0);
    if (RuntimeLogAtLeast(RuntimeLogMode::VERIFY)) {
        std::printf("[MEM_RECLAIM] phase=%s trim=%d rss=%ldkB data=%ldkB\n",
                    phase != nullptr ? phase : "unknown",
                    reclaimed,
                    read_process_status_kb("VmRSS:"),
                    read_process_status_kb("VmData:"));
    }
}

typedef int (*ModuleEntry)();

int run_module_isolated(const char* feature, ModuleEntry entry) {
    // SSNE has no per-model unload API and repeated ssne_release()/initial()
    // cycles leave board-driver/CMA mappings behind on some firmware builds.
    // A short-lived worker process gives every menu invocation a hard kernel
    // resource boundary; exit also cleans up safely after an SDK-side fault.
    reclaim_module_heap("before-fork");
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid < 0) {
        std::perror("[MODULE_ISOLATION] fork failed");
        return -1;
    }
    if (pid == 0) {
        g_signal_received.store(false);
        OfficialPerfReset(feature, 90.0f);
        const int rc = entry();
        OfficialPerfPrintFinal();
        std::fflush(nullptr);
        _exit(rc == 0 ? 0 : 1);
    }

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) continue;
        std::perror("[MODULE_ISOLATION] waitpid failed");
        return -1;
    }
    if (WIFSIGNALED(status)) {
        std::printf("[MODULE_ISOLATION] feature=%s worker terminated by signal=%d\n",
                    feature, WTERMSIG(status));
        return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

}  // namespace

void print_menu() {
    std::cout << "\n======================================================\n";
    std::cout << "          SmartSens SSNE AI Demo 多合一控制台         \n";
    std::cout << "======================================================\n";
    std::cout << "  1. 人脸检测 (Face Detection - SCRFD)\n";
    std::cout << "  2. 目标检测 (Object Detection - YOLOv8 20类)\n";
    std::cout << "  3. 速度检测 (Speed Detection - YOLOv8 3类 + 测速)\n";
    std::cout << "  4. 剪刀石头布 (RPS Detection - 90fps 状态机)\n";
    std::cout << "  5. 光流避障 (Optical Flow Debug - FAST + LK)\n";
    std::cout << "  6. 表情识别 (Facial Expressions - CNN分类)\n";
    std::cout << "  7. 手势识别 (Gesture Detection - CNN分类 + CLAHE)\n";
    std::cout << "  8. 追焦功能 (Focus Tracking - 单目目标追踪聚焦)\n";
    std::cout << "  0. 退出(quit)\n";
    std::cout << "======================================================\n";
    std::cout << "请输入功能编号 (0-8) 并按回车: ";
}

const char* runtime_log_mode_name(RuntimeLogMode mode) {
    switch (mode) {
        case RuntimeLogMode::SILENT:
            return "静默模式";
        case RuntimeLogMode::SUMMARY:
            return "摘要模式";
        case RuntimeLogMode::VERIFY:
            return "验证模式";
        default:
            return "未知模式";
    }
}

RuntimeLogMode choose_runtime_log_mode() {
    while (!g_signal_received.load()) {
        int choice = -1;
        std::cout << "\n======================================================\n";
        std::cout << "              运行日志模式 / Log Mode                \n";
        std::cout << "======================================================\n";
        std::cout << "  说明：业务识别结果始终输出；以下选项只控制额外诊断日志\n";
        std::cout << "  0. 静默模式：仅输出业务结果，最低串口开销\n";
        std::cout << "  1. 摘要模式：业务结果 + 每秒公共管线健康摘要\n";
        std::cout << "  2. 验证模式：业务结果 + 官方口径FPS/P95/T评分/图像质量\n";
        std::cout << "======================================================\n";
        std::cout << "请选择日志模式 (0-2) 并按回车: ";
        std::cin >> choice;

        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "\n[错误] 输入无效，请输入数字编号！\n";
            continue;
        }

        if (choice == 0) return RuntimeLogMode::SILENT;
        if (choice == 1) return RuntimeLogMode::SUMMARY;
        if (choice == 2) return RuntimeLogMode::VERIFY;

        std::cout << "\n[提示] 无效的日志模式选项 (" << choice << ")，请重新选择。\n";
    }

    return RuntimeLogMode::SUMMARY;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    int choice = -1;
    
    setup_signal_handlers();

    // Limit long-lived per-thread allocator arenas. Actual trimming is done
    // only after a module exits, never between menu OSD release and camera
    // pipeline startup.
    mallopt(M_ARENA_MAX, 1);
    mallopt(M_TRIM_THRESHOLD, 64 * 1024);

    std::cout << "\033[2J\033[1;1H";
    
    std::array<int, 2> img_shape = {720, 1280};
    VISUALIZER menu_visualizer;

    while (true) {
        if (g_signal_received.load()) {
            menu_visualizer.Clear();
            menu_visualizer.Release();
            std::cout << "\n>> 收到中断信号，安全退出系统...\n";
            return 0;
        }
        
        clear_stdin_residual();
        
        // The menu only draws one full-screen bitmap. Creating five unused
        // layers consumed and fragmented scarce OSD CMA on every menu return.
        menu_visualizer.Initialize(img_shape, "shared_colorLUT.sscl", 0x100000,
                                   (1u << 2), (1u << 2));
        
        menu_visualizer.DrawBitmap("background.ssbmp", "shared_colorLUT.sscl", 0, 0, 2);

        print_menu();
        std::cin >> choice;

        if (std::cin.fail()) {
            std::cin.clear(); 
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 
            std::cout << "\n[错误] 输入无效，请输入数字编号！\n";
            
            menu_visualizer.Clear();
            menu_visualizer.Release();
            continue;
        }

        menu_visualizer.Clear();
        menu_visualizer.Release();

        if (g_signal_received.load()) {
            std::cout << "\n>> 收到中断信号，安全退出系统...\n";
            return 0;
        }

        if (choice >= 1 && choice <= 8) {
            RuntimeLogMode log_mode = choose_runtime_log_mode();
            SetRuntimeLogMode(log_mode);
            std::cout << "\n>> 当前日志模式: " << runtime_log_mode_name(log_mode) << "\n";
        }

        switch (choice) {
            case 1:
                std::cout << "\n>> 正在启动 [人脸检测] 模块...\n";
                run_module_isolated("face_detection", run_face_detection);
                std::cout << "\n>> [人脸检测] 模块已安全退出，返回主菜单。\n";
                break;
            case 2:
                std::cout << "\n>> 正在启动 [目标检测] 模块...\n";
                run_module_isolated("object_detection", run_object_detection);
                std::cout << "\n>> [目标检测] 模块已安全退出，返回主菜单。\n";
                break;
            case 3:
                std::cout << "\n>> 正在启动 [速度检测] 模块...\n";
                run_module_isolated("speed_detection", run_speed_detection);
                std::cout << "\n>> [速度检测] 模块已安全退出，返回主菜单。\n";
                break;
            case 4:
                std::cout << "\n>> 正在启动 [剪刀石头布] 模块...\n";
                run_module_isolated("rps_detection", run_rps_detection);
                std::cout << "\n>> [剪刀石头布] 模块已安全退出，返回主菜单。\n";
                break;
            case 5:
                std::cout << "\n>> 正在启动 [光流避障] 模块...\n";
                run_module_isolated("optical_flow", run_optical_flow_debug);
                std::cout << "\n>> [光流避障] 模块已安全退出，返回主菜单。\n";
                break;
            case 6:
                std::cout << "\n>> 正在启动 [表情识别] 模块...\n";
                run_module_isolated("facial_expression", run_facial_expressions);
                std::cout << "\n>> [表情识别] 模块已安全退出，返回主菜单。\n";
                break;
            case 7:
                std::cout << "\n>> 正在启动 [手势识别] 模块...\n";
                run_module_isolated("gesture_detection", run_gesture_detection);
                std::cout << "\n>> [手势识别] 模块已安全退出，返回主菜单。\n";
                break;
            case 8:
                std::cout << "\n>> 进入 [追焦功能] 子菜单...\n";
                // Focus modes already have their own worker boundary. Avoid
                // nesting fork() here: nested workers duplicated the submenu
                // runtime/stdio state and made abnormal mode exits propagate
                // as heap/driver corruption in the outer worker.
                run_focus_tracking();
                std::cout << "\n>> 已离开 [追焦功能]，返回主菜单。\n";
                break;
            case 0:
                std::cout << "\n>> 正在退出系统... 再见！\n";
                return 0;
            default:
                std::cout << "\n[提示] 无效的选项 (" << choice << ")，请重新选择。\n";
                break;
        }

        if (choice >= 1 && choice <= 8) {
            reclaim_module_heap("module-exit");
        }
    }
    
    return 0;
}
