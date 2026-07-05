/*
 * @Filename: main.cpp
 * @Description: SSNE AI Demo 多合一统摄主程序 (带全屏彩色 UI 叠加版)
 */

#include <iostream>
#include <string>
#include <limits>
#include "utils.hpp" // 【新增】引入通用可视化器，用于主菜单的 UI 叠加
#include <csignal>

extern int run_face_detection();
extern int run_object_detection();
extern int run_speed_detection();
extern int run_rps_detection();
extern int run_optical_flow_debug();
extern int run_facial_expressions();
extern int run_gesture_detection();
extern int run_focus_tracking();

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

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    int choice = -1;
    
    setup_signal_handlers();
    
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
        
        menu_visualizer.Initialize(img_shape, "shared_colorLUT.sscl");
        
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

        switch (choice) {
            case 1:
                std::cout << "\n>> 正在启动 [人脸检测] 模块...\n";
                run_face_detection();
                std::cout << "\n>> [人脸检测] 模块已安全退出，返回主菜单。\n";
                break;
            case 2:
                std::cout << "\n>> 正在启动 [目标检测] 模块...\n";
                run_object_detection();
                std::cout << "\n>> [目标检测] 模块已安全退出，返回主菜单。\n";
                break;
            case 3:
                std::cout << "\n>> 正在启动 [速度检测] 模块...\n";
                run_speed_detection();
                std::cout << "\n>> [速度检测] 模块已安全退出，返回主菜单。\n";
                break;
            case 4:
                std::cout << "\n>> 正在启动 [剪刀石头布] 模块...\n";
                run_rps_detection();
                std::cout << "\n>> [剪刀石头布] 模块已安全退出，返回主菜单。\n";
                break;
            case 5:
                std::cout << "\n>> 正在启动 [光流避障] 模块...\n";
                run_optical_flow_debug();
                std::cout << "\n>> [光流避障] 模块已安全退出，返回主菜单。\n";
                break;
            case 6:
                std::cout << "\n>> 正在启动 [表情识别] 模块...\n";
                run_facial_expressions();
                std::cout << "\n>> [表情识别] 模块已安全退出，返回主菜单。\n";
                break;
            case 7:
                std::cout << "\n>> 正在启动 [手势识别] 模块...\n";
                run_gesture_detection();
                std::cout << "\n>> [手势识别] 模块已安全退出，返回主菜单。\n";
                break;
            case 8:
                std::cout << "\n>> 进入 [追焦功能] 子菜单...\n";
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
    }
    
    return 0;
}
