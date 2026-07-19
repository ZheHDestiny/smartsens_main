/* RPS module entry point for the unified console. */

#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unistd.h>
#include "common.hpp"
#include "utils.hpp"

using namespace std;

static bool g_exit_flag = false;
static std::mutex g_mtx;

static int ClampParam(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

static void KeyboardListener() {
    std::string line;
    std::cout << "RPS commands: show | help | threshold N | idle_pixels N | "
                 "gap N | gain N | windup N | vote N | q" << std::endl;

    while (true) {
        {
            std::lock_guard<std::mutex> lock(g_mtx);
            if (g_exit_flag || g_signal_received.load()) break;
        }

        if (!nonblocking_getline(line, 100)) continue;

        std::istringstream parser(line);
        std::string command;
        parser >> command;
        if (command.empty()) continue;

        if (command == "q" || command == "Q") {
            std::lock_guard<std::mutex> lock(g_mtx);
            g_exit_flag = true;
            break;
        }
        if (command == "show") {
            PrintRuntimeRpsParams();
            continue;
        }
        if (command == "help") {
            printf("[RPS_PARAM] show | threshold N | idle_pixels N | "
                   "gap N(1-%d) | gain N(1-32) | windup N | vote N | q\n",
                   RPS_MAX_RUNTIME_FRAME_GAP);
            continue;
        }

        int value = 0;
        if (!(parser >> value)) {
            printf("[RPS_PARAM] missing integer value; type help\n");
            continue;
        }

        if (command == "threshold") {
            g_runtime_rps_params.diff_threshold.store(ClampParam(value, 0, 255));
        } else if (command == "idle_pixels") {
            g_runtime_rps_params.idle_min_active_pixels.store(
                ClampParam(value, 0, 360 * 640));
        } else if (command == "gap") {
            g_runtime_rps_params.frame_gap.store(
                ClampParam(value, 1, RPS_MAX_RUNTIME_FRAME_GAP));
        } else if (command == "gain") {
            g_runtime_rps_params.diff_gain.store(ClampParam(value, 1, 32));
        } else if (command == "windup") {
            g_runtime_rps_params.wind_up_frames.store(
                ClampParam(value, 1, TemporalBuffer::CAPACITY));
        } else if (command == "vote") {
            g_runtime_rps_params.vote_frames.store(
                ClampParam(value, 1, TemporalBuffer::CAPACITY));
        } else {
            printf("[RPS_PARAM] unknown command: %s; type help\n", command.c_str());
            continue;
        }
        PrintRuntimeRpsParams();
    }
}

static bool CheckExitFlag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}

int run_rps_detection() {
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        g_exit_flag = false;
    }
    clear_stdin_residual();

    const int image_width = 720;
    const int image_height = 1280;
    array<int, 2> image_shape = {image_width, image_height};
    array<int, 2> model_shape = {360, 640};
    array<int, 2> classifier_input_shape = {720, 1280};
    string model_path = "./app_assets/models/paper.m1model";
    const std::array<float, 4> hand_roi = {0.f, 0.f, 720.f, 1280.f};

    {
        SigintBlocker blocker;
        if (ssne_initial()) {
            fprintf(stderr, "SSNE initialization failed!\n");
            return -1;
        }
    }

    IMAGEPROCESSOR processor;
    processor.Initialize(&image_shape, 0, 720, 0, 1280, 720, 1280);
    if (!processor.IsOpened()) {
        fprintf(stderr, "RPS camera pipeline failed to open!\n");
        SigintBlocker blocker;
        ssne_release();
        return -1;
    }

    RPSCLASSIFIER classifier;
    classifier.Initialize(model_path, &classifier_input_shape, &model_shape);

    RpsResult rps_result;
    VISUALIZER visualizer;
    visualizer.Initialize(image_shape);
    usleep(200000);

    ssne_tensor_t image_sensor;
    memset(&image_sensor, 0, sizeof(image_sensor));
    std::thread listener_thread(KeyboardListener);
    bool result_was_locked = false;

    {
        SigintBlocker blocker;
        while (!CheckExitFlag() && !g_signal_received.load()) {
            processor.GetImage(&image_sensor);
            if (image_sensor.data == nullptr) {
                usleep(10000);
                continue;
            }

            classifier.Predict(&image_sensor, &rps_result);
            visualizer.Draw(rps_result, hand_roi);

            if (rps_result.is_locked && !result_was_locked) {
                const char* human_name =
                    GESTURE_NAMES[static_cast<int>(rps_result.human_gesture)];
                const char* ai_name =
                    GESTURE_NAMES[static_cast<int>(rps_result.ai_counter)];
                printf("[RPS_GAME] human=%s ai=%s confidence=%.2f\n",
                       human_name, ai_name, rps_result.confidence);
            }
            result_was_locked = rps_result.is_locked;
        }
    }

    if (listener_thread.joinable()) listener_thread.join();
    visualizer.Clear();
    visualizer.Release();
    classifier.Release();
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
