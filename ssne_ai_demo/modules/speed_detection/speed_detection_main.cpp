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
static bool g_fps_report_requested = false;
static std::mutex g_mtx;

static void keyboard_listener() {
    std::string input;
    std::cout << "按 'f' + 回车查看 FPS，按 'q' + 回车退出速度检测" << std::endl;
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
            if (input == "f" || input == "F") {
                g_fps_report_requested = true;
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

// Post-processing thresholds. The network remains unchanged; these gates
// are intentionally stricter for the class that most often confuses dark
// handheld objects with a car.
static const float kModelConfThreshold = 0.35f;
static const float kCarConfThreshold = 0.45f;
static const int kRequiredDetectionHits = 3;
static const int kStaticClutterFrames = 90;
static const double kMinPixelWidth = 4.0;
// The speed demo is used with a fixed camera. Validate that a detection has
// real image motion before it may enter the speed tracker, so static objects
// such as a mouse or phone cannot remain a high-confidence false vehicle.
static const int kMotionDownsample = 2;
static const int kMotionReferenceFrames = 6;
static const int kMotionPixelDelta = 18;
static const float kMinMotionChangedRatio = 0.025f;
static const float kMinMotionMeanDelta = 3.0f;
// Keep speed measurement responsive to a short pass. This is deliberately
// independent from the one-second FPS diagnostic report below.
static const double kSpeedUpdateIntervalSec = 0.05;
static const double kDirectionMinDeltaCm = 0.05;
static const float kSpeedDisplayDeadbandCmPerSec = 0.35f;
static const int kMovingSpeedLogIntervalMs = 125;
static const int kIdleSpeedLogIntervalMs = 500;

struct TrackedVehicle {
    double smoothed_x = 0.0;
    double last_calc_x = 0.0;
    std::chrono::steady_clock::time_point last_calc_time;
    float current_speed = 0.0f;
    float smoothed_speed = 0.0f;
    int direction = 0;
    bool initialized = false;
    int missed_frames = 0;
    int class_id = -1;
    int hit_streak = 0;
    int stable_frames = 0;
    double last_pixel_width = 0.0;
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
        g_fps_report_requested = false;
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
    visualizer.Initialize(osd_shape, "shared_colorLUT.sscl", 0x20000,
                          0u, (1u << 0) | (1u << 4));

    ObjectDetectionResult det_result;
    std::thread listener_thread(keyboard_listener);
    ssne_tensor_t img_sensor;
    memset(&img_sensor, 0, sizeof(img_sensor));
    auto last_speed_report = std::chrono::steady_clock::now()
                           - std::chrono::seconds(1);
    const auto stats_start = std::chrono::steady_clock::now();
    std::deque<std::chrono::steady_clock::time_point> frame_times;
    size_t processed_frames = 0;

    const auto calc_last10_fps = [&frame_times]() -> double {
        if (frame_times.size() < 2) return 0.0;
        const double seconds = std::chrono::duration<double>(
            frame_times.back() - frame_times.front()).count();
        return seconds > 0.0
            ? static_cast<double>(frame_times.size() - 1) / seconds
            : 0.0;
    };

    std::map<int, TrackedVehicle> trackers;
    int next_tracker_id = 0;
    const int motion_width = img_shape[0] / kMotionDownsample;
    const int motion_height = img_shape[1] / kMotionDownsample;
    std::vector<uint8_t> motion_current(motion_width * motion_height);
    std::vector<uint8_t> motion_reference(motion_width * motion_height);
    bool motion_reference_ready = false;
    int motion_reference_age = 0;

    const auto sample_motion_frame = [&](const ssne_tensor_t& image) -> bool {
        const uint8_t* source = static_cast<const uint8_t*>(get_data(image));
        if (source == nullptr) return false;
        for (int y = 0; y < motion_height; ++y) {
            const uint8_t* row = source + static_cast<size_t>(y * kMotionDownsample) * img_shape[0];
            uint8_t* dst = motion_current.data() + static_cast<size_t>(y) * motion_width;
            for (int x = 0; x < motion_width; ++x) {
                dst[x] = row[x * kMotionDownsample];
            }
        }
        return true;
    };

    const auto box_has_motion = [&](float x0, float y0, float x1, float y1,
                                    float* changed_ratio, float* mean_delta) -> bool {
        if (changed_ratio != nullptr) *changed_ratio = 0.0f;
        if (mean_delta != nullptr) *mean_delta = 0.0f;
        if (!motion_reference_ready) return false;

        const int gx0 = std::max(0, std::min(motion_width - 1,
            static_cast<int>(x0) / kMotionDownsample));
        const int gy0 = std::max(0, std::min(motion_height - 1,
            static_cast<int>(y0) / kMotionDownsample));
        const int gx1 = std::max(gx0 + 1, std::min(motion_width,
            (static_cast<int>(x1) + kMotionDownsample - 1) / kMotionDownsample));
        const int gy1 = std::max(gy0 + 1, std::min(motion_height,
            (static_cast<int>(y1) + kMotionDownsample - 1) / kMotionDownsample));

        int samples = 0;
        int changed = 0;
        int delta_sum = 0;
        for (int y = gy0; y < gy1; ++y) {
            const size_t row_offset = static_cast<size_t>(y) * motion_width;
            for (int x = gx0; x < gx1; ++x) {
                const int delta = std::abs(static_cast<int>(motion_current[row_offset + x]) -
                                           static_cast<int>(motion_reference[row_offset + x]));
                delta_sum += delta;
                if (delta >= kMotionPixelDelta) ++changed;
                ++samples;
            }
        }
        if (samples == 0) return false;

        const float ratio = static_cast<float>(changed) / samples;
        const float mean = static_cast<float>(delta_sum) / samples;
        if (changed_ratio != nullptr) *changed_ratio = ratio;
        if (mean_delta != nullptr) *mean_delta = mean;
        return ratio >= kMinMotionChangedRatio || mean >= kMinMotionMeanDelta;
    };

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

        if (!sample_motion_frame(img_sensor)) {
            continue;
        }

        detector.Predict(&img_sensor, &det_result, kModelConfThreshold);

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
            if (i >= det_result.class_ids.size() || i >= det_result.scores.size()) {
                continue;
            }
            int cid = det_result.class_ids[i];
            if (cid < 0 || cid > 2) {
                continue;
            }

            const float score = det_result.scores[i];
            const float class_threshold = (cid == 0) ? kCarConfThreshold : kModelConfThreshold;
            if (!std::isfinite(score) || score < class_threshold) {
                continue;
            }

            float x1_orig = det_result.boxes[i][0];
            float y1_orig = det_result.boxes[i][1] + 560.0f;
            float x2_orig = det_result.boxes[i][2];
            float y2_orig = det_result.boxes[i][3] + 560.0f;

            // Reject malformed boxes before any geometry or depth calculation.
            if (x1_orig < 0 || y1_orig < 0 || x2_orig > osd_shape[0] || y2_orig > osd_shape[1] ||
                x1_orig >= x2_orig || y1_orig >= y2_orig ||
                !std::isfinite(x1_orig) || !std::isfinite(y1_orig) ||
                !std::isfinite(x2_orig) || !std::isfinite(y2_orig)) {
                continue;
            }

            // A vehicle used for speed measurement must show real motion in
            // its own ROI. This rejects stationary mouse/phone false alarms
            // before they can create or refresh a tracker.
            if (!box_has_motion(det_result.boxes[i][0], det_result.boxes[i][1],
                                det_result.boxes[i][2], det_result.boxes[i][3],
                                nullptr, nullptr)) {
                continue;
            }

            double u1_u, v1_u, u2_u, v2_u, center_u, center_v;
            UndistortPoint(x1_orig, (y1_orig+y2_orig)/2, u1_u, v1_u);
            UndistortPoint(x2_orig, (y1_orig+y2_orig)/2, u2_u, v2_u);
            UndistortPoint((x1_orig+x2_orig)/2, (y1_orig+y2_orig)/2, center_u, center_v);

            double pixel_width = std::abs(u2_u - u1_u);
            if (!std::isfinite(pixel_width) || pixel_width < kMinPixelWidth) {
                continue;
            }
            double real_len = REAL_LENGTH_CM.count(cid) ? REAL_LENGTH_CM.at(cid) : 7.0;
            double depth_z = (fx * real_len) / pixel_width;
            double real_x = (center_u - cx) * depth_z / fx;

            int best_tracker_id = -1;
            double min_dist = 15.0;

            for (auto& kv : trackers) {
                if (matched_trackers_this_frame.count(kv.first)) {
                    continue;
                }
                if (!kv.second.initialized || kv.second.class_id != cid) {
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
            TrackedVehicle& tracker = trackers[best_tracker_id];
            const int gap_frames = tracker.missed_frames;
            tracker.missed_frames = 0;

            if (!tracker.initialized) {
                tracker.smoothed_x = real_x;
                tracker.last_calc_x = real_x;
                tracker.last_calc_time = now;
                tracker.initialized = true;
                tracker.class_id = cid;
                tracker.hit_streak = 1;
                tracker.stable_frames = 0;
                tracker.last_pixel_width = pixel_width;
                tracker.smoothed_speed = 0.0f;
                tracker.current_speed = 0.0f;
            } else {
                // Every tracker is incremented once at the beginning of the
                // current frame. A value of 1 therefore still means it was
                // matched in the immediately preceding frame. Only values
                // above 1 represent an actual detection gap.
                if (gap_frames > 1) {
                    tracker.hit_streak = 1;
                } else {
                    tracker.hit_streak = std::min(tracker.hit_streak + 1, 1000);
                }

                const double width_change = tracker.last_pixel_width > 0.0
                    ? std::abs(pixel_width - tracker.last_pixel_width) / tracker.last_pixel_width
                    : 1.0;
                if (std::abs(real_x - tracker.last_calc_x) < 1.5 && width_change < 0.08) {
                    tracker.stable_frames = std::min(tracker.stable_frames + 1, 1000);
                } else {
                    tracker.stable_frames = 0;
                }
                tracker.last_pixel_width = pixel_width;

                tracker.smoothed_x = 0.6 * real_x + 0.4 * tracker.smoothed_x;
                double dt_calc = std::chrono::duration<double>(now - tracker.last_calc_time).count();

                if (dt_calc >= kSpeedUpdateIntervalSec) {
                    double dx = tracker.smoothed_x - tracker.last_calc_x;
                    float raw_speed = std::abs(dx) / dt_calc;

                    // Always advance the measurement window. The former
                    // 150 ms / 1.5 cm gate made a low-speed or brief pass
                    // look stationary before it disappeared from the FOV.
                    tracker.last_calc_x = tracker.smoothed_x;
                    tracker.last_calc_time = now;

                    if (raw_speed < 150.0) {
                        // A short EMA rejects coordinate jitter without
                        // suppressing genuine slow toy-car motion.
                        tracker.smoothed_speed = 0.55f * raw_speed
                                               + 0.45f * tracker.smoothed_speed;

                        if (tracker.smoothed_speed < kSpeedDisplayDeadbandCmPerSec) {
                            tracker.smoothed_speed = 0.0f;
                        }

                        tracker.current_speed = tracker.smoothed_speed;
                        if (tracker.smoothed_speed > 0.0f &&
                            std::abs(dx) >= kDirectionMinDeltaCm) {
                            tracker.direction = (dx > 0) ? 1 : -1;
                        }
                    }
                }
            }

            const bool confirmed = tracker.hit_streak >= kRequiredDetectionHits;
            const bool low_confidence_static_car =
                cid == 0 && score < 0.60f &&
                tracker.stable_frames >= kStaticClutterFrames &&
                tracker.current_speed < 1.0f;
            if (!confirmed || low_confidence_static_car) {
                continue;
            }
            boxes_draw.push_back({x1_orig, y1_orig, x2_orig, y2_orig});
            scores_draw.push_back(score);
            ids_draw.push_back(cid);
            speeds_draw.push_back(tracker.current_speed);
            directions_draw.push_back(tracker.direction);
        }

        if (!motion_reference_ready) {
            motion_reference = motion_current;
            motion_reference_ready = true;
            motion_reference_age = 0;
        } else if (++motion_reference_age >= kMotionReferenceFrames) {
            motion_reference = motion_current;
            motion_reference_age = 0;
        }

        visualizer.DrawSpeed(boxes_draw, scores_draw, ids_draw, speeds_draw,
                             directions_draw, 560, 720);

        const auto frame_done = std::chrono::steady_clock::now();
        frame_times.push_back(frame_done);
        if (frame_times.size() > 10) frame_times.pop_front();
        ++processed_frames;

        bool has_measured_speed = false;
        for (float speed : speeds_draw) {
            if (speed >= kSpeedDisplayDeadbandCmPerSec) {
                has_measured_speed = true;
                break;
            }
        }

        const auto report_now = frame_done;
        const int speed_log_interval_ms = has_measured_speed
            ? kMovingSpeedLogIntervalMs
            : kIdleSpeedLogIntervalMs;
        const bool report_speed = !boxes_draw.empty() &&
            std::chrono::duration_cast<std::chrono::milliseconds>(
                report_now - last_speed_report).count() >= speed_log_interval_ms;
        if (report_speed) {
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
            last_speed_report = report_now;
        }

        bool report_fps = false;
        {
            std::lock_guard<std::mutex> lock(g_mtx);
            if (g_fps_report_requested) {
                g_fps_report_requested = false;
                report_fps = true;
            }
        }
        if (report_fps) {
            const double elapsed = std::chrono::duration<double>(
                frame_done - stats_start).count();
            const double average_fps = elapsed > 0.0
                ? static_cast<double>(processed_frames) / elapsed
                : 0.0;

            printf("[SPEED_FPS] last10=%.1f avg=%.1f frames=%zu\n",
                   calc_last10_fps(), average_fps, processed_frames);
        }
    }

    }

    if (listener_thread.joinable()) listener_thread.join();

    const auto stats_end = std::chrono::steady_clock::now();
    (void)stats_end;
    printf("[SPEED_STATS] frames=%zu\n", processed_frames);

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
