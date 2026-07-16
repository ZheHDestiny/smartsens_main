/*
 * @Filename: focus_tracking_main.cpp
 * @Description: Monocular focus tracking demo for grayscale SmartSens camera.
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>

#include "common.hpp"
#include "utils.hpp"

using namespace std;

static bool g_focus_exit = false;
static bool g_focus_pause = false;
static bool g_focus_reset = false;
static bool g_focus_enroll = false;
static bool g_focus_clear_id = false;
static std::mutex g_focus_mtx;

class FocusTimingWindow {
public:
    void Add(float value_ms) {
        if (!std::isfinite(value_ms) || value_ms < 0.0f) return;
        values_[next_] = value_ms;
        next_ = (next_ + 1) % values_.size();
        if (count_ < values_.size()) ++count_;
    }

    float Mean() const {
        if (count_ == 0) return 0.0f;
        float sum = 0.0f;
        for (size_t i = 0; i < count_; ++i) sum += values_[i];
        return sum / static_cast<float>(count_);
    }

    float Percentile(float percentile) const {
        if (count_ == 0) return 0.0f;
        std::array<float, 120> sorted = values_;
        std::sort(sorted.begin(), sorted.begin() + count_);
        const float clamped = std::max(0.0f, std::min(1.0f, percentile));
        const size_t index = static_cast<size_t>(
            std::ceil(clamped * static_cast<float>(count_)) - 1.0f);
        return sorted[std::min(index, count_ - 1)];
    }

private:
    std::array<float, 120> values_ = {};
    size_t next_ = 0;
    size_t count_ = 0;
};

struct ReIdTrackState {
    uint64_t id = 0;
    float cx = 0.0f;
    float cy = 0.0f;
    float eye_distance = 0.0f;
    float similarity = 0.0f;
    int stable_hits = 0;
    uint32_t last_seen_frame = 0;
    uint32_t last_eval_frame = 0;
};

static ReIdTrackState* find_reid_track(vector<ReIdTrackState>* tracks, uint64_t id) {
    if (tracks == nullptr) return nullptr;
    for (size_t i = 0; i < tracks->size(); ++i) {
        if ((*tracks)[i].id == id) return &(*tracks)[i];
    }
    return nullptr;
}

static void assign_reid_tracks(vector<EyePair>* pairs,
                               vector<ReIdTrackState>* tracks,
                               uint64_t* next_track_id,
                               uint32_t frame_count) {
    if (pairs == nullptr || tracks == nullptr || next_track_id == nullptr) return;
    vector<bool> used(tracks->size(), false);
    for (size_t p = 0; p < pairs->size(); ++p) {
        EyePair& pair = (*pairs)[p];
        int best = -1;
        float best_distance = 1e9f;
        for (size_t t = 0; t < tracks->size(); ++t) {
            if (used[t] || frame_count - (*tracks)[t].last_seen_frame > 20) continue;
            const float dx = pair.Cx() - (*tracks)[t].cx;
            const float dy = pair.Cy() - (*tracks)[t].cy;
            const float distance = std::sqrt(dx * dx + dy * dy);
            const float limit = std::max(80.0f, 1.6f * (*tracks)[t].eye_distance);
            if (distance < limit && distance < best_distance) {
                best = static_cast<int>(t);
                best_distance = distance;
            }
        }
        if (best < 0) {
            ReIdTrackState state;
            state.id = (*next_track_id)++;
            state.cx = pair.Cx();
            state.cy = pair.Cy();
            state.eye_distance = pair.EyeDistance();
            state.last_seen_frame = frame_count;
            tracks->push_back(state);
            used.push_back(true);
            pair.track_id = state.id;
        } else {
            ReIdTrackState& state = (*tracks)[best];
            state.cx = pair.Cx();
            state.cy = pair.Cy();
            state.eye_distance = pair.EyeDistance();
            state.last_seen_frame = frame_count;
            used[best] = true;
            pair.track_id = state.id;
        }
    }
    tracks->erase(std::remove_if(tracks->begin(), tracks->end(),
        [frame_count](const ReIdTrackState& state) {
            return frame_count - state.last_seen_frame > 45;
        }), tracks->end());
}

static int find_pair_by_track_id(const vector<EyePair>& pairs, uint64_t track_id) {
    for (size_t i = 0; i < pairs.size(); ++i) {
        if (pairs[i].track_id == track_id) return static_cast<int>(i);
    }
    return -1;
}

static const char* focus_mode_name(FocusTrackingMode mode) {
    switch (mode) {
        case FocusTrackingMode::NO_NPU_TRACKER:
            return "传统视觉追焦";
        case FocusTrackingMode::NPU_MOBILENET:
            return "EyeDet-S + FaceID-S 智能追焦";
        default:
            return "未知追焦模式";
    }
}

static void print_focus_mode_menu() {
    cout << "\n======================================================\n";
    cout << "          追焦功能子菜单 / Focus Tracking            \n";
    cout << "======================================================\n";
    cout << "  1. 传统视觉追焦 (No NPU - 高帧率模板追踪)\n";
    cout << "  2. EyeDet-S + FaceID-S 智能追焦 (双眼检测 + 临时ID)\n";
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

static void focus_keyboard_listener() {
    string input;
    printf("[追焦键盘] ┌─────────────────────────────────┐\n");
    printf("[追焦键盘] │ P/p: 暂停/继续                  │\n");
    printf("[追焦键盘] │ R/r: 重置锁定目标                │\n");
    printf("[追焦键盘] │ E/e: 录入当前目标为 id_tmp       │\n");
    printf("[追焦键盘] │ C/c: 清除临时身份                 │\n");
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
            if (input == "e" || input == "E") {
                g_focus_enroll = true;
            }
            if (input == "c" || input == "C") {
                g_focus_clear_id = true;
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

static bool focus_take_enroll_request() {
    lock_guard<mutex> lock(g_focus_mtx);
    bool request = g_focus_enroll;
    g_focus_enroll = false;
    return request;
}

static bool focus_take_clear_id_request() {
    lock_guard<mutex> lock(g_focus_mtx);
    bool request = g_focus_clear_id;
    g_focus_clear_id = false;
    return request;
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

static int run_focus_tracking_mode(FocusTrackingMode selected_mode) {
    {
        lock_guard<mutex> lock(g_focus_mtx);
        g_focus_exit = false;
        g_focus_pause = false;
        g_focus_reset = false;
        g_focus_enroll = false;
        g_focus_clear_id = false;
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
    tracker_config.position_smooth_alpha = 0.65f;
    tracker_config.template_update_threshold = 0.68f;
    tracker_config.min_focus_score_for_update = 3.0f;
    tracker_config.mode = selected_mode;
    tracker_config.npu_model_path = "./app_assets/models/eyedet_s.m1model";
    tracker_config.npu_interval_frames = 1;

    FocusTracker tracker;
    tracker.Initialize(tracker_config);

    EyeDetFaceIdEngine smart_engine;
    if (tracker_config.mode == FocusTrackingMode::NPU_MOBILENET) {
        if (!ssne_ok || !smart_engine.Initialize(
                "./app_assets/models/eyedet_s.m1model",
                "./app_assets/models/faceid_s.m1model",
                capture_w,
                capture_h)) {
            fprintf(stderr,
                    "[ERROR] EyeDet-S 初始化失败；模式2无法满足模型契约，返回子菜单。\n");
            smart_engine.Release();
            image_processor.Release();
            if (ssne_ok) {
                SigintBlocker blocker;
                ssne_release();
            }
            return -1;
        }
        if (!smart_engine.FaceIdReady()) {
            fprintf(stderr, "[WARN] FaceID-S 不可用；保留 EyeDet 追焦，临时ID已降级。\n");
        }
    }

    VISUALIZER visualizer;
    // Other main demos initialize the visualizer directly.  Do the same here:
    // device-name probing is not reliable across board images, while
    // VISUALIZER itself reports a precise initialization failure if OSD is off.
    visualizer.Initialize(img_shape, "shared_colorLUT.sscl");
    const std::array<float, 4> focus_fov = {
        static_cast<float>(crop_x1), static_cast<float>(crop_y1),
        static_cast<float>(crop_x2 - 1), static_cast<float>(crop_y2 - 1)};
    if (tracker_config.mode == FocusTrackingMode::NPU_MOBILENET) {
        visualizer.DrawFocusFov(focus_fov);
        visualizer.DrawFocusIdentity(false, crop_x2 - 1, crop_y1);
        visualizer.DrawFocusConfidence(0.0f, 0.0f);
        visualizer.DrawFocusEnrollmentFlash(focus_fov, false);
    }
    printf("[FOCUS_OSD] requested capture FOV sensor=(%d,%d)-(%d,%d)\n",
           crop_x1, crop_y1, crop_x2 - 1, crop_y2 - 1);

    printf("[配置] sensor=%dx%d crop=(%d,%d)-(%d,%d) capture=%dx%d proc=%dx%d\n",
           sensor_w, sensor_h, crop_x1, crop_y1, crop_x2, crop_y2, capture_w, capture_h, proc_w, proc_h);
    printf("[模式] %s\n", focus_mode_name(tracker_config.mode));
    printf("[处理] P/p暂停 | R/r重置 | E/e录入id_tmp | C/c清ID | Q/q返回\n\n");

    thread listener_thread(focus_keyboard_listener);

    ssne_tensor_t curr_frame;
    memset(&curr_frame, 0, sizeof(ssne_tensor_t));
    vector<uint8_t> proc_frame((size_t)proc_w * (size_t)proc_h);
    FocusTargetState target;
    EyeDetResult eye_result;
    IdentityResult identity_result;
    uint64_t identity_track_id = 0;
    uint64_t tracker_seed_track_id = 0;
    vector<ReIdTrackState> reid_tracks;
    uint64_t next_reid_track_id = 1;
    uint64_t reid_locked_track_id = 0;
    uint32_t reid_locked_missing_frames = 0;
    size_t reid_round_robin = 0;
    float reid_display_similarity = 0.0f;
    auto last_identity_time = chrono::steady_clock::now() - chrono::seconds(10);
    uint64_t faceid_runs = 0;

    uint32_t frame_count = 0;
    uint32_t locked_count = 0;
    uint32_t last_recover_successes = 0;
    bool last_osd_locked = false;
    bool last_osd_identity_matched = false;
    bool last_enrollment_flash_visible = false;
    int enrollment_complete_flash_frames = 0;
    bool has_last_result_time = false;
    auto start_time = chrono::steady_clock::now();
    auto last_result_time = start_time;
    auto last_summary_time = start_time;

    FocusTimingWindow frame_period_ms;
    FocusTimingWindow result_latency_ms;
    FocusTimingWindow capture_ms;
    FocusTimingWindow preprocess_ms;
    FocusTimingWindow npu_ms;
    FocusTimingWindow postprocess_ms;
    FocusTimingWindow tracker_ms;
    FocusTimingWindow faceid_ms;
    FocusTimingWindow osd_ms;

    const uint32_t tracker_refresh_interval = 5;
    const uint32_t osd_refresh_interval = 3;

    {
        SigintBlocker blocker;
        while (!focus_should_exit()) {
            bool enrollment_sample_accepted = false;
            if (focus_is_paused()) {
                has_last_result_time = false;
                usleep(50000);
                continue;
            }

            const auto capture_begin = chrono::steady_clock::now();
            image_processor.GetImage(&curr_frame);
            const auto capture_end = chrono::steady_clock::now();
            ImagePipelineHealthSnapshot health = image_processor.GetHealthSnapshot();
            if (health.recover_successes != last_recover_successes) {
                last_recover_successes = health.recover_successes;
                tracker.Reset();
                target.Clear();
                smart_engine.ResetSession();
                identity_result.Clear();
                identity_track_id = 0;
                tracker_seed_track_id = 0;
                reid_tracks.clear();
                reid_locked_track_id = 0;
                reid_locked_missing_frames = 0;
                reid_display_similarity = 0.0f;
                last_osd_identity_matched = false;
                if (last_osd_locked) {
                    EyeDetResult empty_result;
                    visualizer.DrawFocusEyes(empty_result, false, crop_x1, crop_y1);
                    last_osd_locked = false;
                }
                if (tracker_config.mode == FocusTrackingMode::NPU_MOBILENET) {
                    visualizer.DrawFocusIdentity(false, crop_x2 - 1, crop_y1);
                }
                if (RuntimeLogEnabled()) {
                    printf("[追焦] camera pipeline 已恢复，清空旧目标并重新选择。\n");
                }
            }

            uint8_t* capture_ptr = nullptr;
            if (curr_frame.data != nullptr) {
                capture_ptr = (uint8_t*)get_data(curr_frame);
            }
            if (capture_ptr == nullptr) {
                usleep(5000);
                continue;
            }
            capture_ms.Add(
                chrono::duration<float, milli>(capture_end - capture_begin).count());
            const auto processing_begin = capture_end;
            bool proc_frame_ready = false;
            auto ensure_proc_frame = [&]() -> uint8_t* {
                if (!proc_frame_ready) {
                    resize_gray_nearest(capture_ptr, capture_w, capture_h,
                                        proc_frame.data(), proc_w, proc_h);
                    proc_frame_ready = true;
                }
                return proc_frame.data();
            };

            if (focus_take_reset_request()) {
                tracker.Reset();
                target.Clear();
                smart_engine.ResetSession();
                identity_result.Clear();
                identity_track_id = 0;
                tracker_seed_track_id = 0;
                reid_tracks.clear();
                reid_locked_track_id = 0;
                reid_locked_missing_frames = 0;
                reid_display_similarity = 0.0f;
                last_osd_identity_matched = false;
                if (tracker_config.mode == FocusTrackingMode::NPU_MOBILENET) {
                    visualizer.DrawFocusIdentity(false, crop_x2 - 1, crop_y1);
                }
                printf("[追焦] 已重置锁定目标，下一帧重新选择。\n");
            }

            if (focus_take_clear_id_request()) {
                smart_engine.ClearEnrollment();
                identity_result.Clear();
                identity_track_id = 0;
                last_osd_identity_matched = false;
                reid_tracks.clear();
                reid_locked_track_id = 0;
                reid_locked_missing_frames = 0;
                reid_display_similarity = 0.0f;
                if (tracker_config.mode == FocusTrackingMode::NPU_MOBILENET) {
                    visualizer.DrawFocusIdentity(false, crop_x2 - 1, crop_y1);
                }
                printf("[FACEID] 已清除 id_tmp、录入样本和身份显示。\n");
            }

            frame_count++;

            bool locked = false;
            if (tracker_config.mode == FocusTrackingMode::NPU_MOBILENET) {
                const bool det_ok = smart_engine.DetectEyes(&curr_frame, &eye_result);
                if (det_ok) {
                    preprocess_ms.Add(eye_result.preprocess_ms);
                    npu_ms.Add(eye_result.npu_ms);
                    postprocess_ms.Add(eye_result.postprocess_ms);
                }
                const bool reid_mode = smart_engine.HasEnrollment() && !smart_engine.IsEnrolling();
                if (det_ok && reid_mode) {
                    assign_reid_tracks(&eye_result.pairs, &reid_tracks,
                                       &next_reid_track_id, frame_count);
                    if (!eye_result.pairs.empty()) {
                        const size_t reid_index = reid_round_robin % eye_result.pairs.size();
                        ++reid_round_robin;
                        EyePair& reid_pair = eye_result.pairs[reid_index];
                        IdentityResult reid_result;
                        const auto face_begin = chrono::steady_clock::now();
                        if (smart_engine.Identify(capture_ptr, capture_w, capture_h,
                                                 reid_pair, &reid_result)) {
                            ReIdTrackState* state = find_reid_track(&reid_tracks,
                                                                      reid_pair.track_id);
                            if (state != nullptr) {
                                state->similarity = state->last_eval_frame == 0
                                    ? reid_result.similarity
                                    : 0.65f * state->similarity + 0.35f * reid_result.similarity;
                                state->last_eval_frame = frame_count;
                                if (state->similarity >= 0.75f) {
                                    ++state->stable_hits;
                                } else {
                                    state->stable_hits = 0;
                                }
                                if (state->stable_hits >= 2) {
                                    if (reid_locked_track_id != state->id) {
                                        printf("[REID] auto-lock track=%llu sim=%.3f hits=%d\n",
                                               static_cast<unsigned long long>(state->id),
                                               state->similarity, state->stable_hits);
                                    }
                                    reid_locked_track_id = state->id;
                                    reid_locked_missing_frames = 0;
                                    reid_display_similarity = state->similarity;
                                    identity_result = reid_result;
                                    identity_result.similarity = state->similarity;
                                    identity_track_id = state->id;
                                    last_identity_time = chrono::steady_clock::now();
                                } else if (reid_locked_track_id == state->id) {
                                    reid_display_similarity = state->similarity;
                                }
                            }
                            ++faceid_runs;
                        }
                        const auto face_end = chrono::steady_clock::now();
                        faceid_ms.Add(chrono::duration<float, milli>(
                            face_end - face_begin).count());
                    }

                    if (reid_locked_track_id != 0) {
                        const int locked_index = find_pair_by_track_id(
                            eye_result.pairs, reid_locked_track_id);
                        if (locked_index >= 0) {
                            eye_result.selected_index = locked_index;
                            reid_locked_missing_frames = 0;
                        } else if (++reid_locked_missing_frames <= 15) {
                            // Preserve the existing tracker target through a short occlusion.
                            eye_result.selected_index = -1;
                        } else {
                            printf("[REID] lock expired track=%llu\n",
                                   static_cast<unsigned long long>(reid_locked_track_id));
                            reid_locked_track_id = 0;
                            reid_display_similarity = 0.0f;
                        }
                    }
                }
                if (det_ok && eye_result.selected_index >= 0) {
                    EyePair& pair = eye_result.pairs[eye_result.selected_index];
                    const float eye_distance = pair.EyeDistance();
                    FocusTargetState detected_target;
                    detected_target.w = tracker_config.target_w;
                    detected_target.h = tracker_config.target_h;
                    detected_target.cx = pair.Cx() * proc_w / capture_w;
                    detected_target.cy = (pair.Cy() + 0.65f * eye_distance) * proc_h / capture_h;
                    detected_target.x = static_cast<int>(std::round(
                        detected_target.cx - 0.5f * tracker_config.target_w));
                    detected_target.y = static_cast<int>(std::round(
                        detected_target.cy - 0.5f * tracker_config.target_h));
                    detected_target.x = std::max(
                        0, std::min(detected_target.x, proc_w - tracker_config.target_w));
                    detected_target.y = std::max(
                        0, std::min(detected_target.y, proc_h - tracker_config.target_h));
                    detected_target.cx =
                        detected_target.x + 0.5f * tracker_config.target_w;
                    detected_target.cy =
                        detected_target.y + 0.5f * tracker_config.target_h;
                    detected_target.confidence = 0.5f * (pair.left.score + pair.right.score);
                    detected_target.locked = true;
                    detected_target.age = target.age + 1;
                    detected_target.lost_frames = 0;
                    target = detected_target;
                    locked = true;

                    if (!tracker.HasTarget() ||
                        tracker_seed_track_id != pair.track_id ||
                        frame_count % tracker_refresh_interval == 0) {
                        const auto tracker_begin = chrono::steady_clock::now();
                        tracker.SetTarget(ensure_proc_frame(), detected_target);
                        tracker_seed_track_id = pair.track_id;
                        const auto tracker_end = chrono::steady_clock::now();
                        tracker_ms.Add(chrono::duration<float, milli>(
                            tracker_end - tracker_begin).count());
                    }

                    if (!reid_mode && identity_track_id != 0 && identity_track_id != pair.track_id) {
                        identity_result.Clear();
                        identity_track_id = 0;
                    }
                    if (!reid_mode &&
                        (frame_count == 1 || frame_count % 6 == 0 || smart_engine.IsEnrolling())) {
                        IdentityResult next_identity;
                        const int samples_before = smart_engine.EnrollmentCount();
                        const auto face_begin = chrono::steady_clock::now();
                        if (smart_engine.Identify(capture_ptr, capture_w, capture_h,
                                                 pair, &next_identity)) {
                            enrollment_sample_accepted =
                                next_identity.enrolled_count > samples_before;
                            if (enrollment_sample_accepted && next_identity.enrolled) {
                                enrollment_complete_flash_frames = 4;
                            }
                            identity_result = next_identity;
                            identity_track_id = pair.track_id;
                            last_identity_time = chrono::steady_clock::now();
                            ++faceid_runs;
                        }
                        const auto face_end = chrono::steady_clock::now();
                        faceid_ms.Add(chrono::duration<float, milli>(
                            face_end - face_begin).count());
                    }
                } else if (det_ok && eye_result.lost_frames > 15) {
                    tracker.Reset();
                    tracker_seed_track_id = 0;
                    target.Clear();
                } else if (tracker.HasTarget()) {
                    const auto tracker_begin = chrono::steady_clock::now();
                    locked = tracker.Update(ensure_proc_frame(), &target);
                    const auto tracker_end = chrono::steady_clock::now();
                    tracker_ms.Add(chrono::duration<float, milli>(
                        tracker_end - tracker_begin).count());
                }
                if (focus_take_enroll_request()) {
                    if (!smart_engine.BeginEnroll()) {
                        printf("[FACEID] 无稳定双眼目标或 FaceID 不可用，无法开始录入。\n");
                    } else {
                        identity_result.Clear();
                        identity_track_id = smart_engine.SelectedTrackId();
                    }
                }
                if (!reid_mode) {
                    const auto identity_age = chrono::duration_cast<chrono::milliseconds>(
                        chrono::steady_clock::now() - last_identity_time).count();
                    if (identity_age > 800) {
                        identity_result.Clear();
                        identity_result.expired = true;
                    }
                }
            } else {
                focus_take_enroll_request();
                const auto tracker_begin = chrono::steady_clock::now();
                locked = tracker.Update(ensure_proc_frame(), &target);
                const auto tracker_end = chrono::steady_clock::now();
                tracker_ms.Add(chrono::duration<float, milli>(
                    tracker_end - tracker_begin).count());
            }
            if (locked) locked_count++;

            const auto osd_begin = chrono::steady_clock::now();
            if (tracker_config.mode == FocusTrackingMode::NPU_MOBILENET) {
                const bool enrollment_flash_visible = enrollment_sample_accepted ||
                    enrollment_complete_flash_frames > 0;
                if (enrollment_flash_visible != last_enrollment_flash_visible) {
                    visualizer.DrawFocusEnrollmentFlash(focus_fov, enrollment_flash_visible);
                    last_enrollment_flash_visible = enrollment_flash_visible;
                }
                if (enrollment_complete_flash_frames > 0) {
                    --enrollment_complete_flash_frames;
                }
                const bool reid_mode = smart_engine.HasEnrollment() && !smart_engine.IsEnrolling();
                const bool identity_matched = reid_mode
                    ? reid_locked_track_id != 0
                    : (identity_result.valid && !identity_result.expired);
                if (identity_matched != last_osd_identity_matched) {
                    visualizer.DrawFocusIdentity(identity_matched,
                                                 crop_x2 - 1, crop_y1);
                    last_osd_identity_matched = identity_matched;
                }
                if (frame_count == 1 || frame_count % osd_refresh_interval == 0) {
                    visualizer.DrawFocusEyes(eye_result, identity_matched,
                                             crop_x1, crop_y1);
                    float eye_confidence = 0.0f;
                    if (eye_result.selected_index >= 0 &&
                        eye_result.selected_index < static_cast<int>(eye_result.pairs.size())) {
                        const EyePair& selected = eye_result.pairs[eye_result.selected_index];
                        eye_confidence = 0.5f * (selected.left.score + selected.right.score);
                    }
                    const float identity_confidence = reid_mode
                        ? reid_display_similarity
                        : (identity_result.enrolled && !identity_result.expired
                            ? identity_result.similarity : 0.0f);
                    visualizer.DrawFocusConfidence(eye_confidence, identity_confidence);
                    last_osd_locked = !eye_result.eyes.empty();
                }
            } else if (tracker_config.mode == FocusTrackingMode::NO_NPU_TRACKER &&
                       locked) {
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
            } else if (!locked && last_osd_locked) {
                vector<array<float, 4>> empty_boxes;
                vector<float> empty_scores;
                vector<int> empty_ids;
                visualizer.Draw(empty_boxes, empty_scores, empty_ids);
                last_osd_locked = false;
            }
            const auto osd_end = chrono::steady_clock::now();
            osd_ms.Add(chrono::duration<float, milli>(osd_end - osd_begin).count());

            const auto result_time = chrono::steady_clock::now();
            result_latency_ms.Add(chrono::duration<float, milli>(
                result_time - processing_begin).count());
            if (has_last_result_time) {
                frame_period_ms.Add(chrono::duration<float, milli>(
                    result_time - last_result_time).count());
            }
            last_result_time = result_time;
            has_last_result_time = true;

            if (RuntimeLogEnabled() &&
                chrono::duration_cast<chrono::milliseconds>(
                    result_time - last_summary_time).count() >= 1000) {
                const float mean_period = frame_period_ms.Mean();
                const float fps = mean_period > 0.001f ? 1000.0f / mean_period : 0.0f;
                const char* id_label =
                    (identity_result.valid && !identity_result.expired) ? "id_tmp" : "unknown";
                printf("[FOCUS] fps=%.1f p95=%.1fms eyes=%zu pairs=%zu score=%.3f id=%s face_runs=%llu lost=%d\n",
                       fps, result_latency_ms.Percentile(0.95f),
                       eye_result.eyes.size(), eye_result.pairs.size(),
                       eye_result.max_class_score, id_label,
                       static_cast<unsigned long long>(faceid_runs),
                       eye_result.lost_frames);
                if (RuntimeLogAtLeast(RuntimeLogMode::VERIFY)) {
                    printf("[FOCUS_PROF] capture=%.2f pre=%.2f npu=%.2f post=%.2f tracker=%.2f face=%.2f osd=%.2f period_p95=%.2f\n",
                           capture_ms.Mean(), preprocess_ms.Mean(), npu_ms.Mean(),
                           postprocess_ms.Mean(), tracker_ms.Mean(), faceid_ms.Mean(),
                           osd_ms.Mean(), frame_period_ms.Percentile(0.95f));
                }
                last_summary_time = result_time;
            }
        }
    }

    {
        lock_guard<mutex> lock(g_focus_mtx);
        g_focus_exit = true;
    }
    if (listener_thread.joinable()) listener_thread.join();

    const float mean_period = frame_period_ms.Mean();
    const float avg_fps = mean_period > 0.001f ? 1000.0f / mean_period : 0.0f;

    printf("\n[统计] 追焦结束\n");
    printf("  • 总帧数: %u\n", frame_count);
    printf("  • 平均帧率: %4.1f FPS\n", avg_fps);
    printf("  • 锁定帧占比: %.1f%%\n", frame_count > 0 ? 100.0f * locked_count / frame_count : 0.0f);
    ImagePipelineHealthSnapshot final_health = image_processor.GetHealthSnapshot();
    printf("  • 健康状态: %s\n", IMAGEPROCESSOR::HealthStateName(final_health.state));
    printf("  • 无效帧: total=%u max_streak=%u recover=%u/%u fail=%u\n",
           final_health.invalid_frame_total,
           final_health.max_invalid_frame_streak,
           final_health.recover_successes,
           final_health.recover_attempts,
           final_health.recover_failures);

    tracker.Reset();
    smart_engine.Release();
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
