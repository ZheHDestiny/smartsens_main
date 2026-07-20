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
#include <malloc.h>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <cerrno>
#include <sys/types.h>
#include <sys/wait.h>

#include "common.hpp"
#include "motion_guard.hpp"
#include "utils.hpp"

using namespace std;

static bool g_focus_exit = false;
static bool g_focus_pause = false;
static bool g_focus_reset = false;
static bool g_focus_enroll = false;
static bool g_focus_clear_id = false;
static std::mutex g_focus_mtx;
static const size_t kMaxReIdTrackStates = 16;

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
    // A crowded/unstable scene can produce short-lived pair associations on
    // every frame. Keep this history strictly bounded so ReID association
    // work cannot grow with run time. Prefer retaining recently seen tracks,
    // which also keeps an active lock intact.
    while (tracks->size() > kMaxReIdTrackStates) {
        size_t oldest = 0;
        for (size_t i = 1; i < tracks->size(); ++i) {
            if ((*tracks)[i].last_seen_frame < (*tracks)[oldest].last_seen_frame) {
                oldest = i;
            }
        }
        tracks->erase(tracks->begin() + oldest);
    }
}

static int find_pair_by_track_id(const vector<EyePair>& pairs, uint64_t track_id) {
    for (size_t i = 0; i < pairs.size(); ++i) {
        if (pairs[i].track_id == track_id) return static_cast<int>(i);
    }
    return -1;
}

static bool is_eye_face_tracking_mode(FocusTrackingMode mode) {
    return mode == FocusTrackingMode::NPU_MOBILENET ||
           mode == FocusTrackingMode::NPU_FLASH;
}

static const char* focus_mode_name(FocusTrackingMode mode) {
    switch (mode) {
        case FocusTrackingMode::NO_NPU_TRACKER:
            return "MotionGuard CPU 多目标风险追焦";
        case FocusTrackingMode::NPU_MOBILENET:
            return "EyeDet-S + FaceID-S 智能追焦";
        case FocusTrackingMode::NPU_FLASH:
            return "EyeDet-Flash + FaceID-S 高帧率智能追焦";
        default:
            return "未知追焦模式";
    }
}

static const char* motion_scene_name_cn(MotionGuardScene scene) {
    return scene == MotionGuardScene::ROADSIDE ? "路侧监控" : "居家守护";
}

static const char* motion_state_name_cn(MotionGuardState state) {
    switch (state) {
        case MotionGuardState::CALIBRATING:   return "背景学习";
        case MotionGuardState::CLEAR:         return "区域安全";
        case MotionGuardState::MOTION:        return "发现移动";
        case MotionGuardState::ZONE_OCCUPIED: return "守护区有人";
        case MotionGuardState::LOITERING:     return "目标长时间滞留";
        case MotionGuardState::PASSING:       return "目标正常经过";
        case MotionGuardState::LINE_CROSSING: return "目标越过警戒线";
        case MotionGuardState::WRONG_WAY:     return "目标逆向移动";
        case MotionGuardState::APPROACHING:   return "目标快速接近";
    }
    return "未知状态";
}

static const char* motion_system_state_name_cn(MotionGuardSystemState state) {
    switch (state) {
        case MotionGuardSystemState::CALIBRATING:     return "背景学习";
        case MotionGuardSystemState::ARMED:           return "已布防";
        case MotionGuardSystemState::CAMERA_UNSTABLE: return "检测到机位变化";
        case MotionGuardSystemState::RECALIBRATING:   return "正在重建背景";
    }
    return "未知系统状态";
}

static const char* motion_display_state_name_cn(const MotionGuardResult& result) {
    return result.system_state == MotionGuardSystemState::ARMED
        ? motion_state_name_cn(result.state)
        : motion_system_state_name_cn(result.system_state);
}

static bool motion_state_is_warning(MotionGuardState state) {
    return state == MotionGuardState::LOITERING ||
           state == MotionGuardState::LINE_CROSSING ||
           state == MotionGuardState::WRONG_WAY ||
           state == MotionGuardState::APPROACHING;
}

static const char* motion_direction_name_cn(const MotionGuardTrack* track) {
    if (track == nullptr) return "无";
    const float ax = std::fabs(track->vx);
    const float ay = std::fabs(track->vy);
    if (std::max(ax, ay) < 0.5f) return "基本静止";
    if (ax >= ay) return track->vx >= 0.0f ? "向右" : "向左";
    return track->vy >= 0.0f ? "向下" : "向上";
}

static void print_motion_state_transition(MotionGuardScene scene,
                                          MotionGuardState previous,
                                          const MotionGuardResult& result) {
    const MotionGuardTrack* selected =
        result.selected_index >= 0 &&
        result.selected_index < static_cast<int>(result.tracks.size())
            ? &result.tracks[result.selected_index] : nullptr;
    const char* scene_name = motion_scene_name_cn(scene);
    if (result.state == MotionGuardState::CALIBRATING) {
        printf("[%s][状态] 正在学习静态背景，请保持机位稳定。\n", scene_name);
        return;
    }
    if (result.state == MotionGuardState::CLEAR) {
        if (previous != MotionGuardState::CALIBRATING &&
            previous != MotionGuardState::CLEAR) {
            printf("[%s][恢复] 关注区域已恢复安全。\n", scene_name);
        } else {
            printf("[%s][状态] 背景学习完成，当前区域安全。\n", scene_name);
        }
        return;
    }
    printf("[%s][%s] %s",
           scene_name,
           motion_state_is_warning(result.state) ? "告警" : "状态",
           motion_state_name_cn(result.state));
    if (selected != nullptr) {
        printf("，关注目标#%llu，方向=%s，风险=%.0f%%",
               static_cast<unsigned long long>(selected->id),
               motion_direction_name_cn(selected), selected->risk);
    }
    printf("。\n");
}

static void print_motion_system_transition(MotionGuardScene scene,
                                           MotionGuardSystemState previous,
                                           MotionGuardSystemState current) {
    const char* scene_name = motion_scene_name_cn(scene);
    if (current == MotionGuardSystemState::CAMERA_UNSTABLE) {
        printf("[%s][相机] 检测到机位变化，暂停目标监测并清除旧框。\n", scene_name);
    } else if (current == MotionGuardSystemState::RECALIBRATING) {
        printf("[%s][相机] 画面已稳定，正在建立新的背景基线。\n", scene_name);
    } else if (current == MotionGuardSystemState::ARMED) {
        if (previous == MotionGuardSystemState::RECALIBRATING ||
            previous == MotionGuardSystemState::CALIBRATING) {
            printf("[%s][相机] 背景建立完成，监测已恢复：区域安全。\n", scene_name);
        } else {
            printf("[%s][相机] 监测已布防。\n", scene_name);
        }
    }
}

static void print_focus_mode_menu() {
    cout << "\n======================================================\n";
    cout << "          追焦功能子菜单 / Focus Tracking            \n";
    cout << "======================================================\n";
    cout << "  1. MotionGuard CPU场景守护 (居家守护/路侧监控)\n";
    cout << "  2. EyeDet-S + FaceID-S 智能追焦 (双眼检测 + 临时ID)\n";
    cout << "  3. EyeDet-Flash + FaceID-S 高帧率追焦 (320x480 GRAY)\n";
    cout << "  0. 返回主菜单\n";
    cout << "======================================================\n";
    cout << "请输入追焦子功能编号 (0-3) 并按回车: ";
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
        if (choice == 3) {
            *mode = FocusTrackingMode::NPU_FLASH;
            return true;
        }

        cout << "\n[提示] 无效的追焦子功能选项 (" << choice << ")，请重新选择。\n";
    }

    return false;
}

static bool choose_motion_guard_scene(MotionGuardScene* scene) {
    if (scene == nullptr) return false;
    while (!g_signal_received.load()) {
        int choice = -1;
        clear_stdin_residual();
        cout << "\n======================================================\n";
        cout << "       MotionGuard CPU 固定机位场景参数              \n";
        cout << "======================================================\n";
        cout << "  1. 居家守护：守护区进入、滞留、快速接近\n";
        cout << "  2. 路侧监控：正常经过、越线、逆行、快速接近\n";
        cout << "  0. 返回追焦子菜单\n";
        cout << "======================================================\n";
        cout << "请输入场景编号 (0-2) 并按回车: ";
        cin >> choice;
        if (cin.fail()) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "\n[错误] 输入无效，请输入数字编号！\n";
            continue;
        }
        if (choice == 0) return false;
        if (choice == 1) {
            *scene = MotionGuardScene::HOME;
            return true;
        }
        if (choice == 2) {
            *scene = MotionGuardScene::ROADSIDE;
            return true;
        }
        cout << "\n[提示] 无效场景编号。\n";
    }
    return false;
}

static void focus_keyboard_listener(FocusTrackingMode mode) {
    string input;
    printf("[追焦键盘] ┌─────────────────────────────────┐\n");
    printf("[追焦键盘] │ P/p: 暂停/继续                  │\n");
    printf("[追焦键盘] │ R/r: 重置锁定目标                │\n");
    if (is_eye_face_tracking_mode(mode)) {
        printf("[追焦键盘] │ E/e: 录入当前目标为 id_tmp       │\n");
        printf("[追焦键盘] │ C/c: 清除临时身份                 │\n");
    }
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
            if (is_eye_face_tracking_mode(mode) &&
                (input == "e" || input == "E")) {
                g_focus_enroll = true;
            }
            if (is_eye_face_tracking_mode(mode) &&
                (input == "c" || input == "C")) {
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

    MotionGuardScene motion_scene = MotionGuardScene::HOME;
    if (selected_mode == FocusTrackingMode::NO_NPU_TRACKER &&
        !choose_motion_guard_scene(&motion_scene)) {
        return 0;
    }

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
    const bool flash_mode = selected_mode == FocusTrackingMode::NPU_FLASH;
    const bool eye_face_mode = is_eye_face_tracking_mode(selected_mode);
    if (eye_face_mode) {
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
    tracker_config.npu_model_path = flash_mode
        ? "./app_assets/models/eyedet_flash.m1model"
        : "./app_assets/models/eyedet_s.m1model";
    tracker_config.npu_interval_frames = 1;

    FocusTracker tracker;
    tracker.Initialize(tracker_config);

    MotionGuard motion_guard;
    MotionGuardResult motion_result;
    uint64_t motion_focus_track_id = 0;
    int motion_focus_missing_frames = 0;
    MotionGuardState last_motion_state = MotionGuardState::CALIBRATING;
    MotionGuardSystemState last_motion_system_state =
        MotionGuardSystemState::CALIBRATING;
    if (tracker_config.mode == FocusTrackingMode::NO_NPU_TRACKER) {
        motion_guard.Initialize(proc_w, proc_h, motion_scene, 90);
        motion_result.tracks.reserve(8);
        printf("[%s] CPU-only，检测网格=180x135；UI白框仅表示当前关注目标。\n",
               motion_scene_name_cn(motion_scene));
        printf("[%s] 顶部状态条给出场景结论，区域框表示规则生效范围，箭头表示目标方向。\n",
               motion_scene_name_cn(motion_scene));
        printf("[%s][状态] 正在学习静态背景，请保持机位稳定。\n",
               motion_scene_name_cn(motion_scene));
    }

    EyeDetFaceIdEngine smart_engine;
    if (eye_face_mode) {
        EyeDetInputConfig eyedet_config;
        if (flash_mode) {
            // Model tensor is NCHW [1, 1, 320, 480]: width=480, height=320.
            eyedet_config.width = 480;
            eyedet_config.height = 320;
            eyedet_config.gray = true;
        }
        if (!ssne_ok || !smart_engine.Initialize(
                tracker_config.npu_model_path,
                "./app_assets/models/faceid_s.m1model",
                capture_w,
                capture_h,
                eyedet_config)) {
            fprintf(stderr,
                    "[ERROR] EyeDet 初始化失败；当前模式无法满足模型契约，返回子菜单。\n");
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
    // This mode only uses small RLE status sprites; do not reserve the menu's
    // 1 MiB-per-image-layer OCM budget on the constrained A1 board.
    visualizer.Initialize(img_shape, "shared_colorLUT.sscl", 0x20000);
    const std::array<float, 4> focus_fov = {
        static_cast<float>(crop_x1), static_cast<float>(crop_y1),
        static_cast<float>(crop_x2 - 1), static_cast<float>(crop_y2 - 1)};
    if (eye_face_mode) {
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
    if (eye_face_mode) {
        printf("[处理] P/p暂停 | R/r重置 | E/e录入id_tmp | C/c清ID | Q/q返回\n\n");
    } else {
        printf("[处理] P/p暂停 | R/r重置背景与轨迹 | Q/q返回\n\n");
    }

    thread listener_thread(focus_keyboard_listener, selected_mode);

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
    // ReID can briefly lose/reacquire a track in a crowded frame.  Texture
    // layer clear+upload is expensive on A1, so avoid repeatedly replacing
    // the ID sprite faster than the display can usefully present it.
    auto last_identity_osd_update = chrono::steady_clock::time_point::min();
    const chrono::milliseconds kIdentityOsdMinUpdateInterval(250);
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
    // Both NPU eye/identity modes keep OSD and FaceID-S below the detector
    // cadence. EyeDet still runs every frame; once a ReID lock exists, the
    // tracker and its last confirmed identity remain valid between samples.
    // Keep the CPU-only MotionGuard cadence unchanged from origin/main.
    const uint32_t osd_refresh_interval = eye_face_mode ? 6 : 3;
    const uint32_t reid_faceid_interval = eye_face_mode ? 2 : 1;
    const uint32_t idle_faceid_interval = flash_mode ? 8 : 6;

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
                if (tracker_config.mode == FocusTrackingMode::NO_NPU_TRACKER) {
                    motion_guard.Reset();
                    motion_focus_track_id = 0;
                    motion_focus_missing_frames = 0;
                    last_motion_state = MotionGuardState::CALIBRATING;
                    last_motion_system_state = MotionGuardSystemState::CALIBRATING;
                }
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
                if (eye_face_mode) {
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
                if (tracker_config.mode == FocusTrackingMode::NO_NPU_TRACKER) {
                    motion_guard.Reset();
                    motion_focus_track_id = 0;
                    motion_focus_missing_frames = 0;
                    last_motion_state = MotionGuardState::CALIBRATING;
                    last_motion_system_state = MotionGuardSystemState::CALIBRATING;
                }
                smart_engine.ResetSession();
                identity_result.Clear();
                identity_track_id = 0;
                tracker_seed_track_id = 0;
                reid_tracks.clear();
                reid_locked_track_id = 0;
                reid_locked_missing_frames = 0;
                reid_display_similarity = 0.0f;
                last_osd_identity_matched = false;
                if (eye_face_mode) {
                    visualizer.DrawFocusIdentity(false, crop_x2 - 1, crop_y1);
                }
                printf("[追焦] 已重置锁定目标，下一帧重新选择。\n");
            }

            if (focus_take_clear_id_request()) {
                smart_engine.ClearEnrollment();
                identity_result.Clear();
                identity_track_id = 0;
                last_osd_identity_matched = false;
                vector<ReIdTrackState>().swap(reid_tracks);
                reid_locked_track_id = 0;
                reid_locked_missing_frames = 0;
                reid_display_similarity = 0.0f;
                reid_round_robin = 0;
                next_reid_track_id = 1;
                enrollment_complete_flash_frames = 0;
                last_enrollment_flash_visible = false;
                if (eye_face_mode) {
                    visualizer.DrawFocusEnrollmentFlash(focus_fov, false);
                    visualizer.DrawFocusIdentity(false, crop_x2 - 1, crop_y1);
                    last_identity_osd_update = chrono::steady_clock::now();
                }
                printf("[FACEID] 已清除 id_tmp、录入样本和身份显示。\n");
            }

            frame_count++;

            bool locked = false;
            if (eye_face_mode) {
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
                    const bool should_run_reid = !eye_result.pairs.empty() &&
                        (reid_locked_track_id == 0 ||
                         frame_count % reid_faceid_interval == 0);
                    if (should_run_reid) {
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
                                    // Two hits are enough to confirm a lock.
                                    // Do not let this counter grow forever in
                                    // a long-running session.
                                    state->stable_hits = std::min(
                                        state->stable_hits + 1, 3);
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
                            ReIdTrackState* expired = find_reid_track(
                                &reid_tracks, reid_locked_track_id);
                            if (expired != nullptr) {
                                expired->stable_hits = 0;
                                expired->similarity = 0.0f;
                                expired->last_eval_frame = 0;
                            }
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
                        (frame_count == 1 ||
                         frame_count % idle_faceid_interval == 0 ||
                         smart_engine.IsEnrolling())) {
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
                        // A new prototype must never inherit association
                        // confidence accumulated for the previous identity.
                        vector<ReIdTrackState>().swap(reid_tracks);
                        reid_locked_track_id = 0;
                        reid_locked_missing_frames = 0;
                        reid_display_similarity = 0.0f;
                        reid_round_robin = 0;
                        next_reid_track_id = 1;
                        identity_result.Clear();
                        identity_track_id = smart_engine.SelectedTrackId();
                        last_osd_identity_matched = false;
                        visualizer.DrawFocusIdentity(false,
                                                     crop_x2 - 1, crop_y1);
                        last_identity_osd_update = chrono::steady_clock::now();
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
                uint8_t* cpu_frame = ensure_proc_frame();
                motion_guard.Process(cpu_frame, frame_count, &motion_result);
                if (motion_result.system_state != last_motion_system_state) {
                    print_motion_system_transition(motion_scene,
                                                   last_motion_system_state,
                                                   motion_result.system_state);
                    last_motion_system_state = motion_result.system_state;
                    last_motion_state = motion_result.state;
                } else if (motion_result.system_state == MotionGuardSystemState::ARMED &&
                           motion_result.state != last_motion_state) {
                    print_motion_state_transition(
                        motion_scene, last_motion_state, motion_result);
                    last_motion_state = motion_result.state;
                }
                if (motion_result.background_ready &&
                    motion_result.selected_index >= 0 &&
                    motion_result.selected_index <
                        static_cast<int>(motion_result.tracks.size())) {
                    const MotionGuardTrack& motion_target =
                        motion_result.tracks[motion_result.selected_index];
                    FocusTargetState motion_seed;
                    motion_seed.w = tracker_config.target_w;
                    motion_seed.h = tracker_config.target_h;
                    motion_seed.cx = motion_target.cx;
                    motion_seed.cy = motion_target.cy;
                    motion_seed.x = static_cast<int>(std::round(
                        motion_seed.cx - 0.5f * motion_seed.w));
                    motion_seed.y = static_cast<int>(std::round(
                        motion_seed.cy - 0.5f * motion_seed.h));
                    motion_seed.x = std::max(
                        0, std::min(motion_seed.x, proc_w - motion_seed.w));
                    motion_seed.y = std::max(
                        0, std::min(motion_seed.y, proc_h - motion_seed.h));
                    motion_seed.cx = motion_seed.x + 0.5f * motion_seed.w;
                    motion_seed.cy = motion_seed.y + 0.5f * motion_seed.h;
                    motion_seed.confidence = motion_target.risk / 100.0f;
                    motion_seed.locked = true;
                    motion_seed.age = target.age + 1;
                    if (!tracker.HasTarget() ||
                        motion_focus_track_id != motion_target.id ||
                        frame_count % 10 == 0) {
                        tracker.SetTarget(cpu_frame, motion_seed);
                        target = motion_seed;
                    } else {
                        tracker.Update(cpu_frame, &target);
                    }
                    motion_focus_track_id = motion_target.id;
                    motion_focus_missing_frames = 0;
                    locked = true;

                } else {
                    ++motion_focus_missing_frames;
                    if (tracker.HasTarget() && motion_focus_track_id != 0 &&
                        motion_focus_missing_frames <= 12) {
                        locked = tracker.Update(cpu_frame, &target);
                    }
                    if (!locked) {
                        tracker.Reset();
                        motion_focus_track_id = 0;
                        motion_focus_missing_frames = 0;
                        target.Clear();
                    }
                }
                const auto tracker_end = chrono::steady_clock::now();
                tracker_ms.Add(chrono::duration<float, milli>(
                    tracker_end - tracker_begin).count());
            }
            if (locked) locked_count++;

            const auto osd_begin = chrono::steady_clock::now();
            if (eye_face_mode) {
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
                const auto identity_osd_now = chrono::steady_clock::now();
                if (identity_matched != last_osd_identity_matched &&
                    (last_identity_osd_update == chrono::steady_clock::time_point::min() ||
                     identity_osd_now - last_identity_osd_update >=
                         kIdentityOsdMinUpdateInterval)) {
                    visualizer.DrawFocusIdentity(identity_matched,
                                                 crop_x2 - 1, crop_y1);
                    last_osd_identity_matched = identity_matched;
                    last_identity_osd_update = identity_osd_now;
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
            } else if (tracker_config.mode == FocusTrackingMode::NO_NPU_TRACKER) {
                if (frame_count == 1 || frame_count % osd_refresh_interval == 0) {
                    visualizer.DrawMotionGuard(motion_result,
                                               crop_x1, crop_y1,
                                               capture_w, capture_h,
                                               proc_w, proc_h);
                }
                last_osd_locked = !motion_result.tracks.empty();
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
                if (tracker_config.mode == FocusTrackingMode::NO_NPU_TRACKER) {
                    const MotionGuardTrack* selected =
                        motion_result.selected_index >= 0 &&
                        motion_result.selected_index <
                            static_cast<int>(motion_result.tracks.size())
                            ? &motion_result.tracks[motion_result.selected_index] : nullptr;
                    printf("[%s] 状态=%s FPS=%.1f 有效目标=%d",
                           motion_scene_name_cn(motion_scene),
                           motion_display_state_name_cn(motion_result),
                           fps, motion_result.active_targets);
                    if (selected != nullptr) {
                        printf(" 关注目标=#%llu 方向=%s 风险=%.0f%%",
                               static_cast<unsigned long long>(selected->id),
                               motion_direction_name_cn(selected),
                               selected->risk);
                    }
                    printf("\n");
                    if (RuntimeLogAtLeast(RuntimeLogMode::VERIFY)) {
                        printf("[MOTION_VERIFY] p95=%.1fms bg=%d fg=%.3f "
                               "internal_tracks=%zu vx=%.2f vy=%.2f growth=%.3f\n",
                               result_latency_ms.Percentile(0.95f),
                               motion_result.background_ready ? 1 : 0,
                               motion_result.foreground_ratio,
                               motion_result.tracks.size(),
                               selected != nullptr ? selected->vx : 0.0f,
                               selected != nullptr ? selected->vy : 0.0f,
                               selected != nullptr ? selected->area_growth : 0.0f);
                    }
                } else {
                    const char* id_label =
                        (identity_result.valid && !identity_result.expired)
                            ? "id_tmp" : "unknown";
                    printf("[FOCUS] fps=%.1f p95=%.1fms eyes=%zu pairs=%zu "
                           "score=%.3f id=%s face_runs=%llu lost=%d\n",
                           fps, result_latency_ms.Percentile(0.95f),
                           eye_result.eyes.size(), eye_result.pairs.size(),
                           eye_result.max_class_score, id_label,
                           static_cast<unsigned long long>(faceid_runs),
                           eye_result.lost_frames);
                }
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

    visualizer.Clear();
    visualizer.Release();
    tracker.Reset();
    smart_engine.Release();
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

static int run_focus_mode_isolated(FocusTrackingMode mode) {
    malloc_trim(0);
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid < 0) {
        std::perror("[FOCUS_ISOLATION] fork failed");
        return -1;
    }
    if (pid == 0) {
        g_signal_received.store(false);
        const char* feature = mode == FocusTrackingMode::NPU_MOBILENET
            ? "focus_eyedet_s" : (mode == FocusTrackingMode::NPU_FLASH
                ? "focus_eyedet_flash" : "focus_motion_guard");
        OfficialPerfReset(feature, 90.0f);
        const int rc = run_focus_tracking_mode(mode);
        OfficialPerfPrintFinal();
        std::fflush(nullptr);
        _exit(rc == 0 ? 0 : 1);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) continue;
        std::perror("[FOCUS_ISOLATION] waitpid failed");
        return -1;
    }
    if (WIFSIGNALED(status)) {
        printf("[FOCUS_ISOLATION] worker terminated by signal=%d\n", WTERMSIG(status));
        return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

int run_focus_tracking() {
    while (!g_signal_received.load()) {
        FocusTrackingMode selected_mode = FocusTrackingMode::NO_NPU_TRACKER;
        if (!choose_focus_mode(&selected_mode)) {
            return 0;
        }

        // Modes 8-1/8-2/8-3 switch inside this submenu and therefore do not
        // pass through main.cpp's module-exit reclaim point. Return the heap
        // left by MotionGuard/FaceID scratch vectors before loading the next
        // (potentially two-model) mode.
        malloc_trim(0);
        cout << "\n>> 正在启动 [" << focus_mode_name(selected_mode) << "] 子功能...\n";
        int ret = run_focus_mode_isolated(selected_mode);
        malloc_trim(0);
        if (g_signal_received.load()) {
            return ret;
        }
    }

    return 0;
}
