#pragma once

#include <cstdint>
#include <vector>

enum class MotionGuardScene : int {
    HOME = 0,
    ROADSIDE = 1
};

enum class MotionGuardState : int {
    CALIBRATING = 0,
    CLEAR,
    MOTION,
    ZONE_OCCUPIED,
    LOITERING,
    PASSING,
    LINE_CROSSING,
    WRONG_WAY,
    APPROACHING
};

// Camera health and target activity are intentionally separate. A fixed-camera
// application must never describe a moved camera as an object event.
enum class MotionGuardSystemState : int {
    CALIBRATING = 0,
    ARMED,
    CAMERA_UNSTABLE,
    RECALIBRATING
};

enum MotionGuardEvent : uint32_t {
    MOTION_EVENT_NONE       = 0,
    MOTION_EVENT_INTRUSION  = 1u << 0,
    MOTION_EVENT_LOITERING  = 1u << 1,
    MOTION_EVENT_LINE_CROSS = 1u << 2,
    MOTION_EVENT_WRONG_WAY  = 1u << 3,
    MOTION_EVENT_APPROACH   = 1u << 4
};

struct MotionGuardTrack {
    uint64_t id = 0;
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float vx = 0.0f;
    float vy = 0.0f;
    float risk = 0.0f;
    float confidence = 0.0f;
    float area_growth = 0.0f;
    float ttc_frames = 0.0f;
    uint32_t events = MOTION_EVENT_NONE;
    int age = 0;
    int missed = 0;
    int dwell_frames = 0;
};

struct MotionGuardResult {
    std::vector<MotionGuardTrack> tracks;
    int selected_index = -1;
    int active_targets = 0;
    bool background_ready = false;
    MotionGuardScene scene = MotionGuardScene::HOME;
    MotionGuardSystemState system_state = MotionGuardSystemState::CALIBRATING;
    MotionGuardState state = MotionGuardState::CALIBRATING;
    float foreground_ratio = 0.0f;
    float zone_x1 = 0.0f;
    float zone_y1 = 0.0f;
    float zone_x2 = 0.0f;
    float zone_y2 = 0.0f;
    float line_y = -1.0f;

    void ClearFrame() {
        tracks.clear();
        selected_index = -1;
        active_targets = 0;
        foreground_ratio = 0.0f;
    }
};

class MotionGuard {
public:
    MotionGuard();

    void Initialize(int width, int height, MotionGuardScene scene, int nominal_fps);
    void Reset();
    bool Process(const uint8_t* gray, uint32_t frame_id, MotionGuardResult* result);
    MotionGuardScene Scene() const { return scene_; }

private:
    struct Config {
        int detection_interval = 2;
        int base_threshold = 16;
        int min_blob_area = 20;
        int max_tracks = 8;
        int max_missed = 6;
        int background_warmup = 30;
        int camera_stable_frames = 6;
        int camera_arm_cooldown_frames = 12;
        int background_shift = 6;
        int loiter_frames = 210;
        float min_association_distance = 20.0f;
        float association_scale = 1.8f;
        float approach_threshold = 0.025f;
        float zone_x1 = 0.15f;
        float zone_y1 = 0.10f;
        float zone_x2 = 0.85f;
        float zone_y2 = 0.95f;
        float line_y = -1.0f;
        float legal_direction_y = 0.0f;
    };

    struct Blob {
        int x1 = 0;
        int y1 = 0;
        int x2 = 0;
        int y2 = 0;
        int area = 0;
        float cx = 0.0f;
        float cy = 0.0f;
    };

    struct TrackState {
        MotionGuardTrack output;
        float last_area = 0.0f;
        float last_line_side = 0.0f;
        int line_event_hold = 0;
        int approach_hits = 0;
        int wrong_way_hits = 0;
        int motion_hits = 0;
        bool motion_confirmed = false;
    };

    int width_ = 0;
    int height_ = 0;
    int low_w_ = 0;
    int low_h_ = 0;
    int nominal_fps_ = 70;
    MotionGuardScene scene_ = MotionGuardScene::HOME;
    Config cfg_;
    uint64_t next_track_id_ = 1;
    int warmup_frames_ = 0;
    MotionGuardState stable_state_ = MotionGuardState::CALIBRATING;
    MotionGuardState pending_state_ = MotionGuardState::CALIBRATING;
    int pending_state_frames_ = 0;
    MotionGuardSystemState system_state_ = MotionGuardSystemState::CALIBRATING;
    int camera_stable_frames_ = 0;
    int camera_arm_cooldown_frames_ = 0;
    int disturbance_hits_ = 0;
    int foreground_grid_cells_ = 0;
    int frame_delta_grid_cells_ = 0;
    float frame_delta_ratio_ = 0.0f;
    bool have_previous_frame_ = false;
    bool initialized_ = false;

    std::vector<uint8_t> low_gray_;
    std::vector<uint8_t> temporal_gray_;
    std::vector<uint8_t> previous_frame_gray_;
    std::vector<uint16_t> background_q8_;
    std::vector<uint8_t> noise_;
    std::vector<uint8_t> raw_mask_;
    std::vector<uint8_t> filtered_mask_;
    std::vector<uint8_t> previous_mask_;
    std::vector<uint8_t> visited_;
    std::vector<int> flood_queue_;
    std::vector<Blob> blobs_;
    std::vector<TrackState> tracks_;

    void ConfigurePreset();
    void Downsample2x(const uint8_t* gray);
    float UpdateBackgroundAndMask();
    bool CameraDisturbanceCandidate(float foreground_ratio) const;
    void StartRecalibration();
    void FilterMask();
    void ExtractBlobs();
    void PredictOnly();
    void AssociateAndUpdate(uint32_t frame_id);
    void UpdateRiskAndResult(MotionGuardResult* result);
    bool InsideZone(float cx, float cy) const;
    float BlobIoU(const MotionGuardTrack& track, const Blob& blob) const;
};
