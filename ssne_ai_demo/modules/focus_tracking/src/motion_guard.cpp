#include "motion_guard.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

float ClampFloat(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

int StatePriority(MotionGuardState state) {
    switch (state) {
        case MotionGuardState::APPROACHING:   return 80;
        case MotionGuardState::WRONG_WAY:     return 70;
        case MotionGuardState::LOITERING:     return 60;
        case MotionGuardState::LINE_CROSSING: return 50;
        case MotionGuardState::ZONE_OCCUPIED: return 40;
        case MotionGuardState::PASSING:       return 30;
        case MotionGuardState::MOTION:        return 20;
        case MotionGuardState::CLEAR:         return 10;
        case MotionGuardState::CALIBRATING:   return 0;
    }
    return 0;
}

}  // namespace

float MotionGuard::BlobIoU(const MotionGuardTrack& track, const Blob& blob) const {
    const float x1 = std::max(track.x * 0.5f, static_cast<float>(blob.x1));
    const float y1 = std::max(track.y * 0.5f, static_cast<float>(blob.y1));
    const float x2 = std::min((track.x + track.w) * 0.5f,
                              static_cast<float>(blob.x2 + 1));
    const float y2 = std::min((track.y + track.h) * 0.5f,
                              static_cast<float>(blob.y2 + 1));
    const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float track_area = std::max(1.0f, track.w * track.h * 0.25f);
    const float blob_area = static_cast<float>(
        std::max(1, (blob.x2 - blob.x1 + 1) * (blob.y2 - blob.y1 + 1)));
    return intersection / std::max(1.0f, track_area + blob_area - intersection);
}

MotionGuard::MotionGuard() {}

void MotionGuard::Initialize(int width, int height,
                             MotionGuardScene scene, int nominal_fps) {
    width_ = width;
    height_ = height;
    low_w_ = std::max(1, width_ / 2);
    low_h_ = std::max(1, height_ / 2);
    scene_ = scene;
    nominal_fps_ = std::max(30, nominal_fps);
    ConfigurePreset();

    const size_t pixels = static_cast<size_t>(low_w_) * low_h_;
    low_gray_.assign(pixels, 0);
    temporal_gray_.assign(pixels, 0);
    previous_frame_gray_.assign(pixels, 0);
    background_q8_.assign(pixels, 0);
    noise_.assign(pixels, 8);
    raw_mask_.assign(pixels, 0);
    filtered_mask_.assign(pixels, 0);
    previous_mask_.assign(pixels, 0);
    visited_.assign(pixels, 0);
    flood_queue_.clear();
    flood_queue_.reserve(pixels);
    blobs_.clear();
    blobs_.reserve(32);
    tracks_.clear();
    tracks_.reserve(cfg_.max_tracks);
    next_track_id_ = 1;
    warmup_frames_ = 0;
    stable_state_ = MotionGuardState::CALIBRATING;
    pending_state_ = MotionGuardState::CALIBRATING;
    pending_state_frames_ = 0;
    system_state_ = MotionGuardSystemState::CALIBRATING;
    camera_stable_frames_ = 0;
    camera_arm_cooldown_frames_ = 0;
    disturbance_hits_ = 0;
    foreground_grid_cells_ = 0;
    frame_delta_grid_cells_ = 0;
    frame_delta_ratio_ = 0.0f;
    have_previous_frame_ = false;
    initialized_ = width_ > 0 && height_ > 0;
}

void MotionGuard::ConfigurePreset() {
    cfg_ = Config();
    if (scene_ == MotionGuardScene::HOME) {
        cfg_.base_threshold = 15;
        cfg_.min_blob_area = 22;
        cfg_.background_shift = 6;
        cfg_.loiter_frames = nominal_fps_ * 3;
        // The home preset describes a visible protected area, rather than
        // treating nearly the entire image as an unexplained "intrusion".
        cfg_.zone_x1 = 0.20f;
        cfg_.zone_y1 = 0.35f;
        cfg_.zone_x2 = 0.80f;
        cfg_.zone_y2 = 0.92f;
        cfg_.line_y = -1.0f;
        cfg_.legal_direction_y = 0.0f;
    } else {
        cfg_.base_threshold = 18;
        cfg_.min_blob_area = 12;
        cfg_.background_shift = 7;
        cfg_.loiter_frames = nominal_fps_ * 5;
        cfg_.zone_x1 = 0.15f;
        cfg_.zone_y1 = 0.25f;
        cfg_.zone_x2 = 0.85f;
        cfg_.zone_y2 = 0.95f;
        cfg_.line_y = 0.68f;
        cfg_.legal_direction_y = 1.0f;
    }
}

void MotionGuard::Reset() {
    std::fill(background_q8_.begin(), background_q8_.end(), 0);
    std::fill(noise_.begin(), noise_.end(), 8);
    std::fill(raw_mask_.begin(), raw_mask_.end(), 0);
    std::fill(filtered_mask_.begin(), filtered_mask_.end(), 0);
    std::fill(temporal_gray_.begin(), temporal_gray_.end(), 0);
    std::fill(previous_frame_gray_.begin(), previous_frame_gray_.end(), 0);
    std::fill(previous_mask_.begin(), previous_mask_.end(), 0);
    tracks_.clear();
    blobs_.clear();
    next_track_id_ = 1;
    warmup_frames_ = 0;
    stable_state_ = MotionGuardState::CALIBRATING;
    pending_state_ = MotionGuardState::CALIBRATING;
    pending_state_frames_ = 0;
    system_state_ = MotionGuardSystemState::CALIBRATING;
    camera_stable_frames_ = 0;
    camera_arm_cooldown_frames_ = 0;
    disturbance_hits_ = 0;
    foreground_grid_cells_ = 0;
    frame_delta_grid_cells_ = 0;
    frame_delta_ratio_ = 0.0f;
    have_previous_frame_ = false;
}

void MotionGuard::Downsample2x(const uint8_t* gray) {
    int delta_sum = 0;
    int delta_grid_sum[9] = {};
    for (int y = 0; y < low_h_; ++y) {
        const int sy = std::min(height_ - 2, y * 2);
        const uint8_t* row0 = gray + sy * width_;
        const uint8_t* row1 = row0 + width_;
        uint8_t* dst = low_gray_.data() + y * low_w_;
        for (int x = 0; x < low_w_; ++x) {
            const int sx = std::min(width_ - 2, x * 2);
            const int sum = row0[sx] + row0[sx + 1] +
                            row1[sx] + row1[sx + 1];
            const uint8_t sampled = static_cast<uint8_t>((sum + 2) >> 2);
            const size_t index = static_cast<size_t>(y) * low_w_ + x;
            // A 3:1 temporal IIR filter removes sensor grain and one-frame
            // flicker without materially delaying a person or vehicle.
            const uint8_t smooth = warmup_frames_ == 0
                ? sampled
                : static_cast<uint8_t>((3 * sampled + temporal_gray_[index] + 2) >> 2);
            temporal_gray_[index] = smooth;
            dst[x] = smooth;
            if (have_previous_frame_) {
                const int difference = std::abs(
                    static_cast<int>(smooth) - previous_frame_gray_[index]);
                delta_sum += difference;
                const int gx = std::min(2, x * 3 / std::max(1, low_w_));
                const int gy = std::min(2, y * 3 / std::max(1, low_h_));
                delta_grid_sum[gy * 3 + gx] += difference;
            }
            previous_frame_gray_[index] = smooth;
        }
    }
    const int grid_area = std::max(1, low_w_ * low_h_ / 9);
    frame_delta_grid_cells_ = 0;
    for (int i = 0; i < 9; ++i) {
        // Mean intensity is robust to high-frequency sensor grain: random
        // noise affects individual pixels, while camera motion shifts the
        // average luminance of textured blocks coherently.
        if (delta_grid_sum[i] / grid_area > 5) ++frame_delta_grid_cells_;
    }
    frame_delta_ratio_ = have_previous_frame_
        ? static_cast<float>(delta_sum) /
              std::max<size_t>(1, low_gray_.size() * 255) : 0.0f;
    have_previous_frame_ = true;
}

float MotionGuard::UpdateBackgroundAndMask() {
    const size_t pixels = low_gray_.size();
    if (warmup_frames_ == 0) {
        for (size_t i = 0; i < pixels; ++i) {
            background_q8_[i] = static_cast<uint16_t>(low_gray_[i]) << 8;
        }
    }

    size_t foreground = 0;
    int foreground_grid[9] = {};
    const bool warming = warmup_frames_ < cfg_.background_warmup;
    const int update_shift = warming ? 3 : cfg_.background_shift;
    for (size_t i = 0; i < pixels; ++i) {
        const int pixel = low_gray_[i];
        const int background = background_q8_[i] >> 8;
        const int difference = std::abs(pixel - background);
        const int threshold = std::max(cfg_.base_threshold,
                                       static_cast<int>(noise_[i]) * 2);
        const bool is_foreground = !warming && difference > threshold;
        raw_mask_[i] = is_foreground ? 1 : 0;
        foreground += is_foreground ? 1 : 0;
        if (is_foreground) {
            const int x = static_cast<int>(i % low_w_);
            const int y = static_cast<int>(i / low_w_);
            const int gx = std::min(2, x * 3 / std::max(1, low_w_));
            const int gy = std::min(2, y * 3 / std::max(1, low_h_));
            ++foreground_grid[gy * 3 + gx];
        }

        if (!is_foreground || warming) {
            const int target = pixel << 8;
            background_q8_[i] = static_cast<uint16_t>(
                static_cast<int>(background_q8_[i]) +
                ((target - static_cast<int>(background_q8_[i])) >> update_shift));
            const int noise_delta = difference - noise_[i];
            noise_[i] = static_cast<uint8_t>(ClampFloat(
                static_cast<float>(static_cast<int>(noise_[i]) + (noise_delta >> 4)),
                3.0f, 32.0f));
        }
    }
    ++warmup_frames_;

    const float ratio = pixels > 0
        ? static_cast<float>(foreground) / static_cast<float>(pixels) : 0.0f;
    const int grid_area = std::max(1, low_w_ * low_h_ / 9);
    foreground_grid_cells_ = 0;
    for (int i = 0; i < 9; ++i) {
        if (foreground_grid[i] > grid_area / 50) ++foreground_grid_cells_;
    }
    return ratio;
}

bool MotionGuard::CameraDisturbanceCandidate(float foreground_ratio) const {
    // A moving person normally occupies one to three grid cells. Camera shake
    // changes edges throughout the image, so require a wider distribution.
    const bool frame_wide_change = frame_delta_ratio_ > 0.010f &&
                                   frame_delta_grid_cells_ >= 5;
    const bool background_wide_change = foreground_ratio > 0.25f &&
                                        foreground_grid_cells_ >= 4;
    return frame_wide_change || background_wide_change;
}

void MotionGuard::StartRecalibration() {
    tracks_.clear();
    blobs_.clear();
    std::fill(background_q8_.begin(), background_q8_.end(), 0);
    std::fill(noise_.begin(), noise_.end(), 8);
    std::fill(raw_mask_.begin(), raw_mask_.end(), 0);
    std::fill(filtered_mask_.begin(), filtered_mask_.end(), 0);
    std::fill(previous_mask_.begin(), previous_mask_.end(), 0);
    warmup_frames_ = 0;
    stable_state_ = MotionGuardState::CALIBRATING;
    pending_state_ = MotionGuardState::CALIBRATING;
    pending_state_frames_ = 0;
    system_state_ = MotionGuardSystemState::CAMERA_UNSTABLE;
    camera_stable_frames_ = 0;
    camera_arm_cooldown_frames_ = 0;
    disturbance_hits_ = 0;
}

void MotionGuard::FilterMask() {
    std::fill(filtered_mask_.begin(), filtered_mask_.end(), 0);
    for (int y = 1; y < low_h_ - 1; ++y) {
        for (int x = 1; x < low_w_ - 1; ++x) {
            int count = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                const uint8_t* row = raw_mask_.data() + (y + dy) * low_w_;
                count += row[x - 1] + row[x] + row[x + 1];
            }
            filtered_mask_[y * low_w_ + x] = count >= 3 ? 1 : 0;
        }
    }
    raw_mask_.swap(filtered_mask_);
    std::fill(filtered_mask_.begin(), filtered_mask_.end(), 0);
    for (int y = 1; y < low_h_ - 1; ++y) {
        for (int x = 1; x < low_w_ - 1; ++x) {
            const int index = y * low_w_ + x;
            filtered_mask_[index] =
                raw_mask_[index] || raw_mask_[index - 1] || raw_mask_[index + 1] ||
                raw_mask_[index - low_w_] || raw_mask_[index + low_w_];
        }
    }

    // Require a connected foreground island to be present in two adjacent
    // detection instants. The one-pixel neighbourhood preserves moving
    // targets while rejecting isolated sensor flicker and compression noise.
    raw_mask_.swap(filtered_mask_);
    std::fill(filtered_mask_.begin(), filtered_mask_.end(), 0);
    for (int y = 1; y < low_h_ - 1; ++y) {
        for (int x = 1; x < low_w_ - 1; ++x) {
            const int index = y * low_w_ + x;
            if (!raw_mask_[index]) continue;
            bool seen_before = false;
            for (int dy = -1; dy <= 1 && !seen_before; ++dy) {
                const uint8_t* previous = previous_mask_.data() + (y + dy) * low_w_;
                for (int dx = -1; dx <= 1; ++dx) {
                    if (previous[x + dx]) {
                        seen_before = true;
                        break;
                    }
                }
            }
            filtered_mask_[index] = seen_before ? 1 : 0;
        }
    }
    // Keep the un-gated connected mask as temporal history. Keeping the
    // already gated mask would make a new target impossible to promote.
    previous_mask_ = raw_mask_;
}

void MotionGuard::ExtractBlobs() {
    blobs_.clear();
    std::fill(visited_.begin(), visited_.end(), 0);
    const int max_area = low_w_ * low_h_ / 2;
    for (int y = 1; y < low_h_ - 1; ++y) {
        for (int x = 1; x < low_w_ - 1; ++x) {
            const int start = y * low_w_ + x;
            if (!filtered_mask_[start] || visited_[start]) continue;
            flood_queue_.clear();
            flood_queue_.push_back(start);
            visited_[start] = 1;
            Blob blob;
            blob.x1 = blob.x2 = x;
            blob.y1 = blob.y2 = y;
            int sum_x = 0;
            int sum_y = 0;

            for (size_t q = 0; q < flood_queue_.size(); ++q) {
                const int index = flood_queue_[q];
                const int px = index % low_w_;
                const int py = index / low_w_;
                ++blob.area;
                sum_x += px;
                sum_y += py;
                blob.x1 = std::min(blob.x1, px);
                blob.y1 = std::min(blob.y1, py);
                blob.x2 = std::max(blob.x2, px);
                blob.y2 = std::max(blob.y2, py);
                const int neighbors[4] = {index - 1, index + 1,
                                          index - low_w_, index + low_w_};
                for (int n = 0; n < 4; ++n) {
                    const int next = neighbors[n];
                    if (!visited_[next] && filtered_mask_[next]) {
                        visited_[next] = 1;
                        flood_queue_.push_back(next);
                    }
                }
            }
            if (blob.area < cfg_.min_blob_area || blob.area > max_area) continue;
            const int box_area = (blob.x2 - blob.x1 + 1) * (blob.y2 - blob.y1 + 1);
            if (box_area > blob.area * 12) continue;
            blob.cx = static_cast<float>(sum_x) / blob.area;
            blob.cy = static_cast<float>(sum_y) / blob.area;
            blobs_.push_back(blob);
        }
    }
    std::sort(blobs_.begin(), blobs_.end(), [](const Blob& a, const Blob& b) {
        return a.area > b.area;
    });
    if (blobs_.size() > static_cast<size_t>(cfg_.max_tracks * 2)) {
        blobs_.resize(cfg_.max_tracks * 2);
    }

    // A moving person or vehicle is often split into several foreground
    // islands by clothes, reflections and low-texture areas. Merge close
    // islands before tracking so the UI describes one moving target instead
    // of exposing implementation fragments as many unrelated boxes.
    const int merge_gap = scene_ == MotionGuardScene::HOME ? 7 : 5;
    bool merged = true;
    while (merged) {
        merged = false;
        for (size_t i = 0; i < blobs_.size() && !merged; ++i) {
            for (size_t j = i + 1; j < blobs_.size(); ++j) {
                const bool near_x = blobs_[i].x1 <= blobs_[j].x2 + merge_gap &&
                                    blobs_[j].x1 <= blobs_[i].x2 + merge_gap;
                const bool near_y = blobs_[i].y1 <= blobs_[j].y2 + merge_gap &&
                                    blobs_[j].y1 <= blobs_[i].y2 + merge_gap;
                if (!near_x || !near_y) continue;
                Blob& a = blobs_[i];
                const Blob& b = blobs_[j];
                a.x1 = std::min(a.x1, b.x1);
                a.y1 = std::min(a.y1, b.y1);
                a.x2 = std::max(a.x2, b.x2);
                a.y2 = std::max(a.y2, b.y2);
                a.area += b.area;
                a.cx = 0.5f * (a.x1 + a.x2);
                a.cy = 0.5f * (a.y1 + a.y2);
                blobs_.erase(blobs_.begin() + j);
                merged = true;
                break;
            }
        }
    }
    std::sort(blobs_.begin(), blobs_.end(), [](const Blob& a, const Blob& b) {
        return a.area > b.area;
    });
    if (blobs_.size() > static_cast<size_t>(cfg_.max_tracks)) {
        blobs_.resize(cfg_.max_tracks);
    }
}

void MotionGuard::PredictOnly() {
    for (size_t i = 0; i < tracks_.size(); ++i) {
        MotionGuardTrack& track = tracks_[i].output;
        track.cx = ClampFloat(track.cx + track.vx, 0.0f, static_cast<float>(width_ - 1));
        track.cy = ClampFloat(track.cy + track.vy, 0.0f, static_cast<float>(height_ - 1));
        track.x = ClampFloat(track.cx - 0.5f * track.w, 0.0f,
                             static_cast<float>(std::max(0, width_ - 1)));
        track.y = ClampFloat(track.cy - 0.5f * track.h, 0.0f,
                             static_cast<float>(std::max(0, height_ - 1)));
        ++track.age;
    }
}

void MotionGuard::AssociateAndUpdate(uint32_t) {
    std::vector<bool> blob_used(blobs_.size(), false);
    for (size_t t = 0; t < tracks_.size(); ++t) {
        TrackState& state = tracks_[t];
        MotionGuardTrack& track = state.output;
        int best = -1;
        float best_cost = std::numeric_limits<float>::max();
        for (size_t b = 0; b < blobs_.size(); ++b) {
            if (blob_used[b]) continue;
            const float bx = 2.0f * blobs_[b].cx;
            const float by = 2.0f * blobs_[b].cy;
            const float dx = bx - (track.cx + track.vx * cfg_.detection_interval);
            const float dy = by - (track.cy + track.vy * cfg_.detection_interval);
            const float distance = std::sqrt(dx * dx + dy * dy);
            const float limit = std::max(cfg_.min_association_distance,
                cfg_.association_scale * std::max(track.w, track.h));
            if (distance > limit) continue;
            const float iou = BlobIoU(track, blobs_[b]);
            const float cost = distance / std::max(1.0f, limit) - 0.7f * iou;
            if (cost < best_cost) {
                best_cost = cost;
                best = static_cast<int>(b);
            }
        }
        if (best < 0) {
            ++track.missed;
            track.cx += track.vx * cfg_.detection_interval;
            track.cy += track.vy * cfg_.detection_interval;
            ++track.age;
            continue;
        }

        blob_used[best] = true;
        const Blob& blob = blobs_[best];
        const float measured_x = 2.0f * blob.x1;
        const float measured_y = 2.0f * blob.y1;
        const float measured_w = 2.0f * (blob.x2 - blob.x1 + 1);
        const float measured_h = 2.0f * (blob.y2 - blob.y1 + 1);
        const float measured_cx = measured_x + 0.5f * measured_w;
        const float measured_cy = measured_y + 0.5f * measured_h;
        const float instant_vx = (measured_cx - track.cx) / cfg_.detection_interval;
        const float instant_vy = (measured_cy - track.cy) / cfg_.detection_interval;
        track.vx = 0.60f * track.vx + 0.40f * instant_vx;
        track.vy = 0.60f * track.vy + 0.40f * instant_vy;
        track.cx = 0.35f * track.cx + 0.65f * measured_cx;
        track.cy = 0.35f * track.cy + 0.65f * measured_cy;
        track.w = 0.45f * track.w + 0.55f * measured_w;
        track.h = 0.45f * track.h + 0.55f * measured_h;
        track.x = track.cx - 0.5f * track.w;
        track.y = track.cy - 0.5f * track.h;
        const float area = std::max(1.0f, track.w * track.h);
        const float growth_raw = tracks_[t].last_area > 1.0f
            ? (area - tracks_[t].last_area) / tracks_[t].last_area : 0.0f;
        const float growth = ClampFloat(growth_raw, -0.25f, 0.25f);
        track.area_growth = 0.75f * track.area_growth + 0.25f * growth;
        tracks_[t].last_area = area;
        track.confidence = ClampFloat(
            static_cast<float>(blob.area) /
            std::max(1.0f, measured_w * measured_h * 0.25f), 0.0f, 1.0f);
        track.missed = 0;
        const float speed = std::sqrt(track.vx * track.vx + track.vy * track.vy);
        if (speed >= 0.45f) {
            state.motion_hits = std::min(state.motion_hits + 1, 8);
        } else {
            state.motion_hits = std::max(0, state.motion_hits - 1);
        }
        if (state.motion_hits >= 3) state.motion_confirmed = true;
        ++track.age;
    }

    for (size_t b = 0; b < blobs_.size() &&
                       tracks_.size() < static_cast<size_t>(cfg_.max_tracks); ++b) {
        if (blob_used[b]) continue;
        const Blob& blob = blobs_[b];
        TrackState state;
        state.output.id = next_track_id_++;
        state.output.x = 2.0f * blob.x1;
        state.output.y = 2.0f * blob.y1;
        state.output.w = 2.0f * (blob.x2 - blob.x1 + 1);
        state.output.h = 2.0f * (blob.y2 - blob.y1 + 1);
        state.output.cx = state.output.x + 0.5f * state.output.w;
        state.output.cy = state.output.y + 0.5f * state.output.h;
        state.output.confidence = 0.6f;
        state.output.age = 1;
        state.last_area = state.output.w * state.output.h;
        state.last_line_side = cfg_.line_y >= 0.0f
            ? state.output.cy / height_ - cfg_.line_y : 0.0f;
        tracks_.push_back(state);
    }

    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
        [this](const TrackState& state) {
            return state.output.missed > cfg_.max_missed;
        }), tracks_.end());
}

bool MotionGuard::InsideZone(float cx, float cy) const {
    const float nx = cx / std::max(1, width_);
    const float ny = cy / std::max(1, height_);
    return nx >= cfg_.zone_x1 && nx <= cfg_.zone_x2 &&
           ny >= cfg_.zone_y1 && ny <= cfg_.zone_y2;
}

void MotionGuard::UpdateRiskAndResult(MotionGuardResult* result) {
    result->tracks.clear();
    result->selected_index = -1;
    result->active_targets = 0;
    MotionGuardState observed_state = result->background_ready
        ? MotionGuardState::CLEAR : MotionGuardState::CALIBRATING;
    float best_risk = -1.0f;
    for (size_t i = 0; i < tracks_.size(); ++i) {
        TrackState& state = tracks_[i];
        MotionGuardTrack& track = state.output;
        track.events = MOTION_EVENT_NONE;
        float risk = std::min(20.0f, 2.5f * std::sqrt(
            track.vx * track.vx + track.vy * track.vy));
        const bool inside = InsideZone(track.cx, track.cy);
        // Never expose a predicted-but-not-measured track to callers or OSD.
        // This is the final guard against a stale frame leaving a residual box.
        const bool stable_track = track.age >= 6 && track.missed == 0;
        const bool eligible_track = stable_track && state.motion_confirmed;
        if (inside && eligible_track) {
            ++track.dwell_frames;
            track.events |= MOTION_EVENT_INTRUSION;
            risk += scene_ == MotionGuardScene::HOME ? 35.0f : 12.0f;
        } else {
            track.dwell_frames = 0;
        }
        if (track.dwell_frames > cfg_.loiter_frames) {
            track.events |= MOTION_EVENT_LOITERING;
            risk += 25.0f;
        }
        if (eligible_track && track.area_growth > cfg_.approach_threshold) {
            state.approach_hits = std::min(state.approach_hits + 1, 8);
        } else {
            state.approach_hits = std::max(0, state.approach_hits - 1);
        }
        if (state.approach_hits >= 3) {
            track.events |= MOTION_EVENT_APPROACH;
            risk += 30.0f;
            track.ttc_frames = 2.0f / std::max(0.001f, track.area_growth);
        } else {
            track.ttc_frames = 0.0f;
        }

        if (scene_ == MotionGuardScene::ROADSIDE && cfg_.line_y >= 0.0f) {
            const float side = track.cy / std::max(1, height_) - cfg_.line_y;
            if (state.last_line_side * side < 0.0f && track.age > 3) {
                state.line_event_hold = nominal_fps_ / 2;
            }
            state.last_line_side = side;
            if (state.line_event_hold > 0) {
                --state.line_event_hold;
                track.events |= MOTION_EVENT_LINE_CROSS;
                risk += 40.0f;
            }
            if (cfg_.legal_direction_y * track.vy < -0.45f && track.age > 8 &&
                track.missed == 0) {
                state.wrong_way_hits = std::min(state.wrong_way_hits + 1, 8);
            } else {
                state.wrong_way_hits = std::max(0, state.wrong_way_hits - 1);
            }
            if (state.wrong_way_hits >= 4) {
                track.events |= MOTION_EVENT_WRONG_WAY;
                risk += 35.0f;
            }
        }
        risk += std::min(15.0f, track.age * 0.15f);
        risk -= track.missed * 8.0f;
        track.risk = ClampFloat(risk, 0.0f, 100.0f);
        if (track.missed != 0) continue;
        result->tracks.push_back(track);
        if (eligible_track) {
            ++result->active_targets;
        }
        if (eligible_track && track.risk > best_risk) {
            best_risk = track.risk;
            result->selected_index = static_cast<int>(result->tracks.size() - 1);
        }
    }

    if (result->selected_index >= 0) {
        const MotionGuardTrack& selected = result->tracks[result->selected_index];
        if (selected.events & MOTION_EVENT_APPROACH) {
            observed_state = MotionGuardState::APPROACHING;
        } else if (selected.events & MOTION_EVENT_WRONG_WAY) {
            observed_state = MotionGuardState::WRONG_WAY;
        } else if (selected.events & MOTION_EVENT_LOITERING) {
            observed_state = MotionGuardState::LOITERING;
        } else if (selected.events & MOTION_EVENT_LINE_CROSS) {
            observed_state = MotionGuardState::LINE_CROSSING;
        } else if (scene_ == MotionGuardScene::HOME &&
                   selected.events & MOTION_EVENT_INTRUSION) {
            observed_state = MotionGuardState::ZONE_OCCUPIED;
        } else {
            observed_state = scene_ == MotionGuardScene::HOME
                ? MotionGuardState::MOTION : MotionGuardState::PASSING;
        }
    }

    if (observed_state == stable_state_) {
        pending_state_ = observed_state;
        pending_state_frames_ = 0;
    } else {
        if (pending_state_ != observed_state) {
            pending_state_ = observed_state;
            pending_state_frames_ = 1;
        } else {
            ++pending_state_frames_;
        }
        const bool urgent = StatePriority(observed_state) >=
                            StatePriority(MotionGuardState::LINE_CROSSING);
        const int confirm_frames = urgent ? 2 : 4;
        if (pending_state_frames_ >= confirm_frames) {
            stable_state_ = observed_state;
            pending_state_frames_ = 0;
        }
    }
    result->state = stable_state_;
}

bool MotionGuard::Process(const uint8_t* gray, uint32_t frame_id,
                          MotionGuardResult* result) {
    if (!initialized_ || gray == nullptr || result == nullptr) return false;
    result->ClearFrame();
    result->scene = scene_;
    result->system_state = system_state_;
    result->tracks.reserve(cfg_.max_tracks);
    result->zone_x1 = cfg_.zone_x1;
    result->zone_y1 = cfg_.zone_y1;
    result->zone_x2 = cfg_.zone_x2;
    result->zone_y2 = cfg_.zone_y2;
    result->line_y = cfg_.line_y;

    Downsample2x(gray);
    const bool detect_now = frame_id % cfg_.detection_interval == 0 ||
                            warmup_frames_ == 0;
    if (!detect_now) {
        if (system_state_ == MotionGuardSystemState::ARMED) PredictOnly();
        result->background_ready = system_state_ == MotionGuardSystemState::ARMED;
        result->system_state = system_state_;
        if (system_state_ == MotionGuardSystemState::ARMED) {
            UpdateRiskAndResult(result);
        } else {
            result->state = MotionGuardState::CALIBRATING;
        }
        return true;
    }

    result->foreground_ratio = UpdateBackgroundAndMask();
    const bool disturbance = CameraDisturbanceCandidate(result->foreground_ratio);
    if (system_state_ == MotionGuardSystemState::ARMED) {
        if (camera_arm_cooldown_frames_ > 0) {
            --camera_arm_cooldown_frames_;
            disturbance_hits_ = 0;
        } else {
            disturbance_hits_ = disturbance ? disturbance_hits_ + 1 :
                std::max(0, disturbance_hits_ - 1);
            if (disturbance_hits_ >= 2) {
                StartRecalibration();
            }
        }
    } else if (system_state_ == MotionGuardSystemState::CAMERA_UNSTABLE) {
        if (disturbance) {
            camera_stable_frames_ = 0;
        } else {
            ++camera_stable_frames_;
        }
        if (camera_stable_frames_ >= cfg_.camera_stable_frames) {
            // The camera has stopped: the following warm-up establishes this
            // new view as the baseline before events are allowed again.
            system_state_ = MotionGuardSystemState::RECALIBRATING;
            warmup_frames_ = 0;
            std::fill(background_q8_.begin(), background_q8_.end(), 0);
            std::fill(noise_.begin(), noise_.end(), 8);
            std::fill(raw_mask_.begin(), raw_mask_.end(), 0);
            std::fill(filtered_mask_.begin(), filtered_mask_.end(), 0);
            std::fill(previous_mask_.begin(), previous_mask_.end(), 0);
        }
    } else if (system_state_ == MotionGuardSystemState::RECALIBRATING) {
        if (disturbance) {
            StartRecalibration();
        } else if (warmup_frames_ >= cfg_.background_warmup) {
            system_state_ = MotionGuardSystemState::ARMED;
            camera_arm_cooldown_frames_ = cfg_.camera_arm_cooldown_frames;
            stable_state_ = MotionGuardState::CLEAR;
            pending_state_ = MotionGuardState::CLEAR;
            pending_state_frames_ = 0;
        }
    } else if (system_state_ == MotionGuardSystemState::CALIBRATING &&
               warmup_frames_ >= cfg_.background_warmup) {
        system_state_ = MotionGuardSystemState::ARMED;
        camera_arm_cooldown_frames_ = cfg_.camera_arm_cooldown_frames;
        stable_state_ = MotionGuardState::CLEAR;
        pending_state_ = MotionGuardState::CLEAR;
        pending_state_frames_ = 0;
    }

    if (system_state_ != MotionGuardSystemState::ARMED) {
        tracks_.clear();
        blobs_.clear();
        result->background_ready = false;
        result->system_state = system_state_;
        result->state = MotionGuardState::CALIBRATING;
        return true;
    }

    FilterMask();
    ExtractBlobs();
    AssociateAndUpdate(frame_id);
    result->background_ready = true;
    result->system_state = system_state_;
    UpdateRiskAndResult(result);
    return true;

}
