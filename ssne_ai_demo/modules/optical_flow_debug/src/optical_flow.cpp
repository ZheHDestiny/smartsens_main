/*
 * @Filename: optical_flow.cpp
 * @Description: Optical flow implementation with FAST corner detection
 */
#include "common.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>

static const int fast_circle_row[16] = {
    -3, -3, -2, -1,  0,  1,  2,  3,
     3,  3,  2,  1,  0, -1, -2, -3
};
static const int fast_circle_col[16] = {
     0,  1,  2,  3,  3,  3,  2,  1,
     0, -1, -2, -3, -3, -3, -2, -1
};

static float fast9_score(const uint8_t* image, int width, int height,
                         int row, int col, int threshold) {
    const int center_val = image[row * width + col];
    const int high = center_val + threshold;
    const int low  = center_val - threshold;

    int bright_count = 0;
    int dark_count = 0;

    int val0 = image[(row + fast_circle_row[0])  * width + (col + fast_circle_col[0])];
    int val4 = image[(row + fast_circle_row[4])  * width + (col + fast_circle_col[4])];
    int val8 = image[(row + fast_circle_row[8])  * width + (col + fast_circle_col[8])];
    int val12 = image[(row + fast_circle_row[12]) * width + (col + fast_circle_col[12])];

    if (val0 > high) bright_count++; else if (val0 < low) dark_count++;
    if (val4 > high) bright_count++; else if (val4 < low) dark_count++;
    if (val8 > high) bright_count++; else if (val8 < low) dark_count++;
    if (val12 > high) bright_count++; else if (val12 < low) dark_count++;

    if (bright_count < 2 && dark_count < 2) return 0.0f;

    int circle_val[16];
    for (int i = 0; i < 16; i++) {
        circle_val[i] = image[(row + fast_circle_row[i]) * width +
                              (col + fast_circle_col[i])];
    }

    int consecutive_bright = 0;
    int max_bright = 0;
    float score_bright = 0.0f;
    for (int i = 0; i < 25; i++) {
        int idx = i % 16;
        if (circle_val[idx] > high) {
            consecutive_bright++;
            score_bright += (float)(circle_val[idx] - center_val);
            if (consecutive_bright > max_bright) max_bright = consecutive_bright;
        } else {
            consecutive_bright = 0;
            score_bright = 0.0f;
        }
    }

    int consecutive_dark = 0;
    int max_dark = 0;
    float score_dark = 0.0f;
    for (int i = 0; i < 25; i++) {
        int idx = i % 16;
        if (circle_val[idx] < low) {
            consecutive_dark++;
            score_dark += (float)(center_val - circle_val[idx]);
            if (consecutive_dark > max_dark) max_dark = consecutive_dark;
        } else {
            consecutive_dark = 0;
            score_dark = 0.0f;
        }
    }

    if (max_bright >= 9) {
        float total = 0.0f;
        for (int i = 0; i < 16; i++) {
            if (circle_val[i] > high) total += (float)(circle_val[i] - center_val);
        }
        return total;
    }

    if (max_dark >= 9) {
        float total = 0.0f;
        for (int i = 0; i < 16; i++) {
            if (circle_val[i] < low) total += (float)(center_val - circle_val[i]);
        }
        return total;
    }

    return 0.0f;
}

static void nms_features(std::vector<FeaturePoint>& features, int radius,
                         int img_width, int img_height) {
    if (features.empty()) return;

    std::sort(features.begin(), features.end(),
              [](const FeaturePoint& a, const FeaturePoint& b) {
                  return a.score > b.score;
              });

    std::vector<bool> suppressed(features.size(), false);
    int cell_size = std::max(1, radius);
    int grid_cols = (img_width + cell_size - 1) / cell_size;
    int grid_rows = (img_height + cell_size - 1) / cell_size;
    std::vector<int> grid(grid_cols * grid_rows, -1); 

    std::vector<FeaturePoint> result;
    result.reserve(features.size());

    for (int i = 0; i < (int)features.size(); i++) {
        if (suppressed[i]) continue;

        int cx = (int)(features[i].x / cell_size);
        int cy = (int)(features[i].y / cell_size);

        bool is_max = true;
        for (int dy = -1; dy <= 1 && is_max; dy++) {
            for (int dx = -1; dx <= 1 && is_max; dx++) {
                int nx = cx + dx;
                int ny = cy + dy;
                if (nx < 0 || nx >= grid_cols || ny < 0 || ny >= grid_rows) continue;

                int idx = grid[ny * grid_cols + nx];
                if (idx >= 0) {
                    float dist_x = features[i].x - result[idx].x;
                    float dist_y = features[i].y - result[idx].y;
                    if (dist_x * dist_x + dist_y * dist_y < radius * radius) {
                        is_max = false;
                    }
                }
            }
        }

        if (is_max) {
            int result_idx = (int)result.size();
            grid[cy * grid_cols + cx] = result_idx;
            result.push_back(features[i]);
        }
    }

    features.swap(result);
}

static void grid_sample(std::vector<FeaturePoint>& features,
                        int img_width, int img_height,
                        int grid_size, int max_per_cell) {
    if (features.empty() || grid_size <= 0) return;

    float cell_w = (float)img_width / grid_size;
    float cell_h = (float)img_height / grid_size;

    std::vector<std::vector<int>> cells(grid_size * grid_size);

    for (int i = 0; i < (int)features.size(); i++) {
        int gx = std::min((int)(features[i].x / cell_w), grid_size - 1);
        int gy = std::min((int)(features[i].y / cell_h), grid_size - 1);
        cells[gy * grid_size + gx].push_back(i);
    }

    std::vector<FeaturePoint> result;
    result.reserve(features.size());

    for (auto& cell : cells) {
        int count = std::min((int)cell.size(), max_per_cell);
        for (int j = 0; j < count; j++) {
            result.push_back(features[cell[j]]);
        }
    }

    features.swap(result);
}

static float median_value(std::vector<float>* values) {
    if (values == nullptr || values->empty()) return 0.0f;
    const size_t middle = values->size() / 2;
    std::nth_element(values->begin(), values->begin() + middle, values->end());
    float median = (*values)[middle];
    if ((values->size() & 1U) == 0U) {
        std::nth_element(values->begin(), values->begin() + middle - 1, values->end());
        median = 0.5f * (median + (*values)[middle - 1]);
    }
    return median;
}


void OPTICALFLOW::Initialize(int width, int height) {
    width_ = width;
    height_ = height;

    max_features = 120;
    fast_threshold = 22;
    feature_scan_step = 1;
    nms_radius = 8;
    grid_size = 5;
    grid_max_per_cell = 6;

    pyramid_levels = 2;        
    lk_win_size = 4;           
    lk_max_iter = 10;          
    lk_epsilon = 0.05f;        
    lk_min_eig = 2e-4f;        

    printf("[OPTICALFLOW] Initialized: %d x %d, pyramid=%d, win=%d\n",
           width_, height_, pyramid_levels, lk_win_size);
}

void OPTICALFLOW::DetectFeatures(const uint8_t* frame,
                                  std::vector<FeaturePoint>& features) {
    features.clear();

    if (frame == nullptr) return;

    const int margin = 4;
    const int scan_step = std::max(1, feature_scan_step);
    const int grid_cols = std::max(1, grid_size);
    const int grid_rows = std::max(1,
        (grid_cols * height_ + width_ / 2) / width_);
    const int per_cell = std::max(1, grid_max_per_cell);
    const int cell_count = grid_cols * grid_rows;
    std::vector<std::vector<FeaturePoint>> cells(cell_count);
    for (auto& cell : cells) cell.reserve(per_cell);

    // Scene-complexity invariant FAST: each spatial cell owns a tiny fixed
    // quota. A cluttered road scene therefore stops scanning a cell as soon
    // as enough usable corners are found instead of collecting and sorting
    // thousands of candidates. Sparse scenes still receive the full scan,
    // where FAST's four-pixel early rejection is inexpensive.
    for (int gy = 0; gy < grid_rows; ++gy) {
        const int y0 = std::max(margin, gy * height_ / grid_rows);
        const int y1 = std::min(height_ - margin, (gy + 1) * height_ / grid_rows);
        for (int gx = 0; gx < grid_cols; ++gx) {
            const int x0 = std::max(margin, gx * width_ / grid_cols);
            const int x1 = std::min(width_ - margin, (gx + 1) * width_ / grid_cols);
            std::vector<FeaturePoint>& cell = cells[gy * grid_cols + gx];
            for (int row = y0; row < y1 && static_cast<int>(cell.size()) < per_cell;
                 row += scan_step) {
                for (int col = x0; col < x1 && static_cast<int>(cell.size()) < per_cell;
                     col += scan_step) {
                    const float score = fast9_score(frame, width_, height_, row, col,
                                                    fast_threshold);
                    if (score <= 0.0f) continue;
                    bool separated = true;
                    for (const auto& accepted : cell) {
                        const float dx = accepted.x - col;
                        const float dy = accepted.y - row;
                        if (dx * dx + dy * dy < nms_radius * nms_radius) {
                            separated = false;
                            break;
                        }
                    }
                    if (separated) {
                        cell.emplace_back(static_cast<float>(col),
                                          static_cast<float>(row), score);
                    }
                }
            }
        }
    }

    features.reserve(std::min(max_features, cell_count * per_cell));
    // Round-robin extraction keeps every image region represented before any
    // cell contributes its second point.
    for (int rank = 0; rank < per_cell && static_cast<int>(features.size()) < max_features;
         ++rank) {
        for (const auto& cell : cells) {
            if (rank < static_cast<int>(cell.size())) {
                features.push_back(cell[rank]);
                if (static_cast<int>(features.size()) >= max_features) break;
            }
        }
    }

}

float OPTICALFLOW::Interp(const uint8_t* img, int w, int h, float x, float y) {
    int ix = (int)x;
    int iy = (int)y;

    if (ix < 0) ix = 0;
    if (iy < 0) iy = 0;
    if (ix >= w - 1) ix = w - 2;
    if (iy >= h - 1) iy = h - 2;

    float fx = x - ix;
    float fy = y - iy;

    const uint8_t* row0 = img + iy * w;
    const uint8_t* row1 = row0 + w;

    float v00 = row0[ix];
    float v10 = row0[ix + 1];
    float v01 = row1[ix];
    float v11 = row1[ix + 1];

    return (1.0f - fx) * (1.0f - fy) * v00 +
           fx * (1.0f - fy) * v10 +
           (1.0f - fx) * fy * v01 +
           fx * fy * v11;
}

void OPTICALFLOW::BuildPyramid(const uint8_t* frame,
                                std::vector<std::vector<uint8_t>>& pyramid) {
    int levels = pyramid_levels;
    pyramid.resize(levels);
    pyr_widths_.resize(levels);
    pyr_heights_.resize(levels);

    int w0 = width_;
    int h0 = height_;
    // Keep level 0 as a contiguous copy. Spatial denoising here used to scan
    // two full 720x540 frames on every iteration; the LK patch and the robust
    // multi-point risk estimator already suppress isolated sensor noise.
    pyramid[0].assign(frame, frame + w0 * h0);
    pyr_widths_[0] = w0;
    pyr_heights_[0] = h0;

    for (int lv = 1; lv < levels; lv++) {
        int pw = pyr_widths_[lv - 1];
        int ph = pyr_heights_[lv - 1];
        int nw = pw / 2;
        int nh = ph / 2;
        if (nw < 4 || nh < 4) {
            pyramid.resize(lv);
            pyr_widths_.resize(lv);
            pyr_heights_.resize(lv);
            break;
        }

        pyramid[lv].resize(nw * nh);
        pyr_widths_[lv] = nw;
        pyr_heights_[lv] = nh;

        const uint8_t* src = pyramid[lv - 1].data();
        uint8_t* dst = pyramid[lv].data();

        for (int r = 0; r < nh; r++) {
            for (int c = 0; c < nw; c++) {
                int sr = r * 2;
                int sc = c * 2;
                int sum = (int)src[sr * pw + sc] +
                          (int)src[sr * pw + sc + 1] +
                          (int)src[(sr + 1) * pw + sc] +
                          (int)src[(sr + 1) * pw + sc + 1];
                dst[r * nw + c] = (uint8_t)((sum + 2) / 4);
            }
        }
    }
}

bool OPTICALFLOW::TrackPointSingleLevel(const uint8_t* prev, const uint8_t* curr,
                                         int w, int h,
                                         float px, float py,
                                         float& cx, float& cy,
                                         float* photometric_error) {
    int win = lk_win_size;

    if (px - win < 1 || px + win >= w - 1 ||
        py - win < 1 || py + win >= h - 1) {
        return false;
    }

    float H00 = 0, H01 = 0, H11 = 0;  
    int patch_size = (2 * win + 1);
    int patch_n = patch_size * patch_size;

    float Ix_buf[225], Iy_buf[225], I_buf[225];  

    if (patch_n > 225) return false;  

    int idx = 0;
    for (int dy = -win; dy <= win; dy++) {
        for (int dx = -win; dx <= win; dx++) {
            float x = px + dx;
            float y = py + dy;

            float ix = (Interp(prev, w, h, x + 1, y) - Interp(prev, w, h, x - 1, y)) * 0.5f;
            float iy = (Interp(prev, w, h, x, y + 1) - Interp(prev, w, h, x, y - 1)) * 0.5f;
            float iv = Interp(prev, w, h, x, y);

            Ix_buf[idx] = ix;
            Iy_buf[idx] = iy;
            I_buf[idx] = iv;

            H00 += ix * ix;
            H01 += ix * iy;
            H11 += iy * iy;

            idx++;
        }
    }

    float trace = H00 + H11;
    float det = H00 * H11 - H01 * H01;
    float disc = trace * trace - 4.0f * det;
    if (disc < 0) disc = 0;
    float lambda_min = 0.5f * (trace - std::sqrt(disc));

    if (lambda_min < lk_min_eig * patch_n) {
        return false;  
    }

    if (std::fabs(det) < 1e-10f) return false;
    float inv_det = 1.0f / det;
    float Hi00 = H11 * inv_det;
    float Hi01 = -H01 * inv_det;
    float Hi11 = H00 * inv_det;

    for (int iter = 0; iter < lk_max_iter; iter++) {
        if (cx - win < 0 || cx + win >= w - 1 ||
            cy - win < 0 || cy + win >= h - 1) {
            return false;
        }

        float b0 = 0, b1 = 0;
        float absolute_error = 0.0f;
        idx = 0;
        for (int dy = -win; dy <= win; dy++) {
            for (int dx = -win; dx <= win; dx++) {
                float jv = Interp(curr, w, h, cx + dx, cy + dy);
                float diff = I_buf[idx] - jv;

                b0 += diff * Ix_buf[idx];
                b1 += diff * Iy_buf[idx];
                absolute_error += std::fabs(diff);
                idx++;
            }
        }

        if (photometric_error != nullptr) {
            *photometric_error = absolute_error / patch_n;
        }

        float ddx = Hi00 * b0 + Hi01 * b1;
        float ddy = Hi01 * b0 + Hi11 * b1;

        cx += ddx;
        cy += ddy;

        if (ddx * ddx + ddy * ddy < lk_epsilon * lk_epsilon) {
            break;
        }
    }

    if (cx < 0 || cx >= w - 1 || cy < 0 || cy >= h - 1) {
        return false;
    }

    return true;
}

void OPTICALFLOW::ComputeFlow(const uint8_t* prev_frame, const uint8_t* curr_frame,
                               std::vector<FeaturePoint>& features) {
    if (prev_frame == nullptr || curr_frame == nullptr) return;

    if (features.empty()) return;

    if (!pyramid_cache_valid_) {
        BuildPyramid(prev_frame, pyramid_prev_);
    }
    BuildPyramid(curr_frame, pyramid_curr_);

    int levels = (int)pyramid_prev_.size();
    if (levels <= 0 || pyramid_curr_.size() != pyramid_prev_.size()) return;
    for (auto& f : features) {
        if (!f.tracked || !std::isfinite(f.x) || !std::isfinite(f.y) ||
            f.x < 1.0f || f.x >= width_ - 1.0f ||
            f.y < 1.0f || f.y >= height_ - 1.0f) {
            f.tracked = false;
            f.dx = 0.0f;
            f.dy = 0.0f;
            continue;
        }
        const float old_x = f.x;
        const float old_y = f.y;
        float scale = 1.0f / (1 << (levels - 1));
        float px = f.x * scale;  
        float py = f.y * scale;
        float cx = px;  
        float cy = py;

        bool success = true;
        float photometric_error = 0.0f;

        for (int lv = levels - 1; lv >= 0; lv--) {
            const uint8_t* prev_img = pyramid_prev_[lv].data();
            const uint8_t* curr_img = pyramid_curr_[lv].data();
            int w = pyr_widths_[lv];
            int h = pyr_heights_[lv];

            success = TrackPointSingleLevel(prev_img, curr_img, w, h,
                                            px, py, cx, cy,
                                            lv == 0 ? &photometric_error : nullptr);
            if (!success) break;

            if (lv > 0) {
                px *= 2.0f;
                py *= 2.0f;
                cx *= 2.0f;
                cy *= 2.0f;
            }
        }

        if (success) {
            const float flow_x = cx - old_x;
            const float flow_y = cy - old_y;
            const float flow_sq = flow_x * flow_x + flow_y * flow_y;
            // A final patch residual catches occlusion and gross mismatches at
            // a fraction of the cost of running a complete backward LK pass.
            // The regional median and support threshold reject remaining
            // isolated outliers before they can become an obstacle alert.
            if (photometric_error > 20.0f || flow_sq > 900.0f) {
                f.dx = 0.0f;
                f.dy = 0.0f;
                f.tracked = false;
                continue;
            }
            f.dx = flow_x;
            f.dy = flow_y;
            f.x = cx;           
            f.y = cy;
            f.tracked = true;
        } else {
            f.dx = 0.0f;
            f.dy = 0.0f;
            f.tracked = false;
        }
    }

    // Reuse this frame's pyramid as the next frame's previous pyramid.
    // ResetHistory() is called whenever the frame sequence is interrupted or
    // feature detection replaces the active tracks.
    pyramid_prev_.swap(pyramid_curr_);
    pyramid_cache_valid_ = true;
}

void OPTICALFLOW::ResetHistory() {
    pyramid_cache_valid_ = false;
}

void OPTICALFLOW::Release() {
    pyramid_prev_.clear();
    pyramid_curr_.clear();
    pyr_widths_.clear();
    pyr_heights_.clear();
    pyramid_cache_valid_ = false;
    printf("[OPTICALFLOW] Released\n");
}

void OBSTACLE_DETECTOR::Initialize(int width, int height) {
    width_ = width;
    height_ = height;

    ttc_threshold = 2.0f;
    divergence_threshold = 0.45f;
    frame_interval_seconds_ = 1.0f / 75.0f;
    std::fill(smoothed_danger_, smoothed_danger_ + ObstacleInfo::REGION_COUNT, 0.0f);
    std::fill(latched_obstacle_, latched_obstacle_ + ObstacleInfo::REGION_COUNT, false);

    printf("[OBSTACLE_DETECTOR] Initialized: %d x %d\n", width_, height_);
}

void OBSTACLE_DETECTOR::SetFrameInterval(float seconds) {
    if (std::isfinite(seconds) && seconds > 0.002f && seconds < 0.2f) {
        frame_interval_seconds_ = seconds;
    }
}

float OBSTACLE_DETECTOR::ComputeTTC(float x, float y, float dx, float dy) const {
    float foe_x = width_ / 2.0f;
    float foe_y = height_ / 2.0f;

    // Radial direction points away from the focus of expansion. An approaching
    // object/camera produces positive expansion in this direction.
    float dist_x = x - foe_x;
    float dist_y = y - foe_y;
    float dist = std::sqrt(dist_x * dist_x + dist_y * dist_y);

    if (dist < 1.0f) return 1e6f;  

    dist_x /= dist;
    dist_y /= dist;

    float radial_velocity = dx * dist_x + dy * dist_y;

    if (radial_velocity < 0.03f) return 1e6f;

    float ttc = dist * frame_interval_seconds_ / radial_velocity;
    return std::max(0.0f, ttc);
}

void OBSTACLE_DETECTOR::DetectObstacles(const std::vector<FeaturePoint>& features,
                                        ObstacleInfo& obstacle_info) {
    std::vector<float> global_x;
    std::vector<float> global_y;
    global_x.reserve(features.size());
    global_y.reserve(features.size());
    for (const auto& f : features) {
        if (f.tracked && std::isfinite(f.dx) && std::isfinite(f.dy)) {
            global_x.push_back(f.dx);
            global_y.push_back(f.dy);
        }
    }
    obstacle_info.tracking_quality = features.empty() ? 0.0f :
        static_cast<float>(global_x.size()) / features.size();
    obstacle_info.global_dx = median_value(&global_x);
    obstacle_info.global_dy = median_value(&global_y);

    std::vector<float> region_ttc[ObstacleInfo::REGION_COUNT];
    std::vector<float> region_radial[ObstacleInfo::REGION_COUNT];
    const float region_width = width_ / 3.0f;
    for (const auto& f : features) {
        if (!f.tracked || !std::isfinite(f.x) || !std::isfinite(f.y) ||
            !std::isfinite(f.dx) || !std::isfinite(f.dy) ||
            f.x < 0.0f || f.x >= width_ || f.y < 0.0f || f.y >= height_) {
            continue;
        }
        const int region = std::max(0, std::min(static_cast<int>(f.x / region_width),
                                                static_cast<int>(ObstacleInfo::REGION_COUNT) - 1));
        const float flow_x = f.dx - obstacle_info.global_dx;
        const float flow_y = f.dy - obstacle_info.global_dy;
        const float rx = f.x - width_ * 0.5f;
        const float ry = f.y - height_ * 0.5f;
        const float radius = std::sqrt(rx * rx + ry * ry);
        if (radius < 20.0f) continue;
        const float radial = (flow_x * rx + flow_y * ry) / radius;
        if (radial <= 0.03f) continue;
        region_radial[region].push_back(radial);
        region_ttc[region].push_back(ComputeTTC(f.x, f.y, flow_x, flow_y));
    }

    obstacle_info.priority = ObstacleInfo::CLEAR;
    obstacle_info.most_dangerous_region = ObstacleInfo::CENTER;
    obstacle_info.safest_region = ObstacleInfo::CENTER;
    float max_danger = -1.0f;
    float min_danger = 2.0f;
    for (int i = 0; i < ObstacleInfo::REGION_COUNT; ++i) {
        obstacle_info.support_count[i] = static_cast<int>(region_radial[i].size());
        const bool enough_support = obstacle_info.support_count[i] >= 3;
        float raw_danger = 0.0f;
        float robust_ttc = -1.0f;
        if (enough_support) {
            const float median_radial = median_value(&region_radial[i]);
            robust_ttc = median_value(&region_ttc[i]);
            const float ttc_risk = robust_ttc < ttc_threshold
                ? 1.0f - robust_ttc / ttc_threshold : 0.0f;
            const float expansion_risk = std::min(1.0f,
                std::max(0.0f, (median_radial - divergence_threshold) /
                               (2.0f * divergence_threshold)));
            raw_danger = std::max(ttc_risk, 0.65f * expansion_risk);
        }
        smoothed_danger_[i] = 0.68f * smoothed_danger_[i] + 0.32f * raw_danger;
        if (latched_obstacle_[i]) {
            latched_obstacle_[i] = smoothed_danger_[i] > 0.18f;
        } else {
            latched_obstacle_[i] = enough_support && smoothed_danger_[i] > 0.35f;
        }
        obstacle_info.danger_level[i] = smoothed_danger_[i];
        obstacle_info.ttc_seconds[i] = enough_support ? robust_ttc : -1.0f;
        obstacle_info.has_obstacle[i] = latched_obstacle_[i];
        if (obstacle_info.danger_level[i] > max_danger) {
            max_danger = obstacle_info.danger_level[i];
            obstacle_info.most_dangerous_region = i;
        }
        if (obstacle_info.danger_level[i] < min_danger) {
            min_danger = obstacle_info.danger_level[i];
            obstacle_info.safest_region = i;
        }
        if (latched_obstacle_[i]) {
            const bool emergency = (robust_ttc > 0.0f && robust_ttc < 1.0f) ||
                                   smoothed_danger_[i] > 0.70f;
            obstacle_info.priority = std::min(obstacle_info.priority,
                emergency ? static_cast<int>(ObstacleInfo::EMERGENCY)
                          : static_cast<int>(ObstacleInfo::CAUTION));
        }
    }
}

void OBSTACLE_DETECTOR::Release() {
    std::fill(smoothed_danger_, smoothed_danger_ + ObstacleInfo::REGION_COUNT, 0.0f);
    std::fill(latched_obstacle_, latched_obstacle_ + ObstacleInfo::REGION_COUNT, false);
    printf("[OBSTACLE_DETECTOR] Released\n");
}
