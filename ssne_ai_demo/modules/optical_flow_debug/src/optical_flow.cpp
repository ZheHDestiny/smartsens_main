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


void OPTICALFLOW::Initialize(int width, int height) {
    width_ = width;
    height_ = height;

    max_features = 120;
    fast_threshold = 22;
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
    features.reserve(2000);

    for (int row = margin; row < height_ - margin; row++) {
        for (int col = margin; col < width_ - margin; col++) {
            float score = fast9_score(frame, width_, height_, row, col, fast_threshold);
            if (score > 0.0f) {
                features.emplace_back((float)col, (float)row, score);
            }
        }
    }

    nms_features(features, nms_radius, width_, height_);

    if (grid_size > 0) {
        grid_sample(features, width_, height_, grid_size, grid_max_per_cell);
    }

    if ((int)features.size() > max_features) {
        std::sort(features.begin(), features.end(),
                  [](const FeaturePoint& a, const FeaturePoint& b) {
                      return a.score > b.score;
                  });
        features.resize(max_features);
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
                                         float& cx, float& cy) {
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
        idx = 0;
        for (int dy = -win; dy <= win; dy++) {
            for (int dx = -win; dx <= win; dx++) {
                float jv = Interp(curr, w, h, cx + dx, cy + dy);
                float diff = I_buf[idx] - jv;

                b0 += diff * Ix_buf[idx];
                b1 += diff * Iy_buf[idx];
                idx++;
            }
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

    BuildPyramid(prev_frame, pyramid_prev_);
    BuildPyramid(curr_frame, pyramid_curr_);

    int levels = (int)pyramid_prev_.size();
    int tracked_count = 0;

    for (auto& f : features) {
        float scale = 1.0f / (1 << (levels - 1));
        float px = f.x * scale;  
        float py = f.y * scale;
        float cx = px;  
        float cy = py;

        bool success = true;

        for (int lv = levels - 1; lv >= 0; lv--) {
            const uint8_t* prev_img = pyramid_prev_[lv].data();
            const uint8_t* curr_img = pyramid_curr_[lv].data();
            int w = pyr_widths_[lv];
            int h = pyr_heights_[lv];

            success = TrackPointSingleLevel(prev_img, curr_img, w, h,
                                            px, py, cx, cy);
            if (!success) break;

            if (lv > 0) {
                px *= 2.0f;
                py *= 2.0f;
                cx *= 2.0f;
                cy *= 2.0f;
            }
        }

        if (success) {
            f.dx = cx - f.x;   
            f.dy = cy - f.y;
            f.x = cx;           
            f.y = cy;
            f.tracked = true;
            tracked_count++;
        } else {
            f.dx = 0.0f;
            f.dy = 0.0f;
            f.tracked = false;
        }
    }
}

void OPTICALFLOW::Release() {
    pyramid_prev_.clear();
    pyramid_curr_.clear();
    pyr_widths_.clear();
    pyr_heights_.clear();
    printf("[OPTICALFLOW] Released\n");
}

void OBSTACLE_DETECTOR::Initialize(int width, int height) {
    width_ = width;
    height_ = height;

    ttc_threshold = 0.8f;
    divergence_threshold = 0.65f;

    printf("[OBSTACLE_DETECTOR] Initialized: %d x %d\n", width_, height_);
}

float OBSTACLE_DETECTOR::ComputeTTC(float x, float y, float dx, float dy) {
    float foe_x = width_ / 2.0f;
    float foe_y = height_ / 2.0f;

    float dist_x = foe_x - x;
    float dist_y = foe_y - y;
    float dist = std::sqrt(dist_x * dist_x + dist_y * dist_y);

    if (dist < 1.0f) return 1e6f;  

    dist_x /= dist;
    dist_y /= dist;

    float radial_velocity = dx * dist_x + dy * dist_y;

    if (radial_velocity < 0.01f) return 1e6f;  

    float ttc = dist / radial_velocity;
    return std::max(0.0f, ttc);
}

float OBSTACLE_DETECTOR::ComputeDivergence(const std::vector<FeaturePoint>& features) {
    if (features.empty()) return 0.0f;

    float div_sum = 0.0f;
    int count = 0;

    for (const auto& f : features) {
        if (!f.tracked) continue;

        float dist_x = f.x - (width_ / 2.0f);
        float dist_y = f.y - (height_ / 2.0f);
        float dist = std::sqrt(dist_x * dist_x + dist_y * dist_y);

        if (dist > 1.0f) {
            dist_x /= dist;
            dist_y /= dist;
            float radial = f.dx * dist_x + f.dy * dist_y;
            div_sum += radial;
            count++;
        }
    }

    return count > 0 ? div_sum / count : 0.0f;
}

void OBSTACLE_DETECTOR::DetectObstacles(const std::vector<FeaturePoint>& features,
                                        ObstacleInfo& obstacle_info) {
    for (int i = 0; i < ObstacleInfo::REGION_COUNT; i++) {
        obstacle_info.danger_level[i] = 0.0f;
        obstacle_info.has_obstacle[i] = false;
    }
    obstacle_info.priority = 4;  

    if (features.empty()) {
        return;
    }

    float region_width = width_ / 3.0f;
    float min_ttc[ObstacleInfo::REGION_COUNT] = {1e6f, 1e6f, 1e6f};
    float region_div_sum[ObstacleInfo::REGION_COUNT] = {0.0f, 0.0f, 0.0f};
    int region_div_cnt[ObstacleInfo::REGION_COUNT] = {0, 0, 0};

    for (const auto& f : features) {
        if (!f.tracked) continue;

        int region = (int)(f.x / region_width);
        region = std::min(region, (int)ObstacleInfo::REGION_COUNT - 1);

        float ttc = ComputeTTC(f.x, f.y, f.dx, f.dy);
        min_ttc[region] = std::min(min_ttc[region], ttc);

        float dist_x = f.x - (width_ / 2.0f);
        float dist_y = f.y - (height_ / 2.0f);
        float dist = std::sqrt(dist_x * dist_x + dist_y * dist_y);
        if (dist > 1.0f) {
            dist_x /= dist;
            dist_y /= dist;
            float radial = f.dx * dist_x + f.dy * dist_y;
            region_div_sum[region] += radial;
            region_div_cnt[region]++;
        }
    }

    int max_priority = 4;  

    for (int i = 0; i < ObstacleInfo::REGION_COUNT; i++) {
        float region_divergence =
            (region_div_cnt[i] > 0) ? (region_div_sum[i] / region_div_cnt[i]) : 0.0f;

        if (min_ttc[i] < ttc_threshold) {
            obstacle_info.has_obstacle[i] = true;
            obstacle_info.danger_level[i] = 1.0f - (min_ttc[i] / ttc_threshold);
            max_priority = std::min(max_priority, 0);  
        } else if (region_divergence > divergence_threshold) {
            obstacle_info.has_obstacle[i] = true;
            obstacle_info.danger_level[i] = std::min(1.0f, region_divergence / (2.0f * divergence_threshold));
            max_priority = std::min(max_priority, 1);  
        }
    }

    obstacle_info.priority = max_priority;

    int most_dangerous = ObstacleInfo::CENTER;
    float max_danger = -1.0f;
    for (int i = 0; i < ObstacleInfo::REGION_COUNT; i++) {
        if (obstacle_info.danger_level[i] > max_danger) {
            max_danger = obstacle_info.danger_level[i];
            most_dangerous = i;
        }
    }
    obstacle_info.most_dangerous_region = most_dangerous;
}

void OBSTACLE_DETECTOR::Release() {
    printf("[OBSTACLE_DETECTOR] Released\n");
}