/*
 * @Filename: utils.cpp
 * @Description: SSNE AI Demo 统一工具函数与可视化器实现
 */

#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "log.hpp"

std::atomic<bool> g_signal_received{false};
std::atomic<int> g_runtime_log_mode{static_cast<int>(RuntimeLogMode::SUMMARY)};

using namespace fdevice;
using OsdQR = sst::device::osd::OsdQuadRangle;

const int VISUALIZER::MAX_FEATURE_MARKERS;
const int VISUALIZER::MAX_FLOW_ARROWS;

static inline std::array<float, 4> to_original(float x1, float y1, float x2, float y2, int offset_y) {
    return {x1, y1 + offset_y, x2, y2 + offset_y};
}

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static int flow_magnitude_color(float dx, float dy) {
    (void)dx;
    (void)dy;
    return 1; 
}

static bool same_eye_box(const EyeBox& a, const EyeBox& b) {
    const float epsilon = 0.01f;
    return std::fabs(a.x1 - b.x1) < epsilon &&
           std::fabs(a.y1 - b.y1) < epsilon &&
           std::fabs(a.x2 - b.x2) < epsilon &&
           std::fabs(a.y2 - b.y2) < epsilon;
}

static const uint8_t* focus_glyph_rows(char ch) {
    static const uint8_t glyph_blank[7] = {0, 0, 0, 0, 0, 0, 0};
    static const uint8_t glyph_a[7] = {0x0e, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11};
    static const uint8_t glyph_b[7] = {0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e};
    static const uint8_t glyph_c[7] = {0x0e, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0e};
    static const uint8_t glyph_e[7] = {0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x1f};
    static const uint8_t glyph_f[7] = {0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10};
    static const uint8_t glyph_g[7] = {0x0e, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0f};
    static const uint8_t glyph_h[7] = {0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11};
    static const uint8_t glyph_l[7] = {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f};
    static const uint8_t glyph_r[7] = {0x1e, 0x11, 0x11, 0x1e, 0x14, 0x12, 0x11};
    static const uint8_t glyph_s[7] = {0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e};
    static const uint8_t glyph_v[7] = {0x11, 0x11, 0x11, 0x11, 0x0a, 0x0a, 0x04};
    static const uint8_t glyph_y[7] = {0x11, 0x11, 0x0a, 0x04, 0x04, 0x04, 0x04};
    static const uint8_t glyph_z[7] = {0x1f, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1f};
    static const uint8_t glyph_u[7] = {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e};
    static const uint8_t glyph_n[7] = {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11};
    static const uint8_t glyph_k[7] = {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11};
    static const uint8_t glyph_o[7] = {0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e};
    static const uint8_t glyph_w[7] = {0x11, 0x11, 0x11, 0x15, 0x15, 0x1b, 0x11};
    static const uint8_t glyph_i[7] = {0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1f};
    static const uint8_t glyph_d[7] = {0x1e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1e};
    static const uint8_t glyph_t[7] = {0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04};
    static const uint8_t glyph_m[7] = {0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11};
    static const uint8_t glyph_p[7] = {0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10};
    static const uint8_t glyph_underscore[7] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f};
    switch (ch) {
        case ' ': return glyph_blank;
        case 'A': return glyph_a;
        case 'B': return glyph_b;
        case 'C': return glyph_c;
        case 'E': return glyph_e;
        case 'F': return glyph_f;
        case 'G': return glyph_g;
        case 'H': return glyph_h;
        case 'L': return glyph_l;
        case 'R': return glyph_r;
        case 'S': return glyph_s;
        case 'V': return glyph_v;
        case 'Y': return glyph_y;
        case 'Z': return glyph_z;
        case 'U': return glyph_u;
        case 'N': return glyph_n;
        case 'K': return glyph_k;
        case 'O': return glyph_o;
        case 'W': return glyph_w;
        case 'I': return glyph_i;
        case 'D': return glyph_d;
        case 'T': return glyph_t;
        case 'M': return glyph_m;
        case 'P': return glyph_p;
        case '_': return glyph_underscore;
        default: return nullptr;
    }
}

static bool write_motion_status_bitmap(const char* path, const char* text) {
    if (path == nullptr || text == nullptr) return false;
    const int scale = 3;
    const int glyph_width = 5 * scale;
    const int spacing = 2 * scale;
    const int length = static_cast<int>(std::strlen(text));
    // Keep the longest label ("HOME/ROAD APPROACH") fully inside the bitmap.
    const uint32_t width = 288;
    const uint32_t height = 36;
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height, 5);
    const int total_width = length > 0
        ? length * glyph_width + (length - 1) * spacing : 0;
    const int origin_x = std::max(4, (static_cast<int>(width) - total_width) / 2);
    for (int glyph = 0; glyph < length; ++glyph) {
        const uint8_t* rows = focus_glyph_rows(text[glyph]);
        if (rows == nullptr) continue;
        const int glyph_x = origin_x + glyph * (glyph_width + spacing);
        for (int row = 0; row < 7; ++row) {
            for (int col = 0; col < 5; ++col) {
                if ((rows[row] & (1u << (4 - col))) == 0) continue;
                for (int py = 0; py < scale; ++py) {
                    const int y = 7 + row * scale + py;
                    for (int px = 0; px < scale; ++px) {
                        const int x = glyph_x + col * scale + px;
                        if (x >= 0 && x < static_cast<int>(width) &&
                            y >= 0 && y < static_cast<int>(height)) {
                            pixels[static_cast<size_t>(y) * width + x] = 4;
                        }
                    }
                }
            }
        }
    }
    const uint32_t header[4] = {0x5353424dU, width, height, 30};
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(header), sizeof(header));
    file.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
    return file.good();
}

static const char* motion_status_bitmap(MotionGuardScene scene,
                                        MotionGuardState state,
                                        MotionGuardSystemState system_state) {
    static bool ready[2][12] = {};
    static char paths[2][12][64] = {};
    static const char* const home_text[9] = {
        "HOME LEARN", "HOME SAFE", "HOME MOTION", "HOME ZONE",
        "HOME LOITER", "HOME MOTION", "HOME CROSS", "HOME WRONG",
        "HOME APPROACH"
    };
    static const char* const road_text[9] = {
        "ROAD LEARN", "ROAD CLEAR", "ROAD MOTION", "ROAD ZONE",
        "ROAD LOITER", "ROAD PASS", "ROAD CROSS", "ROAD WRONG",
        "ROAD APPROACH"
    };
    const int scene_index = scene == MotionGuardScene::ROADSIDE ? 1 : 0;
    int status_index = 0;
    const char* text = nullptr;
    if (system_state == MotionGuardSystemState::CAMERA_UNSTABLE) {
        status_index = 1;
        text = scene_index == 0 ? "HOME CAMERA" : "ROAD CAMERA";
    } else if (system_state == MotionGuardSystemState::RECALIBRATING) {
        status_index = 2;
        text = scene_index == 0 ? "HOME RELEARN" : "ROAD RELEARN";
    } else if (system_state == MotionGuardSystemState::CALIBRATING) {
        status_index = 0;
        text = scene_index == 0 ? "HOME LEARN" : "ROAD LEARN";
    } else {
        status_index = 3 + std::max(0, std::min(8, static_cast<int>(state)));
        text = scene_index == 0 ? home_text[status_index - 3]
                                : road_text[status_index - 3];
    }
    if (!ready[scene_index][status_index]) {
        std::snprintf(paths[scene_index][status_index],
                      sizeof(paths[scene_index][status_index]),
                      "/tmp/ssne_motion_%d_%d.ssbmp", scene_index, status_index);
        ready[scene_index][status_index] =
            write_motion_status_bitmap(paths[scene_index][status_index], text);
    }
    return ready[scene_index][status_index]
        ? paths[scene_index][status_index] : nullptr;
}

static const char* optical_status_bitmap(const ObstacleInfo& obstacle_info) {
    static bool ready[3][3] = {};
    static char paths[3][3][64] = {};
    const int region = std::max(0, std::min(2, obstacle_info.most_dangerous_region));
    const int priority_index = obstacle_info.priority == ObstacleInfo::EMERGENCY ? 2 :
        (obstacle_info.priority == ObstacleInfo::CAUTION ? 1 : 0);
    const char* text = "SAFE";
    if (priority_index == 1) {
        text = region == ObstacleInfo::LEFT ? "CAUTION L" :
               (region == ObstacleInfo::CENTER ? "CAUTION C" : "CAUTION R");
    } else if (priority_index == 2) {
        text = region == ObstacleInfo::LEFT ? "BRAKE L" :
               (region == ObstacleInfo::CENTER ? "BRAKE C" : "BRAKE R");
    }
    if (!ready[priority_index][region]) {
        std::snprintf(paths[priority_index][region],
                      sizeof(paths[priority_index][region]),
                      "/tmp/ssne_optical_%d_%d.ssbmp", priority_index, region);
        ready[priority_index][region] =
            write_motion_status_bitmap(paths[priority_index][region], text);
    }
    return ready[priority_index][region] ? paths[priority_index][region] : nullptr;
}

static bool write_focus_status_bitmap(const char* path, const char* text) {
    if (path == nullptr || text == nullptr) return false;
    const int scale = 4;
    const int glyph_width = 5 * scale;
    const int spacing = 2 * scale;
    const int length = static_cast<int>(std::strlen(text));
    const uint32_t width = 200;
    const uint32_t height = 44;
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height, 5);
    const int total_width = length > 0 ? length * glyph_width + (length - 1) * spacing : 0;
    const int origin_x = std::max(4, (static_cast<int>(width) - total_width) / 2);

    for (int glyph = 0; glyph < length; ++glyph) {
        const uint8_t* rows = focus_glyph_rows(text[glyph]);
        if (rows == nullptr) continue;
        const int glyph_x = origin_x + glyph * (glyph_width + spacing);
        for (int row = 0; row < 7; ++row) {
            for (int col = 0; col < 5; ++col) {
                if ((rows[row] & (1u << (4 - col))) == 0) continue;
                for (int py = 0; py < scale; ++py) {
                    const int y = 8 + row * scale + py;
                    if (y < 0 || y >= static_cast<int>(height)) continue;
                    for (int px = 0; px < scale; ++px) {
                        const int x = glyph_x + col * scale + px;
                        if (x >= 0 && x < static_cast<int>(width)) {
                            pixels[static_cast<size_t>(y) * width + x] = 4;
                        }
                    }
                }
            }
        }
    }

    const uint32_t header[4] = {0x5353424dU, width, height, 30};
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(header), sizeof(header));
    file.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
    return file.good();
}

static const char* focus_status_bitmap(bool identity_matched) {
    static const char* const kUnknownPath = "/tmp/ssne_focus_unknown.ssbmp";
    static const char* const kIdentityPath = "/tmp/ssne_focus_id_tmp.ssbmp";
    static bool unknown_ready = false;
    static bool identity_ready = false;
    bool* ready = identity_matched ? &identity_ready : &unknown_ready;
    const char* path = identity_matched ? kIdentityPath : kUnknownPath;
    if (!*ready) {
        *ready = write_focus_status_bitmap(path,
                                           identity_matched ? "ID_TMP" : "UNKNOWN");
        if (!*ready) {
            std::cerr << "[FOCUS_OSD] failed to create status bitmap: " << path << std::endl;
        }
    }
    return *ready ? path : nullptr;
}

static const char* focus_face_bitmap() {
    static const char* const kPath = "/tmp/ssne_focus_face.ssbmp";
    static bool ready = false;
    if (ready) return kPath;
    const uint32_t width = 32;
    const uint32_t height = 18;
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height, 5);
    const auto put = [&pixels, width, height](int x, int y) {
        if (x >= 0 && x < static_cast<int>(width) && y >= 0 && y < static_cast<int>(height)) {
            pixels[static_cast<size_t>(y) * width + x] = 4;
        }
    };
    // Square face, two eyes, nose dot, and a three-step smile arc.
    for (int x = 1; x <= 29; ++x) { put(x, 1); put(x, 16); }
    for (int y = 1; y <= 16; ++y) { put(1, y); put(29, y); }
    for (int y = 5; y <= 6; ++y) {
        for (int x = 7; x <= 8; ++x) put(x, y);
        for (int x = 21; x <= 22; ++x) put(x, y);
    }
    for (int y = 8; y <= 9; ++y) {
        for (int x = 14; x <= 15; ++x) put(x, y);
    }
    for (int x = 8; x <= 11; ++x) put(x, 11);
    for (int x = 12; x <= 18; ++x) put(x, 13);
    for (int x = 19; x <= 22; ++x) put(x, 11);

    const uint32_t header[4] = {0x5353424dU, width, height, 30};
    std::ofstream file(kPath, std::ios::binary | std::ios::trunc);
    if (!file) return nullptr;
    file.write(reinterpret_cast<const char*>(header), sizeof(header));
    file.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
    ready = file.good();
    if (!ready) {
        std::cerr << "[FOCUS_OSD] failed to create face bitmap: " << kPath << std::endl;
    }
    return ready ? kPath : nullptr;
}

void VISUALIZER::Initialize(std::array<int, 2>& in_img_shape,
                            const std::string& bitmap_lut_path,
                            int image_dma_size,
                            uint32_t image_layer_mask,
                            uint32_t layer_creation_mask) {
    m_width = in_img_shape[0];
    m_height = in_img_shape[1];

    const char* lut_path = nullptr;
    if (!bitmap_lut_path.empty()) {
        m_bitmap_lut_path_full = "./app_assets/" + bitmap_lut_path;
        lut_path = m_bitmap_lut_path_full.c_str();
    }
    
    osd_device.Initialize(m_width, m_height, lut_path, image_dma_size,
                          image_layer_mask, layer_creation_mask);
    enabled_ = osd_device.IsEnabled();
    last_optical_priority_ = -1;
    last_optical_region_ = -1;
    focus_face_icon_drawn_ = false;
    last_drawn_emotion = EmotionClass::NUM_CLASSES;
    candidate_emotion = EmotionClass::NUM_CLASSES;
    emotion_hold_count = 0;
    
    if (!enabled_) {
        std::cerr << "[VISUALIZER] Warning: OSD device not enabled or initialization failed." << std::endl;
    }
}

void VISUALIZER::Release() {
    // Release also owns partially initialized devices. Checking only enabled_
    // leaked DMA when every requested layer failed during CMA pressure.
    osd_device.Release();
    enabled_ = false;
    last_optical_priority_ = -1;
    last_optical_region_ = -1;
    focus_face_icon_drawn_ = false;
    last_drawn_emotion = EmotionClass::NUM_CLASSES;
    candidate_emotion = EmotionClass::NUM_CLASSES;
    emotion_hold_count = 0;
}

void VISUALIZER::Clear() {
    if (!enabled_) return;
    std::vector<OsdQR> empty;
    osd_device.Draw(empty, LAYER_FEATURES);
    osd_device.Draw(empty, LAYER_BITMAP);
    osd_device.Draw(empty, LAYER_OBSTACLES);
    osd_device.Draw(empty, LAYER_SAFEDIR);
    osd_device.Draw(empty, LAYER_MASK);
    osd_device.ClearLayer(LAYER_FACE_ICON);
    focus_face_icon_drawn_ = false;
}

void VISUALIZER::Draw() {
    if (!enabled_) return;
    std::vector<OsdQR> quad_rangle_vec;
    OsdQR q;
    q.color = 0;                         
    q.box = {100, 100, 200, 200};        
    q.border = 3;                        
    q.alpha = fdevice::TYPE_ALPHA75;     
    q.type = fdevice::TYPE_HOLLOW;       
    quad_rangle_vec.emplace_back(q);
    osd_device.Draw(quad_rangle_vec);
}

void VISUALIZER::DrawFixedSquare(int x_min, int y_min, int x_max, int y_max, int layer_id) {
    if (!enabled_) return;
    int abs_x_min = x_min, abs_y_min = y_min, abs_x_max = x_max, abs_y_max = y_max;
    if (abs_x_min > abs_x_max) std::swap(abs_x_min, abs_x_max);
    if (abs_y_min > abs_y_max) std::swap(abs_y_min, abs_y_max);

    abs_x_min = std::max(0, std::min(abs_x_min, m_width - 1));
    abs_y_min = std::max(0, std::min(abs_y_min, m_height - 1));
    abs_x_max = std::max(0, std::min(abs_x_max, m_width - 1));
    abs_y_max = std::max(0, std::min(abs_y_max, m_height - 1));

    std::vector<std::array<float, 4>> square_box;
    square_box.push_back({static_cast<float>(abs_x_min), static_cast<float>(abs_y_min),
                          static_cast<float>(abs_x_max), static_cast<float>(abs_y_max)});

    osd_device.Draw(square_box, 0, layer_id, fdevice::TYPE_SOLID, fdevice::TYPE_ALPHA100, 2);
}

void VISUALIZER::DrawBitmap(const std::string& bitmap_path, const std::string& lut_path, int pos_x, int pos_y, int layer_id) {
    if (!enabled_) return;
    std::string full_bitmap_path = "./app_assets/" + bitmap_path;
    const char* full_lut_path = nullptr;
    std::string lut_full_path;
    if (!lut_path.empty()) {
        lut_full_path = "./app_assets/" + lut_path;
        full_lut_path = lut_full_path.c_str();
    }
    osd_device.DrawTexture(full_bitmap_path.c_str(), full_lut_path, layer_id, pos_x, pos_y);
}


void VISUALIZER::Draw(const std::vector<std::array<float, 4>>& boxes) {
    if (!enabled_) return;
    std::vector<OsdQR> quad_rangle_vec;
    for (size_t i = 0; i < boxes.size(); i++) {
        OsdQR q;
        q.box = {boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]};
        q.color = 1;                         
        q.border = 3;                        
        q.alpha = fdevice::TYPE_ALPHA75;     
        q.type = fdevice::TYPE_HOLLOW;       
        q.layer_id = DETECTION_LAYER_ID;     
        quad_rangle_vec.emplace_back(q);     
    }
    osd_device.Draw(quad_rangle_vec, DETECTION_LAYER_ID);
}

void VISUALIZER::Draw(const std::vector<std::array<float, 4>>& boxes, 
                      const std::vector<float>& scores, 
                      const std::vector<int>& class_ids) {
                      
    if (boxes.empty()) {
        std::vector<sst::device::osd::OsdQuadRangle> empty_quads;
        osd_device.Draw(empty_quads, DETECTION_LAYER_ID);
        return;
    }

    std::vector<sst::device::osd::OsdQuadRangle> quad_rangles;
    
    for (size_t i = 0; i < boxes.size(); ++i) {
        sst::device::osd::OsdQuadRangle q_box;
        q_box.box = boxes[i];
        q_box.border = 3;  
        q_box.layer_id = DETECTION_LAYER_ID;
        q_box.type = fdevice::TYPE_HOLLOW;
        q_box.alpha = fdevice::TYPE_ALPHA75;
        
        int color_idx = ((class_ids[i] + i * 2) % 6) + 1; 
        q_box.color = color_idx;
        quad_rangles.push_back(q_box);

        sst::device::osd::OsdQuadRangle q_conf;
        float x1 = boxes[i][0];
        float y1 = boxes[i][1];
        float x2 = boxes[i][2];
        float y2 = boxes[i][3];
        float score = clampf(scores[i], 0.0f, 1.0f);
        float conf_width = (x2 - x1) * score;
        const float conf_gap = 8.0f;
        const float conf_h = 8.0f;
        float conf_y2 = y1 - conf_gap;
        float conf_y1 = conf_y2 - conf_h;
        if (conf_y1 < 0.0f) {
            conf_y1 = y2 + conf_gap;
            conf_y2 = conf_y1 + conf_h;
            if (conf_y2 > static_cast<float>(m_height - 1)) {
                conf_y2 = y2 - conf_gap;
                conf_y1 = conf_y2 - conf_h;
            }
        }
        conf_y1 = clampf(conf_y1, 0.0f, static_cast<float>(m_height - 1));
        conf_y2 = clampf(conf_y2, 0.0f, static_cast<float>(m_height - 1));

        q_conf.box = {x1, conf_y1, x1 + conf_width, conf_y2};
        q_conf.border = 0; 
        q_conf.layer_id = DETECTION_LAYER_ID;
        q_conf.type = fdevice::TYPE_SOLID; 
        q_conf.alpha = fdevice::TYPE_ALPHA100;
        q_conf.color = (color_idx == 1) ? 6 : 1; 

        quad_rangles.push_back(q_conf);
    }
    
    osd_device.Draw(quad_rangles, DETECTION_LAYER_ID);
}

void VISUALIZER::DrawFocusFov(const std::array<float, 4>& fov) {
    if (!enabled_) return;
    std::vector<OsdQR> quads;
    OsdQR q;
    q.box = fov;
    q.border = 2;
    q.layer_id = LAYER_OBSTACLES;
    q.type = fdevice::TYPE_HOLLOW;
    q.alpha = fdevice::TYPE_ALPHA50;
    q.color = 23;
    quads.push_back(q);
    osd_device.Draw(quads, LAYER_OBSTACLES);
}

void VISUALIZER::DrawFocusEyes(const EyeDetResult& result,
                               bool identity_matched,
                               int offset_x,
                               int offset_y) {
    if (!enabled_) return;
    std::vector<OsdQR> quads;
    const EyePair* selected = nullptr;
    if (result.selected_index >= 0 &&
        result.selected_index < static_cast<int>(result.pairs.size())) {
        selected = &result.pairs[result.selected_index];
    }

    // The A1 OSD supports at most four quadrangles on a layer scanline.
    // Render the selected pair first, then at most two supplementary eyes.
    // This bounds command/DMA use even when a crowded scene produces many
    // detector candidates.
    const size_t kMaxFocusEyeQuads = 4;
    for (int pass = 0; pass < 2 && quads.size() < kMaxFocusEyeQuads; ++pass) {
        const bool want_selected = pass == 0;
        for (size_t i = 0; i < result.eyes.size() && quads.size() < kMaxFocusEyeQuads; ++i) {
            const EyeBox& eye = result.eyes[i];
            const bool selected_eye = selected != nullptr &&
                (same_eye_box(eye, selected->left) || same_eye_box(eye, selected->right));
            if (selected_eye != want_selected) continue;

            OsdQR q;
            q.box = {eye.x1 + offset_x, eye.y1 + offset_y,
                     eye.x2 + offset_x, eye.y2 + offset_y};
            q.layer_id = DETECTION_LAYER_ID;
            q.type = fdevice::TYPE_HOLLOW;
            if (!selected_eye) {
                q.border = 2;
                q.color = 29;
                q.alpha = fdevice::TYPE_ALPHA50;
            } else if (identity_matched) {
                q.border = 5;
                q.color = 4;
                q.alpha = fdevice::TYPE_ALPHA100;
            } else {
                q.border = 4;
                q.color = 23;
                q.alpha = fdevice::TYPE_ALPHA75;
            }
            quads.push_back(q);
        }
    }
    // A1's layer flush submits new primitives but is not a guaranteed
    // replacement for primitives submitted by a previous frame.  Clearing
    // this layer first is therefore required: without it old eye boxes can
    // accumulate in the driver, causing residual boxes and eventually an OSD
    // command/DMA stall during a long-running tracking session.
    std::vector<OsdQR> empty;
    osd_device.Draw(empty, DETECTION_LAYER_ID);
    osd_device.Draw(quads, DETECTION_LAYER_ID);
}

void VISUALIZER::DrawFocusIdentity(bool identity_matched,
                                   int fov_right,
                                   int fov_top) {
    if (!enabled_) return;
    const char* bitmap = focus_status_bitmap(identity_matched);
    if (bitmap == nullptr) return;
    osd_device.ClearLayer(LAYER_BITMAP);
    osd_device.DrawTexture(bitmap, nullptr, LAYER_BITMAP,
                           fov_right - 204, fov_top + 4,
                           fdevice::TYPE_ALPHA100);
    const char* face = focus_face_icon_drawn_ ? nullptr : focus_face_bitmap();
    if (face != nullptr) {
        osd_device.ClearLayer(LAYER_FACE_ICON);
        osd_device.DrawTexture(face, nullptr, LAYER_FACE_ICON,
                               11, std::max(4, m_height - 24),
                               fdevice::TYPE_ALPHA100);
        focus_face_icon_drawn_ = true;
    }
}

void VISUALIZER::DrawFocusConfidence(float eye_confidence,
                                     float identity_confidence) {
    if (!enabled_) return;
    const float eye = clampf(eye_confidence, 0.0f, 1.0f);
    const float identity = clampf(identity_confidence, 0.0f, 1.0f);
    const int icon_x1 = 12;
    const int icon_x2 = 42;
    const int bar_x1 = 56;
    const int bar_x2 = std::max(bar_x1 + 20, m_width - 12);
    const int bar_h = 14;
    const int first_y = std::max(4, m_height - 46);
    const int second_y = std::max(4, m_height - 23);
    std::vector<OsdQR> quads;
    const auto add = [&quads](float x1, float y1, float x2, float y2,
                              fdevice::QUADRANGLETYPE type, int border) {
        OsdQR q;
        q.box = {x1, y1, x2, y2};
        q.border = border;
        q.layer_id = LAYER_SAFEDIR;
        q.type = type;
        q.alpha = fdevice::TYPE_ALPHA100;
        q.color = 4;
        quads.push_back(q);
    };
    // First row: eye outline, pupil, bar outline, and fill (four quads).
    add(icon_x1, first_y, icon_x2, first_y + bar_h, fdevice::TYPE_HOLLOW, 2);
    add(24, first_y + 5, 30, first_y + 9, fdevice::TYPE_SOLID, 0);
    add(bar_x1, first_y, bar_x2, first_y + bar_h, fdevice::TYPE_HOLLOW, 1);
    if (eye > 0.0f) {
        add(bar_x1 + 3, first_y + 3,
            bar_x1 + 3 + std::max(1, static_cast<int>((bar_x2 - bar_x1 - 6) * eye)),
            first_y + bar_h - 3, fdevice::TYPE_SOLID, 0);
    }
    // The square expression face is a texture on the image layer; only the
    // bar is kept here so this graphic row stays well under OSD quad limits.
    add(bar_x1, second_y, bar_x2, second_y + bar_h, fdevice::TYPE_HOLLOW, 1);
    if (identity > 0.0f) {
        add(bar_x1 + 3, second_y + 3,
            bar_x1 + 3 + std::max(1, static_cast<int>((bar_x2 - bar_x1 - 6) * identity)),
            second_y + bar_h - 3, fdevice::TYPE_SOLID, 0);
    }
    // See DrawFocusEyes(): clear the dynamic layer before each replacement.
    // The calls are adjacent in the same processing thread; the OSD refresh
    // is already rate-limited by the focus module, so this prevents stale
    // bars without materially increasing display flicker.
    std::vector<OsdQR> empty;
    osd_device.Draw(empty, LAYER_SAFEDIR);
    osd_device.Draw(quads, LAYER_SAFEDIR);
}

void VISUALIZER::DrawFocusEnrollmentFlash(const std::array<float, 4>& fov,
                                           bool visible) {
    if (!enabled_) return;
    std::vector<OsdQR> quads;
    if (visible) {
        OsdQR flash;
        flash.box = fov;
        flash.border = 0;
        flash.layer_id = LAYER_MASK;
        flash.type = fdevice::TYPE_SOLID;
        flash.alpha = fdevice::TYPE_ALPHA25;
        flash.color = 4;  // white, visible even when RGB output is unavailable
        quads.push_back(flash);
    }
    // This layer is stateful in the A1 OSD driver. Replace the previous
    // enrollment pulse explicitly so repeated E/C cycles cannot accumulate
    // full-frame mask primitives.
    std::vector<OsdQR> empty;
    osd_device.Draw(empty, LAYER_MASK);
    if (quads.empty()) return;
    osd_device.Draw(quads, LAYER_MASK);
}

void VISUALIZER::DrawMotionGuard(const MotionGuardResult& result,
                                 int crop_x,
                                 int crop_y,
                                 int crop_width,
                                 int crop_height,
                                 int process_width,
                                 int process_height) {
    if (!enabled_ || process_width <= 0 || process_height <= 0) return;
    const float scale_x = static_cast<float>(crop_width) / process_width;
    const float scale_y = static_cast<float>(crop_height) / process_height;
    const bool warning = result.state == MotionGuardState::LOITERING ||
                         result.state == MotionGuardState::LINE_CROSSING ||
                         result.state == MotionGuardState::WRONG_WAY ||
                         result.state == MotionGuardState::APPROACHING;

    const int scene_value = static_cast<int>(result.scene);
    const int state_value = static_cast<int>(result.state);
    const int system_state_value = static_cast<int>(result.system_state);
    if (scene_value != last_motion_scene_ || state_value != last_motion_state_ ||
        system_state_value != last_motion_system_state_) {
        const char* bitmap = motion_status_bitmap(result.scene, result.state,
                                                  result.system_state);
        osd_device.ClearLayer(LAYER_BITMAP);
        if (bitmap != nullptr) {
            osd_device.DrawTexture(bitmap, nullptr, LAYER_BITMAP,
                                   std::max(0, crop_x + crop_width - 292),
                                   crop_y + 4, fdevice::TYPE_ALPHA100);
        }
        last_motion_scene_ = scene_value;
        last_motion_state_ = state_value;
        last_motion_system_state_ = system_state_value;
    }

    std::vector<OsdQR> guide;
    OsdQR zone;
    zone.box = {
        crop_x + result.zone_x1 * crop_width,
        crop_y + result.zone_y1 * crop_height,
        crop_x + result.zone_x2 * crop_width,
        crop_y + result.zone_y2 * crop_height
    };
    zone.border = warning ? 5 : 2;
    zone.layer_id = LAYER_OBSTACLES;
    zone.type = fdevice::TYPE_HOLLOW;
    zone.alpha = warning ? fdevice::TYPE_ALPHA100 : fdevice::TYPE_ALPHA50;
    zone.color = warning ? 4 : 23;
    guide.push_back(zone);
    if (result.line_y >= 0.0f) {
        OsdQR line;
        const float y = crop_y + result.line_y * crop_height;
        line.box = {static_cast<float>(crop_x), y - 1.0f,
                    static_cast<float>(crop_x + crop_width - 1), y + 1.0f};
        line.border = 0;
        line.layer_id = LAYER_OBSTACLES;
        line.type = fdevice::TYPE_SOLID;
        line.alpha = fdevice::TYPE_ALPHA75;
        line.color = 4;
        guide.push_back(line);
    }
    std::vector<OsdQR> empty;
    osd_device.Draw(empty, LAYER_OBSTACLES);
    osd_device.Draw(guide, LAYER_OBSTACLES);

    std::vector<OsdQR> boxes;
    if (result.selected_index >= 0 &&
        result.selected_index < static_cast<int>(result.tracks.size())) {
        const MotionGuardTrack& track = result.tracks[result.selected_index];
        OsdQR box;
        box.box = {
            clampf(crop_x + track.x * scale_x,
                   static_cast<float>(crop_x),
                   static_cast<float>(crop_x + crop_width - 1)),
            clampf(crop_y + track.y * scale_y,
                   static_cast<float>(crop_y),
                   static_cast<float>(crop_y + crop_height - 1)),
            clampf(crop_x + (track.x + track.w) * scale_x,
                   static_cast<float>(crop_x),
                   static_cast<float>(crop_x + crop_width - 1)),
            clampf(crop_y + (track.y + track.h) * scale_y,
                   static_cast<float>(crop_y),
                   static_cast<float>(crop_y + crop_height - 1))
        };
        box.layer_id = DETECTION_LAYER_ID;
        box.type = fdevice::TYPE_HOLLOW;
        box.border = warning ? 7 : 4;
        box.alpha = fdevice::TYPE_ALPHA100;
        box.color = 4;
        boxes.push_back(box);
    }
    osd_device.Draw(empty, DETECTION_LAYER_ID);
    osd_device.Draw(boxes, DETECTION_LAYER_ID);

    // A compact, axis-aligned direction arrow is attached to the current
    // semantic target. Other internal tracks are deliberately not drawn.
    std::vector<OsdQR> direction;
    if (result.selected_index >= 0 &&
        result.selected_index < static_cast<int>(result.tracks.size())) {
        const MotionGuardTrack& track = result.tracks[result.selected_index];
        const float speed = std::sqrt(track.vx * track.vx + track.vy * track.vy);
        if (speed >= 0.5f) {
            const float cx = crop_x + track.cx * scale_x;
            const float cy = crop_y + track.cy * scale_y;
            const bool horizontal = std::fabs(track.vx) >= std::fabs(track.vy);
            const float sign = horizontal
                ? (track.vx >= 0.0f ? 1.0f : -1.0f)
                : (track.vy >= 0.0f ? 1.0f : -1.0f);
            const auto add = [&direction](float x1, float y1, float x2, float y2) {
                OsdQR q;
                q.box = {std::min(x1, x2), std::min(y1, y2),
                         std::max(x1, x2), std::max(y1, y2)};
                q.border = 0;
                q.layer_id = LAYER_SAFEDIR;
                q.type = fdevice::TYPE_SOLID;
                q.alpha = fdevice::TYPE_ALPHA100;
                q.color = 4;
                direction.push_back(q);
            };
            if (horizontal) {
                const float tip = clampf(cx + sign * 42.0f,
                                         crop_x + 12.0f, crop_x + crop_width - 12.0f);
                add(std::min(cx, tip), cy - 2, std::max(cx, tip), cy + 2);
                add(tip - sign * 9, cy - 9, tip, cy - 3);
                add(tip - sign * 9, cy + 3, tip, cy + 9);
            } else {
                const float tip = clampf(cy + sign * 42.0f,
                                         crop_y + 12.0f, crop_y + crop_height - 12.0f);
                add(cx - 2, std::min(cy, tip), cx + 2, std::max(cy, tip));
                add(cx - 9, tip - sign * 9, cx - 3, tip);
                add(cx + 3, tip - sign * 9, cx + 9, tip);
            }
        }
    }
    osd_device.Draw(empty, LAYER_SAFEDIR);
    osd_device.Draw(direction, LAYER_SAFEDIR);
}

void VISUALIZER::DrawSpeed(const std::vector<std::array<float, 4>>& boxes,
                           const std::vector<float>& scores,
                           const std::vector<int>& class_ids,
                           const std::vector<float>& speeds,
                           const std::vector<int>& directions,
                           int crop_offset_y,
                           int crop_height) {
    if (!enabled_) return;
    if (crop_height > 0) {
        DrawMask(crop_offset_y, crop_height);
    } else {
        std::vector<OsdQR> empty;
        osd_device.Draw(empty, LAYER_MASK);
    }

    std::vector<OsdQR> quad_rangle_vec;

    // OSD layer 0 has a limited number of quadrangles. Keep the three most
    // confident detections and validate every primitive before submitting it.
    // This also prevents a fast-moving arrow from producing an off-screen
    // quadrangle, which makes OsdDevice::Draw() fail with ret=-1.
    const size_t detection_count = std::min(boxes.size(), class_ids.size());
    std::vector<size_t> draw_order(detection_count);
    for (size_t i = 0; i < detection_count; ++i) draw_order[i] = i;
    std::sort(draw_order.begin(), draw_order.end(), [&](size_t lhs, size_t rhs) {
        const float lhs_score = lhs < scores.size() ? scores[lhs] : 0.0f;
        const float rhs_score = rhs < scores.size() ? scores[rhs] : 0.0f;
        return lhs_score > rhs_score;
    });
    if (draw_order.size() > 3) draw_order.resize(3);

    const int osd_width = std::max(1, m_width);
    const int osd_height = std::max(1, m_height);
    const auto clamp_x = [osd_width](int value) {
        return std::max(0, std::min(value, osd_width - 1));
    };
    const auto clamp_y = [osd_height](int value) {
        return std::max(0, std::min(value, osd_height - 1));
    };
    const auto append_solid_rect = [&](int x0, int y0, int x1, int y1,
                                       int color, fdevice::ALPHATYPE alpha) {
        x0 = clamp_x(x0);
        x1 = clamp_x(x1);
        y0 = clamp_y(y0);
        y1 = clamp_y(y1);
        if (x1 <= x0 || y1 <= y0) return;
        OsdQR rect;
        rect.box = {static_cast<float>(x0), static_cast<float>(y0),
                    static_cast<float>(x1), static_cast<float>(y1)};
        rect.border = 0;
        rect.color = color;
        rect.alpha = alpha;
        rect.type = fdevice::TYPE_SOLID;
        rect.layer_id = DETECTION_LAYER_ID;
        quad_rangle_vec.emplace_back(rect);
    };

    for (size_t order_pos = 0; order_pos < draw_order.size(); ++order_pos) {
        const size_t i = draw_order[order_pos];
        int xmin = clamp_x(static_cast<int>(boxes[i][0]));
        int ymin = clamp_y(static_cast<int>(boxes[i][1]));
        int xmax = clamp_x(static_cast<int>(boxes[i][2]));
        int ymax = clamp_y(static_cast<int>(boxes[i][3]));
        if (xmax <= xmin || ymax <= ymin) continue;
        
        int cid = class_ids[i];
        int box_color;
        int box_border;
        fdevice::ALPHATYPE box_alpha;

        if (cid == 0) { 
            box_color = 1;     
            box_border = 2;
            box_alpha = fdevice::TYPE_ALPHA100; // 100%不透明，最亮
        } else if (cid == 1) { 
            box_color = 6;     
            box_border = 4;    
            box_alpha = fdevice::TYPE_ALPHA75;  // 75%透明度，硬件混频后自然呈现标准灰色
        } else {               
            box_color = 3;     
            box_border = 6; 
            box_alpha = fdevice::TYPE_ALPHA50;  // 50%透明度，沉入背景，呈现深色/黑色
        }
        
        OsdQR q_box;
        q_box.box = {static_cast<float>(xmin), static_cast<float>(ymin),
                     static_cast<float>(xmax), static_cast<float>(ymax)};
        q_box.border = box_border; 
        q_box.color = box_color; 
        q_box.alpha = box_alpha;
        q_box.type = fdevice::TYPE_HOLLOW; 
        q_box.layer_id = DETECTION_LAYER_ID;
        quad_rangle_vec.emplace_back(q_box);

        const float speed = i < speeds.size() ? speeds[i] : 0.0f;
        const int dir = i < directions.size() ? directions[i] : 0;
        if (speed > 1.0f) { 
            int center_x = (xmin + xmax) / 2;
            const int arrow_y = clamp_y(std::max(10, ymin - 15));
            const int arrow_length = std::min(80, 20 + static_cast<int>(speed * 2));
            const int tip_x = clamp_x(center_x + ((dir >= 0 ? 1 : -1) * arrow_length));

            // Keep both the speed-proportional tail and the arrow head. The
            // three-vehicle cap still bounds this layer to at most nine
            // validated quadrangles (three boxes plus two markers each).
            if (dir >= 0) {
                append_solid_rect(center_x, arrow_y - 2, tip_x, arrow_y + 2,
                                  box_color, box_alpha);
                append_solid_rect(tip_x - 6, arrow_y - 8, tip_x, arrow_y + 8,
                                  box_color, box_alpha);
            } else {
                append_solid_rect(tip_x, arrow_y - 2, center_x, arrow_y + 2,
                                  box_color, box_alpha);
                append_solid_rect(tip_x, arrow_y - 8, tip_x + 6, arrow_y + 8,
                                  box_color, box_alpha);
            }
        }
    }
    osd_device.Draw(quad_rangle_vec, DETECTION_LAYER_ID);
}
void VISUALIZER::DrawMask(int crop_offset_y, int crop_height) {
    if (!enabled_) return;
    std::vector<OsdQR> masks;
    if (crop_offset_y > 0) {
        OsdQR top_mask;
        top_mask.box = {0, 0, (float)m_width, (float)crop_offset_y};
        top_mask.border = 0;
        top_mask.color = 0; 
        top_mask.alpha = fdevice::TYPE_ALPHA75; 
        top_mask.type = fdevice::TYPE_SOLID;
        masks.emplace_back(top_mask);
    }
    if (crop_offset_y + crop_height < m_height) {
        OsdQR bottom_mask;
        bottom_mask.box = {0, (float)(crop_offset_y + crop_height), (float)m_width, (float)m_height};
        bottom_mask.border = 0;
        bottom_mask.color = 0; 
        bottom_mask.alpha = fdevice::TYPE_ALPHA75;
        bottom_mask.type = fdevice::TYPE_SOLID;
        masks.emplace_back(bottom_mask);
    }
    osd_device.Draw(masks, LAYER_MASK);
}

void VISUALIZER::DrawAll(const std::vector<FeaturePoint>& features, const ObstacleInfo& obstacle_info, int crop_offset_y, uint32_t frame_count) {
    if (!enabled_) return;
    DrawMask(crop_offset_y, 540);
    DrawFeaturePoints(features, crop_offset_y);
    DrawObstacleRegions(obstacle_info, crop_offset_y, frame_count);
    DrawSafeDirection(obstacle_info, crop_offset_y);

    // The label is a generated monochrome bitmap so that it remains readable
    // on grayscale OSD hardware; it is regenerated only once per state.
    if (last_optical_priority_ != obstacle_info.priority ||
        last_optical_region_ != obstacle_info.most_dangerous_region) {
        const char* bitmap = optical_status_bitmap(obstacle_info);
        std::vector<OsdQR> clear;
        osd_device.Draw(clear, LAYER_BITMAP);
        if (bitmap != nullptr) {
            osd_device.DrawTexture(bitmap, nullptr, LAYER_BITMAP,
                                   m_width - 292, crop_offset_y + 4);
        }
        last_optical_priority_ = obstacle_info.priority;
        last_optical_region_ = obstacle_info.most_dangerous_region;
    }
}

void VISUALIZER::DrawObstacleRegions(const ObstacleInfo& obstacle_info, int crop_offset_y, uint32_t frame_count) {
    if (!enabled_) return;
    std::vector<OsdQR> regions;
    float region_w = (float)m_width / 3.0f;
    float crop_height = 540.0f; 

    for (int i = 0; i < ObstacleInfo::REGION_COUNT; i++) {
        const float x_start = region_w * i;
        const float x_end = region_w * (i + 1);
        const bool emergency = obstacle_info.priority == ObstacleInfo::EMERGENCY &&
                               obstacle_info.most_dangerous_region == i;
        const bool caution = obstacle_info.has_obstacle[i];
        const bool flash_on = !emergency || ((frame_count / 5) % 2 == 0);

        OsdQR q;
        q.box = {x_start + 2, static_cast<float>(crop_offset_y) + 2,
                 x_end - 2, static_cast<float>(crop_offset_y) + crop_height - 2};
        q.border = emergency ? 10 : (caution ? 6 : 2);
        q.color = (emergency || caution) ? 4 : 29;
        q.alpha = (emergency || caution) && flash_on
            ? fdevice::TYPE_ALPHA100 : fdevice::TYPE_ALPHA25;
        q.type = fdevice::TYPE_HOLLOW;
        q.layer_id = LAYER_OBSTACLES;
        regions.emplace_back(q);
    }
    osd_device.Draw(regions, LAYER_OBSTACLES);
}


void VISUALIZER::DrawFeaturePoints(const std::vector<FeaturePoint>& features, int crop_offset_y) {
    if (!enabled_) return;
    std::vector<OsdQR> markers;
    markers.reserve(MAX_FEATURE_MARKERS);
    for (const auto& fp : features) {
        if (!fp.tracked || !std::isfinite(fp.x) || !std::isfinite(fp.y)) continue;
        float r = 3.0f; 
        float x1 = clampf(fp.x - r, 0, m_width - 1);
        float y1 = clampf(fp.y - r, 0, m_height - 1);
        float x2 = clampf(fp.x + r, 0, m_width - 1);
        float y2 = clampf(fp.y + r, 0, m_height - 1);

        OsdQR q;
        auto b = to_original(x1, y1, x2, y2, crop_offset_y);
        for (int k = 0; k < 4; ++k) {
            if (k % 2 == 0) b[k] = clampf(b[k], 0.0f, (float)(m_width - 1));
            else b[k] = clampf(b[k], 0.0f, (float)(m_height - 1));
        }
        q.box = b;
        q.border = 2; 
        q.color = flow_magnitude_color(fp.dx, fp.dy);
        q.alpha = fdevice::TYPE_ALPHA100;
        q.type = fdevice::TYPE_HOLLOW;
        markers.emplace_back(q);
        if (static_cast<int>(markers.size()) >= MAX_FEATURE_MARKERS) break;
    }
    osd_device.Draw(markers, LAYER_FEATURES);
}


void VISUALIZER::DrawSafeDirection(const ObstacleInfo& obstacle_info, int crop_offset_y) {
    if (!enabled_) return;
    std::vector<OsdQR> indicators;
    float region_w = (float)m_width / 3.0f;
    float indicator_y = (float)crop_offset_y + 540.0f - 44.0f;
    float indicator_h = 24.0f;
    float cx = region_w * obstacle_info.safest_region + region_w / 2.0f;
    float arrow_w = region_w * 0.34f;

    OsdQR q;
    q.box = {cx - arrow_w, indicator_y, cx + arrow_w, indicator_y + indicator_h};
    q.border = 3;
    q.color = 4;
    q.alpha = fdevice::TYPE_ALPHA100;
    q.type = fdevice::TYPE_HOLLOW;
    q.layer_id = LAYER_SAFEDIR;
    indicators.emplace_back(q);

    osd_device.Draw(indicators, LAYER_SAFEDIR);
}

int VISUALIZER::GestureColor(GestureClass g) {
    switch (g) {
        case GestureClass::ROCK:     return C_ROCK;
        case GestureClass::PAPER:    return C_PAPER;
        case GestureClass::SCISSORS: return C_SCISSORS;
        default:                     return C_IDLE;
    }
}

void VISUALIZER::CommitBoxes(int layer_id, int border, int color, std::vector<std::array<float,4>>& boxes) {
    osd_device.Draw(boxes, border, layer_id, fdevice::TYPE_HOLLOW, fdevice::TYPE_ALPHA75, color);
}

void VISUALIZER::DrawRock(int layer_id, int cx, int y0, int color, bool active) {
    std::vector<std::array<float,4>> boxes;
    const int W = 160;   
    const int H = ICON_H;
    int x0 = cx - W / 2;
    int x1 = cx + W / 2;

    int knuckle_h = static_cast<int>(H * 0.18f);  
    int knuckle_w = (W - 12) / 3;                 
    int knuckle_y0 = y0;
    int knuckle_y1 = y0 + knuckle_h;

    for (int k = 0; k < 3; k++) {
        int kx0 = x0 + k * (knuckle_w + 6);
        int kx1 = kx0 + knuckle_w;
        boxes.push_back({ (float)kx0, (float)knuckle_y0, (float)kx1, (float)knuckle_y1 });
    }

    int palm_y0 = knuckle_y1 + 10;
    int palm_y1 = y0 + static_cast<int>(H * 0.75f);
    boxes.push_back({ (float)x0, (float)palm_y0, (float)x1, (float)palm_y1 });

    int wrist_y0 = palm_y1 + 6;
    int wrist_y1 = wrist_y0 + 24;
    int wrist_x0 = x0 + 20;
    int wrist_x1 = x1 - 20;
    boxes.push_back({ (float)wrist_x0, (float)wrist_y0, (float)wrist_x1, (float)wrist_y1 });

    int border = active ? 10 : 2;
    CommitBoxes(layer_id, border, active ? color : C_IDLE, boxes);
}

void VISUALIZER::DrawPaper(int layer_id, int cx, int y0, int color, bool active) {
    std::vector<std::array<float,4>> boxes;
    const int W = 170;   
    const int H = ICON_H;
    int x0 = cx - W / 2;

    // A1 accepts at most four quadrangles intersecting one scanline on a
    // graphic layer. Use a clear three-finger palm symbol instead of five
    // parallel rectangles, which made the final add fail intermittently.
    const int finger_count = 3;
    int finger_w  = 36;
    int finger_gap = (W - finger_count * finger_w) / (finger_count - 1);
    int finger_h  = static_cast<int>(H * 0.58f);    

    for (int f = 0; f < finger_count; f++) {
        int fx0 = x0 + f * (finger_w + finger_gap);
        int fx1 = fx0 + finger_w;
        int fy0 = y0;
        int fy1 = y0 + finger_h;
        boxes.push_back({ (float)fx0, (float)fy0, (float)fx1, (float)fy1 });
    }

    int palm_y0 = y0 + finger_h + 4;
    int palm_y1 = palm_y0 + static_cast<int>(H * 0.22f);
    boxes.push_back({ (float)x0, (float)palm_y0, (float)(x0 + W), (float)palm_y1 });

    int border = active ? 10 : 2;
    CommitBoxes(layer_id, border, active ? color : C_IDLE, boxes);
}

void VISUALIZER::DrawScissors(int layer_id, int cx, int y0, int color, bool active) {
    // A1 graphic OSD has a small per-submit rectangle budget.  The old
    // scissors icon used 11 rectangles; when the budget was exceeded the
    // whole layer was rejected, so scissors disappeared for both players.
    // Keep the icon recognizable with four axis-aligned rectangles.
    std::vector<std::array<float,4>> boxes;
    const int W = 160;
    const int H = ICON_H;
    int x0 = cx - W / 2;
    int x1 = cx + W / 2;

    const int finger_w = 30;
    const int finger_y1 = y0 + static_cast<int>(H * 0.42f);
    boxes.push_back({ (float)(x0 + 8), (float)y0,
                      (float)(x0 + 8 + finger_w), (float)finger_y1 });
    boxes.push_back({ (float)(x1 - 8 - finger_w), (float)y0,
                      (float)(x1 - 8), (float)finger_y1 });

    const int palm_y0 = y0 + static_cast<int>(H * 0.43f);
    const int palm_y1 = y0 + static_cast<int>(H * 0.75f);
    boxes.push_back({ (float)(x0 + 10), (float)palm_y0,
                      (float)(x1 - 10), (float)palm_y1 });

    const int wrist_y0 = y0 + static_cast<int>(H * 0.79f);
    boxes.push_back({ (float)(cx - 42), (float)wrist_y0,
                      (float)(cx + 42), (float)(y0 + H) });

    int border = active ? 10 : 2;
    CommitBoxes(layer_id, border, active ? color : C_IDLE, boxes);
}

void VISUALIZER::DrawIdle(int layer_id, int cx, int y0) {
    std::vector<std::array<float,4>> boxes;
    const int W = 140;
    const int H = ICON_H;
    int x0 = cx - W / 2;
    int x1 = cx + W / 2;

    boxes.push_back({ (float)x0, (float)y0, (float)x1, (float)(y0 + H * 2 / 3) });

    int dot_size = 20;
    int dot_cx   = cx;
    int dot_cy   = y0 + H / 3;
    boxes.push_back({ (float)(dot_cx - dot_size / 2), (float)(dot_cy - dot_size / 2),
                      (float)(dot_cx + dot_size / 2), (float)(dot_cy + dot_size / 2) });

    CommitBoxes(layer_id, 2, C_IDLE, boxes);
}

void VISUALIZER::DrawGestureIcon(int layer_id, int cx, int y0, GestureClass gesture, bool active) {
    if (!enabled_) return;
    int color = GestureColor(gesture);
    switch (gesture) {
        case GestureClass::ROCK:     DrawRock(layer_id, cx, y0, color, active); break;
        case GestureClass::PAPER:    DrawPaper(layer_id, cx, y0, color, active); break;
        case GestureClass::SCISSORS: DrawScissors(layer_id, cx, y0, color, active); break;
        default:                     DrawIdle(layer_id, cx, y0); break;
    }
}

void VISUALIZER::DrawStatusBar(int layer_id, GameState state, GestureClass locked) {
    if (!enabled_) return;
    std::vector<std::array<float,4>> boxes;
    boxes.push_back({(float)STATUS_X1, (float)STATUS_Y1, (float)STATUS_X2, (float)STATUS_Y2});

    int border = 2;
    int color  = C_IDLE;

    switch (state) {
        case GameState::IDLE:      border = 2;  color = C_IDLE;           break;
        case GameState::WIND_UP:   border = 6;  color = C_WINDUP;         break;
        case GameState::PREDICTED:
        case GameState::DISPLAY:   border = 10; color = GestureColor(locked); break;
    }
    CommitBoxes(layer_id, border, color, boxes);
}

void VISUALIZER::DrawConfBar(int layer_id, float confidence, int color) {
    if (!enabled_) return;
    float w = CONF_MAXW * std::max(0.f, std::min(1.f, confidence));
    std::vector<std::array<float,4>> boxes;
    if (w >= 2.f) {
        boxes.push_back({(float)CONF_X0, (float)CONF_Y1, (float)(CONF_X0 + w), (float)CONF_Y2});
    }
    CommitBoxes(layer_id, 3, color, boxes);
}

void VISUALIZER::Draw(const RpsResult& result, const std::array<float, 4>& hand_roi) {
    if (!enabled_) return;

    std::vector<std::array<float,4>> roi_vec = { hand_roi };
    CommitBoxes(0, 2, C_PAPER, roi_vec);

    switch (result.game_state) {
        case GameState::IDLE:
            DrawGestureIcon(1, HUMAN_CX, ICON_Y0, GestureClass::IDLE, false);
            break;
        case GameState::WIND_UP:
            if (result.human_gesture != GestureClass::IDLE)
                DrawGestureIcon(1, HUMAN_CX, ICON_Y0, result.human_gesture, true);
            else
                DrawGestureIcon(1, HUMAN_CX, ICON_Y0, GestureClass::IDLE, false);
            break;
        case GameState::PREDICTED:
        case GameState::DISPLAY:
            DrawGestureIcon(1, HUMAN_CX, ICON_Y0, result.human_gesture, true);
            break;
    }
    if (result.is_locked && result.ai_counter != GestureClass::IDLE) {
        DrawGestureIcon(3, AI_CX, ICON_Y0, result.ai_counter, true);
    } else {
        DrawGestureIcon(3, AI_CX, ICON_Y0, GestureClass::IDLE, false);
    }

    // Layer 2 is a hardware RLE layer and rejects vector quadrangles even if
    // requested otherwise. Keep both gesture pictures on known graphic layers
    // (human=1, AI=3), then combine status and confidence into one submission
    // on layer 4 so no required visual element is lost.
    std::vector<OsdQR> hud;
    OsdQR status;
    status.box = {static_cast<float>(STATUS_X1), static_cast<float>(STATUS_Y1),
                  static_cast<float>(STATUS_X2), static_cast<float>(STATUS_Y2)};
    status.layer_id = 4;
    status.type = fdevice::TYPE_HOLLOW;
    status.alpha = fdevice::TYPE_ALPHA75;
    status.border = 2;
    status.color = C_IDLE;
    switch (result.game_state) {
        case GameState::WIND_UP:
            status.border = 6;
            status.color = C_WINDUP;
            break;
        case GameState::PREDICTED:
        case GameState::DISPLAY:
            status.border = 10;
            status.color = GestureColor(result.human_gesture);
            break;
        default:
            break;
    }
    hud.push_back(status);

    const float confidence = clampf(result.confidence, 0.0f, 1.0f);
    const float confidence_width = CONF_MAXW * confidence;
    if (confidence_width >= 2.0f) {
        OsdQR confidence_bar;
        confidence_bar.box = {
            static_cast<float>(CONF_X0), static_cast<float>(CONF_Y1),
            static_cast<float>(CONF_X0) + confidence_width,
            static_cast<float>(CONF_Y2)};
        confidence_bar.layer_id = 4;
        confidence_bar.type = fdevice::TYPE_SOLID;
        confidence_bar.alpha = fdevice::TYPE_ALPHA75;
        confidence_bar.border = 0;
        confidence_bar.color = result.is_locked
            ? GestureColor(result.human_gesture) : C_IDLE;
        hud.push_back(confidence_bar);
    }
    osd_device.Draw(hud, 4);
}

namespace utils {
    EmotionClass ArgmaxProbsEmotion(const float probs[4], float* out_conf) {
        int best_idx = 0; float best_prob = probs[0];
        for (int i = 1; i < 4; i++) {
            if (probs[i] > best_prob) { best_prob = probs[i]; best_idx = i; }
        }
        if (out_conf) *out_conf = best_prob;
        return static_cast<EmotionClass>(best_idx);
    }
    HandGestureClass ArgmaxProbsHandGesture(const float probs[6], float* out_conf) {
        int best_idx = 0; float best_prob = probs[0];
        for (int i = 1; i < 6; i++) {
            if (probs[i] > best_prob) { best_prob = probs[i]; best_idx = i; }
        }
        if (out_conf) *out_conf = best_prob;
        return static_cast<HandGestureClass>(best_idx);
    }
}

int VISUALIZER::EmotionColor(EmotionClass e) {
    switch (e) {
        case EmotionClass::SURPRISE: return C_SURPRISE;
        case EmotionClass::HAPPY:    return C_HAPPY;
        case EmotionClass::SAD:      return C_SAD;
        default:                     return C_NEUTRAL;
    }
}

void VISUALIZER::DrawEmotionIcon(int layer_id, EmotionClass emotion) {
    if (!enabled_) return;
    if (emotion != candidate_emotion) {
        candidate_emotion = emotion;
        emotion_hold_count = 1;
    } else {
        emotion_hold_count++;
    }

    if (emotion_hold_count >= 3 && candidate_emotion != last_drawn_emotion) {
        last_drawn_emotion = candidate_emotion;
        std::string filename = "";
        switch (candidate_emotion) {
            case EmotionClass::SURPRISE: filename = "surprise.ssbmp"; break;
            case EmotionClass::HAPPY:    filename = "happy.ssbmp"; break;
            case EmotionClass::SAD:      filename = "sad.ssbmp"; break;
            case EmotionClass::NEUTRAL:  filename = "neutral.ssbmp"; break;
            default:                     filename = "neutral.ssbmp"; break;
        }
        DrawBitmap(filename, "shared_colorLUT.sscl", 630, 10, layer_id);
    }
}

void VISUALIZER::DrawSimple(const EmotionResult& result, const std::array<float, 4>& hand_roi) {
    if (!enabled_) return;
    std::vector<std::array<float, 4>> roi_vec = { hand_roi };
    CommitBoxes(0, 3, C_NEUTRAL, roi_vec);
    DrawEmotionIcon(2, result.emotion);
}

int VISUALIZER::HandGestureColor(HandGestureClass g) {
    (void)g;
    return C_GESTURE1; 
}

void VISUALIZER::DrawResultLines(int layer_id_primary, int layer_id_secondary, HandGestureClass gesture, const std::array<float, 4>& hand_roi) {
    if (!enabled_) return;

    // Classes are the displayed numbers 0..5. The previous 8-pixel-high
    // boxes also used border=8, producing a degenerate inner quadrangle. The
    // common OSD safety guard correctly rejects that geometry. Keep the guard
    // and submit bars with a valid inner/outer shape instead.
    const int num_lines = std::max(0, std::min(5, static_cast<int>(gesture)));
    
    std::vector<std::array<float, 4>> boxes_primary;
    std::vector<std::array<float, 4>> boxes_secondary;

    if (num_lines > 0) {
        float roi_x1 = hand_roi[0];
        float roi_y1 = hand_roi[1];
        float roi_w = hand_roi[2] - hand_roi[0];
        const float line_spacing = 25.f;
        const float line_half_height = 7.f;

        for (int i = 0; i < num_lines; ++i) {
            float line_y = roi_y1 - 15.f - line_spacing * i;
            if (line_y < 10.f) line_y = 10.f; 
            std::array<float, 4> box = {
                roi_x1 + 20.f, line_y - line_half_height,
                roi_x1 + roi_w - 20.f, line_y + line_half_height
            };
            if (i < 4) {
                boxes_primary.push_back(box);
            } 
            else {
                boxes_secondary.push_back(box);
            }
        }
    }
    CommitBoxes(layer_id_primary, 3, C_GESTURE0, boxes_primary);
    CommitBoxes(layer_id_secondary, 3, C_GESTURE0, boxes_secondary);

    // Class zero has no tally bars, so give it a compact fixed marker. This
    // also makes a missing result layer distinguishable from a valid zero.
    if (num_lines == 0) {
        std::vector<std::array<float, 4>> zero_marker = {{
            hand_roi[0] + 20.f, hand_roi[1] - 37.f,
            hand_roi[0] + 62.f, hand_roi[1] - 19.f
        }};
        CommitBoxes(layer_id_secondary, 3, C_GESTURE1, zero_marker);
    }
}

void VISUALIZER::Draw(const HandGestureResult& result, const std::array<float, 4>& hand_roi) {
    if (!enabled_) return;
    std::vector<std::array<float, 4>> roi_vec = { hand_roi };
    CommitBoxes(0, 4, C_GESTURE1, roi_vec);
    DrawResultLines(1, 3, result.gesture, hand_roi);
}
