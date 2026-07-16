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
    static const uint8_t glyph_u[7] = {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e};
    // Keep each text scanline within the A1 OSD limit of four rectangles.
    // This simplified N uses one vertical stroke plus one diagonal stroke.
    static const uint8_t glyph_n[7] = {0x10, 0x18, 0x14, 0x12, 0x11, 0x11, 0x11};
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

void VISUALIZER::Initialize(std::array<int, 2>& in_img_shape, const std::string& bitmap_lut_path) {
    m_width = in_img_shape[0];
    m_height = in_img_shape[1];

    const char* lut_path = nullptr;
    if (!bitmap_lut_path.empty()) {
        m_bitmap_lut_path_full = "./app_assets/" + bitmap_lut_path;
        lut_path = m_bitmap_lut_path_full.c_str();
    }
    
    osd_device.Initialize(m_width, m_height, lut_path);
    enabled_ = osd_device.IsEnabled();
    
    if (!enabled_) {
        std::cerr << "[VISUALIZER] Warning: OSD device not enabled or initialization failed." << std::endl;
    }
}

void VISUALIZER::Release() {
    if (enabled_) osd_device.Release();
    enabled_ = false;
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

    for (size_t i = 0; i < result.eyes.size(); ++i) {
        const EyeBox& eye = result.eyes[i];
        const bool selected_eye = selected != nullptr &&
            (same_eye_box(eye, selected->left) || same_eye_box(eye, selected->right));
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
    if (!quads.empty()) {
        std::vector<OsdQR> empty;
        osd_device.Draw(empty, DETECTION_LAYER_ID);
    }
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
    const char* face = focus_face_bitmap();
    if (face != nullptr) {
        osd_device.ClearLayer(LAYER_FACE_ICON);
        osd_device.DrawTexture(face, nullptr, LAYER_FACE_ICON,
                               11, std::max(4, m_height - 24),
                               fdevice::TYPE_ALPHA100);
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
    osd_device.Draw(quads, LAYER_MASK);
}

void VISUALIZER::DrawSpeed(const std::vector<std::array<float, 4>>& boxes,
                           const std::vector<float>& scores,
                           const std::vector<int>& class_ids,
                           const std::vector<float>& speeds,
                           const std::vector<int>& directions,
                           int crop_offset_y,
                           int crop_height) {
    (void)scores;
    if (!enabled_) return;
    if (crop_height > 0) {
        DrawMask(crop_offset_y, crop_height);
    } else {
        std::vector<OsdQR> empty;
        osd_device.Draw(empty, LAYER_MASK);
    }

    std::vector<OsdQR> quad_rangle_vec;

    for (size_t i = 0; i < boxes.size(); i++) {
        int xmin = static_cast<int>(boxes[i][0]);
        int ymin = static_cast<int>(boxes[i][1]);
        int xmax = static_cast<int>(boxes[i][2]);
        int ymax = static_cast<int>(boxes[i][3]);
        
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
        q_box.box = {static_cast<float>(xmin), static_cast<float>(ymin), static_cast<float>(xmax), static_cast<float>(ymax)};
        q_box.border = box_border; 
        q_box.color = box_color; 
        q_box.alpha = box_alpha;
        q_box.type = fdevice::TYPE_HOLLOW; 
        q_box.layer_id = DETECTION_LAYER_ID;
        quad_rangle_vec.emplace_back(q_box);

        float speed = speeds[i];
        int dir = directions[i]; 
        if (speed > 1.0f) { 
            int center_x = (xmin + xmax) / 2;
            int arrow_y = std::max(10, ymin - 15);
            int arrow_length = 20 + static_cast<int>(speed * 2); 
            int tip_x = center_x + (dir * arrow_length);
            
            OsdQR q_stem;
            q_stem.box = {static_cast<float>(std::min(center_x, tip_x)), static_cast<float>(arrow_y - 2), 
                          static_cast<float>(std::max(center_x, tip_x)), static_cast<float>(arrow_y + 2)};
            q_stem.border = 0;
            q_stem.color = box_color;
            q_stem.alpha = box_alpha;
            q_stem.type = fdevice::TYPE_SOLID;
            q_stem.layer_id = DETECTION_LAYER_ID;
            quad_rangle_vec.emplace_back(q_stem);

            OsdQR q_tip;
            if (dir > 0) {
                q_tip.box = {static_cast<float>(tip_x - 6), static_cast<float>(arrow_y - 8), 
                             static_cast<float>(tip_x), static_cast<float>(arrow_y + 8)};
            } else {       
                q_tip.box = {static_cast<float>(tip_x), static_cast<float>(arrow_y - 8), 
                             static_cast<float>(tip_x + 6), static_cast<float>(arrow_y + 8)};
            }
            q_tip.border = 0;
            q_tip.color = box_color;
            q_tip.alpha = box_alpha;
            q_tip.type = fdevice::TYPE_SOLID;
            q_tip.layer_id = DETECTION_LAYER_ID;
            quad_rangle_vec.emplace_back(q_tip);
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
    (void)features;
    if (!enabled_) return;
    DrawMask(crop_offset_y, 540); 
    DrawObstacleRegions(obstacle_info, crop_offset_y, frame_count);
    if (obstacle_info.priority == 4) {
        std::vector<OsdQR> empty;
        osd_device.Draw(empty, LAYER_SAFEDIR);
    } else {
       std::vector<OsdQR> empty;
       osd_device.Draw(empty, LAYER_SAFEDIR); //
    }
}

void VISUALIZER::DrawObstacleRegions(const ObstacleInfo& obstacle_info, int crop_offset_y, uint32_t frame_count) {
    if (!enabled_) return;
    std::vector<OsdQR> regions;
    float region_w = (float)m_width / 3.0f;
    float crop_height = 540.0f; 

    for (int i = 0; i < ObstacleInfo::REGION_COUNT; i++) {
        if (obstacle_info.has_obstacle[i]) {
            bool flash_on = (frame_count / 5) % 2 == 0;
            
            if (flash_on) {
                float x_start = region_w * i;
                float x_end = region_w * (i + 1);
                OsdQR q;
                q.box = {x_start + 2, (float)crop_offset_y + 2,x_end - 2,(float)crop_offset_y + crop_height - 2};
                
                q.border = 12; 
                q.color = 0; 
                q.alpha = fdevice::TYPE_ALPHA100; 
                q.type = fdevice::TYPE_HOLLOW;
                regions.emplace_back(q);
            }
        }
    }
    osd_device.Draw(regions, LAYER_OBSTACLES);
}


void VISUALIZER::DrawFeaturePoints(const std::vector<FeaturePoint>& features, int crop_offset_y) {
    if (!enabled_) return;
    std::vector<OsdQR> markers;
    for (const auto& fp : features) {
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
    }
    osd_device.Draw(markers, LAYER_FEATURES);
}


void VISUALIZER::DrawSafeDirection(const ObstacleInfo& obstacle_info, int crop_offset_y) {
    if (!enabled_) return;
    std::vector<OsdQR> indicators;
    if (obstacle_info.priority != 0) {
        osd_device.Draw(indicators, LAYER_SAFEDIR); 
        return;
    }

    int safest = 0;
    float min_danger = obstacle_info.danger_level[0];
    for (int i = 1; i < ObstacleInfo::REGION_COUNT; i++) {
        if (obstacle_info.danger_level[i] < min_danger) {
            min_danger = obstacle_info.danger_level[i];
            safest = i;
        }
    }

    float region_w = (float)m_width / 3.0f;
    float indicator_y = (float)crop_offset_y + 540.0f - 60.0f;
    float indicator_h = 30.0f;
    float cx = region_w * safest + region_w / 2.0f;  
    float arrow_w = region_w * 0.4f;  

    OsdQR q;
    q.box = {cx - arrow_w, indicator_y, cx + arrow_w, indicator_y + indicator_h};
    q.border = 3;
    q.color = 1; 
    q.alpha = fdevice::TYPE_ALPHA100;
    q.type = fdevice::TYPE_HOLLOW;
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

    int finger_w  = 22;                             
    int finger_gap = (W - 5 * finger_w) / 4;        
    int finger_h  = static_cast<int>(H * 0.58f);    

    for (int f = 0; f < 5; f++) {
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
    std::vector<std::array<float,4>> boxes;
    const int W = 160;
    const int H = ICON_H;
    int x0 = cx - W / 2;
    int x1 = cx + W / 2;

    int finger_w = 28;
    int finger_h = static_cast<int>(H * 0.35f);

    int lf_x0 = x0 + 8;
    int lf_x1 = lf_x0 + finger_w;
    boxes.push_back({ (float)lf_x0, (float)y0, (float)lf_x1, (float)(y0 + finger_h) });

    int rf_x1 = x1 - 8;
    int rf_x0 = rf_x1 - finger_w;
    boxes.push_back({ (float)rf_x0, (float)y0, (float)rf_x1, (float)(y0 + finger_h) });

    int fork_y0 = y0 + finger_h;
    int fork_h  = static_cast<int>(H * 0.28f);
    int fork_y1 = fork_y0 + fork_h;

    int steps = 4;
    for (int s = 0; s < steps; s++) {
        float t0 = (float)s / steps;
        float t1 = (float)(s + 1) / steps;
        int lx0 = lf_x0 + (int)((cx - 26 - lf_x0) * t0);
        int lx1 = lf_x1 + (int)((cx -  6 - lf_x1) * t0);
        int sy0 = fork_y0 + (int)(fork_h * t0);
        int sy1 = fork_y0 + (int)(fork_h * t1);
        boxes.push_back({ (float)lx0, (float)sy0, (float)lx1, (float)sy1 });

        int rx0 = rf_x0 + (int)((cx +  6 - rf_x0) * t0);
        int rx1 = rf_x1 + (int)((cx + 26 - rf_x1) * t0);
        boxes.push_back({ (float)rx0, (float)sy0, (float)rx1, (float)sy1 });
    }

    int palm_y0 = fork_y1 + 4;
    int palm_y1 = palm_y0 + static_cast<int>(H * 0.18f);
    int palm_x0 = x0 + 10;
    int palm_x1 = x1 - 10;
    boxes.push_back({ (float)palm_x0, (float)palm_y0, (float)palm_x1, (float)palm_y1 });

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
        DrawGestureIcon(2, AI_CX, ICON_Y0, result.ai_counter, true);
    } else {
        DrawGestureIcon(2, AI_CX, ICON_Y0, GestureClass::IDLE, false);
    }

    DrawStatusBar(3, result.game_state, result.human_gesture);
    int conf_color = (result.is_locked) ? GestureColor(result.human_gesture) : C_IDLE;
    DrawConfBar(4, result.confidence, conf_color);
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
    
    int num_lines = static_cast<int>(gesture);
    
    std::vector<std::array<float, 4>> boxes_primary;
    std::vector<std::array<float, 4>> boxes_secondary;

    if (num_lines > 0) {
        float roi_x1 = hand_roi[0];
        float roi_y1 = hand_roi[1];
        float roi_w = hand_roi[2] - hand_roi[0];
        float line_spacing = 25.f; // 线条间距
        float line_thickness = 4.f; // 线条厚度

        for (int i = 0; i < num_lines; ++i) {
            float line_y = roi_y1 - 15.f - line_spacing * i;
            if (line_y < 10.f) line_y = 10.f; 
            std::array<float, 4> box = {roi_x1 + 20.f, line_y - line_thickness, roi_x1 + roi_w - 20.f, line_y + line_thickness};
            if (i < 4) {
                boxes_primary.push_back(box);
            } 
            else {
                boxes_secondary.push_back(box);
            }
        }
    }
    CommitBoxes(layer_id_primary, 8, C_GESTURE0, boxes_primary);
    CommitBoxes(layer_id_secondary, 8, C_GESTURE0, boxes_secondary);
}

void VISUALIZER::Draw(const HandGestureResult& result, const std::array<float, 4>& hand_roi) {
    if (!enabled_) return;
    std::vector<std::array<float, 4>> roi_vec = { hand_roi };
    CommitBoxes(0, 4, C_GESTURE1, roi_vec);
    DrawResultLines(1, 3, result.gesture, hand_roi);
}
