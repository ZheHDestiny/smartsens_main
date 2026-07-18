#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

namespace {

const float kEyeScoreThreshold = 0.40f;
const float kNmsThreshold = 0.45f;
const float kIdentityThreshold = 0.75f;
const int kMaxEyes = 16;
const int kMaxPairs = 4;
const int kEnrollSamples = 20;

float ClampFloat(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

void PrintByteDistribution(const char* name, const uint8_t* data, size_t bytes) {
    if (name == nullptr || data == nullptr || bytes == 0) return;
    const size_t samples = std::min<size_t>(4096, bytes);
    const size_t step = std::max<size_t>(1, bytes / samples);
    uint8_t min_value = 255;
    uint8_t max_value = 0;
    uint64_t sum = 0;
    size_t count = 0;
    for (size_t offset = 0; offset < bytes; offset += step) {
        const uint8_t value = data[offset];
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        sum += value;
        ++count;
    }
    printf("[EYEDET][INPUT] %s bytes=%zu min=%u max=%u mean=%.2f samples=%zu\n",
           name, bytes, min_value, max_value,
           count > 0 ? static_cast<float>(sum) / count : 0.0f, count);
}

}  // namespace

EyeDetFaceIdEngine::EyeDetFaceIdEngine() {
    std::memset(&det_canvas_, 0, sizeof(det_canvas_));
    std::memset(&det_input_, 0, sizeof(det_input_));
    std::memset(det_outputs_, 0, sizeof(det_outputs_));
    std::memset(&face_roi_, 0, sizeof(face_roi_));
    std::memset(&face_input_, 0, sizeof(face_input_));
    std::memset(face_output_, 0, sizeof(face_output_));
    prototype_.fill(0.0f);
}

EyeDetFaceIdEngine::~EyeDetFaceIdEngine() {
    Release();
}

bool EyeDetFaceIdEngine::Initialize(const std::string& eyedet_path,
                                    const std::string& faceid_path,
                                    int capture_w,
                                    int capture_h,
                                    const EyeDetInputConfig& eyedet_config) {
    Release();
    capture_w_ = capture_w;
    capture_h_ = capture_h;
    det_input_w_ = eyedet_config.width;
    det_input_h_ = eyedet_config.height;
    det_input_gray_ = eyedet_config.gray;
    if (capture_w_ <= 0 || capture_h_ <= 0 ||
        det_input_w_ <= 0 || det_input_h_ <= 0 ||
        det_input_w_ % 32 != 0 || det_input_h_ % 32 != 0) {
        fprintf(stderr, "[EYEDET] invalid capture/model geometry capture=%dx%d input=%dx%d\n",
                capture_w_, capture_h_, det_input_w_, det_input_h_);
        return false;
    }

    det_candidate_limit_ =
        static_cast<size_t>(det_input_w_ / 8) * (det_input_h_ / 8) +
        static_cast<size_t>(det_input_w_ / 16) * (det_input_h_ / 16) +
        static_cast<size_t>(det_input_w_ / 32) * (det_input_h_ / 32);
    det_candidates_.reserve(det_candidate_limit_);
    nms_order_.reserve(det_candidate_limit_);
    nms_keep_.reserve(kMaxEyes);
    pair_candidates_.reserve(kMaxEyes * (kMaxEyes - 1) / 2);
    if (access(eyedet_path.c_str(), R_OK) != 0) {
        fprintf(stderr, "[EYEDET] model file is not readable: %s\n", eyedet_path.c_str());
        return false;
    }

    // Contract shared by EyeDet-S and EyeDet-Flash: resize with one scale,
    // write at the upper-left corner, and zero-fill only the remaining right
    // or bottom area.  Never centre-pad, stretch, or divide by 255 here.
    letterbox_scale_ = std::min(
        static_cast<float>(det_input_w_) / static_cast<float>(capture_w_),
        static_cast<float>(det_input_h_) / static_cast<float>(capture_h_));
    letterbox_w_ = static_cast<int>(std::floor(capture_w_ * letterbox_scale_ + 0.5f));
    letterbox_h_ = static_cast<int>(std::floor(capture_h_ * letterbox_scale_ + 0.5f));
    resize_x0_.resize(letterbox_w_);
    resize_x1_.resize(letterbox_w_);
    resize_wx_.resize(letterbox_w_);
    resize_y0_.resize(letterbox_h_);
    resize_y1_.resize(letterbox_h_);
    resize_wy_.resize(letterbox_h_);

    const float inv_scale_x =
        static_cast<float>(capture_w_) / static_cast<float>(letterbox_w_);
    const float inv_scale_y =
        static_cast<float>(capture_h_) / static_cast<float>(letterbox_h_);
    for (int x = 0; x < letterbox_w_; ++x) {
        const float source_x = ClampFloat(
            (static_cast<float>(x) + 0.5f) * inv_scale_x - 0.5f,
            0.0f, static_cast<float>(capture_w_ - 1));
        const int x0 = static_cast<int>(source_x);
        resize_x0_[x] = x0;
        resize_x1_[x] = std::min(x0 + 1, capture_w_ - 1);
        resize_wx_[x] = static_cast<uint16_t>(
            ClampFloat((source_x - x0) * 256.0f + 0.5f, 0.0f, 256.0f));
    }
    for (int y = 0; y < letterbox_h_; ++y) {
        const float source_y = ClampFloat(
            (static_cast<float>(y) + 0.5f) * inv_scale_y - 0.5f,
            0.0f, static_cast<float>(capture_h_ - 1));
        const int y0 = static_cast<int>(source_y);
        resize_y0_[y] = y0;
        resize_y1_[y] = std::min(y0 + 1, capture_h_ - 1);
        resize_wy_[y] = static_cast<uint16_t>(
            ClampFloat((source_y - y0) * 256.0f + 0.5f, 0.0f, 256.0f));
    }

    eyedet_pipe_ = GetAIPreprocessPipe();
    if (eyedet_pipe_ == nullptr) {
        fprintf(stderr, "[EYEDET] GetAIPreprocessPipe failed\n");
        return false;
    }
    eyedet_model_id_ = ssne_loadmodel(const_cast<char*>(eyedet_path.c_str()), SSNE_STATIC_ALLOC);
    const int det_inputs = ssne_get_model_input_num(eyedet_model_id_);
    int det_dtype = -1;
    const int det_dtype_ret = ssne_get_model_input_dtype(eyedet_model_id_, &det_dtype);
    int det_mean[3] = {0, 0, 0};
    int det_std[3] = {0, 0, 0};
    int det_uint8 = 0;
    const int det_norm_ret = ssne_get_model_normalize_params(
        eyedet_model_id_, det_mean, det_std, &det_uint8);
    // Both EyeDet variants are calibrated with direct Y8 values in [0,255].
    // EyeDet-S expands Y8 to RGB3; EyeDet-Flash remains a single GRAY plane.
    // In either case, do not apply SDK fixed-point normalization.
    Clear(eyedet_pipe_);
    det_canvas_ = create_tensor(det_input_w_, det_input_h_, SSNE_Y_8, SSNE_BUF_AI);
    det_input_ = create_tensor(det_input_w_, det_input_h_,
                               det_input_gray_ ? SSNE_Y_8 : SSNE_RGB,
                               SSNE_BUF_AI);
    const size_t det_required_bytes = static_cast<size_t>(det_input_w_) *
        static_cast<size_t>(det_input_h_) * (det_input_gray_ ? 1u : 3u);
    eyedet_ready_ = det_inputs == 1 && det_dtype_ret == 0 &&
                    get_data(det_canvas_) != nullptr && get_data(det_input_) != nullptr &&
                    get_width(det_input_) == static_cast<uint32_t>(det_input_w_) &&
                    get_height(det_input_) == static_cast<uint32_t>(det_input_h_) &&
                    get_data_format(det_input_) == (det_input_gray_ ? SSNE_Y_8 : SSNE_RGB) &&
                    get_mem_size(det_input_) >= det_required_bytes;
    printf("[EYEDET] model=%s id=%u inputs=%d dtype=%d dtype_ret=%d "
           "norm_ret=%d pipe=clear(no_normalize) format=%s mean=%d,%d,%d std=%d,%d,%d uint8=%d ready=%d\n",
           eyedet_path.c_str(), eyedet_model_id_, det_inputs, det_dtype, det_dtype_ret,
           det_norm_ret, det_input_gray_ ? "GRAY" : "RGB", det_mean[0], det_mean[1], det_mean[2],
           det_std[0], det_std[1], det_std[2], det_uint8, eyedet_ready_ ? 1 : 0);
    printf("[INPUT][EYEDET] %ux%u dtype=%u format=%u bytes=%zu letterbox=%dx%d scale=%.6f\n",
           get_width(det_input_), get_height(det_input_), get_data_type(det_input_),
           get_data_format(det_input_), get_mem_size(det_input_), letterbox_w_, letterbox_h_,
           letterbox_scale_);
    if (!eyedet_ready_) {
        fprintf(stderr, "[EYEDET] input/model contract check failed\n");
        return false;
    }

    faceid_pipe_ = GetAIPreprocessPipe();
    if (faceid_pipe_ != nullptr && access(faceid_path.c_str(), R_OK) == 0) {
        faceid_model_id_ = ssne_loadmodel(const_cast<char*>(faceid_path.c_str()), SSNE_STATIC_ALLOC);
        const int face_inputs = ssne_get_model_input_num(faceid_model_id_);
        int face_dtype = -1;
        const int face_dtype_ret = ssne_get_model_input_dtype(faceid_model_id_, &face_dtype);
        int face_mean[3] = {0, 0, 0};
        int face_std[3] = {0, 0, 0};
        int face_uint8 = 0;
        const int face_norm_ret = ssne_get_model_normalize_params(
            faceid_model_id_, face_mean, face_std, &face_uint8);
        // FaceID-S uses the same direct Y8->RGB3 pixel-value contract.
        Clear(faceid_pipe_);
        face_roi_ = create_tensor(kFaceSize, kFaceSize, SSNE_Y_8, SSNE_BUF_AI);
        face_input_ = create_tensor(kFaceSize, kFaceSize, SSNE_RGB, SSNE_BUF_AI);
        faceid_ready_ = faceid_model_id_ != eyedet_model_id_ &&
                        face_inputs == 1 && face_dtype_ret == 0 &&
                        get_data(face_roi_) != nullptr && get_data(face_input_) != nullptr &&
                        get_width(face_input_) == kFaceSize && get_height(face_input_) == kFaceSize &&
                        get_data_format(face_input_) == SSNE_RGB &&
                        get_mem_size(face_input_) >= static_cast<size_t>(3 * kFaceSize * kFaceSize);
        printf("[FACEID] model=%s id=%u inputs=%d dtype=%d dtype_ret=%d "
               "norm_ret=%d pipe=clear(no_normalize) mean=%d,%d,%d std=%d,%d,%d uint8=%d ready=%d\n",
               faceid_path.c_str(), faceid_model_id_, face_inputs, face_dtype, face_dtype_ret,
               face_norm_ret, face_mean[0], face_mean[1], face_mean[2],
               face_std[0], face_std[1], face_std[2], face_uint8, faceid_ready_ ? 1 : 0);
        if (get_data(face_input_) != nullptr) {
            printf("[INPUT][FACEID] %ux%u dtype=%u format=%u bytes=%zu\n",
                   get_width(face_input_), get_height(face_input_), get_data_type(face_input_),
                   get_data_format(face_input_), get_mem_size(face_input_));
        }
    }
    if (!faceid_ready_) {
        fprintf(stderr, "[FACEID] unavailable; EyeDet tracking remains enabled\n");
    }

    initialized_ = true;
    return true;
}

bool EyeDetFaceIdEngine::PrepareEyeDetInput(const uint8_t* src) {
    uint8_t* canvas = static_cast<uint8_t*>(get_data(det_canvas_));
    if (src == nullptr || canvas == nullptr) return false;
    std::memset(canvas, 0, static_cast<size_t>(det_input_w_) * det_input_h_);
    for (int y = 0; y < letterbox_h_; ++y) {
        const uint8_t* row0 = src + resize_y0_[y] * capture_w_;
        const uint8_t* row1 = src + resize_y1_[y] * capture_w_;
        const uint32_t wy = resize_wy_[y];
        const uint32_t inv_wy = 256u - wy;
        uint8_t* row = canvas + y * det_input_w_;
        for (int x = 0; x < letterbox_w_; ++x) {
            const uint32_t wx = resize_wx_[x];
            const uint32_t inv_wx = 256u - wx;
            const uint32_t top =
                row0[resize_x0_[x]] * inv_wx + row0[resize_x1_[x]] * wx;
            const uint32_t bottom =
                row1[resize_x0_[x]] * inv_wx + row1[resize_x1_[x]] * wx;
            row[x] = static_cast<uint8_t>(
                (top * inv_wy + bottom * wy + 32768u) >> 16);
        }
    }
    const int ret = RunAiPreprocessPipe(eyedet_pipe_, det_canvas_, det_input_);
    if (ret == 0 && !det_preprocess_logged_) {
        PrintByteDistribution("capture_y8", src,
                              static_cast<size_t>(capture_w_ * capture_h_));
        PrintByteDistribution("letterbox_y8", canvas,
                              static_cast<size_t>(det_input_w_) * det_input_h_);
        PrintByteDistribution(det_input_gray_ ? "model_gray_storage" : "model_rgb_storage",
                              static_cast<const uint8_t*>(get_data(det_input_)),
                              get_mem_size(det_input_));
        det_preprocess_logged_ = true;
    }
    return ret == 0;
}

float EyeDetFaceIdEngine::Sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    const float e = std::exp(x);
    return e / (1.0f + e);
}

float EyeDetFaceIdEngine::IoU(const EyeBox& a, const EyeBox& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float area_a = std::max(0.0f, a.Width()) * std::max(0.0f, a.Height());
    const float area_b = std::max(0.0f, b.Width()) * std::max(0.0f, b.Height());
    const float denom = area_a + area_b - inter;
    return denom > 0.0f ? inter / denom : 0.0f;
}

float EyeDetFaceIdEngine::IntersectionOverMinArea(const EyeBox& a, const EyeBox& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float area_a = std::max(0.0f, a.Width()) * std::max(0.0f, a.Height());
    const float area_b = std::max(0.0f, b.Width()) * std::max(0.0f, b.Height());
    const float min_area = std::min(area_a, area_b);
    return min_area > 0.0f ? inter / min_area : 0.0f;
}

bool EyeDetFaceIdEngine::ValidateEyeDetOutputs() {
    const uint32_t widths[6] = {
        static_cast<uint32_t>(det_input_w_ / 8),
        static_cast<uint32_t>(det_input_w_ / 16),
        static_cast<uint32_t>(det_input_w_ / 32),
        static_cast<uint32_t>(det_input_w_ / 8),
        static_cast<uint32_t>(det_input_w_ / 16),
        static_cast<uint32_t>(det_input_w_ / 32)};
    const uint32_t heights[6] = {
        static_cast<uint32_t>(det_input_h_ / 8),
        static_cast<uint32_t>(det_input_h_ / 16),
        static_cast<uint32_t>(det_input_h_ / 32),
        static_cast<uint32_t>(det_input_h_ / 8),
        static_cast<uint32_t>(det_input_h_ / 16),
        static_cast<uint32_t>(det_input_h_ / 32)};
    size_t bytes[6] = {};
    for (int i = 0; i < 6; ++i) {
        const size_t channels = i < 3 ? 1u : 64u;
        bytes[i] = static_cast<size_t>(widths[i]) * heights[i] * channels * sizeof(float);
    }
    bool valid = true;
    for (int i = 0; i < 6; ++i) {
        const bool item_valid = get_data(det_outputs_[i]) != nullptr &&
                                get_width(det_outputs_[i]) == widths[i] &&
                                get_height(det_outputs_[i]) == heights[i] &&
                                get_data_type(det_outputs_[i]) == SSNE_FLOAT32 &&
                                get_mem_size(det_outputs_[i]) >= bytes[i];
        valid = valid && item_valid;
        if (!det_contract_logged_ || !item_valid) {
            printf("[OUTPUT][EYEDET][%d] %ux%u dtype=%u format=%u bytes=%zu expected>=%zu ok=%d\n",
                   i, get_width(det_outputs_[i]), get_height(det_outputs_[i]),
                   get_data_type(det_outputs_[i]), get_data_format(det_outputs_[i]),
                   get_mem_size(det_outputs_[i]), bytes[i], item_valid ? 1 : 0);
        }
    }
    det_contract_logged_ = true;
    return valid;
}

void EyeDetFaceIdEngine::DecodeHead(const float* cls,
                                    const float* reg,
                                    int width,
                                    int height,
                                    int stride,
                                    std::vector<EyeBox>* candidates,
                                    float* max_class_score) const {
    if (cls == nullptr || reg == nullptr || candidates == nullptr) return;
    float softmax[16];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int spatial = y * width + x;
            if (candidates->size() >= det_candidate_limit_) return;
            if (!std::isfinite(cls[spatial])) continue;
            const float score = Sigmoid(cls[spatial]);
            if (max_class_score != nullptr) {
                *max_class_score = std::max(*max_class_score, score);
            }
            if (score < kEyeScoreThreshold) continue;
            float dist[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            bool regression_valid = true;
            for (int side = 0; side < 4; ++side) {
                const float* logits = reg + spatial * 64 + side * 16;
                for (int bin = 0; bin < 16; ++bin) {
                    if (!std::isfinite(logits[bin])) {
                        regression_valid = false;
                        break;
                    }
                }
                if (!regression_valid) break;
                float max_logit = logits[0];
                for (int bin = 1; bin < 16; ++bin) max_logit = std::max(max_logit, logits[bin]);
                float sum = 0.0f;
                for (int bin = 0; bin < 16; ++bin) {
                    softmax[bin] = std::exp(std::max(-20.0f, logits[bin] - max_logit));
                    sum += softmax[bin];
                }
                if (!std::isfinite(sum) || sum <= 1e-12f) {
                    regression_valid = false;
                    break;
                }
                for (int bin = 0; bin < 16; ++bin) {
                    dist[side] += static_cast<float>(bin) * softmax[bin] / sum;
                }
            }
            if (!regression_valid ||
                !std::isfinite(dist[0]) || !std::isfinite(dist[1]) ||
                !std::isfinite(dist[2]) || !std::isfinite(dist[3])) {
                continue;
            }
            const float ax = static_cast<float>(x) + 0.5f;
            const float ay = static_cast<float>(y) + 0.5f;
            EyeBox box;
            box.x1 = ClampFloat((ax - dist[0]) * stride, 0.0f, static_cast<float>(det_input_w_));
            box.y1 = ClampFloat((ay - dist[1]) * stride, 0.0f, static_cast<float>(det_input_h_));
            box.x2 = ClampFloat((ax + dist[2]) * stride, 0.0f, static_cast<float>(det_input_w_));
            box.y2 = ClampFloat((ay + dist[3]) * stride, 0.0f, static_cast<float>(det_input_h_));
            box.score = score;
            if (box.Width() >= 2.0f && box.Height() >= 2.0f) candidates->push_back(box);
        }
    }
}

void EyeDetFaceIdEngine::NmsAndMap(const std::vector<EyeBox>& candidates,
                                   std::vector<EyeBox>* eyes) {
    eyes->clear();
    nms_order_.resize(candidates.size());
    std::iota(nms_order_.begin(), nms_order_.end(), 0);
    std::sort(nms_order_.begin(), nms_order_.end(), [&](int a, int b) {
        return candidates[a].score > candidates[b].score;
    });
    nms_keep_.clear();
    for (size_t i = 0; i < nms_order_.size() && nms_keep_.size() < kMaxEyes; ++i) {
        const int idx = nms_order_[i];
        bool suppressed = false;
        for (size_t k = 0; k < nms_keep_.size(); ++k) {
            const EyeBox& candidate = candidates[idx];
            const EyeBox& selected = candidates[nms_keep_[k]];
            const bool overlap_duplicate = IoU(candidate, selected) > kNmsThreshold;
            const bool nested_duplicate =
                IntersectionOverMinArea(candidate, selected) > 0.75f;
            if (overlap_duplicate || nested_duplicate) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) nms_keep_.push_back(idx);
    }
    eyes->reserve(nms_keep_.size());
    for (size_t i = 0; i < nms_keep_.size(); ++i) {
        EyeBox box = candidates[nms_keep_[i]];
        // Padding is always at the right/bottom. Do not map detections whose
        // centre belongs to the zero-filled region back onto a capture edge.
        if (box.Cx() >= static_cast<float>(letterbox_w_) ||
            box.Cy() >= static_cast<float>(letterbox_h_)) {
            continue;
        }
        box.x1 = ClampFloat(box.x1 / letterbox_scale_, 0.0f, static_cast<float>(capture_w_));
        box.y1 = ClampFloat(box.y1 / letterbox_scale_, 0.0f, static_cast<float>(capture_h_));
        box.x2 = ClampFloat(box.x2 / letterbox_scale_, 0.0f, static_cast<float>(capture_w_));
        box.y2 = ClampFloat(box.y2 / letterbox_scale_, 0.0f, static_cast<float>(capture_h_));
        if (box.Width() >= 2.0f && box.Height() >= 2.0f) eyes->push_back(box);
    }
}

void EyeDetFaceIdEngine::PairEyes(const std::vector<EyeBox>& eyes,
                                  std::vector<EyePair>* pairs) {
    pairs->clear();
    pair_candidates_.clear();
    for (size_t i = 0; i < eyes.size(); ++i) {
        for (size_t j = i + 1; j < eyes.size(); ++j) {
            EyePair pair;
            pair.left = eyes[i].Cx() <= eyes[j].Cx() ? eyes[i] : eyes[j];
            pair.right = eyes[i].Cx() <= eyes[j].Cx() ? eyes[j] : eyes[i];
            const float dx = pair.right.Cx() - pair.left.Cx();
            const float dy = pair.right.Cy() - pair.left.Cy();
            const float distance = std::sqrt(dx * dx + dy * dy);
            const float mean_width = 0.5f * (pair.left.Width() + pair.right.Width());
            const float width_ratio = pair.left.Width() / std::max(1.0f, pair.right.Width());
            if (distance < 10.0f || mean_width <= 0.0f) continue;
            if (distance / mean_width < 1.25f || distance / mean_width > 6.5f) continue;
            if (std::fabs(dy) > 0.40f * distance) continue;
            if (width_ratio < 0.45f || width_ratio > 2.2f) continue;
            pair.score = pair.left.score + pair.right.score + 0.002f * distance;
            pair_candidates_.push_back(pair);
        }
    }
    std::sort(pair_candidates_.begin(), pair_candidates_.end(), [](const EyePair& a, const EyePair& b) {
        return a.score > b.score;
    });
    bool used[kMaxEyes] = {};
    for (size_t c = 0; c < pair_candidates_.size() && pairs->size() < kMaxPairs; ++c) {
        int left_idx = -1;
        int right_idx = -1;
        for (size_t i = 0; i < eyes.size(); ++i) {
            if (eyes[i].x1 == pair_candidates_[c].left.x1 &&
                eyes[i].y1 == pair_candidates_[c].left.y1) left_idx = i;
            if (eyes[i].x1 == pair_candidates_[c].right.x1 &&
                eyes[i].y1 == pair_candidates_[c].right.y1) right_idx = i;
        }
        if (left_idx < 0 || right_idx < 0 || used[left_idx] || used[right_idx]) continue;
        used[left_idx] = true;
        used[right_idx] = true;
        pairs->push_back(pair_candidates_[c]);
    }
}

void EyeDetFaceIdEngine::SelectPair(EyeDetResult* result) {
    result->selected_index = -1;
    if (result->pairs.empty()) {
        if (selected_valid_) ++selected_lost_frames_;
        if (selected_lost_frames_ > 15) {
            selected_valid_ = false;
            enrolling_ = false;
            enroll_track_id_ = 0;
            enroll_samples_.clear();
        }
        result->lost_frames = selected_lost_frames_;
        return;
    }

    int selected = 0;
    if (selected_valid_) {
        float best_distance = 1e9f;
        for (size_t i = 0; i < result->pairs.size(); ++i) {
            const float dx = result->pairs[i].Cx() - selected_cx_;
            const float dy = result->pairs[i].Cy() - selected_cy_;
            const float distance = std::sqrt(dx * dx + dy * dy);
            if (distance < best_distance) {
                best_distance = distance;
                selected = static_cast<int>(i);
            }
        }
        const float association_limit = std::max(80.0f, 1.5f * selected_eye_distance_);
        if (best_distance > association_limit) {
            if (enrolling_) {
                ++selected_lost_frames_;
                result->lost_frames = selected_lost_frames_;
                return;
            }
            selected_valid_ = false;
        }
    }
    if (!selected_valid_) {
        selected_track_id_ = next_track_id_++;
        selected = 0;
    }
    selected_valid_ = true;
    selected_lost_frames_ = 0;
    selected_cx_ = result->pairs[selected].Cx();
    selected_cy_ = result->pairs[selected].Cy();
    selected_eye_distance_ = result->pairs[selected].EyeDistance();
    result->pairs[selected].track_id = selected_track_id_;
    result->selected_index = selected;
    result->lost_frames = 0;
}

bool EyeDetFaceIdEngine::DetectEyes(ssne_tensor_t* capture_y8, EyeDetResult* out) {
    if (out == nullptr) return false;
    out->Clear();
    if (!eyedet_ready_ || capture_y8 == nullptr) return false;
    const auto total_begin = std::chrono::steady_clock::now();
    const uint8_t* src = static_cast<const uint8_t*>(get_data(*capture_y8));
    const size_t capture_bytes = static_cast<size_t>(capture_w_) * capture_h_;
    const bool capture_valid = src != nullptr &&
        get_width(*capture_y8) == static_cast<uint32_t>(capture_w_) &&
        get_height(*capture_y8) == static_cast<uint32_t>(capture_h_) &&
        get_data_format(*capture_y8) == SSNE_Y_8 &&
        get_mem_size(*capture_y8) >= capture_bytes;
    if (!capture_valid || !PrepareEyeDetInput(src)) {
        ++det_failures_;
        return false;
    }
    const auto preprocess_end = std::chrono::steady_clock::now();
    out->preprocess_ms =
        std::chrono::duration<float, std::milli>(preprocess_end - total_begin).count();
    const auto begin = preprocess_end;
    ssne_tensor_t input[1] = {det_input_};
    const int infer_ret = ssne_inference(eyedet_model_id_, 1, input);
    if (infer_ret != 0) {
        fprintf(stderr, "[EYEDET] ssne_inference failed ret=%d\n", infer_ret);
        ++det_failures_;
        return false;
    }
    const int output_ret = ssne_getoutput(eyedet_model_id_, 6, det_outputs_);
    const auto end = std::chrono::steady_clock::now();
    out->npu_ms = std::chrono::duration<float, std::milli>(end - begin).count();
    ++det_runs_;
    if (output_ret != 0 || !ValidateEyeDetOutputs()) {
        fprintf(stderr, "[EYEDET] ssne_getoutput/contract failed ret=%d\n", output_ret);
        ++det_failures_;
        return false;
    }

    det_candidates_.clear();
    DecodeHead(static_cast<const float*>(get_data(det_outputs_[0])),
               static_cast<const float*>(get_data(det_outputs_[3])),
               det_input_w_ / 8, det_input_h_ / 8, 8, &det_candidates_,
               &out->max_class_score);
    DecodeHead(static_cast<const float*>(get_data(det_outputs_[1])),
               static_cast<const float*>(get_data(det_outputs_[4])),
               det_input_w_ / 16, det_input_h_ / 16, 16, &det_candidates_,
               &out->max_class_score);
    DecodeHead(static_cast<const float*>(get_data(det_outputs_[2])),
               static_cast<const float*>(get_data(det_outputs_[5])),
               det_input_w_ / 32, det_input_h_ / 32, 32, &det_candidates_,
               &out->max_class_score);
    out->candidates_before_nms = static_cast<int>(det_candidates_.size());
    NmsAndMap(det_candidates_, &out->eyes);
    PairEyes(out->eyes, &out->pairs);
    SelectPair(out);
    const auto total_end = std::chrono::steady_clock::now();
    out->postprocess_ms =
        std::chrono::duration<float, std::milli>(total_end - end).count();
    out->total_ms =
        std::chrono::duration<float, std::milli>(total_end - total_begin).count();
    return true;
}

bool EyeDetFaceIdEngine::BuildFaceRoi(const uint8_t* src,
                                      int width,
                                      int height,
                                      const EyePair& pair) {
    if (src == nullptr || width <= 0 || height <= 0) return false;
    const float lx = pair.left.Cx();
    const float ly = pair.left.Cy();
    const float rx = pair.right.Cx();
    const float ry = pair.right.Cy();
    const float dx = rx - lx;
    const float dy = ry - ly;
    const float distance = std::sqrt(dx * dx + dy * dy);
    if (distance < 16.0f || std::fabs(dy) > 0.40f * distance) return false;

    const float cx = 0.5f * (lx + rx);
    const float cy = 0.5f * (ly + ry);
    const float raw_x1 = cx - 1.45f * distance;
    const float raw_y1 = cy - 0.90f * distance;
    const float raw_x2 = cx + 1.45f * distance;
    const float raw_y2 = cy + 2.20f * distance;
    const float side = std::max(raw_x2 - raw_x1, raw_y2 - raw_y1);
    if (side < 24.0f) return false;
    const float roi_cx = 0.5f * (raw_x1 + raw_x2);
    const float roi_cy = 0.5f * (raw_y1 + raw_y2);
    const float roi_x1 = roi_cx - 0.5f * side;
    const float roi_y1 = roi_cy - 0.5f * side;

    int histogram[256] = {0};
    int histogram_count = 0;
    const int clip_x1 = std::max(0, static_cast<int>(std::floor(roi_x1)));
    const int clip_y1 = std::max(0, static_cast<int>(std::floor(roi_y1)));
    const int clip_x2 = std::min(width, static_cast<int>(std::ceil(roi_x1 + side)));
    const int clip_y2 = std::min(height, static_cast<int>(std::ceil(roi_y1 + side)));
    for (int y = clip_y1; y < clip_y2; ++y) {
        for (int x = clip_x1; x < clip_x2; ++x) {
            ++histogram[src[y * width + x]];
            ++histogram_count;
        }
    }
    if (histogram_count < 64) return false;
    int cumulative = 0;
    int median = 0;
    for (; median < 256; ++median) {
        cumulative += histogram[median];
        if (cumulative * 2 >= histogram_count) break;
    }

    uint8_t* dst = static_cast<uint8_t*>(get_data(face_roi_));
    if (dst == nullptr) return false;
    const float step = side / static_cast<float>(kFaceSize);
    for (int oy = 0; oy < kFaceSize; ++oy) {
        for (int ox = 0; ox < kFaceSize; ++ox) {
            const float sx0f = roi_x1 + ox * step;
            const float sy0f = roi_y1 + oy * step;
            const float sx1f = roi_x1 + (ox + 1) * step;
            const float sy1f = roi_y1 + (oy + 1) * step;
            const int sx0 = static_cast<int>(std::floor(sx0f));
            const int sy0 = static_cast<int>(std::floor(sy0f));
            const int sx1 = std::max(sx0 + 1, static_cast<int>(std::ceil(sx1f)));
            const int sy1 = std::max(sy0 + 1, static_cast<int>(std::ceil(sy1f)));
            int sum = 0;
            int count = 0;
            for (int sy = sy0; sy < sy1; ++sy) {
                for (int sx = sx0; sx < sx1; ++sx) {
                    sum += (sx >= 0 && sx < width && sy >= 0 && sy < height)
                               ? src[sy * width + sx]
                               : median;
                    ++count;
                }
            }
            dst[oy * kFaceSize + ox] = static_cast<uint8_t>(sum / std::max(1, count));
        }
    }
    return RunAiPreprocessPipe(faceid_pipe_, face_roi_, face_input_) == 0;
}

bool EyeDetFaceIdEngine::Normalize(std::array<float, kEmbeddingSize>* value) {
    float sum = 0.0f;
    for (int i = 0; i < kEmbeddingSize; ++i) sum += (*value)[i] * (*value)[i];
    if (!std::isfinite(sum) || sum < 1e-12f) return false;
    const float inv = 1.0f / std::sqrt(sum);
    for (int i = 0; i < kEmbeddingSize; ++i) (*value)[i] *= inv;
    return true;
}

float EyeDetFaceIdEngine::Cosine(const std::array<float, kEmbeddingSize>& a,
                                 const std::array<float, kEmbeddingSize>& b) {
    float sum = 0.0f;
    for (int i = 0; i < kEmbeddingSize; ++i) sum += a[i] * b[i];
    return sum;
}

bool EyeDetFaceIdEngine::ReadEmbedding(std::array<float, kEmbeddingSize>* embedding) {
    if (embedding == nullptr || get_data(face_output_[0]) == nullptr ||
        get_data_type(face_output_[0]) != SSNE_FLOAT32 ||
        get_mem_size(face_output_[0]) < static_cast<size_t>(kEmbeddingSize * sizeof(float))) {
        return false;
    }
    if (!face_contract_logged_) {
        printf("[OUTPUT][FACEID] %ux%u dtype=%u format=%u bytes=%zu expected>=%zu\n",
               get_width(face_output_[0]), get_height(face_output_[0]),
               get_data_type(face_output_[0]), get_data_format(face_output_[0]),
               get_mem_size(face_output_[0]), static_cast<size_t>(kEmbeddingSize * sizeof(float)));
        face_contract_logged_ = true;
    }
    const float* data = static_cast<const float*>(get_data(face_output_[0]));
    for (int i = 0; i < kEmbeddingSize; ++i) (*embedding)[i] = data[i];
    return Normalize(embedding);
}

void EyeDetFaceIdEngine::ConsumeEnrollment(
    const std::array<float, kEmbeddingSize>& embedding) {
    if (!enrolling_) return;
    if (!enroll_samples_.empty() && Cosine(enroll_samples_.back(), embedding) > 0.9999f) return;
    enroll_samples_.push_back(embedding);
    printf("[FACEID] enrollment sample %zu/%d track=%llu\n", enroll_samples_.size(),
           kEnrollSamples, static_cast<unsigned long long>(enroll_track_id_));
    if (enroll_samples_.size() < kEnrollSamples) return;

    std::array<float, kEmbeddingSize> initial;
    initial.fill(0.0f);
    for (size_t i = 0; i < enroll_samples_.size(); ++i) {
        for (int d = 0; d < kEmbeddingSize; ++d) initial[d] += enroll_samples_[i][d];
    }
    Normalize(&initial);
    std::vector<std::pair<float, size_t>> ranked;
    for (size_t i = 0; i < enroll_samples_.size(); ++i) {
        ranked.push_back(std::make_pair(Cosine(initial, enroll_samples_[i]), i));
    }
    std::sort(ranked.begin(), ranked.end(), [](const std::pair<float, size_t>& a,
                                                const std::pair<float, size_t>& b) {
        return a.first > b.first;
    });
    const size_t keep = ranked.size() - ranked.size() / 5;
    prototype_.fill(0.0f);
    for (size_t i = 0; i < keep; ++i) {
        const std::array<float, kEmbeddingSize>& sample = enroll_samples_[ranked[i].second];
        for (int d = 0; d < kEmbeddingSize; ++d) prototype_[d] += sample[d];
    }
    prototype_valid_ = Normalize(&prototype_);
    enrolling_ = false;
    printf("[FACEID] enrollment complete kept=%zu/%zu prototype=%d threshold=%.2f\n",
           keep, ranked.size(), prototype_valid_ ? 1 : 0, kIdentityThreshold);
}

bool EyeDetFaceIdEngine::Identify(const uint8_t* capture_y8,
                                  int width,
                                  int height,
                                  const EyePair& pair,
                                  IdentityResult* out) {
    if (out == nullptr) return false;
    out->Clear();
    out->enrolled = prototype_valid_;
    out->enrolled_count = EnrollmentCount();
    if (!faceid_ready_ ||
        (enrolling_ && (pair.track_id == 0 || pair.track_id != enroll_track_id_)) ||
        !BuildFaceRoi(capture_y8, width, height, pair)) {
        return false;
    }
    const auto begin = std::chrono::steady_clock::now();
    ssne_tensor_t input[1] = {face_input_};
    const int infer_ret = ssne_inference(faceid_model_id_, 1, input);
    if (infer_ret != 0) {
        ++face_failures_;
        return false;
    }
    const int output_ret = ssne_getoutput(faceid_model_id_, 1, face_output_);
    const auto end = std::chrono::steady_clock::now();
    out->npu_ms = std::chrono::duration<float, std::milli>(end - begin).count();
    ++face_runs_;
    std::array<float, kEmbeddingSize> embedding;
    if (output_ret != 0 || !ReadEmbedding(&embedding)) {
        fprintf(stderr, "[FACEID] ssne_getoutput/embedding failed ret=%d\n", output_ret);
        ++face_failures_;
        return false;
    }
    ConsumeEnrollment(embedding);
    out->enrolled_count = EnrollmentCount();
    out->enrolled = prototype_valid_;
    if (prototype_valid_) {
        out->similarity = Cosine(prototype_, embedding);
        out->valid = out->similarity >= kIdentityThreshold;
        out->label = out->valid ? "id_tmp" : "unknown";
    } else {
        out->label = enrolling_ ? "enrolling" : "unknown";
    }
    return true;
}

bool EyeDetFaceIdEngine::BeginEnroll() {
    if (!faceid_ready_ || !selected_valid_ || selected_track_id_ == 0) return false;
    ClearEnrollment();
    enroll_samples_.reserve(kEnrollSamples);
    enrolling_ = true;
    enroll_track_id_ = selected_track_id_;
    printf("[FACEID] begin enrollment track=%llu target_samples=%d\n",
           static_cast<unsigned long long>(enroll_track_id_), kEnrollSamples);
    return true;
}

void EyeDetFaceIdEngine::ClearEnrollment() {
    enrolling_ = false;
    enroll_track_id_ = 0;
    // Enrollment is a short-lived transaction. Release its retained heap
    // capacity as well as its logical contents so repeated E/C cycles return
    // to the same memory baseline.
    std::vector<std::array<float, kEmbeddingSize>>().swap(enroll_samples_);
    prototype_.fill(0.0f);
    prototype_valid_ = false;
}

void EyeDetFaceIdEngine::ResetSession() {
    selected_valid_ = false;
    selected_track_id_ = 0;
    selected_cx_ = 0.0f;
    selected_cy_ = 0.0f;
    selected_eye_distance_ = 0.0f;
    selected_lost_frames_ = 0;
    ClearEnrollment();
}

void EyeDetFaceIdEngine::Release() {
    ResetSession();
    if (det_canvas_.data != nullptr) release_tensor(det_canvas_);
    if (det_input_.data != nullptr) release_tensor(det_input_);
    if (face_roi_.data != nullptr) release_tensor(face_roi_);
    if (face_input_.data != nullptr) release_tensor(face_input_);
    std::memset(&det_canvas_, 0, sizeof(det_canvas_));
    std::memset(&det_input_, 0, sizeof(det_input_));
    std::memset(&face_roi_, 0, sizeof(face_roi_));
    std::memset(&face_input_, 0, sizeof(face_input_));
    for (int i = 0; i < 6; ++i) det_outputs_[i].data = nullptr;
    face_output_[0].data = nullptr;
    if (eyedet_pipe_ != nullptr) ReleaseAIPreprocessPipe(eyedet_pipe_);
    if (faceid_pipe_ != nullptr) ReleaseAIPreprocessPipe(faceid_pipe_);
    eyedet_pipe_ = nullptr;
    faceid_pipe_ = nullptr;
    initialized_ = false;
    eyedet_ready_ = false;
    faceid_ready_ = false;
    det_contract_logged_ = false;
    det_preprocess_logged_ = false;
    face_contract_logged_ = false;
    resize_x0_.clear();
    resize_x1_.clear();
    resize_y0_.clear();
    resize_y1_.clear();
    resize_wx_.clear();
    resize_wy_.clear();
    det_candidate_limit_ = 0;
    std::vector<EyeBox>().swap(det_candidates_);
    std::vector<int>().swap(nms_order_);
    std::vector<int>().swap(nms_keep_);
    std::vector<EyePair>().swap(pair_candidates_);
}
