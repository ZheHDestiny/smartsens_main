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

uint8_t BilinearGray(const uint8_t* src, int width, int height, float x, float y) {
    x = ClampFloat(x, 0.0f, static_cast<float>(width - 1));
    y = ClampFloat(y, 0.0f, static_cast<float>(height - 1));
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const float top = src[y0 * width + x0] * (1.0f - fx) + src[y0 * width + x1] * fx;
    const float bottom = src[y1 * width + x0] * (1.0f - fx) + src[y1 * width + x1] * fx;
    return static_cast<uint8_t>(ClampFloat(top * (1.0f - fy) + bottom * fy + 0.5f, 0.0f, 255.0f));
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
                                    int capture_h) {
    Release();
    capture_w_ = capture_w;
    capture_h_ = capture_h;
    if (capture_w_ <= 0 || capture_h_ <= 0) return false;
    if (access(eyedet_path.c_str(), R_OK) != 0) {
        fprintf(stderr, "[EYEDET] model file is not readable: %s\n", eyedet_path.c_str());
        return false;
    }

    letterbox_scale_ = static_cast<float>(kDetSize) /
                       static_cast<float>(std::max(capture_w_, capture_h_));
    letterbox_w_ = static_cast<int>(std::floor(capture_w_ * letterbox_scale_ + 0.5f));
    letterbox_h_ = static_cast<int>(std::floor(capture_h_ * letterbox_scale_ + 0.5f));

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
    const int det_set_norm_ret = SetNormalize(eyedet_pipe_, eyedet_model_id_);
    det_canvas_ = create_tensor(kDetSize, kDetSize, SSNE_Y_8, SSNE_BUF_AI);
    det_input_ = create_tensor(kDetSize, kDetSize, SSNE_RGB, SSNE_BUF_AI);
    eyedet_ready_ = det_inputs == 1 && det_dtype_ret == 0 && det_set_norm_ret == 0 &&
                    get_data(det_canvas_) != nullptr && get_data(det_input_) != nullptr &&
                    get_width(det_input_) == kDetSize && get_height(det_input_) == kDetSize &&
                    get_data_format(det_input_) == SSNE_RGB &&
                    get_mem_size(det_input_) >= static_cast<size_t>(3 * kDetSize * kDetSize);
    printf("[EYEDET] model=%s id=%u inputs=%d dtype=%d dtype_ret=%d "
           "norm_ret=%d set_norm_ret=%d mean=%d,%d,%d std=%d,%d,%d uint8=%d ready=%d\n",
           eyedet_path.c_str(), eyedet_model_id_, det_inputs, det_dtype, det_dtype_ret,
           det_norm_ret, det_set_norm_ret, det_mean[0], det_mean[1], det_mean[2],
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
        const int face_set_norm_ret = SetNormalize(faceid_pipe_, faceid_model_id_);
        face_roi_ = create_tensor(kFaceSize, kFaceSize, SSNE_Y_8, SSNE_BUF_AI);
        face_input_ = create_tensor(kFaceSize, kFaceSize, SSNE_RGB, SSNE_BUF_AI);
        faceid_ready_ = faceid_model_id_ != eyedet_model_id_ &&
                        face_inputs == 1 && face_dtype_ret == 0 && face_set_norm_ret == 0 &&
                        get_data(face_roi_) != nullptr && get_data(face_input_) != nullptr &&
                        get_width(face_input_) == kFaceSize && get_height(face_input_) == kFaceSize &&
                        get_data_format(face_input_) == SSNE_RGB &&
                        get_mem_size(face_input_) >= static_cast<size_t>(3 * kFaceSize * kFaceSize);
        printf("[FACEID] model=%s id=%u inputs=%d dtype=%d dtype_ret=%d "
               "norm_ret=%d set_norm_ret=%d mean=%d,%d,%d std=%d,%d,%d uint8=%d ready=%d\n",
               faceid_path.c_str(), faceid_model_id_, face_inputs, face_dtype, face_dtype_ret,
               face_norm_ret, face_set_norm_ret, face_mean[0], face_mean[1], face_mean[2],
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
    std::memset(canvas, 0, static_cast<size_t>(kDetSize * kDetSize));
    const float inv_scale_x = static_cast<float>(capture_w_) / static_cast<float>(letterbox_w_);
    const float inv_scale_y = static_cast<float>(capture_h_) / static_cast<float>(letterbox_h_);
    for (int y = 0; y < letterbox_h_; ++y) {
        const float sy = (static_cast<float>(y) + 0.5f) * inv_scale_y - 0.5f;
        uint8_t* row = canvas + y * kDetSize;
        for (int x = 0; x < letterbox_w_; ++x) {
            const float sx = (static_cast<float>(x) + 0.5f) * inv_scale_x - 0.5f;
            row[x] = BilinearGray(src, capture_w_, capture_h_, sx, sy);
        }
    }
    return RunAiPreprocessPipe(eyedet_pipe_, det_canvas_, det_input_) == 0;
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

bool EyeDetFaceIdEngine::ValidateEyeDetOutputs() {
    const uint32_t widths[6] = {80, 40, 20, 80, 40, 20};
    const uint32_t heights[6] = {80, 40, 20, 80, 40, 20};
    const size_t bytes[6] = {
        80u * 80u * 4u, 40u * 40u * 4u, 20u * 20u * 4u,
        80u * 80u * 64u * 4u, 40u * 40u * 64u * 4u, 20u * 20u * 64u * 4u};
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
                                    std::vector<EyeBox>* candidates) const {
    if (cls == nullptr || reg == nullptr || candidates == nullptr) return;
    float softmax[16];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int spatial = y * width + x;
            const float score = Sigmoid(cls[spatial]);
            if (score < kEyeScoreThreshold) continue;
            float dist[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int side = 0; side < 4; ++side) {
                const float* logits = reg + spatial * 64 + side * 16;
                float max_logit = logits[0];
                for (int bin = 1; bin < 16; ++bin) max_logit = std::max(max_logit, logits[bin]);
                float sum = 0.0f;
                for (int bin = 0; bin < 16; ++bin) {
                    softmax[bin] = std::exp(std::max(-20.0f, logits[bin] - max_logit));
                    sum += softmax[bin];
                }
                for (int bin = 0; bin < 16; ++bin) {
                    dist[side] += static_cast<float>(bin) * softmax[bin] / sum;
                }
            }
            const float ax = static_cast<float>(x) + 0.5f;
            const float ay = static_cast<float>(y) + 0.5f;
            EyeBox box;
            box.x1 = ClampFloat((ax - dist[0]) * stride, 0.0f, static_cast<float>(kDetSize));
            box.y1 = ClampFloat((ay - dist[1]) * stride, 0.0f, static_cast<float>(kDetSize));
            box.x2 = ClampFloat((ax + dist[2]) * stride, 0.0f, static_cast<float>(kDetSize));
            box.y2 = ClampFloat((ay + dist[3]) * stride, 0.0f, static_cast<float>(kDetSize));
            box.score = score;
            if (box.Width() >= 2.0f && box.Height() >= 2.0f) candidates->push_back(box);
        }
    }
}

void EyeDetFaceIdEngine::NmsAndMap(const std::vector<EyeBox>& candidates,
                                   std::vector<EyeBox>* eyes) const {
    eyes->clear();
    std::vector<int> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return candidates[a].score > candidates[b].score;
    });
    std::vector<int> keep;
    for (size_t i = 0; i < order.size() && keep.size() < kMaxEyes; ++i) {
        const int idx = order[i];
        bool suppressed = false;
        for (size_t k = 0; k < keep.size(); ++k) {
            if (IoU(candidates[idx], candidates[keep[k]]) > kNmsThreshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) keep.push_back(idx);
    }
    eyes->reserve(keep.size());
    for (size_t i = 0; i < keep.size(); ++i) {
        EyeBox box = candidates[keep[i]];
        box.x1 = ClampFloat(box.x1 / letterbox_scale_, 0.0f, static_cast<float>(capture_w_));
        box.y1 = ClampFloat(box.y1 / letterbox_scale_, 0.0f, static_cast<float>(capture_h_));
        box.x2 = ClampFloat(box.x2 / letterbox_scale_, 0.0f, static_cast<float>(capture_w_));
        box.y2 = ClampFloat(box.y2 / letterbox_scale_, 0.0f, static_cast<float>(capture_h_));
        if (box.Width() >= 2.0f && box.Height() >= 2.0f) eyes->push_back(box);
    }
}

void EyeDetFaceIdEngine::PairEyes(const std::vector<EyeBox>& eyes,
                                  std::vector<EyePair>* pairs) const {
    pairs->clear();
    std::vector<EyePair> candidates;
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
            candidates.push_back(pair);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const EyePair& a, const EyePair& b) {
        return a.score > b.score;
    });
    std::vector<bool> used(eyes.size(), false);
    for (size_t c = 0; c < candidates.size() && pairs->size() < kMaxPairs; ++c) {
        int left_idx = -1;
        int right_idx = -1;
        for (size_t i = 0; i < eyes.size(); ++i) {
            if (eyes[i].x1 == candidates[c].left.x1 && eyes[i].y1 == candidates[c].left.y1) left_idx = i;
            if (eyes[i].x1 == candidates[c].right.x1 && eyes[i].y1 == candidates[c].right.y1) right_idx = i;
        }
        if (left_idx < 0 || right_idx < 0 || used[left_idx] || used[right_idx]) continue;
        used[left_idx] = true;
        used[right_idx] = true;
        pairs->push_back(candidates[c]);
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
    const uint8_t* src = static_cast<const uint8_t*>(get_data(*capture_y8));
    if (src == nullptr || !PrepareEyeDetInput(src)) {
        ++det_failures_;
        return false;
    }
    const auto begin = std::chrono::steady_clock::now();
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

    std::vector<EyeBox> candidates;
    candidates.reserve(256);
    DecodeHead(static_cast<const float*>(get_data(det_outputs_[0])),
               static_cast<const float*>(get_data(det_outputs_[3])), 80, 80, 8, &candidates);
    DecodeHead(static_cast<const float*>(get_data(det_outputs_[1])),
               static_cast<const float*>(get_data(det_outputs_[4])), 40, 40, 16, &candidates);
    DecodeHead(static_cast<const float*>(get_data(det_outputs_[2])),
               static_cast<const float*>(get_data(det_outputs_[5])), 20, 20, 32, &candidates);
    NmsAndMap(candidates, &out->eyes);
    PairEyes(out->eyes, &out->pairs);
    SelectPair(out);
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
    if (!faceid_ready_ || pair.track_id == 0 ||
        (enrolling_ && pair.track_id != enroll_track_id_) ||
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
    enrolling_ = true;
    enroll_track_id_ = selected_track_id_;
    printf("[FACEID] begin enrollment track=%llu target_samples=%d\n",
           static_cast<unsigned long long>(enroll_track_id_), kEnrollSamples);
    return true;
}

void EyeDetFaceIdEngine::ClearEnrollment() {
    enrolling_ = false;
    enroll_track_id_ = 0;
    enroll_samples_.clear();
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
    face_contract_logged_ = false;
}
