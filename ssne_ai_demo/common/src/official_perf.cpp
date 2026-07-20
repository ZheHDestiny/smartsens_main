#include "official_perf.hpp"
#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {
using Clock = std::chrono::steady_clock;
constexpr float kBinMs = 0.5f;
constexpr int kBinCount = 2001;

struct PerfState {
    bool active = false, in_frame = false, have_first = false, have_last_finish = false;
    char feature[48] = "unknown";
    float sensor_fps = 90.0f;
    uint64_t completed = 0, period_count = 0;
    double period_sum_ms = 0.0;
    uint32_t latency_hist[kBinCount] = {0};
    uint32_t period_hist[kBinCount] = {0};
    Clock::time_point first_begin, frame_begin, last_finish, last_report;
};
PerfState g_perf;

void add_hist(uint32_t* hist, float ms) {
    int bin = static_cast<int>(ms / kBinMs + 0.5f);
    bin = std::max(0, std::min(bin, kBinCount - 1));
    if (hist[bin] != UINT32_MAX) ++hist[bin];
}
float percentile(const uint32_t* hist, uint64_t count, float q) {
    if (count == 0) return 0.0f;
    const uint64_t rank = std::max<uint64_t>(1, static_cast<uint64_t>(std::ceil(q * count)));
    uint64_t seen = 0;
    for (int i = 0; i < kBinCount; ++i) {
        seen += hist[i];
        if (seen >= rank) return i * kBinMs;
    }
    return (kBinCount - 1) * kBinMs;
}
void print_snapshot(const char* phase, Clock::time_point now) {
    if (!RuntimeLogAtLeast(RuntimeLogMode::VERIFY) || !g_perf.have_first || g_perf.completed == 0) return;
    const float elapsed_s = std::chrono::duration<float>(now - g_perf.first_begin).count();
    const float fps_app = elapsed_s > 0.001f ? g_perf.completed / elapsed_s : 0.0f;
    const float ratio = g_perf.sensor_fps > 0.0f ? fps_app / g_perf.sensor_fps : 0.0f;
    const float drop_pct = 100.0f * std::max(0.0f, 1.0f - ratio);
    const int realtime_score = static_cast<int>(std::floor(10.0f * std::min(ratio, 1.0f)));
    const float sensor_period_ms = 1000.0f / g_perf.sensor_fps;
    const float latency_p95_ms = percentile(g_perf.latency_hist, g_perf.completed, 0.95f);
    const float latency_t = latency_p95_ms / sensor_period_ms;
    const int latency_score = std::max(0, std::min(10, static_cast<int>(std::floor(11.0f - latency_t))));
    const float period_mean_ms = g_perf.period_count ? g_perf.period_sum_ms / g_perf.period_count : 0.0f;
    const float period_p95_ms = percentile(g_perf.period_hist, g_perf.period_count, 0.95f);
    const float jitter_pct = period_mean_ms > 0.001f ?
        100.0f * std::max(0.0f, period_p95_ms - period_mean_ms) / period_mean_ms : 0.0f;
    std::printf("[OFFICIAL_PERF] phase=%s feature=%s FPS_sensor=%.1f FPS_app=%.1f "
                "R=%.3f drop=%.1f%% realtime_score=%d/10 latency_p95=%.2fms "
                "T=%.2fms P95/T=%.2f latency_score=%d/10 period_mean=%.2fms "
                "period_p95=%.2fms jitter=%.1f%% drop_ok=%d jitter_ok=%d samples=%llu\n",
                phase, g_perf.feature, g_perf.sensor_fps, fps_app, ratio, drop_pct,
                realtime_score, latency_p95_ms, sensor_period_ms, latency_t, latency_score,
                period_mean_ms, period_p95_ms, jitter_pct,
                drop_pct <= 5.0f ? 1 : 0, jitter_pct <= 20.0f ? 1 : 0,
                static_cast<unsigned long long>(g_perf.completed));
}
}  // namespace

void OfficialPerfReset(const char* name, float sensor_fps) {
    g_perf = PerfState();
    g_perf.active = true;
    g_perf.sensor_fps = sensor_fps > 0.0f ? sensor_fps : 90.0f;
    if (name && name[0]) {
        std::strncpy(g_perf.feature, name, sizeof(g_perf.feature) - 1);
        g_perf.feature[sizeof(g_perf.feature) - 1] = '\0';
    }
    g_perf.last_report = Clock::now();
}
void OfficialPerfFinishPreviousFrame() {
    if (!g_perf.active || !g_perf.in_frame) return;
    const auto now = Clock::now();
    add_hist(g_perf.latency_hist, std::chrono::duration<float, std::milli>(now - g_perf.frame_begin).count());
    ++g_perf.completed;
    if (g_perf.have_last_finish) {
        const float ms = std::chrono::duration<float, std::milli>(now - g_perf.last_finish).count();
        add_hist(g_perf.period_hist, ms);
        g_perf.period_sum_ms += ms;
        ++g_perf.period_count;
    }
    g_perf.last_finish = now;
    g_perf.have_last_finish = true;
    g_perf.in_frame = false;
    if (now - g_perf.last_report >= std::chrono::seconds(1)) {
        print_snapshot("runtime", now);
        g_perf.last_report = now;
    }
}
void OfficialPerfBeginFrame() {
    if (!g_perf.active) return;
    const auto now = Clock::now();
    if (!g_perf.have_first) { g_perf.first_begin = now; g_perf.have_first = true; }
    g_perf.frame_begin = now;
    g_perf.in_frame = true;
}
void OfficialPerfPrintFinal() {
    OfficialPerfFinishPreviousFrame();
    print_snapshot("final", Clock::now());
    g_perf.active = false;
    g_perf.in_frame = false;
}
