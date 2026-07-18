#include "arduino_bridge.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

namespace {

uint8_t ClampPercent(float value) {
    value = std::max(0.0f, std::min(100.0f, value));
    return static_cast<uint8_t>(std::lround(value));
}

uint8_t XorChecksum(const char* begin, const char* end) {
    uint8_t value = 0;
    while (begin < end) value ^= static_cast<uint8_t>(*begin++);
    return value;
}

bool SameFeedback(const ArduinoFeedback& lhs, const ArduinoFeedback& rhs) {
    return lhs.hint == rhs.hint &&
           lhs.speed_percent == rhs.speed_percent &&
           lhs.risk_percent == rhs.risk_percent;
}

}  // namespace

ArduinoFeedback MakeArduinoFeedback(const ObstacleInfo& info) {
    float max_risk = 0.0f;
    for (int i = 0; i < ObstacleInfo::REGION_COUNT; ++i) {
        max_risk = std::max(max_risk, info.danger_level[i]);
    }

    ArduinoFeedback result = {
        VehicleHint::STOP,
        0,
        ClampPercent(max_risk * 100.0f)
    };

    if (info.tracking_quality < 0.25f ||
        info.priority == ObstacleInfo::EMERGENCY) {
        return result;
    }

    if (info.priority == ObstacleInfo::CLEAR) {
        // With no obstacle, small regional risk differences must not make the
        // indication weave left and right.
        result.hint = VehicleHint::FORWARD;
    } else {
        if (info.safest_region == ObstacleInfo::LEFT) {
            result.hint = VehicleHint::LEFT;
        } else if (info.safest_region == ObstacleInfo::RIGHT) {
            result.hint = VehicleHint::RIGHT;
        } else {
            result.hint = VehicleHint::FORWARD;
        }
    }

    // This is a recommendation, not a motor PWM command. Higher optical-flow
    // risk produces a lower LED blink frequency on the Arduino.
    result.speed_percent = ClampPercent((1.0f - max_risk) * 100.0f);
    if (result.speed_percent < 20) result.speed_percent = 20;
    return result;
}

ArduinoBridge::ArduinoBridge()
    : fd_(-1),
      sequence_(0),
      last_feedback_{VehicleHint::STOP, 0, 100},
      have_last_feedback_(false),
      last_send_(std::chrono::steady_clock::now() - std::chrono::seconds(1)),
      last_connect_attempt_(std::chrono::steady_clock::now() - std::chrono::seconds(2)) {}

ArduinoBridge::~ArduinoBridge() {
    Close();
}

bool ArduinoBridge::ConfigurePort(int fd) {
    termios tty;
    if (tcgetattr(fd, &tty) != 0) return false;

    cfmakeraw(&tty);
    cfsetispeed(&tty, B115200);
    cfsetospeed(&tty, B115200);
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag |= CLOCAL | CREAD;
    tty.c_cflag &= ~(PARENB | CSTOPB | CRTSCTS);
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;
    return tcsetattr(fd, TCSANOW, &tty) == 0;
}

bool ArduinoBridge::TryConnect() {
    const auto now = std::chrono::steady_clock::now();
    if (now - last_connect_attempt_ < std::chrono::seconds(1)) return false;
    last_connect_attempt_ = now;

    const char* override_path = std::getenv("A1_ARDUINO_PORT");
    const char* candidates[] = {"/dev/ttyACM0", "/dev/ttyACM1",
                                "/dev/ttyUSB0", "/dev/ttyUSB1"};
    const int candidate_count = static_cast<int>(sizeof(candidates) / sizeof(candidates[0]));
    const int attempts = (override_path != nullptr && override_path[0] != '\0') ? 1 : candidate_count;

    for (int i = 0; i < attempts; ++i) {
        const char* path = override_path != nullptr && override_path[0] != '\0'
            ? override_path : candidates[i];
        const int candidate_fd = open(path, O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
        if (candidate_fd < 0) continue;
        // ttyS0 is also the A1 system console in the current image. Its device
        // tree already fixes 115200 8N1; applying cfmakeraw() here would also
        // change stdin/stdout line handling for the menu. Preserve its console
        // termios and only configure dedicated USB/other UART devices.
        const bool is_system_console = std::strcmp(path, "/dev/ttyS0") == 0;
        if (!is_system_console && !ConfigurePort(candidate_fd)) {
            close(candidate_fd);
            continue;
        }
        fd_ = candidate_fd;
        device_path_ = path;
        have_last_feedback_ = false;
        if (!is_system_console) tcflush(fd_, TCIOFLUSH);
        std::printf("[Arduino] 已连接 %s (115200 8N1)\n", device_path_.c_str());
        return true;
    }
    return false;
}

bool ArduinoBridge::Start() {
    if (fd_ >= 0) return true;
    const bool connected = TryConnect();
    if (!connected) {
        std::printf("[Arduino] 未连接可用串口；光流继续运行，将每秒重试。"
                    "可用 A1_ARDUINO_PORT 指定设备。\n");
    }
    return connected;
}

bool ArduinoBridge::SendPacket(const ArduinoFeedback& feedback) {
    if (fd_ < 0) return false;

    char payload[64];
    const int payload_len = std::snprintf(
        payload, sizeof(payload), "OF,%lu,%c,%u,%u",
        static_cast<unsigned long>(sequence_++),
        static_cast<char>(feedback.hint),
        static_cast<unsigned int>(feedback.speed_percent),
        static_cast<unsigned int>(feedback.risk_percent));
    if (payload_len <= 0 || payload_len >= static_cast<int>(sizeof(payload))) return false;

    const uint8_t checksum = XorChecksum(payload, payload + payload_len);
    char packet[80];
    const int packet_len = std::snprintf(packet, sizeof(packet),
                                         "@%s*%02X\n", payload, checksum);
    if (packet_len <= 0 || packet_len >= static_cast<int>(sizeof(packet))) return false;

    int written = 0;
    while (written < packet_len) {
        const ssize_t count = write(fd_, packet + written, packet_len - written);
        if (count > 0) {
            written += static_cast<int>(count);
            continue;
        }
        if (count < 0 && errno == EINTR) continue;
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
        return false;
    }
    return written == packet_len;
}

void ArduinoBridge::Update(const ArduinoFeedback& feedback, bool force) {
    if (fd_ < 0 && !TryConnect()) return;

    const auto now = std::chrono::steady_clock::now();
    const bool changed = !have_last_feedback_ || !SameFeedback(feedback, last_feedback_);
    const bool heartbeat_due = now - last_send_ >= std::chrono::milliseconds(200);
    if (!force && !changed && !heartbeat_due) return;

    if (!SendPacket(feedback)) {
        std::fprintf(stderr, "[Arduino] 写入失败，关闭串口并等待重连: %s\n",
                     std::strerror(errno));
        Close();
        return;
    }
    last_feedback_ = feedback;
    have_last_feedback_ = true;
    last_send_ = now;
}

void ArduinoBridge::SendStop() {
    const ArduinoFeedback stop = {VehicleHint::STOP, 0, 100};
    Update(stop, true);
    if (fd_ >= 0) tcdrain(fd_);
}

void ArduinoBridge::Close() {
    if (fd_ >= 0) close(fd_);
    fd_ = -1;
    device_path_.clear();
    have_last_feedback_ = false;
}

bool ArduinoBridge::IsConnected() const {
    return fd_ >= 0;
}

const std::string& ArduinoBridge::DevicePath() const {
    return device_path_;
}
