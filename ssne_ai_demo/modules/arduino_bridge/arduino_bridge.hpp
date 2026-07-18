#ifndef ARDUINO_BRIDGE_HPP
#define ARDUINO_BRIDGE_HPP

#include <chrono>
#include <cstdint>
#include <string>

#include "common.hpp"

enum class VehicleHint : char {
    LEFT = 'L',
    FORWARD = 'F',
    RIGHT = 'R',
    STOP = 'S'
};

struct ArduinoFeedback {
    VehicleHint hint;
    uint8_t speed_percent;
    uint8_t risk_percent;
};

// Convert the existing optical-flow result without changing its thresholds or
// state machine. Poor tracking and EMERGENCY are deliberately fail-safe STOP.
ArduinoFeedback MakeArduinoFeedback(const ObstacleInfo& info);

class ArduinoBridge {
public:
    ArduinoBridge();
    ~ArduinoBridge();

    // A1_ARDUINO_PORT overrides USB auto-discovery (for example /dev/ttyACM0).
    // Failure is non-fatal: Update() periodically retries in the background.
    bool Start();
    void Update(const ArduinoFeedback& feedback, bool force = false);
    void SendStop();
    void Close();
    bool IsConnected() const;
    const std::string& DevicePath() const;

private:
    bool TryConnect();
    bool ConfigurePort(int fd);
    bool SendPacket(const ArduinoFeedback& feedback);

    int fd_;
    uint32_t sequence_;
    std::string device_path_;
    ArduinoFeedback last_feedback_;
    bool have_last_feedback_;
    std::chrono::steady_clock::time_point last_send_;
    std::chrono::steady_clock::time_point last_connect_attempt_;
};

#endif
