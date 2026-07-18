// A1 optical-flow feedback display for Arduino UNO R3.
// Hardware Serial/D0 RX: 115200 8N1. D1 TX is not connected to A1.
// D4=left green, D7=red stop, D8=right green.

const uint8_t PIN_LEFT_GREEN = 4;
const uint8_t PIN_STOP_RED = 7;
const uint8_t PIN_RIGHT_GREEN = 8;

const unsigned long LINK_TIMEOUT_MS = 750;

char lineBuffer[64];
uint8_t lineLength = 0;
char currentHint = 'S';
uint8_t currentSpeed = 0;
unsigned long lastPacketMs = 0;
unsigned long lastBlinkMs = 0;
bool blinkOn = false;

void showStop() {
  digitalWrite(PIN_LEFT_GREEN, LOW);
  digitalWrite(PIN_RIGHT_GREEN, LOW);
  digitalWrite(PIN_STOP_RED, HIGH);
}

uint8_t hexValue(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'A' && value <= 'F') return value - 'A' + 10;
  if (value >= 'a' && value <= 'f') return value - 'a' + 10;
  return 255;
}

bool parsePercent(const char* text, uint8_t* result) {
  if (text == NULL || text[0] == '\0') return false;
  char* end = NULL;
  const long value = strtol(text, &end, 10);
  if (*end != '\0' || value < 0 || value > 100) return false;
  *result = (uint8_t)value;
  return true;
}

bool parsePacket(char* line) {
  if (line[0] != '@') return false;
  char* star = strchr(line, '*');
  if (star == NULL || star[1] == '\0' || star[2] == '\0' || star[3] != '\0') return false;

  uint8_t checksum = 0;
  for (char* cursor = line + 1; cursor < star; ++cursor) checksum ^= (uint8_t)*cursor;
  const uint8_t high = hexValue(star[1]);
  const uint8_t low = hexValue(star[2]);
  if (high > 15 || low > 15 || checksum != (uint8_t)((high << 4) | low)) return false;

  *star = '\0';
  char* save = NULL;
  char* type = strtok_r(line + 1, ",", &save);
  char* sequence = strtok_r(NULL, ",", &save);
  char* hint = strtok_r(NULL, ",", &save);
  char* speed = strtok_r(NULL, ",", &save);
  char* risk = strtok_r(NULL, ",", &save);
  if (type == NULL || strcmp(type, "OF") != 0 || sequence == NULL ||
      hint == NULL || hint[1] != '\0' || speed == NULL || risk == NULL ||
      strtok_r(NULL, ",", &save) != NULL) return false;
  if (hint[0] != 'L' && hint[0] != 'F' && hint[0] != 'R' && hint[0] != 'S') return false;

  uint8_t speedValue = 0;
  uint8_t riskValue = 0;
  if (!parsePercent(speed, &speedValue) || !parsePercent(risk, &riskValue)) return false;

  const bool directionChanged = currentHint != hint[0];
  currentHint = hint[0];
  currentSpeed = speedValue;
  lastPacketMs = millis();
  // Heartbeats arrive every 200 ms. Resetting the blink phase for every
  // heartbeat would prevent slow blink intervals from ever completing.
  if (directionChanged) {
    lastBlinkMs = lastPacketMs;
    blinkOn = true;
  }
  return true;
}

void receivePackets() {
  while (Serial.available() > 0) {
    const char value = (char)Serial.read();
    if (value == '\r') continue;
    if (value == '\n') {
      lineBuffer[lineLength] = '\0';
      parsePacket(lineBuffer);
      lineLength = 0;
    } else if (lineLength < sizeof(lineBuffer) - 1) {
      lineBuffer[lineLength++] = value;
    } else {
      lineLength = 0;
    }
  }
}

void updateLeds() {
  const unsigned long now = millis();
  if (currentHint == 'S' || now - lastPacketMs > LINK_TIMEOUT_MS) {
    currentHint = 'S';
    currentSpeed = 0;
    showStop();
    return;
  }

  // 20..100% maps to about 1.2..5.0 Hz; one toggle is half a blink period.
  const unsigned long frequencyMilliHz = 200UL + 48UL * currentSpeed;
  const unsigned long toggleIntervalMs = 500000UL / frequencyMilliHz;
  if (now - lastBlinkMs >= toggleIntervalMs) {
    lastBlinkMs = now;
    blinkOn = !blinkOn;
  }

  digitalWrite(PIN_STOP_RED, LOW);
  digitalWrite(PIN_LEFT_GREEN,
               blinkOn && (currentHint == 'L' || currentHint == 'F') ? HIGH : LOW);
  digitalWrite(PIN_RIGHT_GREEN,
               blinkOn && (currentHint == 'R' || currentHint == 'F') ? HIGH : LOW);
}

void setup() {
  pinMode(PIN_LEFT_GREEN, OUTPUT);
  pinMode(PIN_STOP_RED, OUTPUT);
  pinMode(PIN_RIGHT_GREEN, OUTPUT);
  showStop();
  Serial.begin(115200);
  lastPacketMs = millis();
}

void loop() {
  receivePackets();
  updateLeds();
}
