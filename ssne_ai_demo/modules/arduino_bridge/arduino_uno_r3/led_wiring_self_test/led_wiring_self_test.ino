// Temporary wiring self-test for Arduino UNO R3.
// D4 = left green, D7 = center red, D8 = right green.

const uint8_t PIN_LEFT_GREEN = 4;
const uint8_t PIN_STOP_RED = 7;
const uint8_t PIN_RIGHT_GREEN = 8;

void show(bool leftOn, bool redOn, bool rightOn) {
  digitalWrite(PIN_LEFT_GREEN, leftOn ? HIGH : LOW);
  digitalWrite(PIN_STOP_RED, redOn ? HIGH : LOW);
  digitalWrite(PIN_RIGHT_GREEN, rightOn ? HIGH : LOW);
}

void setup() {
  pinMode(PIN_LEFT_GREEN, OUTPUT);
  pinMode(PIN_STOP_RED, OUTPUT);
  pinMode(PIN_RIGHT_GREEN, OUTPUT);
  show(false, false, false);
}

void loop() {
  show(true, false, false);   // Left green only.
  delay(1500);

  show(false, true, false);   // Center red only.
  delay(1500);

  show(false, false, true);   // Right green only.
  delay(1500);

  show(true, false, true);    // Both green LEDs.
  delay(1500);

  show(true, true, true);     // All three: verifies every output together.
  delay(1500);

  show(false, false, false);  // Visible separator before repeating.
  delay(1000);
}
