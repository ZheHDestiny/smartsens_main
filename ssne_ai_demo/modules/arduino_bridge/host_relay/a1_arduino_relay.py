#!/usr/bin/env python3
"""Pure serial relay between the A1 console and an Arduino UNO R3.

This production relay deliberately has no camera, UVC, OpenCV, OSD or image
API code. The PC owns both USB serial ports, mirrors the A1 console, sends
keyboard lines to A1, and forwards only checksum-valid @OF packets to Arduino.
"""

import argparse
import codecs
import sys
import threading
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("缺少 pyserial，请执行: python -m pip install pyserial", file=sys.stderr)
    raise SystemExit(2)


BAUD_RATE = 115200
MAX_CONSOLE_LINE = 4096
OUTPUT_LOCK = threading.Lock()


def locked_print(message, file=None):
    stream = file if file is not None else sys.stdout
    with OUTPUT_LOCK:
        print(message, file=stream, flush=True)


def valid_packet_from_line(line):
    """Return one normalized @OF packet, or None for ordinary console text."""
    start = line.rfind(b"@OF,")
    if start < 0:
        return None
    packet = line[start:].strip(b"\r\n")
    star = packet.rfind(b"*")
    if star < 2 or len(packet) != star + 3:
        return None

    try:
        received_checksum = int(packet[star + 1:star + 3], 16)
    except ValueError:
        return None

    payload = packet[1:star]
    checksum = 0
    for value in payload:
        checksum ^= value
    if checksum != received_checksum:
        return None

    fields = payload.split(b",")
    if len(fields) != 5 or fields[0] != b"OF":
        return None
    if not fields[1].isdigit() or fields[2] not in (b"L", b"F", b"R", b"S"):
        return None
    if not fields[3].isdigit() or not fields[4].isdigit():
        return None
    if int(fields[3]) > 100 or int(fields[4]) > 100:
        return None
    return packet + b"\n"


def print_ports():
    ports = list(list_ports.comports())
    if not ports:
        print("没有发现串口。请检查 A1/Arduino 数据线和驱动。")
        return
    for port in ports:
        print("{}\t{}\t{}".format(port.device, port.description, port.hwid))


def open_port(name):
    return serial.Serial(
        port=name,
        baudrate=BAUD_RATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.05,
        write_timeout=1.0,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )


class ReconnectingPort:
    """Thread-safe serial endpoint that survives A1 reset/re-enumeration."""

    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.port = None
        self.lock = threading.Lock()
        self.last_attempt = 0.0
        self.last_error_report = 0.0
        self.ever_connected = False

    def _disconnect_locked(self, reason):
        if self.port is not None:
            try:
                self.port.close()
            except (serial.SerialException, OSError):
                pass
        self.port = None
        locked_print("\n[{}串口断开] {}；将自动重连。".format(
            self.role, reason), file=sys.stderr)

    def ensure_connected(self):
        with self.lock:
            if self.port is not None and self.port.is_open:
                return True
            now = time.monotonic()
            if now - self.last_attempt < 0.5:
                return False
            self.last_attempt = now
            try:
                self.port = open_port(self.name)
            except (serial.SerialException, OSError) as exc:
                self.port = None
                if now - self.last_error_report >= 3.0:
                    self.last_error_report = now
                    locked_print("[中继] 暂时无法打开{}串口 {}: {}；继续重试。".format(
                        self.role, self.name, exc), file=sys.stderr)
                return False
            reconnect = self.ever_connected
            self.ever_connected = True
            locked_print("[中继] {}串口{}: {}".format(
                self.role, "已重连" if reconnect else "已连接", self.name))
            return True

    def read(self):
        if not self.ensure_connected():
            time.sleep(0.05)
            return b""
        with self.lock:
            try:
                return self.port.read(self.port.in_waiting or 1)
            except (serial.SerialException, OSError) as exc:
                self._disconnect_locked(exc)
                return b""

    def write(self, payload):
        if not self.ensure_connected():
            return False
        with self.lock:
            try:
                self.port.write(payload)
                self.port.flush()
                return True
            except (serial.SerialException, OSError) as exc:
                self._disconnect_locked(exc)
                return False

    def close(self):
        with self.lock:
            if self.port is not None:
                try:
                    self.port.close()
                except (serial.SerialException, OSError):
                    pass
                self.port = None


def send_a1_line(a1_link, command, announce=True):
    # CRLF works both at the boot-console activation prompt and in BusyBox
    # canonical shell mode. CR alone can stop working after a program changes
    # and restores tty line discipline.
    payload = command.encode("utf-8", errors="replace") + b"\r\n"
    ok = a1_link.write(payload)
    if announce:
        locked_print("[发送到A1] {!r} (CRLF){}".format(
            command, "" if ok else " [等待串口重连]"))
    return ok


class AutoOpticalLauncher:
    """Prompt-driven launcher; avoids blind fixed-delay menu keystrokes."""

    def __init__(self, a1_link, program, log_mode):
        self.a1_link = a1_link
        self.program = program
        self.log_mode = str(log_mode)
        self.buffer = ""
        self.state = "wake"
        self.wake_attempts = 0
        self.program_sent_at = 0.0
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._wake_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _wake_loop(self):
        # Prefer a detected shell prompt. If firmware does not echo an idle
        # prompt, use one bounded fallback launch after three wake attempts;
        # never queue the program command continuously into a running menu.
        while not self.stop_event.is_set():
            if self.state == "wake":
                send_a1_line(self.a1_link, "", announce=False)
                self.wake_attempts += 1
                if self.wake_attempts >= 3:
                    send_a1_line(self.a1_link, self.program, announce=False)
                    self.state = "wait_menu"
                    self.program_sent_at = time.monotonic()
                self.stop_event.wait(2.0)
            elif (self.state == "wait_menu" and
                  time.monotonic() - self.program_sent_at > 8.0):
                self.state = "wake"
                self.wake_attempts = 0
            else:
                self.stop_event.wait(0.2)

    def feed(self, text):
        self.buffer = (self.buffer + text)[-8192:]
        if "Please press Enter to activate this console" in self.buffer:
            self.state = "wake"
            self.wake_attempts = 0
            send_a1_line(self.a1_link, "", announce=False)
            self.buffer = ""
            return
        if (self.buffer.endswith("# ") or self.buffer.endswith("$ ")) and \
                self.state in ("wake", "ready", "packets"):
            self.state = "wait_menu"
            self.program_sent_at = time.monotonic()
            self.buffer = ""
            send_a1_line(self.a1_link, self.program)
            return
        if self.state not in ("log", "output") and "请输入功能编号" in self.buffer:
            self.state = "log"
            self.buffer = ""
            send_a1_line(self.a1_link, "5")
        elif self.state == "log" and "请选择日志模式" in self.buffer:
            self.state = "output"
            self.buffer = ""
            send_a1_line(self.a1_link, self.log_mode)
        elif self.state == "output" and "请选择输出模式" in self.buffer:
            self.state = "packets"
            self.buffer = ""
            send_a1_line(self.a1_link, "2")

    def packet_seen(self):
        if self.state != "ready":
            self.state = "ready"
            locked_print("\n[中继] 光流 @OF 握手完成，Arduino 链路已贯通。")


def console_reader(a1_link, arduino_port, stop_event, launcher=None):
    pending = bytearray()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    forwarded = 0

    while not stop_event.is_set():
        try:
            chunk = a1_link.read()
        except (serial.SerialException, OSError) as exc:
            locked_print("\n[A1读取失败] {}".format(exc), file=sys.stderr)
            time.sleep(0.1)
            continue
        if not chunk:
            continue

        text = decoder.decode(chunk, final=False)
        if text:
            with OUTPUT_LOCK:
                sys.stdout.write(text)
                sys.stdout.flush()
            if launcher is not None:
                launcher.feed(text)
        pending.extend(chunk)

        while b"\n" in pending:
            raw_line, _, remainder = pending.partition(b"\n")
            pending = bytearray(remainder)
            packet = valid_packet_from_line(raw_line)
            if packet is None:
                continue
            try:
                arduino_port.write(packet)
                arduino_port.flush()
                forwarded += 1
                if forwarded == 1:
                    locked_print("\n[中继] 已向 Arduino 转发首个合法 OF 数据包。")
                if launcher is not None:
                    launcher.packet_seen()
            except serial.SerialException as exc:
                locked_print("\n[Arduino写入失败] {}".format(exc), file=sys.stderr)
                stop_event.set()
                return

        if len(pending) > MAX_CONSOLE_LINE:
            del pending[:-MAX_CONSOLE_LINE]


def keyboard_writer(a1_link, stop_event):
    while not stop_event.is_set():
        line = sys.stdin.readline()
        if line == "":
            stop_event.set()
            return
        command = line.rstrip("\r\n")
        if command.strip().lower() in ("/quit", "/exit"):
            locked_print("[中继] 正在关闭并释放 COM 端口...")
            stop_event.set()
            return
        try:
            send_a1_line(a1_link, command)
        except (serial.SerialException, OSError) as exc:
            locked_print("\n[A1写入失败] {}".format(exc), file=sys.stderr)
            stop_event.set()
            return


def main():
    parser = argparse.ArgumentParser(description="A1 到 Arduino 纯串口中继（无图像功能）")
    parser.add_argument("--list", action="store_true", help="列出电脑串口后退出")
    parser.add_argument("--a1", help="A1 控制台串口，例如 COM4")
    parser.add_argument("--arduino", help="Arduino USB 串口，例如 COM8")
    parser.add_argument("--auto-optical", action="store_true",
                        help="自动唤醒A1、启动ssne_ai_demo并进入光流Arduino模式")
    parser.add_argument("--program", default="cd /app_demo && ./ssne_ai_demo",
                        help="--auto-optical 使用的板端启动命令")
    parser.add_argument("--log-mode", choices=("0", "1", "2"), default="0",
                        help="自动启动时选择的运行日志模式，默认0")
    args = parser.parse_args()

    if args.list:
        print_ports()
        return 0
    if not args.a1 or not args.arduino:
        parser.error("必须指定 --a1 和 --arduino；先用 --list 查看")
    if args.a1 == args.arduino:
        parser.error("A1 和 Arduino 必须使用不同串口")

    arduino_port = None
    a1_link = ReconnectingPort(args.a1, "A1")
    try:
        arduino_port = open_port(args.arduino)
        print("[中继] Arduino: {}，等待 USB 复位完成...".format(args.arduino))
        time.sleep(2.5)
        arduino_port.reset_input_buffer()
        if not a1_link.ensure_connected():
            print("[中继] A1串口暂不可用；脚本保持运行，端口释放后自动连接。")
    except (serial.SerialException, OSError) as exc:
        a1_link.close()
        if arduino_port is not None:
            arduino_port.close()
        print("打开串口失败: {}".format(exc), file=sys.stderr)
        return 1

    print("[中继] A1: {} <-> PC -> Arduino: {} (115200 8N1)".format(
        args.a1, args.arduino))
    print("[中继] 此窗口就是 A1 shell/菜单终端；输入 /quit 退出。")

    stop_event = threading.Event()
    launcher = None
    if args.auto_optical:
        launcher = AutoOpticalLauncher(a1_link, args.program, args.log_mode)
        launcher.start()
        print("[中继] 自动模式：等待A1提示并进入 5 -> 日志{} -> Arduino模式2。".format(
            args.log_mode))
    reader = threading.Thread(
        target=console_reader,
        args=(a1_link, arduino_port, stop_event, launcher),
        daemon=True,
    )
    keyboard = threading.Thread(
        target=keyboard_writer,
        args=(a1_link, stop_event),
        daemon=True,
    )
    reader.start()
    keyboard.start()

    try:
        while not stop_event.wait(0.2):
            pass
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        if launcher is not None:
            launcher.stop()
        a1_link.close()
        arduino_port.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
