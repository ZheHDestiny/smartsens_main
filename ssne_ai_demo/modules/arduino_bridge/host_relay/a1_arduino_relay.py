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


def console_reader(a1_port, arduino_port, stop_event):
    pending = bytearray()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    forwarded = 0

    while not stop_event.is_set():
        try:
            chunk = a1_port.read(a1_port.in_waiting or 1)
        except serial.SerialException as exc:
            locked_print("\n[A1读取失败] {}".format(exc), file=sys.stderr)
            stop_event.set()
            return
        if not chunk:
            continue

        text = decoder.decode(chunk, final=False)
        if text:
            with OUTPUT_LOCK:
                sys.stdout.write(text)
                sys.stdout.flush()
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
            except serial.SerialException as exc:
                locked_print("\n[Arduino写入失败] {}".format(exc), file=sys.stderr)
                stop_event.set()
                return

        if len(pending) > MAX_CONSOLE_LINE:
            del pending[:-MAX_CONSOLE_LINE]


def keyboard_writer(a1_port, stop_event):
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
            # Buildroot/BusyBox serial console expects the terminal Enter key.
            a1_port.write(command.encode("utf-8", errors="replace") + b"\r")
            a1_port.flush()
            locked_print("[发送到A1] {!r} (CR)".format(command))
        except serial.SerialException as exc:
            locked_print("\n[A1写入失败] {}".format(exc), file=sys.stderr)
            stop_event.set()
            return


def main():
    parser = argparse.ArgumentParser(description="A1 到 Arduino 纯串口中继（无图像功能）")
    parser.add_argument("--list", action="store_true", help="列出电脑串口后退出")
    parser.add_argument("--a1", help="A1 控制台串口，例如 COM4")
    parser.add_argument("--arduino", help="Arduino USB 串口，例如 COM8")
    args = parser.parse_args()

    if args.list:
        print_ports()
        return 0
    if not args.a1 or not args.arduino:
        parser.error("必须指定 --a1 和 --arduino；先用 --list 查看")
    if args.a1 == args.arduino:
        parser.error("A1 和 Arduino 必须使用不同串口")

    arduino_port = None
    a1_port = None
    try:
        arduino_port = open_port(args.arduino)
        print("[中继] Arduino: {}，等待 USB 复位完成...".format(args.arduino))
        time.sleep(2.5)
        arduino_port.reset_input_buffer()
        a1_port = open_port(args.a1)
    except serial.SerialException as exc:
        if a1_port is not None:
            a1_port.close()
        if arduino_port is not None:
            arduino_port.close()
        print("打开串口失败: {}".format(exc), file=sys.stderr)
        return 1

    print("[中继] A1: {} <-> PC -> Arduino: {} (115200 8N1)".format(
        args.a1, args.arduino))
    print("[中继] 此窗口就是 A1 shell/菜单终端；输入 /quit 退出。")

    stop_event = threading.Event()
    reader = threading.Thread(
        target=console_reader,
        args=(a1_port, arduino_port, stop_event),
        daemon=True,
    )
    keyboard = threading.Thread(
        target=keyboard_writer,
        args=(a1_port, stop_event),
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
        a1_port.close()
        arduino_port.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
