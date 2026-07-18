#!/usr/bin/env python3
"""Test the three optical-flow LEDs through the Arduino USB serial port."""

import argparse
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("缺少 pyserial，请执行: py -m pip install pyserial", file=sys.stderr)
    raise SystemExit(2)


BAUD_RATE = 115200
HEARTBEAT_SECONDS = 0.15


def make_packet(sequence, direction, speed, risk=0):
    payload = "OF,{},{},{},{}".format(sequence, direction, speed, risk)
    checksum = 0
    for value in payload.encode("ascii"):
        checksum ^= value
    return "@{}*{:02X}\n".format(payload, checksum).encode("ascii")


def available_ports():
    return list(list_ports.comports())


def likely_arduino_ports(ports):
    result = []
    for port in ports:
        description = (port.description or "").lower()
        manufacturer = (port.manufacturer or "").lower()
        if port.vid == 0x2341 or "arduino" in description or "arduino" in manufacturer:
            result.append(port)
    return result


def print_ports(ports):
    if not ports:
        print("没有发现串口。请确认 USB-A 转 USB-B 数据线已经插好。")
        return
    print("电脑当前发现的串口：")
    for port in ports:
        print("  {}  {}  {}".format(port.device, port.description, port.hwid))


def choose_port(requested):
    if requested:
        return requested

    ports = available_ports()
    candidates = likely_arduino_ports(ports)
    if len(candidates) == 1:
        print("自动选择 Arduino 串口：{} ({})".format(
            candidates[0].device, candidates[0].description))
        return candidates[0].device

    print_ports(ports)
    if not ports:
        raise SystemExit(1)
    if len(candidates) > 1:
        print("检测到多个可能的 Arduino，请用 --port COM号 指定。", file=sys.stderr)
    else:
        print("无法自动确认 Arduino，请用 --port COM号 指定。", file=sys.stderr)
    raise SystemExit(1)


def hold_state(port, sequence, direction, speed, duration, message):
    print(message)
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        port.write(make_packet(sequence, direction, speed))
        port.flush()
        sequence += 1
        time.sleep(HEARTBEAT_SECONDS)
    return sequence


def main():
    parser = argparse.ArgumentParser(description="通过USB串口依次测试三盏Arduino指示灯")
    parser.add_argument("--port", help="Arduino 串口，例如 Windows 的 COM5")
    parser.add_argument("--list", action="store_true", help="列出串口后退出")
    args = parser.parse_args()

    if args.list:
        print_ports(available_ports())
        return 0

    port_name = choose_port(args.port)
    try:
        port = serial.Serial(port_name, BAUD_RATE, timeout=0.2, write_timeout=1.0)
    except serial.SerialException as exc:
        print("无法打开 {}：{}".format(port_name, exc), file=sys.stderr)
        print("请关闭 Arduino IDE 的串口监视器以及其他占用该 COM 口的软件。", file=sys.stderr)
        return 1

    try:
        print("已打开 {}，等待 Arduino USB 连接引起的自动复位……".format(port_name))
        time.sleep(2.5)
        port.reset_input_buffer()
        sequence = 1
        sequence = hold_state(port, sequence, "S", 0, 3.0,
                              "[1/5] 中间红灯应常亮 3 秒")
        sequence = hold_state(port, sequence, "L", 100, 4.0,
                              "[2/5] 左绿色灯应快速闪烁 4 秒")
        sequence = hold_state(port, sequence, "R", 100, 4.0,
                              "[3/5] 右绿色灯应快速闪烁 4 秒")
        sequence = hold_state(port, sequence, "F", 60, 4.0,
                              "[4/5] 两个绿色灯应同时闪烁 4 秒")
        hold_state(port, sequence, "S", 0, 3.0,
                   "[5/5] 回到中间红灯常亮")
        print("测试发送完成。若灯光与提示一致，三路接线和最终 Arduino 程序均正常。")
    except (serial.SerialException, KeyboardInterrupt) as exc:
        if isinstance(exc, serial.SerialException):
            print("串口通信失败：{}".format(exc), file=sys.stderr)
            return 1
        print("\n测试已取消。")
    finally:
        port.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
