#!/usr/bin/env python3
"""serial_test.py - checks the connection to the machine. Does NOT move anything.
    With MACHINE = "fluidnc" in settings.py: connects to FLUIDNC_HOST and asks only $I (firmware info) and ? (state).
    With MACHINE = "marlin": sends ONLY M115 (firmware info):
    python serial_test.py              (SERIAL_PORT from settings.py, several speeds, two line endings)
    python serial_test.py COM9         (another port)
"""
import sys, time

try:
    import settings as S
    port = S.SERIAL_PORT
except Exception:
    S, port = None, "COM9"

if S is not None and getattr(S, "MACHINE", "marlin").lower() == "fluidnc" and len(sys.argv) < 2:
    import machine
    m = machine.FluidNCLink()
    m.connect()                                  # prints the firmware and the state, stops with a hint if unreachable
    m.disconnect()
    print("  -> the machine answers. Connection works.")
    raise SystemExit

import serial
from serial.tools import list_ports
if len(sys.argv) >= 2:
    port = sys.argv[1]

BAUDS = [115200, 250000, 57600, 38400, 19200, 9600]
ENDINGS = [b"\n", b"\r\n"]

print("ports found on this PC:")
for p in list_ports.comports():
    print(f"  {p.device}: {p.description}")
print(f"\ntesting {port} (sends only M115)")

found = False
for baud in BAUDS:
    for end in ENDINGS:
        try:
            ser = serial.Serial(port, baud, timeout=0.5, write_timeout=3)
        except Exception as e:
            print(f"{baud:7d} baud: cannot open {port}: {e}")
            print("  -> another program has the port open, or the name is wrong")
            raise SystemExit(1)
        time.sleep(2.5)                              # the board resets when the port opens
        ser.reset_input_buffer()
        ser.write(b"M115" + end)
        raw = b""
        t0 = time.time()
        while time.time() - t0 < 3:
            raw += ser.read(256)
            if b"ok" in raw.lower():
                break
        ser.close()
        label = "CRLF" if end == b"\r\n" else "LF"
        if raw:
            text = raw.decode("ascii", errors="replace").strip()
            print(f"{baud:7d} baud, {label}: got {len(raw)} bytes: {text[:160]!r}")
            if b"ok" in raw.lower():
                print("  -> the machine answers. Connection works at this speed and line ending.")
                found = True
                break
        else:
            print(f"{baud:7d} baud, {label}: no answer")
    if found:
        break

if not found:
    print("\nNO 'ok' from the machine at any speed.")
    print("Check: the controller's power light is on (not only the PC's USB), the controller is switched on")
    print("for the whole test, the USB cable is in the controller's board port, and no other program has COM9.")
