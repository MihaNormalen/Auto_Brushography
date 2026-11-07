import time
"""
machine.py - the serial link to the Lerdge K, plus the few things every program does with the machine:
open it and home it, and lift + park the toolhead.
"""
import re, time, threading
import serial
from settings import SERIAL_PORT, SERIAL_BAUD, SERIAL_TIMEOUT, HOME_LIFT_Z, ACCEL_TRAVEL, FEED_TRAVEL

_IGNORE = [re.compile(p) for p in (r'^echo:', r'^busy:', r'^T:\d', r'^Cap:', r'^\s*$')]
VERBOSE = True
LINE_HOOK = None      # optional callable(line) - lets a monitor watch G-code without printing it


class LerdgeSerial:
    def __init__(self, port=SERIAL_PORT, baud=SERIAL_BAUD, timeout=SERIAL_TIMEOUT):
        self.port, self.baud, self.timeout = port, baud, timeout
        self.ser = None
        self._lock = threading.Lock()
        self._log = []
        self.last_error = None

    def connect(self):
        print(f"Connecting to {self.port} @ {self.baud} baud...")
        self.ser = serial.Serial(self.port, self.baud, timeout=1, write_timeout=5)
        time.sleep(3.0)                                  # opening the port resets many boards: wait until it answers
        self.ser.reset_input_buffer()
        time.sleep(2)
        time.sleep(0.1); self.ser.reset_input_buffer()
        print("Connected.")
        print(f"Firmware: {self.query('M115', timeout=5)[:120]}")

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close(); print("Disconnected.")

    def __enter__(self): self.connect(); return self
    def __exit__(self, *_): self.disconnect()

    @staticmethod
    def _is_ok(raw):
        s = raw.strip().lower()
        return s == 'ok' or s.startswith('ok ') or s.startswith('ok\t')

    def send_line(self, line, wait_ok=True, timeout=None):
        line = line.strip()
        if LINE_HOOK and line:
            try: LINE_HOOK(line)
            except Exception: pass
        if not line or line.startswith(';'):
            return ''
        if ';' in line:
            line = line[:line.index(';')].strip()
        if not line:
            return ''
        if timeout is None:
            cmd = line.split()[0].upper()
            if cmd in ("G28", "G29", "M400", "G4"): timeout = 120
            elif cmd in ("G0", "G1"):               timeout = max(self.timeout, 30)
            else:                                   timeout = self.timeout
        with self._lock:
            if VERBOSE: print(f"  >> {line}")
            self._log.append(f">> {line}")
            self.ser.write((line + '\n').encode('ascii'))
            if not wait_ok:
                return ''
            resp, deadline, got_ok = [], time.time() + timeout, False
            while time.time() < deadline:
                try:
                    raw = self.ser.readline().decode('ascii', errors='replace').strip()
                except Exception as e:
                    print(f"  Serial read error: {e}"); break
                if not raw: continue
                self._log.append(f"<< {raw}")
                if VERBOSE: print(f"  << {raw}")
                if raw.lower().startswith(('error', '!!')):
                    self.last_error = raw
                    print(f"  MACHINE ERROR: {raw}"); got_ok = True; break
                if self._is_ok(raw):
                    self.ok_seen = True
                    resp.append(raw); got_ok = True; break
                if not any(p.match(raw) for p in _IGNORE):
                    resp.append(raw)
            if not got_ok:
                if not getattr(self, 'ok_seen', False):
                    raise SystemExit('STOPPED: the machine gave no answer at all (no ok to the first command). '
                                     'Check the power, that no other program has COM9 open, and the baud rate. Nothing was moved.')
                print(f"  WARNING: no ok for {line!r} within {timeout}s — check baud (115200 / 250000)")
            return '\n'.join(resp)

    def query(self, cmd, timeout=None):
        return self.send_line(cmd, True, timeout or self.timeout)

    def lift_to_safe(self):     self.send_line(f"G0 Z{HOME_LIFT_Z:.3f} F3000")      # high: the next step is homing X and Y
    def flush_motion(self):     self.send_line("M400", timeout=120)
    def wait_motion_complete(self): self.flush_motion(); return True
    def send_stroke(self, lines):
        for l in lines: self.send_line(l)

    def home_z(self):
        print("  Homing Z..."); self.send_line("G28 Z", timeout=60); self.lift_to_safe()
    def home_xy(self):
        print("  Homing XY..."); self.send_line("G28 XY", timeout=60)
    def home_all(self):
        self.home_z(); self.home_xy()

    def set_absolute(self): self.send_line("G90")
    def set_units_mm(self): self.send_line("G21")


def open_machine(home=True, quiet=True):
    """Connects, sets absolute mm, and (unless home=False) homes: Z first, a lift to HOME_LIFT_Z, then XY. Returns the machine."""
    global VERBOSE
    if quiet:
        VERBOSE = False                                            # do not print every G-code line
    m = LerdgeSerial()
    m.connect()
    m.set_absolute()
    m.set_units_mm()
    if home:
        m.home_all()
    return m


def park(machine, x, y, z_lift, feed=FEED_TRAVEL, accel=ACCEL_TRAVEL, timeout=60):
    """Lift to z_lift (machine Z) and go to X Y at travel speed AND travel acceleration. A stroke runs with the (low) painting
    acceleration; without restoring the travel acceleration the way out takes about twice as long, and a wet stroke dries meanwhile."""
    machine.send_line("M400")
    machine.send_line(f"M204 P{accel} T{accel}")
    machine.send_line(f"G0 Z{z_lift:.3f} F3000")
    machine.send_line(f"G0 X{x:.3f} Y{y:.3f} F{feed}", timeout=timeout)
