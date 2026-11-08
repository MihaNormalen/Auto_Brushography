"""
machine.py - the link to the machine, plus the few things every program does with it: open it and home it, and lift + park the toolhead.
  MACHINE = "marlin"   the Lerdge K on a serial port (LerdgeSerial)
  MACHINE = "fluidnc"  a FluidNC controller over WiFi (FluidNCLink, telnet). The programs keep writing Marlin G-code; the link
                       translates the few commands FluidNC does not know (M400, M204, M115, G4 in ms, G28 homing).
Machine() gives the one chosen in settings.py.
"""
import re, socket, time, threading
import settings as S
from settings import SERIAL_PORT, SERIAL_BAUD, SERIAL_TIMEOUT, HOME_LIFT_Z, ACCEL_TRAVEL, FEED_TRAVEL

_IGNORE = [re.compile(p) for p in (r'^echo:', r'^busy:', r'^T:\d', r'^Cap:', r'^\s*$', r'^<', r'^\[MSG:')]
VERBOSE = True
LINE_HOOK = None      # optional callable(line) - lets a monitor watch G-code without printing it
WAIT_HOOK = None      # optional callable() - called often while waiting for the machine (FluidNC); the live preview draws its
                      #   window from it, because on macOS a window may only be drawn from the main thread, which is the one waiting


class LerdgeSerial:
    def __init__(self, port=SERIAL_PORT, baud=SERIAL_BAUD, timeout=SERIAL_TIMEOUT):
        self.port, self.baud, self.timeout = port, baud, timeout
        self.ser = None
        self._lock = threading.Lock()
        self._log = []
        self.last_error = None

    def connect(self):
        import serial
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
        resp = ''
        for l in self._translate(line):
            resp = self._send_raw(l, wait_ok, timeout)
        return resp

    def _translate(self, line):
        """The line(s) this firmware gets for one line of the programs' (Marlin) G-code."""
        return [line]

    def _write(self, data: bytes):
        self.ser.write(data)

    def _readline(self) -> str:
        """One line from the machine, '' after about 1 s without one."""
        return self.ser.readline().decode('ascii', errors='replace').strip()

    def _no_answer_hint(self):
        return (f'Check the power, that no other program has {self.port} open, and the baud rate.')

    def _send_raw(self, line, wait_ok=True, timeout=None):
        if timeout is None:
            cmd = line.split()[0].upper()
            if cmd in ("G28", "G29", "M400", "G4") or cmd.startswith("$H"): timeout = 120
            elif cmd in ("G0", "G1"):               timeout = max(self.timeout, 30)
            else:                                   timeout = self.timeout
        with self._lock:
            if VERBOSE: print(f"  >> {line}")
            self._log.append(f">> {line}")
            self._write((line + '\n').encode('ascii'))
            if not wait_ok:
                return ''
            resp, deadline, got_ok = [], time.time() + timeout, False
            while time.time() < deadline:
                try:
                    raw = self._readline()
                except Exception as e:
                    print(f"  read error: {e}"); break
                if not raw: continue
                self._log.append(f"<< {raw}")
                if VERBOSE: print(f"  << {raw}")
                if raw.lower().startswith(('error', '!!', 'alarm')):
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
                                     + self._no_answer_hint() + ' Nothing was moved.')
                print(f"  WARNING: no ok for {line!r} within {timeout}s")
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


class FluidNCLink(LerdgeSerial):
    """A FluidNC controller over WiFi (telnet, port 23). Same use as LerdgeSerial: send_line(), query(), home_all(), ...
    FluidNC speaks Grbl G-code; the Marlin commands the programs send are translated here:
      M400       -> G4 P0.01, then wait until the status says Idle (FluidNC has no M400)
      M204       -> nothing: FluidNC takes the acceleration from its own config (acceleration_mm_per_sec2)
      M115       -> $I (firmware info)
      G4 P<ms>   -> G4 P<seconds> (Grbl's P is seconds)
      G28 <axes> -> FLUIDNC_ZERO_MACRO (zero.g on the SD card: drives into the end stops and sets X/Y/Z from them), once per
                    connection; else $H<axes> if FLUIDNC_HOMING; else nothing (the position FluidNC keeps is used)
      G0 / G1    -> G1 at the current feed (Marlin runs G0 at the feed too; FluidNC would run it at max rate), never faster
                    than FLUIDNC_MAX_FEED: the small steppers lose steps at full speed (2026-11-07: 2.6 mm lost in two loops at
                    2000 mm/min, nothing at 600)
    BACKLASH_MM: the play in X and Y. Each axis remembers which way it last moved; when a move reverses it, a take-up move
    (that axis only, by the play) goes first, and while an axis moves in - its coordinates are sent shifted by the play.
    Reference: the + side, which is how zero.g leaves both axes (its last moves are +X +Y).
    Every G0/G1 target outside MACHINE_LIMITS_MM is refused (like a firmware error): without limit switches nothing else stops
    the machine from running into its ends."""

    def __init__(self, host=None, port=None, timeout=SERIAL_TIMEOUT):
        super().__init__(timeout=timeout)
        self.host = host or S.FLUIDNC_HOST
        self.tcp_port = port or getattr(S, "FLUIDNC_PORT", 23)
        self.homing = bool(getattr(S, "FLUIDNC_HOMING", False))
        self.zero_macro = getattr(S, "FLUIDNC_ZERO_MACRO", None)
        self._zeroed = False
        self.limits = getattr(S, "MACHINE_LIMITS_MM", None)
        self.sock, self._buf = None, b""
        self._said_no_homing = False
        self.max_feed = float(getattr(S, "FLUIDNC_MAX_FEED", None) or float("inf"))   # None = no cap
        self.backlash = {k: float(v) for k, v in (getattr(S, "BACKLASH_MM", None) or {}).items() if k in "XY" and v}
        self.pos = {}                                # where the programs' G-code has put each axis (machine mm, uncompensated)
        self.dir = {"X": 1, "Y": 1}                  # which way each axis last moved
        self.feed = min(self.max_feed, float(FEED_TRAVEL))   # the modal feed rate of the programs' G-code (mm/min)

    def connect(self):
        print(f"Connecting to FluidNC at {self.host}:{self.tcp_port}...")
        try:
            self.sock = socket.create_connection((self.host, self.tcp_port), timeout=10)
        except OSError as e:
            raise SystemExit(f"STOPPED: cannot reach FluidNC at {self.host}:{self.tcp_port} ({e}). "
                             "Is it switched on and in the same WiFi? Nothing was moved.")
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(0.1)
        time.sleep(0.3)
        self._drain()
        print("Connected.")
        info = self.query('$I', timeout=5)
        print(f"Firmware: {info.splitlines()[0][:120] if info else '?'}")
        rep = self.report()
        state = rep[1:].split("|", 1)[0].rstrip(">")
        self._take_position(rep)
        print(f"State: {state}")
        if state.startswith(("Alarm", "Door")):
            raise SystemExit(f"STOPPED: FluidNC is in {state}. Check where the head is, unlock it ($X in the WebUI or the console) "
                             "and set the position if it was lost, then run again. Nothing was moved.")

    def disconnect(self):
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None
                print("Disconnected.")

    def _no_answer_hint(self):
        return f'Is FluidNC at {self.host} switched on, and no other program (WebUI job, another telnet) sending to it?'

    # ---- the raw link
    def _write(self, data: bytes):
        self.sock.sendall(data)

    def _readline(self) -> str:
        end = time.time() + 1.0
        while b"\n" not in self._buf:
            if time.time() > end:
                return ''
            try:
                d = self.sock.recv(4096)
            except socket.timeout:
                if WAIT_HOOK is not None:
                    try:
                        WAIT_HOOK()
                    except Exception:
                        pass
                continue
            if not d:
                raise ConnectionError("FluidNC closed the connection")
            self._buf += d
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode('ascii', errors='replace').strip()

    def _drain(self) -> str:
        """Throws away whatever the controller has sent and nobody has read; returns it (for the log)."""
        got = self._buf
        self.sock.settimeout(0.2)
        try:
            while True:
                d = self.sock.recv(4096)
                if not d:
                    break
                got += d
        except (socket.timeout, BlockingIOError):
            pass
        self.sock.settimeout(0.1)
        self._buf = b""
        return got.decode("ascii", errors="replace")

    def report(self, timeout=3.0):
        """The whole '?' status report, e.g. <Idle|WPos:0.000,156.000,12.000|FS:0,0> ('' if none came)."""
        with self._lock:
            self._write(b"?")
            end = time.time() + timeout
            while time.time() < end:
                raw = self._readline()
                if raw.startswith("<"):
                    return raw
                if raw:
                    self._log.append(f"<< {raw}")
        return ''

    def status(self, timeout=3.0):
        """The machine state from a '?' status report: Idle, Run, Hold:0, Alarm, ... ('' if none came)."""
        return self.report(timeout)[1:].split("|", 1)[0].rstrip(">")

    def _take_position(self, rep):
        """Starts the backlash bookkeeping from a status report: the axes as FluidNC has them, both last moved in +."""
        m = re.search(r"WPos:(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)", rep or "")
        if m:
            self.pos = {"X": float(m.group(1)), "Y": float(m.group(2)), "Z": float(m.group(3))}
            self.dir = {"X": 1, "Y": 1}

    def _phys(self, ax, v, d):
        """The coordinate actually sent for machine position v while the axis moves in direction d."""
        return v - self.backlash.get(ax, 0.0) if d < 0 else v

    def run_macro(self, name, timeout=240.0):
        """Runs a G-code file from FluidNC's SD card and waits until it has finished (while a file runs, the status report
        carries an SD: field; between two of its lines the state can read Idle for a moment)."""
        print(f"  running {name} on the controller...")
        self.last_error = None
        self._send_raw(f"$SD/Run={name}", wait_ok=False)     # FluidNC answers ok only when the file has ENDED
        end, started = time.time() + timeout, time.time()
        while time.time() < end:
            r = self.report()
            busy = "|SD:" in r or not r.startswith("<Idle")
            if r.startswith(("<Alarm", "<Door")):
                raise SystemExit(f"STOPPED: FluidNC went into {r} while running {name}.")
            if not busy and time.time() - started > 1.0:
                time.sleep(0.3)
                with self._lock:
                    late = self._drain()                  # its late ok (or error) must not answer the next command
                errs = [l for l in late.splitlines() + self._log[-20:] if "error" in l.lower() or "alarm" in l.lower()]
                if errs:
                    raise SystemExit(f"STOPPED: FluidNC reported a problem running {name}: {errs[-1].strip()}")
                print(f"  {name} done: {r}")
                self._take_position(r)
                return r
            time.sleep(0.25)
        raise SystemExit(f"STOPPED: {name} did not finish within {timeout:.0f} s.")

    def wait_idle(self, timeout=120.0):
        end = time.time() + timeout
        while time.time() < end:
            st = self.status()
            if st == "Idle":
                return True
            if st.startswith(("Alarm", "Door")):
                self.last_error = f"FluidNC state {st}"
                print(f"  MACHINE ERROR: {self.last_error}")
                return False
            time.sleep(0.1)
        print(f"  WARNING: the machine did not become Idle within {timeout:.0f}s")
        return False

    # ---- Marlin -> FluidNC
    def _translate(self, line):
        words = line.split()
        cmd = words[0].upper()
        if cmd == "M400":
            return ["G4 P0.01", "<idle>"]
        if cmd == "M204":
            return []
        if cmd == "M115":
            return ["$I"]
        if cmd == "G4":
            p = re.search(r"P(-?[\d.]+)", line, re.I)
            sec = float(p.group(1)) / 1000.0 if p else float((re.search(r"S(-?[\d.]+)", line, re.I) or [0, 0])[1])
            return [f"G4 P{max(sec, 0.01):.3f}"]
        if cmd == "G28":
            axes = "".join(c for c in "".join(words[1:]).upper() if c in "XYZ")
            if self.zero_macro:
                if self._zeroed:
                    return []                        # zero.g did all axes at once
                self._zeroed = True
                return ["<macro>" + self.zero_macro]
            if self.homing:
                return [f"$H{axes}"]
            if not self._said_no_homing:
                print("  no homing (FLUIDNC_HOMING = False): the position FluidNC keeps is used as it is")
                self._said_no_homing = True
            return []
        if cmd == "G29":
            return []
        if cmd in ("G0", "G1"):
            words = dict(re.findall(r"([XYZF])\s*(-?[\d.]+)", line.upper()))
            if "F" in words:
                self.feed = float(words.pop("F"))
            if not words:
                return []                            # only a new feed rate: remembered, sent with the next move
            for ax, v in words.items():
                lo, hi = self.limits[ax] if self.limits else (-1e9, 1e9)
                if not lo <= float(v) <= hi:
                    return [f"<refuse>{ax} {float(v):.1f} is outside MACHINE_LIMITS_MM ({lo:g} to {hi:g}): {line}"]
            feed = f"F{min(self.feed, self.max_feed):g}"
            out, take = [], {}
            for ax in "XY":
                if ax in words and ax in self.pos and self.backlash.get(ax):
                    d = float(words[ax]) - self.pos[ax]
                    nd = 1 if d > 1e-6 else -1 if d < -1e-6 else self.dir[ax]
                    if nd != self.dir[ax]:
                        self.dir[ax] = nd
                        take[ax] = self._phys(ax, self.pos[ax], nd)      # reversal: take up the play first, in place
            if take:
                out.append("G1 " + " ".join(f"{a}{v:.3f}" for a, v in take.items()) + f" {feed}")
            sent = {a: (self._phys(a, float(v), self.dir[a]) if a in "XY" and a in self.pos else float(v)) for a, v in words.items()}
            for a, v in words.items():
                self.pos[a] = float(v)
            axes = " ".join(f"{a}{sent[a]:.3f}" for a in "XYZ" if a in sent)
            return out + [f"G1 {axes} {feed}"]
        return [line]

    def _send_raw(self, line, wait_ok=True, timeout=None):
        if line.startswith("<macro>"):
            self.run_macro(line[len("<macro>"):])
            return ''
        if line == "<idle>":
            self.wait_idle(timeout or 120)
            return ''
        if line.startswith("<refuse>"):
            self.last_error = "error: " + line[len("<refuse>"):]
            print(f"  MACHINE ERROR (not sent): {line[len('<refuse>'):]}")
            return ''
        return super()._send_raw(line, wait_ok, timeout)


def Machine(**kw):
    """The machine chosen in settings.py (MACHINE = "fluidnc" or "marlin"), not connected yet."""
    if getattr(S, "MACHINE", "marlin").lower() == "fluidnc":
        return FluidNCLink(**kw)
    return LerdgeSerial(**kw)


def open_machine(home=True, quiet=True):
    """Connects, sets absolute mm, and (unless home=False) homes: Z first, a lift to HOME_LIFT_Z, then XY. Returns the machine."""
    global VERBOSE
    if quiet:
        VERBOSE = False                                            # do not print every G-code line
    m = Machine()
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
