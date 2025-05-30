# driver.py

import car.msgParser
import car.carState
import car.carControl
import keyboard    # pip install keyboard
import time

class Driver(object):
    """
    Manual mode: WASD keys + speed-aware steering
    with smooth RPM+speed-based auto gearbox (1–6 forward, –1 reverse once stopped).
    Auto mode: TORCS-style AI (steer, gear, speed).
    """
    # General
    MAX_GEAR        = 6
    SPEED_STEP      = 30      # km/h per gear

    # Smooth shift parameters
    AUTO_SHIFT_UP_RPM     = 7000
    AUTO_SHIFT_DOWN_RPM   = 3000
    SHIFT_COOLDOWN        = 0.5   # seconds between gear shifts

    # Steering parameters (more forgiving)
    STEER_MAX        = 1.6     # full lock at low speed
    STEER_MIN_FACTOR = 0.7     # retain more lock at top speed
    STEER_SMOOTH_IN  = 0.80    # ramp‐in factor when key held
    STEER_SMOOTH_OUT = 0.60    # recenter factor when released
    STEER_DEADZONE   = 0.02    # snap small values to zero

    # Throttle/brake smoothing
    THROTTLE_INC     = 0.03    # per‐tick throttle increase
    THROTTLE_DEC     = 0.01    # per‐tick throttle decay
    BRAKE_HOLD       = 0.8     # brake value when holding S

    def __init__(self, stage, manual_mode=True):
        self.stage        = stage
        self.manual_mode  = manual_mode

        self.parser       = car.msgParser.MsgParser()
        self.state        = car.carState.CarState()
        self.control      = car.carControl.CarControl()

        # shared AI settings
        self.max_speed    = 200.0
        self.steer_lock   = 0.785398

        # manual state
        self.steer_val    = 0.0
        self.accel_val    = 0.0
        self.gear_val     = 1

        # shift cooldown timer
        self._last_shift  = time.time() - self.SHIFT_COOLDOWN

    def init(self):
        """Return the rangefinder angles (unchanged)."""
        angles = [0]*19
        for i in range(5):
            angles[i]      = -90 + i*15
            angles[18-i]   =  90 - i*15
        for i in range(5,9):
            angles[i]      = -20 + (i-5)*5
            angles[18-i]   =  20 - (i-5)*5
        return self.parser.stringify({'init': angles})

    def drive(self, msg):
        """Called each tick with the sensor string from TORCS."""
        self.state.setFromMsg(msg)
        if self.manual_mode:
            return self._manual_drive()
        else:
            return self._auto_drive()

    def _manual_drive(self):
        speed = abs(self.state.getSpeedX())
        rpm   = self.state.getRpm() or 0.0

        # 1) Steering via A/D with speed‑aware smoothing
        speed_factor = max(self.STEER_MIN_FACTOR,
                           1.0 - speed/self.max_speed)
        if keyboard.is_pressed("a"):
            target, smooth =  self.STEER_MAX * speed_factor, self.STEER_SMOOTH_IN
        elif keyboard.is_pressed("d"):
            target, smooth = -self.STEER_MAX * speed_factor, self.STEER_SMOOTH_IN
        else:
            target, smooth = 0.0, self.STEER_SMOOTH_OUT

        self.steer_val += (target - self.steer_val) * smooth
        if abs(self.steer_val) < self.STEER_DEADZONE:
            self.steer_val = 0.0

        # 2) Throttle/Brake/Reverse via W/S
        if keyboard.is_pressed("w"):
            # always go forward
            if self.gear_val < 0:
                self.gear_val = 1
            self.control.setBrake(0.0)
            self.accel_val = min(self.accel_val + self.THROTTLE_INC, 1.0)

        elif keyboard.is_pressed("s"):
            if self.gear_val > 0 and speed > 0.5:
                # braking
                self.control.setBrake(self.BRAKE_HOLD)
                self.accel_val = 0.0
            else:
                # reverse
                self.gear_val = -1
                self.control.setBrake(0.0)
                self.accel_val = min(self.accel_val + self.THROTTLE_INC, 1.0)
        else:
            # coast
            self.control.setBrake(0.0)
            self.accel_val = max(self.accel_val - self.THROTTLE_DEC, 0.0)

        # 3) Smooth RPM+speed‑based gearbox (forward only)
        now = time.time()
        if self.gear_val > 0 and (now - self._last_shift) >= self.SHIFT_COOLDOWN:
            up_speed_thr   = self.gear_val * self.SPEED_STEP
            down_speed_thr = (self.gear_val - 1) * self.SPEED_STEP

            # upshift?
            if rpm  > self.AUTO_SHIFT_UP_RPM and speed > up_speed_thr and self.gear_val < self.MAX_GEAR:
                self.gear_val += 1
                self._last_shift = now

            # downshift?
            elif (rpm  < self.AUTO_SHIFT_DOWN_RPM or speed < down_speed_thr) and self.gear_val > 1:
                self.gear_val -= 1
                self._last_shift = now

        # 4) Apply controls
        self.control.setSteer(self.steer_val)
        self.control.setAccel(self.accel_val)
        self.control.setGear(self.gear_val)

        return self.control.toMsg()

    def _auto_drive(self):
        # TORCS-style AI control
        # 1) steer
        angle, pos = self.state.angle, self.state.trackPos
        self.control.setSteer((angle - 0.5*pos) / self.steer_lock)
        # 2) gear
        rpm  = self.state.getRpm() or 0.0
        gear = self.state.getGear() or 1
        if rpm > self.AUTO_SHIFT_UP_RPM and gear < self.MAX_GEAR:
            gear += 1
        elif rpm < self.AUTO_SHIFT_DOWN_RPM and gear > 1:
            gear -= 1
        self.control.setGear(gear)
        # 3) speed
        speed = self.state.getSpeedX()
        if speed < self.max_speed:
            self.control.setBrake(0.0)
            self.control.setAccel(1.0)
        else:
            self.control.setAccel(0.0)
            self.control.setBrake(0.2)
        return self.control.toMsg()

    def onShutDown(self):
        pass

    def onRestart(self):
        pass
