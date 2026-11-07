#!/usr/bin/env python3
"""
mayhem_autopilot.py - Computer-vision autopilot macro for Roblox "Motorcycle Mayhem".

Per-frame pipeline
------------------
1. Capture   : grab ONLY the region of interest (central highway ahead of the bike)
               from the Roblox client area with `mss` (fast GDI/BitBlt capture).
2. Road mask : downscale -> HSV, threshold grey asphalt, keep the road blob under the
               bike and build a per-row "drivable corridor" (left..right road edge).
3. Obstacles : non-asphalt blobs inside the corridor (thin lane markings removed),
               extracted with contour detection -> bounding boxes.
4. Columns   : the forward path is split into perspective-aware Left / Center / Right
               columns; each gets a proximity-weighted obstacle density.
5. Decision  : Center blocked  -> dodge toward the side with the LOWEST density.
               Center clear    -> lane-keeping PD controller that follows the lane
                                  centre traced from road-mask / lane-marking scanlines
                                  (Canny + Hough line fallback).
6. Actuation : analog steering -> PWM-modulated A/D key presses via `pydirectinput`.

Hotkeys (global, work while Roblox is focused)
----------------------------------------------
  ESC or Q : emergency kill-switch (releases every key and exits)
  F8       : pause / resume
  F9       : save a full client-area screenshot to ./snapshots (for offline tuning)
  Mouse in top-left screen corner also aborts (pydirectinput fail-safe).

Usage
-----
  python mayhem_autopilot.py --debug               # live, with debug window
  python mayhem_autopilot.py --debug --dry-run     # live vision only, no key presses
  python mayhem_autopilot.py --image shot.png      # offline test on a screenshot
  python mayhem_autopilot.py --image shot.png --tune   # HSV/threshold trackbars
  python mayhem_autopilot.py --dump-config my.json # write default config to edit
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from ctypes import wintypes
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np

try:
    import mss
except ImportError:  # only required for live mode
    mss = None

try:
    import pydirectinput
except ImportError:  # only required for live mode without --dry-run
    pydirectinput = None


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    # --- window / ROI (fractions of the Roblox client area) -----------------
    window_title: str = "Roblox"
    roi_top: float = 0.08        # just below Roblox top stats bar (sees vehicles far ahead)
    roi_bottom: float = 0.50     # just above the player's motorcycle (rider is at y=0.52)
    roi_left: float = 0.15       # crops to the 5-lane highway area
    roi_right: float = 0.85
    process_width: int = 320     # ROI is downscaled to this width for speed
    resize_interp: str = "linear"  # "linear" (fast) or "area" (smoother, ~4 ms slower)
    bike_x: float = 0.50         # bike's horizontal position inside the ROI (0..1)

    # --- ego vehicle (player rider / bike exclusion mask) -------------------
    bike_mask_height: float = 0.10  # safety margin at bottom of ROI (rider's helmet peeks in)
    bike_mask_width: float = 0.10   # fraction of ROI width around bike_x (rider is ~4% wide)
    first_person: bool = False      # if True, sets minimal mask (for 1st person camera)

    # --- road (asphalt) HSV thresholds --------------------------------------
    road_s_max: int = 40         # asphalt is dark grey (S <= 40 in Motorcycle Mayhem)
    road_v_min: int = 35         # neither pitch black...
    road_v_max: int = 115        # ...nor bright white
    min_road_width_frac: float = 0.10

    # --- lane markings --------------------------------------------------------
    marking_s_max: int = 50      # white paint: low saturation, high value
    marking_v_min: int = 170     # white dashed paint in Motorcycle Mayhem
    marking_max_width_frac: float = 0.035  # blobs thinner than this = paint, not car
    use_lane_markings: bool = True

    # --- obstacle detection ---------------------------------------------------
    detect_mode: str = "diff"    # "diff" = anything not asphalt, "color" = saturated/dark
    car_s_min: int = 80          # used by "color" mode
    car_v_dark_max: int = 40     # used by "color" mode (tyres, windows, black cars)
    min_obstacle_area: float = 0.003   # fraction of processed ROI area
    max_obstacle_area: float = 0.60

    # --- column geometry (fractions of ROI width) ---------------------------
    center_col_bottom: float = 0.20    # centre column width matching Lane 3 (center lane)
    center_col_top: float = 0.10       # ...and at the far (top) edge -> perspective
    side_col_scale: float = 1.0        # side column width relative to centre column
    proximity_weight: float = 3.0      # near obstacles weigh up to (1 + this) x more

    # --- avoidance logic ------------------------------------------------------
    center_block_thresh: float = 0.05  # centre density that triggers a dodge
    release_ratio: float = 0.5         # stop dodging when centre < thresh * ratio
    side_clear_thresh: float = 0.04    # a side is "clear" below this density
    min_side_road: float = 0.35        # side needs this much road to be usable
    tie_margin: float = 0.01           # densities closer than this are a tie
    switch_margin: float = 0.10        # flip dodge direction if target gets this much worse
    dodge_commit_s: float = 0.25       # dodge nudge duration
    dodge_strength: float = 0.45       # gentle PWM pulsed nudge (prevents sharp spin-out)
    brake_when_boxed_in: bool = True
    brake_density: float = 0.15        # centre density that triggers braking when boxed in

    # --- lane tracking --------------------------------------------------------
    lane_method: str = "hybrid"  # "mask", "canny" or "hybrid" (mask, canny fallback)
    lane_scanlines: int = 14
    lane_top_frac: float = 0.25  # ignore rows above this (too far / noisy)
    lane_min_points: int = 4
    lane_fit_degree: int = 1
    canny_low: int = 60
    canny_high: int = 160
    hough_threshold: int = 18
    hough_max_slope: float = 3.0       # |dx/dy| above this = near-horizontal, ignored
    default_lane_half_width_frac: float = 0.08  # Canny fallback: half lane width / ROI width
    lane_count: int = 5                # lanes across the road (initial lane-width guess)

    # --- PD controller --------------------------------------------------------
    kp: float = 0.85                   # proportional gain (moderate for smooth tracking)
    kd: float = 0.05                   # derivative damping
    k_heading: float = 0.60
    lane_smoothing: float = 0.5        # EMA alpha on the lane error
    deadzone: float = 0.06
    full_lock: float = 0.95            # only continuous hold if steer > 0.95
    pwm_period: float = 0.05           # 50 ms cycle for responsive, smooth tapping
    min_pulse_s: float = 0.025         # shortest key tap; shorter taps fall between game frames

    # --- keys / runtime -------------------------------------------------------
    left_key: str = "a"
    right_key: str = "d"
    throttle_key: str = "w"
    brake_key: str = "s"
    hold_throttle: bool = True
    require_focus: bool = True         # only press keys while Roblox is foreground
    target_fps: int = 90               # 0 = uncapped
    window_refresh_s: float = 1.0
    countdown_s: int = 3

    # --- debug ----------------------------------------------------------------
    debug_scale: float = 2.0
    debug_window_pos: list | None = None   # [x, y]; None = top-left of the Roblox window
    debug_topmost: bool = True             # keep the debug window above the game
    debug_exclude_from_capture: bool = True  # hide it from our own screen grab (Win10 2004+)


def load_config(path: str | None, cfg: Config | None = None) -> Config:
    """Apply a JSON override file on top of `cfg` (defaults if None)."""
    cfg = cfg or Config()
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {f.name for f in fields(Config)}
        for key, value in data.items():
            if key in valid:
                setattr(cfg, key, value)
            else:
                print(f"[config] ignoring unknown key: {key}")
    return cfg


def roi_fracs(cfg: Config) -> tuple[float, float, float, float]:
    """Sanitised (top, bottom, left, right) ROI fractions. Tuner sliders can cross
    top/bottom; an empty crop would crash OpenCV, so enforce a minimum span."""
    span = 0.05
    top = min(max(cfg.roi_top, 0.0), 1.0 - span)
    bottom = min(max(cfg.roi_bottom, top + span), 1.0)
    left = min(max(cfg.roi_left, 0.0), 1.0 - span)
    right = min(max(cfg.roi_right, left + span), 1.0)
    return top, bottom, left, right


# =============================================================================
# Win32 helpers (window lookup, focus, global hotkeys) - pure ctypes
# =============================================================================
IS_WINDOWS = sys.platform == "win32"
VK_ESCAPE, VK_Q, VK_F8, VK_F9 = 0x1B, 0x51, 0x77, 0x78

if IS_WINDOWS:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    for _fn in ("IsWindowVisible", "IsIconic", "IsWindow", "GetWindowTextLengthW"):
        getattr(user32, _fn).argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user32.GetAsyncKeyState.restype = ctypes.c_short
    user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.FindWindowW.restype = wintypes.HWND
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, wintypes.UINT]
    user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    kernel32.GetConsoleWindow.restype = wintypes.HWND


def enable_dpi_awareness() -> None:
    """Make window coordinates match physical pixels on scaled (125%/150%) displays."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def find_game_window(title: str) -> int | None:
    """Return the HWND of the Roblox player window (exact title / class preferred)."""
    console = kernel32.GetConsoleWindow()
    candidates: list[tuple[int, int]] = []

    def callback(hwnd, _lparam):
        if hwnd == console or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if not length:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        cls = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls, 256)
        text = buf.value
        if text == title or cls.value == "WINDOWSCLIENT":
            candidates.append((0, hwnd))          # the real Roblox player
        elif title.lower() in text.lower():
            candidates.append((1, hwnd))          # e.g. a browser tab - last resort
        return True

    user32.EnumWindows(WNDENUMPROC(callback), 0)
    return min(candidates)[1] if candidates else None


def client_rect_on_screen(hwnd: int) -> tuple[int, int, int, int] | None:
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    pt = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return None
    if rect.right <= 0 or rect.bottom <= 0:
        return None
    return pt.x, pt.y, rect.right, rect.bottom


HWND_TOPMOST = -1
SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x0001, 0x0002, 0x0010
WDA_EXCLUDEFROMCAPTURE = 0x00000011


def pin_window(title: str, topmost: bool, exclude_from_capture: bool) -> dict:
    """Keep an OpenCV window above the game (without stealing focus) and optionally
    make it invisible to screen capture, so the bot never 'sees' its own overlay."""
    status = {"topmost": False, "excluded": False}
    if topmost:
        try:
            cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
            status["topmost"] = True
        except (cv2.error, AttributeError):
            pass
    if not IS_WINDOWS:
        return status
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        return status
    if topmost:
        status["topmost"] = bool(user32.SetWindowPos(
            hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE))
    if exclude_from_capture:
        status["excluded"] = bool(user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))
    return status


class Hotkeys:
    """Global hotkeys via GetAsyncKeyState (no hooks, works while the game has focus)."""

    def __init__(self) -> None:
        self._prev: dict[int, bool] = {}

    @staticmethod
    def _down(vk: int) -> bool:
        return bool(user32.GetAsyncKeyState(vk) & 0x8000)

    def _edge(self, vk: int) -> bool:
        now = self._down(vk)
        was = self._prev.get(vk, False)
        self._prev[vk] = now
        return now and not was

    def kill(self) -> bool:
        return self._down(VK_ESCAPE) or self._down(VK_Q)

    def pause_toggled(self) -> bool:
        return self._edge(VK_F8)

    def snapshot(self) -> bool:
        return self._edge(VK_F9)


# =============================================================================
# Screen capture
# =============================================================================
class GameCapture:
    """Captures only the ROI of the Roblox client area; tracks window moves/resizes."""

    def __init__(self, cfg: Config) -> None:
        if mss is None:
            raise RuntimeError("mss is not installed:  pip install mss")
        self.cfg = cfg
        self.sct = (getattr(mss, "MSS", None) or mss.mss)()   # mss>=10.2 renamed mss.mss -> mss.MSS
        self.hwnd: int | None = None
        self.client: tuple[int, int, int, int] | None = None
        self.region: dict | None = None
        self._last_lookup = 0.0

    def refresh(self, force: bool = False) -> bool:
        now = time.perf_counter()
        if not force and self.region and now - self._last_lookup < self.cfg.window_refresh_s:
            return True
        self._last_lookup = now
        if self.hwnd is None or not user32.IsWindow(self.hwnd):
            self.hwnd = find_game_window(self.cfg.window_title)
        if self.hwnd is None:
            self.region = None
            return False
        client = client_rect_on_screen(self.hwnd)
        if client is None:
            self.region = None
            return False
        x, y, w, h = client
        top, bottom, left, right = roi_fracs(self.cfg)
        self.client = client
        self.region = {
            "left": x + int(w * left),
            "top": y + int(h * top),
            "width": max(16, int(w * (right - left))),
            "height": max(16, int(h * (bottom - top))),
        }
        return True

    def grab(self) -> np.ndarray:
        shot = self.sct.grab(self.region)
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])   # BGRA -> BGR

    def grab_full(self) -> np.ndarray | None:
        if not self.client:
            return None
        x, y, w, h = self.client
        shot = self.sct.grab({"left": x, "top": y, "width": w, "height": h})
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])

    def is_foreground(self) -> bool:
        return self.hwnd is not None and user32.GetForegroundWindow() == self.hwnd


def crop_roi(full_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """Crop the ROI out of a full client-area screenshot (offline mode)."""
    h, w = full_bgr.shape[:2]
    top, bottom, left, right = roi_fracs(cfg)
    y0, y1 = int(h * top), int(h * bottom)
    x0, x1 = int(w * left), int(w * right)
    return full_bgr[y0:y1, x0:x1]


# =============================================================================
# Vision
# =============================================================================
COL_NAMES = ("L", "C", "R")
COL_COLORS = ((255, 160, 0), (0, 0, 255), (0, 220, 255))   # BGR: blue, red, yellow


@dataclass
class Obstacle:
    x: int
    y: int
    w: int
    h: int
    area: float
    column: int          # 0 = Left, 1 = Center, 2 = Right


@dataclass
class Perception:
    frame: np.ndarray                    # processed (downscaled) BGR ROI
    obstacles: list[Obstacle]
    density: np.ndarray                  # (3,) proximity-weighted obstacle density L/C/R
    road_frac: np.ndarray                # (3,) fraction of each column that is road
    lane_error: float | None = None      # + = lane centre is right of the bike
    lane_heading: float | None = None    # + = lane curves to the right
    lane_poly: np.ndarray | None = None  # x = polyval(poly, y)
    lane_points: np.ndarray | None = None
    lane_source: str = "none"
    masks: dict = field(default_factory=dict)


class Vision:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._geom_key: tuple | None = None
        self._vy: float | None = None

    # ------------------------------------------------------------------ setup
    def _build_geometry(self, h: int, w: int) -> None:
        """Pre-compute perspective column masks, proximity weights, ego mask, and kernels."""
        c = self.cfg
        t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]      # 0 far .. 1 near
        xs = np.arange(w, dtype=np.float32)[None, :]
        cx = c.bike_x * w
        half_c = 0.5 * w * (c.center_col_top + (c.center_col_bottom - c.center_col_top) * t)
        side = 2.0 * half_c * c.side_col_scale
        lb, rb = cx - half_c, cx + half_c

        # Ego vehicle (rider helmet, body, motorcycle) exclusion zone
        # This prevents the macro from detecting the player's own head/bike as an obstacle.
        ego = np.zeros((h, w), dtype=np.uint8)
        mask_h = 0.08 if c.first_person else c.bike_mask_height
        mask_w = 0.20 if c.first_person else c.bike_mask_width
        if mask_h > 0 and mask_w > 0:
            y_top = max(0, int(h * (1.0 - mask_h)))
            x_b_l = max(0, int(cx - 0.5 * mask_w * w))
            x_b_r = min(w - 1, int(cx + 0.5 * mask_w * w))
            top_w = 0.65 * mask_w * w
            x_t_l = max(0, int(cx - 0.5 * top_w))
            x_t_r = min(w - 1, int(cx + 0.5 * top_w))
            poly = np.array([[x_b_l, h], [x_t_l, y_top], [x_t_r, y_top], [x_b_r, h]], dtype=np.int32)
            cv2.fillPoly(ego, [poly], 255)
            self.ego_poly = poly
        else:
            self.ego_poly = None
        self.ego_mask = ego

        col = np.full((h, w), -1, np.int8)
        col[(xs >= lb - side) & (xs < lb)] = 0
        col[(xs >= lb) & (xs <= rb)] = 1
        col[(xs > rb) & (xs <= rb + side)] = 2

        # Weight drops to 0 inside the ego mask so the bike never contributes to obstacle threat
        ego_f = (ego == 0).astype(np.float32)
        weight = np.broadcast_to(1.0 + c.proximity_weight * t ** 2, (h, w)) * ego_f

        self.col_map = col
        self.col_w = np.stack([(col == i) * weight for i in range(3)]).astype(np.float32)
        self.col_w_sum = self.col_w.reshape(3, -1).sum(axis=1) + 1e-6
        # column boundary lines for the debug overlay: (x_top, x_bottom)
        self.col_lines = [
            (float(v[0, 0]), float(v[-1, 0])) for v in (lb - side, lb, rb, rb + side)
        ]

        self.k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        k = max(5, int(w * 0.02) | 1)
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        km = max(3, int(w * self.cfg.marking_max_width_frac) | 1)
        self.k_mark = cv2.getStructuringElement(cv2.MORPH_RECT, (km, km))
        # Horizontal curb / guardrail erosion kernel to prevent barriers from triggering car detection
        kw = max(5, int(w * 0.04) | 1)
        self.k_curb = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
        self.xs = np.arange(w, dtype=np.float32)[None, :]
        self._geom_key = self._geometry_key(h, w)

    def _geometry_key(self, h: int, w: int) -> tuple:
        """Everything _build_geometry depends on: rebuild only when one of these changes
        (frame size, or a tuner slider that moves the columns / ego mask / kernels)."""
        c = self.cfg
        return (h, w, c.bike_x, c.bike_mask_height, c.bike_mask_width, c.first_person,
                c.center_col_bottom, c.center_col_top, c.side_col_scale, c.proximity_weight,
                c.marking_max_width_frac)

    # ------------------------------------------------------------- road mask
    @staticmethod
    def _main_road_component(road: np.ndarray) -> np.ndarray:
        """Keep the asphalt blob the bike is riding on (most pixels in the bottom band)."""
        h = road.shape[0]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(road, connectivity=8)
        if n <= 1:
            return road
        counts = np.bincount(labels[int(h * 0.6):].ravel(), minlength=n)
        counts[0] = 0
        best = int(counts.argmax()) if counts.max() > 0 else 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
        return np.where(labels == best, 255, 0).astype(np.uint8)

    def _row_spans(self, road_main: np.ndarray):
        """Per-row road edges -> filled corridor (cars inside the road become 'holes')."""
        h, w = road_main.shape
        m = road_main > 0
        has = m.any(axis=1)
        left = m.argmax(axis=1).astype(np.float32)
        right = (w - 1 - m[:, ::-1].argmax(axis=1)).astype(np.float32)
        valid = has & ((right - left) >= self.cfg.min_road_width_frac * w)
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            return np.zeros((h, w), bool), left, right, valid
        rows = np.arange(h)
        # interpolate rows where a vehicle hides the whole road width
        left_i = np.interp(rows, idx, left[idx]).astype(np.float32)
        right_i = np.interp(rows, idx, right[idx]).astype(np.float32)
        corridor = (self.xs >= left_i[:, None]) & (self.xs <= right_i[:, None])
        corridor[: idx[0]] = False          # nothing above the horizon
        return corridor, left_i, right_i, valid

    # --------------------------------------------------------------- process
    def process(self, roi_bgr: np.ndarray) -> Perception:
        c = self.cfg
        h0, w0 = roi_bgr.shape[:2]
        if w0 > c.process_width:
            # INTER_LINEAR is ~6x faster than INTER_AREA for non-integer ratios
            scale = c.process_width / w0
            interp = cv2.INTER_AREA if c.resize_interp == "area" else cv2.INTER_LINEAR
            img = cv2.resize(roi_bgr, (c.process_width, max(8, round(h0 * scale))),
                             interpolation=interp)
        else:
            img = roi_bgr
        h, w = img.shape[:2]
        if self._geom_key != self._geometry_key(h, w):
            self._build_geometry(h, w)

        # un-blurred HSV keeps thin lane paint intact; morphology handles the noise
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        blur = cv2.GaussianBlur(img, (3, 3), 0)      # only used for Canny

        # 1) road mask + corridor ------------------------------------------
        road = cv2.inRange(hsv, (0, 0, c.road_v_min), (180, c.road_s_max, c.road_v_max))
        road = cv2.morphologyEx(road, cv2.MORPH_OPEN, self.k3)
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, self.k3)
        road_main = self._main_road_component(road)
        corridor, left_e, right_e, valid_rows = self._row_spans(road_main)
        corridor_u8 = corridor.astype(np.uint8) * 255

        # 2) lane markings (thin bright paint only) ------------------------
        white = cv2.inRange(hsv, (0, 0, c.marking_v_min), (180, c.marking_s_max, 255))
        yellow = cv2.inRange(hsv, (15, 90, 140), (35, 255, 255))
        paint = cv2.bitwise_and(cv2.bitwise_or(white, yellow), corridor_u8)
        blobs = cv2.morphologyEx(paint, cv2.MORPH_OPEN, self.k_mark)   # wide = car, not paint
        markings = cv2.bitwise_and(paint, cv2.bitwise_not(blobs))

        # 3) obstacles -----------------------------------------------------
        if c.detect_mode == "color":
            cand = cv2.bitwise_or(cv2.inRange(sat, c.car_s_min, 255),
                                  cv2.inRange(val, 0, c.car_v_dark_max))
        else:  # "diff": everything inside the corridor that is not asphalt
            cand = cv2.bitwise_not(road_main)
        # Erode corridor horizontally so guardrails/barriers at the road edge are not counted as cars
        drivable = cv2.erode(corridor_u8, self.k_curb)
        obst = cv2.bitwise_and(cand, drivable)
        obst = cv2.bitwise_and(obst, cv2.bitwise_not(markings))
        if self.ego_mask.any():
            # Cut out the player's own helmet, shoulders, and motorcycle
            obst = cv2.bitwise_and(obst, cv2.bitwise_not(self.ego_mask))
        obst = cv2.morphologyEx(obst, cv2.MORPH_OPEN, self.k3)
        obst = cv2.morphologyEx(obst, cv2.MORPH_CLOSE, self.k_close)   # merge car parts

        contours, _ = cv2.findContours(obst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_a, max_a = c.min_obstacle_area * h * w, c.max_obstacle_area * h * w
        kept = np.zeros_like(obst)
        obstacles: list[Obstacle] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_a <= area <= max_a):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh > 3.5 * bw and bw < 0.05 * w:      # leftover lane dash / pole
                continue
            # Discard any residual contour centered inside the ego exclusion zone
            cx_box = min(w - 1, max(0, int(x + bw / 2.0)))
            cy_box = min(h - 1, max(0, int(y + bh / 2.0)))
            if self.ego_mask[cy_box, cx_box] > 0:
                continue
            # Obstacle center must fall inside one of the 3 driving columns (L, C, R)
            center_col = self.col_map[cy_box, cx_box]
            if center_col < 0:
                continue                              # outside path columns (shoulder/guardrail)
            cols = self.col_map[y:y + bh, x:x + bw]
            counts = np.bincount((cols[cols >= 0]).ravel(), minlength=3)
            if counts.sum() < 0.20 * (bw * bh):
                continue                              # reject marginal boundary grazing
            obstacles.append(Obstacle(x, y, bw, bh, area, int(center_col)))
            cv2.drawContours(kept, [cnt], -1, 255, cv2.FILLED)

        kept_f = kept.astype(np.float32) * (1.0 / 255.0)
        density = (self.col_w * kept_f[None]).reshape(3, -1).sum(axis=1) / self.col_w_sum
        road_frac = (self.col_w * corridor[None]).reshape(3, -1).sum(axis=1) / self.col_w_sum

        p = Perception(img, obstacles, density, road_frac,
                       masks={"road": road_main, "obstacles": kept, "markings": markings, "ego": self.ego_mask})

        # 4) lane tracking -------------------------------------------------
        lane = None
        if c.lane_method in ("mask", "hybrid"):
            lane = self._lane_from_mask(left_e, right_e, valid_rows, markings, h, w)
            if lane:
                p.lane_source = lane[3]
        if lane is None and c.lane_method in ("canny", "hybrid"):
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            lane = self._lane_from_canny(gray, corridor_u8, kept, h, w, p)
            if lane:
                p.lane_source = "canny"
        if lane:
            poly, pts, near_lane_w = lane[:3]
            y_near, y_far = h - 1, int(h * c.lane_top_frac)
            cx = c.bike_x * w
            e_near = float(np.polyval(poly, y_near)) - cx
            e_far = float(np.polyval(poly, y_far)) - cx
            # Perspective: on a STRAIGHT road a lateral offset e_near shrinks toward the
            # vanishing point, so the far offset is e_near * (y_far - vy) / (y_near - vy)
            # even with zero heading error. Only the deviation from that is real
            # heading/curvature; otherwise the heading term cancels part of the lane
            # correction.
            vy = self._vanishing_y(left_e, right_e, valid_rows, h)
            expected_far = e_near * (y_far - vy) / (y_near - vy) if vy is not None else e_near
            lane_half = max(0.5 * near_lane_w, 0.03 * w)
            p.lane_poly, p.lane_points = poly, pts
            p.lane_error = float(np.clip(e_near / lane_half, -1.5, 1.5))
            p.lane_heading = float(np.clip((e_far - expected_far) / lane_half, -1.5, 1.5))
        return p

    # ------------------------------------------------------ vanishing point
    def _vanishing_y(self, left_e, right_e, valid_rows, h):
        """Row (ROI coords, usually negative = above the ROI) where the road edges meet.
        Road width is linear in y on flat ground, so fit width = a*y + b and solve for 0.
        Smoothed across frames; returns the last good value when the fit is unreliable."""
        idx = np.flatnonzero(valid_rows)
        if idx.size >= 8:
            widths = (right_e - left_e)[idx]
            a, b = np.polyfit(idx.astype(np.float32), widths, 1)
            if a > 1e-3:
                vy = float(np.clip(-b / a, -20.0 * h, -1.0))
                self._vy = vy if self._vy is None else 0.8 * self._vy + 0.2 * vy
        return self._vy

    # ------------------------------------------------------- lane: road mask
    def _lane_from_mask(self, left_e, right_e, valid_rows, markings, h, w):
        """Scanline lane tracer: walks up from the bike, locking onto the nearest left
        and right lane markings.

        Lane width is tracked as a FRACTION OF ROAD WIDTH, because both shrink by the
        same perspective factor; a fixed pixel width measured at the bottom is too wide
        at far rows and drags the centre sideways whenever only one line is visible.
        Missing dashes never fall back to the outer road edge (that pulled the bike
        across all lanes). The road centre is used only when no markings exist at all.
        """
        c = self.cfg
        if not valid_rows.any():
            return None
        rows = np.linspace(h - 1, int(h * c.lane_top_frac), c.lane_scanlines).astype(int)
        road_w = np.maximum(right_e - left_e, 1.0)
        ratio = 1.0 / max(1, c.lane_count)            # lane width / road width
        margin = max(3.0, 0.015 * w)
        ref = c.bike_x * w
        pts: list[tuple[float, float]] = []
        near_lane_w = None

        if c.use_lane_markings:
            for r in rows:
                if not valid_rows[r]:
                    continue
                lw = ratio * float(road_w[r])           # expected lane width at this row
                mk = np.flatnonzero(markings[r])
                lb = rb = None
                if mk.size:
                    max_d = 0.85 * lw                   # rejects far lanes and guardrails
                    lm = mk[(mk < ref - margin) & (ref - mk <= max_d)]
                    rm = mk[(mk > ref + margin) & (mk - ref <= max_d)]
                    lb = float(lm.max()) if lm.size else None
                    rb = float(rm.min()) if rm.size else None
                if lb is not None and rb is not None:
                    measured = rb - lb
                    if 0.5 * lw <= measured <= 1.8 * lw:
                        ratio = 0.5 * ratio + 0.5 * measured / float(road_w[r])
                        lw = ratio * float(road_w[r])
                        centre = 0.5 * (lb + rb)
                    elif abs(lb - ref) < abs(rb - ref):
                        centre = lb + 0.5 * lw
                    else:
                        centre = rb - 0.5 * lw
                elif lb is not None:                    # dashed gap on the right
                    centre = lb + 0.5 * lw
                elif rb is not None:                    # dashed gap on the left
                    centre = rb - 0.5 * lw
                else:
                    continue
                if abs(centre - ref) > 1.2 * lw:
                    continue
                pts.append((centre, float(r)))
                ref = 0.65 * ref + 0.35 * centre        # follow the curve smoothly upward
            near_lane_w = ratio * float(road_w[h - 1])
            source = "mask"

        if len(pts) < c.lane_min_points:
            # No usable markings: centre of the road (right on single-lane roads only).
            pts = [(0.5 * float(left_e[r] + right_e[r]), float(r)) for r in rows if valid_rows[r]]
            near_lane_w = float(road_w[h - 1])
            source = "road"
            if len(pts) < c.lane_min_points:
                return None

        pts = np.array(pts, np.float32)
        deg = max(0, min(c.lane_fit_degree, len(pts) - 2))
        poly = np.polyfit(pts[:, 1], pts[:, 0], deg)
        res = np.abs(pts[:, 0] - np.polyval(poly, pts[:, 1]))
        keep = res <= max(2.5 * float(np.median(res)), 0.04 * w)   # reject outliers
        if c.lane_min_points <= keep.sum() < len(pts):
            pts = pts[keep]
            poly = np.polyfit(pts[:, 1], pts[:, 0], deg)
        return poly, pts, near_lane_w, source

    # ------------------------------------------------------ lane: Canny/Hough
    def _lane_from_canny(self, gray, corridor_u8, obstacles_u8, h, w, p: Perception):
        c = self.cfg
        edges = cv2.Canny(gray, c.canny_low, c.canny_high)
        edges[: int(h * c.lane_top_frac)] = 0
        if corridor_u8.any():
            edges = cv2.bitwise_and(edges, cv2.dilate(corridor_u8, self.k_close))
        edges = cv2.bitwise_and(edges, cv2.bitwise_not(cv2.dilate(obstacles_u8, self.k_close)))
        p.masks["edges"] = edges
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, c.hough_threshold,
                                minLineLength=max(8, int(w * 0.08)),
                                maxLineGap=max(4, int(w * 0.04)))
        if lines is None:
            return None
        ref, yb = c.bike_x * w, h - 1
        best_l = best_r = None                 # (distance, a, b) with x = a*y + b
        for x1, y1, x2, y2 in lines.reshape(-1, 4):   # (N,1,4) in OpenCV 4, (N,4) in 5
            dy = float(y2 - y1)
            if abs(dy) < 1.0:
                continue
            a = (x2 - x1) / dy
            if abs(a) > c.hough_max_slope:
                continue
            b = x1 - a * y1
            d = a * yb + b - ref
            if d < 0 and (best_l is None or -d < best_l[0]):
                best_l = (-d, a, b)
            elif d >= 0 and (best_r is None or d < best_r[0]):
                best_r = (d, a, b)
        half_lane = c.default_lane_half_width_frac * w
        if best_l and best_r:
            a, b = 0.5 * (best_l[1] + best_r[1]), 0.5 * (best_l[2] + best_r[2])
        elif best_l:
            a, b = best_l[1], best_l[2] + half_lane
        elif best_r:
            a, b = best_r[1], best_r[2] - half_lane
        else:
            return None
        ys = np.linspace(yb, int(h * c.lane_top_frac), 6, dtype=np.float32)
        pts = np.stack([a * ys + b, ys], axis=1)
        return np.array([a, b]), pts, half_lane * 2.0, "canny"


# =============================================================================
# Decision making
# =============================================================================
class Controller:
    """Obstacle avoidance (column densities) layered over a lane-keeping PD loop."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.dodge_dir = 0           # -1 left, +1 right, 0 none
        self.dodge_until = 0.0
        self.err_f: float | None = None
        self.prev_err: float | None = None
        self.prev_t: float | None = None

    def _reset_pd(self) -> None:
        self.err_f = self.prev_err = None

    def update(self, p: Perception, now: float):
        c = self.cfg
        dens = p.density
        eff = dens.copy()
        for i in (0, 2):                         # a side without road is not an escape
            if p.road_frac[i] < c.min_side_road:
                eff[i] = max(eff[i], 1.0)
        centre = dens[1]
        blocked = centre > c.center_block_thresh

        # --- maintain / release an ongoing dodge ----------------------------
        if self.dodge_dir:
            committed = now < self.dodge_until
            still_blocked = centre > c.center_block_thresh * c.release_ratio
            if not (committed or still_blocked):
                self.dodge_dir = 0
            else:
                tgt, other = (0, 2) if self.dodge_dir < 0 else (2, 0)
                if eff[tgt] > eff[other] + c.switch_margin:   # escape route closed
                    self.dodge_dir = -self.dodge_dir
                    self.dodge_until = now + c.dodge_commit_s

        # --- start a new dodge: steer to the side with the LOWEST density ----
        if not self.dodge_dir and blocked:
            if abs(eff[0] - eff[2]) <= c.tie_margin:
                # Tie: both sides have similar clearance
                # 1) Prefer the side with more drivable road margin
                if abs(p.road_frac[0] - p.road_frac[2]) > 0.05:
                    self.dodge_dir = -1 if p.road_frac[0] > p.road_frac[2] else 1
                elif p.lane_error is not None and abs(p.lane_error) > 0.05:
                    # 2) Dodge towards current lane center
                    self.dodge_dir = -1 if p.lane_error < 0 else 1
                else:
                    # 3) Default: Left (passing lane) is safer than right shoulder/sea
                    self.dodge_dir = -1
            else:
                self.dodge_dir = -1 if eff[0] < eff[2] else 1
            self.dodge_until = now + c.dodge_commit_s

        boxed_in = bool(blocked and min(eff[0], eff[2]) > c.side_clear_thresh)
        brake = bool(c.brake_when_boxed_in and boxed_in and centre > c.brake_density)
        throttle = c.hold_throttle and not brake

        # --- steering ---------------------------------------------------------
        if self.dodge_dir:
            self._reset_pd()
            steer = self.dodge_dir * c.dodge_strength
            action = f"DODGE {'LEFT' if self.dodge_dir < 0 else 'RIGHT'}"
            if boxed_in:
                action += " (boxed in)"
        elif p.lane_error is not None:
            err = p.lane_error + c.k_heading * (p.lane_heading or 0.0)
            a = c.lane_smoothing
            self.err_f = err if self.err_f is None else (1 - a) * self.err_f + a * err
            deriv = 0.0
            if self.prev_err is not None and self.prev_t is not None:
                deriv = (self.err_f - self.prev_err) / max(now - self.prev_t, 1e-3)
            self.prev_err = self.err_f
            steer = c.kp * self.err_f + c.kd * deriv
            action = f"LANE ({p.lane_source})"
        else:
            self._reset_pd()
            steer = 0.0
            action = "NO LANE"
        self.prev_t = now
        if brake:
            action += " +BRAKE"
        return float(np.clip(steer, -1.0, 1.0)), throttle, brake, action


# =============================================================================
# Actuation
# =============================================================================
class KeyActuator:
    """Turns analog steering into PWM key taps; only sends events on state changes."""

    def __init__(self, cfg: Config, enabled: bool) -> None:
        self.cfg = cfg
        self.enabled = enabled
        self.down: set[str] = set()
        if enabled:
            if pydirectinput is None:
                raise RuntimeError("pydirectinput is not installed:  pip install pydirectinput")
            pydirectinput.PAUSE = 0          # crucial: default adds 0.1 s per call

    def _set(self, key: str, state: bool) -> None:
        if state and key not in self.down:
            if self.enabled:
                pydirectinput.keyDown(key)
            self.down.add(key)
        elif not state and key in self.down:
            if self.enabled:
                pydirectinput.keyUp(key)
            self.down.discard(key)

    def apply(self, steer: float, throttle: bool, brake: bool, now: float) -> None:
        c = self.cfg
        mag = abs(steer)
        if mag < c.deadzone:
            on = False
        elif mag >= c.full_lock:
            on = True
        else:
            # Frequency modulation: expand period for low duty cycle so each pulse
            # is at least min_pulse_s long (prevents taps from falling between 60 Hz game frames)
            period = max(c.pwm_period, c.min_pulse_s / max(mag, 1e-4))
            on = (now % period) < (period * mag)
        self._set(c.left_key, on and steer < 0)
        self._set(c.right_key, on and steer > 0)
        self._set(c.brake_key, brake)
        self._set(c.throttle_key, throttle and not brake)

    def release_all(self) -> None:
        c = self.cfg
        all_keys = self.down.union({c.left_key, c.right_key, c.throttle_key, c.brake_key})
        for key in all_keys:
            try:
                if self.enabled:
                    pydirectinput.keyUp(key)
            except Exception:
                pass
        self.down.clear()


# =============================================================================
# Debug overlay
# =============================================================================
def draw_debug(p: Perception, vision: Vision, cfg: Config, hud: dict, show_masks: bool) -> np.ndarray:
    img = p.frame.copy()
    h, w = img.shape[:2]

    # tint road green, detected obstacles red
    tint = img.copy()
    tint[p.masks["road"] > 0] = (0, 160, 0)
    tint[p.masks["obstacles"] > 0] = (0, 0, 255)
    img = cv2.addWeighted(tint, 0.30, img, 0.70, 0)

    s = cfg.debug_scale
    vis = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_NEAREST)
    S = lambda v: int(round(v * s))  # noqa: E731

    # column boundaries
    for i, (xt, xb) in enumerate(vision.col_lines):
        colour = (200, 200, 200) if i in (0, 3) else (255, 255, 255)
        cv2.line(vis, (S(xt), 0), (S(xb), S(h - 1)), colour, 1, cv2.LINE_AA)

    # ego vehicle exclusion zone (player bike & helmet)
    if vision.ego_poly is not None:
        poly_s = (vision.ego_poly.astype(np.float32) * s).astype(np.int32)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [poly_s], (35, 35, 35))
        vis = cv2.addWeighted(overlay, 0.40, vis, 0.60, 0)
        cv2.polylines(vis, [poly_s], True, (0, 255, 255), 1, cv2.LINE_AA)
        p_top = poly_s[1]
        cv2.putText(vis, "EGO BIKE (EXCLUDED)", (max(10, p_top[0] - 65), max(15, p_top[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

    # obstacle bounding boxes
    for o in p.obstacles:
        col = COL_COLORS[o.column]
        cv2.rectangle(vis, (S(o.x), S(o.y)), (S(o.x + o.w), S(o.y + o.h)), col, 2)
        cv2.putText(vis, COL_NAMES[o.column], (S(o.x), max(12, S(o.y) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # lane trace
    if p.lane_poly is not None:
        ys = np.linspace(h - 1, int(h * cfg.lane_top_frac), 20)
        xs = np.polyval(p.lane_poly, ys)
        pts = np.stack([xs * s, ys * s], axis=1).astype(np.int32)
        cv2.polylines(vis, [pts], False, (255, 0, 255), 2, cv2.LINE_AA)
        for x, y in p.lane_points:
            cv2.circle(vis, (S(x), S(y)), 3, (255, 255, 0), -1)
    bx = S(cfg.bike_x * w)
    cv2.drawMarker(vis, (bx, S(h - 1) - 6), (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 14, 2)

    # density bars per column
    for i, d in enumerate(p.density):
        x0 = 10 + i * 46
        bar = int(min(d / max(cfg.center_block_thresh * 2, 1e-6), 1.0) * 60)
        cv2.rectangle(vis, (x0, 120 - bar), (x0 + 30, 120), COL_COLORS[i], -1)
        cv2.rectangle(vis, (x0, 60), (x0 + 30, 120), (255, 255, 255), 1)
        cv2.putText(vis, f"{COL_NAMES[i]} {d:.2f}", (x0 - 2, 136),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # HUD text
    err = "--" if p.lane_error is None else f"{p.lane_error:+.2f}"
    head = "--" if p.lane_heading is None else f"{p.lane_heading:+.2f}"
    lines = [
        f"FPS {hud.get('fps', 0):5.1f}   {hud.get('action', '')}",
        f"lane[{p.lane_source}] off {err} head {head}  steer {hud.get('steer', 0):+.2f}",
        f"keys: {' '.join(sorted(hud.get('keys', []))).upper() or '-'}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(vis, text, (10, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (10, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # steering bar
    W, H = vis.shape[1], vis.shape[0]
    cx, y = W // 2, H - 14
    cv2.rectangle(vis, (cx - 100, y - 6), (cx + 100, y + 6), (80, 80, 80), -1)
    sx = int(cx + hud.get("steer", 0.0) * 100)
    cv2.rectangle(vis, (min(cx, sx), y - 6), (max(cx, sx), y + 6), (0, 255, 255), -1)
    cv2.line(vis, (cx, y - 9), (cx, y + 9), (255, 255, 255), 1)

    if show_masks:
        def tile(m):
            return cv2.resize(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR), (W // 2, H // 2),
                              interpolation=cv2.INTER_NEAREST)
        side = p.masks.get("edges", p.masks["markings"])
        grid = np.vstack([np.hstack([tile(p.masks["road"]), tile(p.masks["obstacles"])]),
                          np.hstack([tile(p.masks["markings"]), tile(side)])])
        vis = np.hstack([vis, grid])
    return vis


# =============================================================================
# Tuning trackbars
# =============================================================================
TUNABLES = [  # (config field, max, scale) -> slider value = field * scale
    ("road_s_max", 255, 1), ("road_v_min", 255, 1), ("road_v_max", 255, 1),
    ("marking_v_min", 255, 1), ("marking_s_max", 255, 1),
    ("car_s_min", 255, 1), ("car_v_dark_max", 255, 1),
    ("bike_mask_height", 100, 100), ("bike_mask_width", 100, 100),
    ("roi_top", 100, 100), ("roi_bottom", 100, 100),
    ("canny_low", 400, 1), ("canny_high", 400, 1),
    ("center_block_thresh", 300, 1000), ("side_clear_thresh", 300, 1000),
    ("kp", 300, 100), ("kd", 100, 100), ("k_heading", 200, 100),
    ("dodge_strength", 100, 100),
]


def create_tuner(cfg: Config) -> None:
    cv2.namedWindow("Tuning", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tuning", 420, 40 * len(TUNABLES))
    for name, vmax, scale in TUNABLES:
        def on_change(v, name=name, scale=scale):
            setattr(cfg, name, v / scale if scale != 1 else int(v))
        cv2.createTrackbar(name, "Tuning", int(round(getattr(cfg, name) * scale)), vmax, on_change)


def save_tuned(cfg: Config, path: str = "tuned_config.json") -> None:
    Path(path).write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"[tune] configuration saved to {path}")


# =============================================================================
# Run modes
# =============================================================================
DEBUG_WIN = "Mayhem Autopilot - Debug"


def run_live(cfg: Config, args) -> None:
    if not IS_WINDOWS:
        sys.exit("Live mode requires Windows (Roblox + DirectInput).")
    enable_dpi_awareness()
    cap = GameCapture(cfg)
    vision, ctrl = Vision(cfg), Controller(cfg)
    act = KeyActuator(cfg, enabled=not args.dry_run)
    keys = Hotkeys()
    show_masks = False

    print("Looking for the Roblox window...")
    while not cap.refresh(force=True):
        if keys.kill():
            return
        time.sleep(0.5)
    print(f"Found Roblox client at {cap.client}; ROI = {cap.region}")
    print("ESC/Q = kill-switch | F8 = pause | F9 = snapshot"
          + ("   [DRY RUN - no keys will be pressed]" if args.dry_run else ""))
    if not args.no_countdown:
        for i in range(cfg.countdown_s, 0, -1):
            print(f"  starting in {i}... (focus the Roblox window)")
            time.sleep(1)
            if keys.kill():
                return

    if args.debug:
        cv2.namedWindow(DEBUG_WIN, cv2.WINDOW_AUTOSIZE)
        pos = cfg.debug_window_pos or (cap.client[0] + 10, cap.client[1] + 10)
        cv2.moveWindow(DEBUG_WIN, int(pos[0]), int(pos[1]))
        if args.tune:
            create_tuner(cfg)

    frame_dt = 1.0 / cfg.target_fps if cfg.target_fps > 0 else 0.0
    paused = False
    pinned_at = None                     # last time the debug window was re-pinned
    fps, last = 0.0, time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            if keys.kill():
                print("Kill-switch pressed - exiting.")
                break
            if keys.pause_toggled():
                paused = not paused
                act.release_all()
                ctrl._reset_pd()
                print("PAUSED" if paused else "RESUMED")
            if not cap.refresh():
                act.release_all()
                time.sleep(0.25)
                continue
            if keys.snapshot():
                full = cap.grab_full()
                if full is not None:
                    Path("snapshots").mkdir(exist_ok=True)
                    out = Path("snapshots") / time.strftime("mayhem_%Y%m%d_%H%M%S.png")
                    cv2.imwrite(str(out), full)
                    print(f"Snapshot saved: {out}")

            try:
                frame = cap.grab()
            except Exception:
                cap.refresh(force=True)
                continue

            p = vision.process(frame)
            steer, throttle, brake, action = ctrl.update(p, t0)
            focused = cap.is_foreground() or not cfg.require_focus
            if paused:
                act.release_all()
                action = "PAUSED (F8)"
            elif not focused:
                act.release_all()
                action = "WAITING FOR ROBLOX FOCUS"
            else:
                act.apply(steer, throttle, brake, t0)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now

            if args.debug:
                hud = {"fps": fps, "action": action, "steer": steer, "keys": act.down}
                cv2.imshow(DEBUG_WIN, draw_debug(p, vision, cfg, hud, show_masks))
                k = cv2.waitKey(1) & 0xFF
                if pinned_at is None or now - pinned_at > 2.0:
                    st = pin_window(DEBUG_WIN, cfg.debug_topmost, cfg.debug_exclude_from_capture)
                    if args.tune:
                        pin_window("Tuning", cfg.debug_topmost, cfg.debug_exclude_from_capture)
                    if pinned_at is None:
                        print(f"Debug window: always-on-top={st['topmost']}, "
                              f"hidden-from-capture={st['excluded']}")
                        if cfg.debug_exclude_from_capture and not st["excluded"]:
                            print("  WARNING: capture exclusion unsupported (needs Windows 10 2004+)."
                                  " Keep the debug window OFF the road area or the bot sees it.")
                    pinned_at = now
                if k in (27, ord("q")):
                    break
                if k == ord("m"):
                    show_masks = not show_masks

            if frame_dt:
                remaining = frame_dt - (time.perf_counter() - t0)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # includes pydirectinput.FailSafeException
        print(f"Stopped: {exc!r}")
    finally:
        act.release_all()
        cv2.destroyAllWindows()
        if args.tune:
            save_tuned(cfg)
        print("All keys released. Bye.")


def run_offline(cfg: Config, args) -> None:
    """Run the vision + decision pipeline on a screenshot or recording (no inputs)."""
    vision, ctrl = Vision(cfg), Controller(cfg)
    show = not args.save_debug or args.tune
    show_masks = args.masks
    if args.tune:
        create_tuner(cfg)

    def step(full: np.ndarray, t: float) -> tuple[np.ndarray, str, float]:
        roi = full if args.roi_only else crop_roi(full, cfg)
        t0 = time.perf_counter()
        p = vision.process(roi)
        steer, throttle, brake, action = ctrl.update(p, t)
        ms = (time.perf_counter() - t0) * 1000
        keys = set()
        if abs(steer) >= cfg.deadzone:
            keys.add(cfg.left_key if steer < 0 else cfg.right_key)
        if brake:
            keys.add(cfg.brake_key)
        elif throttle:
            keys.add(cfg.throttle_key)
        hud = {"fps": 1000.0 / max(ms, 1e-3), "action": action, "steer": steer, "keys": keys}
        dens = " ".join(f"{n}={d:.3f}" for n, d in zip(COL_NAMES, p.density))
        summary = (f"{action:<28} steer={steer:+.2f}  {dens}  obstacles={len(p.obstacles)}"
                   f"  lane={p.lane_source}  {ms:.1f} ms")
        return draw_debug(p, vision, cfg, hud, show_masks), summary, steer

    pinned = False

    def pin_all() -> None:
        for title in (DEBUG_WIN, "Tuning") if args.tune else (DEBUG_WIN,):
            pin_window(title, cfg.debug_topmost, cfg.debug_exclude_from_capture)

    if args.image:
        full = cv2.imread(args.image)
        if full is None:
            sys.exit(f"Cannot read image: {args.image}")
        vis, summary, _ = step(full, time.perf_counter())
        print(summary)
        if args.save_debug:
            cv2.imwrite(args.save_debug, vis)
            print(f"Debug overlay saved to {args.save_debug}")
        while show:
            ctrl.dodge_dir = 0                   # re-evaluate from scratch on each tweak
            vis, _, _ = step(full, time.perf_counter())
            cv2.imshow(DEBUG_WIN, vis)
            k = cv2.waitKey(30) & 0xFF
            if not pinned:
                pin_all()
                pinned = True
            if k in (27, ord("q")) or cv2.getWindowProperty(DEBUG_WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if k == ord("m"):
                show_masks = not show_masks
    else:
        vid = cv2.VideoCapture(args.video)
        if not vid.isOpened():
            sys.exit(f"Cannot open video: {args.video}")
        fps = vid.get(cv2.CAP_PROP_FPS) or 30.0
        n = 0
        while True:
            ok, full = vid.read()
            if not ok:
                break
            vis, summary, _ = step(full, n / fps)
            n += 1
            cv2.imshow(DEBUG_WIN, vis)
            k = cv2.waitKey(1) & 0xFF
            if not pinned:
                pin_all()
                pinned = True
            if k in (27, ord("q")):
                break
            if k == ord("m"):
                show_masks = not show_masks
            if k == ord(" "):
                cv2.waitKey(0)
    cv2.destroyAllWindows()
    if args.tune:
        save_tuned(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="CV autopilot macro for Roblox Motorcycle Mayhem")
    ap.add_argument("--debug", action="store_true", help="show the real-time debug window")
    ap.add_argument("--dry-run", action="store_true", help="live vision, but never press keys")
    ap.add_argument("--config", help="JSON file overriding Config fields")
    ap.add_argument("--dump-config", metavar="PATH", help="write the default config and exit")
    ap.add_argument("--image", help="offline: run on a full client-area screenshot")
    ap.add_argument("--video", help="offline: run on a gameplay recording")
    ap.add_argument("--roi-only", action="store_true", help="offline input is already cropped to the ROI")
    ap.add_argument("--save-debug", metavar="PATH", help="offline image: save the overlay instead of showing it")
    ap.add_argument("--masks", action="store_true", help="start with mask panels visible (toggle: m)")
    ap.add_argument("--tune", action="store_true", help="show threshold trackbars; saves tuned_config.json")
    ap.add_argument("--no-countdown", action="store_true", help="skip the start countdown")
    ap.add_argument("--no-topmost", action="store_true", help="don't keep the debug window always on top")
    ap.add_argument("--fpv", action="store_true", help="first-person view mode (minimal rider mask)")
    args = ap.parse_args()

    if args.dump_config:
        Path(args.dump_config).write_text(json.dumps(asdict(Config()), indent=2), encoding="utf-8")
        print(f"Default config written to {args.dump_config}")
        return
    cfg = load_config(args.config)
    if args.fpv:
        cfg.first_person = True
        cfg.roi_bottom = 0.78
        cfg.bike_mask_height = 0.08
        cfg.bike_mask_width = 0.20
    if args.no_topmost:
        cfg.debug_topmost = False
    if args.image or args.video:
        run_offline(cfg, args)
    else:
        run_live(cfg, args)


if __name__ == "__main__":
    main()
