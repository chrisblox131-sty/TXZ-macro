# Motorcycle Mayhem CV Autopilot

A computer-vision macro for Roblox **Motorcycle Mayhem**. It uses OpenCV to read the screen and `pydirectinput` to press keys.

### Quick Start (Windows)
Double-click either of the batch files:
- **`START_MACRO.bat`**: Interactive launcher with options (Normal Live, 1st-Person FPV, Tuning sliders, Safe Dry-Run, or Install dependencies).
- **`RUN_AUTOPILOT.bat`**: 1-click instant launch directly into live mode with the debug overlay.

Or run from PowerShell/Command Prompt:
```powershell
pip install -r requirements.txt
python mayhem_autopilot.py --debug --dry-run   # 1) check the vision, no key presses
python mayhem_autopilot.py --debug             # 2) let it drive
```

## Hotkeys
| Key | Action |
|---|---|
| **ESC** or **Q** | Emergency kill switch: releases every key and exits |
| **F8** | Pause / resume |
| **F9** | Save a full screenshot to `snapshots/` (use it for offline tuning) |
| **m** (debug window focused) | Show or hide the mask panels (road / obstacles / markings / edges) |
| Mouse in top-left screen corner | `pydirectinput` fail-safe abort |

Keys are only pressed while the Roblox window is in the foreground, so clicking another window pauses input.

## How it works
1. **ROI capture**: captures the forward highway from just below the top stats bar (`roi_top: 0.08`) down to just above the bike (`roi_bottom: 0.50`). In *Motorcycle Mayhem*'s high-chase camera angle, the motorcycle sits at $y=0.52$, meaning your rider and bike are completely outside the capture area.
2. **Ego-Vehicle Exclusion Mask**: an extra safety zone at the bottom-center ensures any wheelies or camera shifts never detect your helmet or handlebars as a car.
3. **Road mask**: calibrated to *Motorcycle Mayhem*'s dark grey asphalt (`HSV V: 35-115, S <= 40`).
4. **Lane tracking**: specifically tuned for multi-lane highways (5 lanes). Dashed lines are tracked per-lane; gaps in dashed lines are cleanly bridged without jumping to the guardrail.
5. **Obstacle detection**: detects vehicles ahead (like trucks, SUVs, and cars). Roadside guardrails and off-road barriers are automatically excluded via centroid checking and boundary margin erosion.
6. **Actuation**: WASD keys are actuated via smooth DirectInput pulses (PWM). Drifting right pulses **A** to steer left; drifting left pulses **D** to steer right; cruising in lane holds **W**; emergency stops pulse **S**.

## Camera View Modes
- **3rd Person (Default)**: The macro uses an ego-vehicle cutout to ignore your rider and bike. You can adjust the cutout in `--tune` via `bike_mask_height` and `bike_mask_width`.
- **1st Person (Recommended for cleanest vision)**: Scroll your mouse wheel all the way forward in Roblox or press `V` to switch to 1st person. Run the macro with `--fpv`:
  ```powershell
  python mayhem_autopilot.py --fpv --debug
  ```
  In 1st person, your rider's head and body are completely removed from view, giving the macro an unobstructed line of sight down the road.

## Tuning
Every game, graphics setting and resolution looks a bit different, so tune on a real frame:
```powershell
python mayhem_autopilot.py --dump-config my.json      # editable defaults
python mayhem_autopilot.py --image snapshots/x.png --tune --masks
python mayhem_autopilot.py --config tuned_config.json --debug
```
- **Rider / Helmet visible in the box**: increase `bike_mask_height` or decrease `roi_bottom` until the yellow `EGO BIKE (EXCLUDED)` trapezoid fully encloses your helmet and shoulders.
- **Road not green in the overlay**: adjust `road_s_max`, `road_v_min` and `road_v_max`.
- **Cars missed or false boxes**: adjust `min_obstacle_area`, or try `detect_mode: "color"` with `car_s_min`.
- **Columns don't line up with the lanes**: adjust `center_col_bottom`, `center_col_top` and `bike_x`.
- **Weaving or sluggish steering**: lower or raise `kp`, `k_heading` and `pwm_period`.
- **Throttle**: if the game accelerates on its own, set `hold_throttle: false`.

> The debug window stays **always on top** (it doesn't take focus) and starts at the top-left of the Roblox window. It is also **hidden from screen capture**, so it can sit over the road without confusing the bot. On Windows 10 2004 and later you see it, but the macro doesn't. Don't click it while driving: keys are only sent while Roblox has focus. Use `--no-topmost` to turn it off, or `debug_window_pos: [x, y]` to move it. If the console says `hidden-from-capture=False`, keep the window off the road area.

Vision takes about 4 ms per frame at the default 320 px processing width. Use of macros is subject to Roblox's Terms of Use and the game's rules.
