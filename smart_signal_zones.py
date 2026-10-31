"""
Smart Traffic Signal Optimization - Road-shaped zones version
-------------------------------------------------------------
You mark each approach road on the junction photo ONCE by clicking its corners.
Every detected vehicle is then given to the road it is standing on.

Flow:
  1. A window opens: click the corners of the NORTH road, press Enter,
     then the same for SOUTH, EAST, WEST (your clicks are saved to a file)
  2. YOLO detects vehicles (approximate count)
  3. Each vehicle goes to the road shape it is inside -> Cycle 1 counts
  4. You type the waiting time per direction (a photo cannot show it).
     (Optional: set ALLOW_MANUAL_CORRECTION = True to also fix counts by hand)
  5. PuLP (0-1 Integer Programming) picks the green signal each cycle
  6. Cycles 2+ update with the formulas:
        V(t+1) = V(t) + Arriving - Discharged
        W(t+1) = W(t) + T     (for directions that did NOT get green)

Usage (PowerShell):
    py -3.12 smart_signal_zones.py Roadjunction.webp
"""

import json
import os
import random
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, PULP_CBC_CMD

# ======================= SETTINGS =======================
DIRECTIONS = ["North", "South", "East", "West"]

# --- Signal simulation ---
CYCLE_DURATION = 30          # T = seconds per cycle
NUM_CYCLES = 6               # how many cycles to simulate
ARRIVING_RANGE = (1, 4)      # new vehicles arriving per cycle (simulated)
DISCHARGE_RANGE = (5, 10)    # vehicles that leave during a green (simulated)
DEFAULT_WAIT = 30            # used if you just press Enter for waiting time

# --- Vehicle detection ---
MODEL_NAME = "yolov8m.pt"       # bigger model than yolov8s: finds more small/partly hidden vehicles
IMG_SIZE = 960                # photo is enlarged to this size before detection (helps small photos)
ALLOW_MANUAL_CORRECTION = False   # False = fully automatic counts. True = ask the operator to fix them
CONFIDENCE = 0.15
IOU_THRESHOLD = 0.5          # lower = removes more overlapping duplicate boxes
                             # (YOLO default is 0.7; put it back if real
                             #  neighbouring vehicles start disappearing)
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

ZONE_COLORS = {                      # BGR colours for drawing
    "North": (255, 120, 0),
    "South": (0, 0, 255),
    "East": (0, 180, 0),
    "West": (0, 165, 255),
}
OUTSIDE_COLOR = (150, 150, 150)      # vehicles not on any marked road


# ======================= PART 1: DRAW THE ROAD ZONES =======================
def render_editor_frame(base, zones, current, direction, scale):
    """Draw finished zones, the points clicked so far, and the instructions."""
    overlay = base.copy()
    for d, poly in zones.items():
        if len(poly) >= 3:
            pts = (np.array(poly) * scale).astype(np.int32)
            cv2.fillPoly(overlay, [pts], ZONE_COLORS[d])
    frame = cv2.addWeighted(overlay, 0.35, base, 0.65, 0)

    for d, poly in zones.items():
        if len(poly) >= 3:
            cx, cy = (np.mean(poly, axis=0) * scale).astype(int)
            cv2.putText(frame, d, (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

    color = ZONE_COLORS[direction]
    pts = [(int(x * scale), int(y * scale)) for x, y in current]
    for p in pts:
        cv2.circle(frame, p, 4, color, -1)
    if len(pts) >= 2:
        cv2.polylines(frame, [np.array(pts, np.int32)], len(pts) >= 3, color, 2)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(frame,
                f"{direction.upper()} road: click its corners | "
                "Enter=done  u=undo  r=reset  s=skip",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def define_zones(image):
    """Open a window and let the user click the 4 road shapes."""
    h, w = image.shape[:2]
    scale = min(1100 / w, 700 / h)
    base = cv2.resize(image, None, fx=scale, fy=scale)

    window = "Define road zones"
    cv2.namedWindow(window)
    zones, current = {}, []

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((int(mx / scale), int(my / scale)))

    cv2.setMouseCallback(window, on_mouse)

    for direction in DIRECTIONS:
        current.clear()
        while True:
            cv2.imshow(window, render_editor_frame(base, zones, current,
                                                   direction, scale))
            key = cv2.waitKey(30) & 0xFF
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise SystemExit("Window closed.")
            if key == 13 and len(current) >= 3:          # Enter
                zones[direction] = list(current)
                break
            elif key in (8, ord("u")) and current:       # undo
                current.pop()
            elif key == ord("r"):                        # reset this road
                current.clear()
            elif key == ord("s"):                        # skip this direction
                zones[direction] = []
                break
            elif key == 27:                              # Esc
                raise SystemExit("Cancelled.")

    cv2.destroyAllWindows()
    return zones


def load_or_define_zones(image, image_path):
    """Reuse saved zones for this photo if they exist."""
    zones_path = os.path.splitext(image_path)[0] + "_zones.json"
    if os.path.exists(zones_path):
        ans = input("Saved road zones found for this photo. "
                    "Use them? (Enter = yes, n = draw again): ").strip().lower()
        if ans != "n":
            with open(zones_path) as f:
                return json.load(f)

    print("\nA window will open. For each road, click the corners of the road")
    print("area where vehicles queue (from the junction outwards), then press Enter.")
    zones = define_zones(image)
    with open(zones_path, "w") as f:
        json.dump(zones, f)
    print(f"Road zones saved to {zones_path}")
    return zones


# ======================= PART 2: DETECTION + ASSIGNMENT =======================
def detect_vehicles(image):
    """Run YOLO once. Returns a list of vehicle boxes."""
    model = YOLO(MODEL_NAME)
    results = model(image, conf=CONFIDENCE, iou=IOU_THRESHOLD, imgsz=IMG_SIZE,
                   agnostic_nms=True, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        if int(box.cls[0]) in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes


def zone_of_point(x, y, zones):
    """Return the road a point is inside (deepest inside wins), or None."""
    best, best_depth = None, -1e9
    for d, poly in zones.items():
        if len(poly) < 3:
            continue
        contour = np.array(poly, dtype=np.int32)
        depth = cv2.pointPolygonTest(contour, (float(x), float(y)), True)
        if depth >= 0 and depth > best_depth:
            best, best_depth = d, depth
    return best


def count_by_zone(boxes, zones):
    """Give each vehicle to the road it stands on.
    First try where it touches the road (bottom-centre), then its box centre."""
    counts = {d: 0 for d in DIRECTIONS}
    assigned = []
    for (x1, y1, x2, y2) in boxes:
        zone = zone_of_point((x1 + x2) // 2, y2, zones)
        if zone is None:
            zone = zone_of_point((x1 + x2) // 2, (y1 + y2) // 2, zones)
        if zone is not None:
            counts[zone] += 1
        assigned.append(zone)
    return counts, assigned


def draw_result(image, boxes, assigned, zones, counts):
    """Road shapes, coloured vehicle boxes and per-road counts."""
    overlay = image.copy()
    for d, poly in zones.items():
        if len(poly) >= 3:
            cv2.fillPoly(overlay, [np.array(poly, np.int32)], ZONE_COLORS[d])
    image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)

    for (x1, y1, x2, y2), zone in zip(boxes, assigned):
        color = ZONE_COLORS[zone] if zone else OUTSIDE_COLOR
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    for d, poly in zones.items():
        if len(poly) >= 3:
            cx, cy = np.mean(poly, axis=0).astype(int)
            cv2.putText(image, f"{d}: {counts[d]}", (cx - 35, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image



def save_result_image(image, base_name):
    """Save the annotated image and CHECK it worked. If the normal file name is
    locked (for example still open in a photo viewer on Windows), save under a
    new name instead. Returns the file name, or None if nothing could be saved."""
    names = [f"roadzones_{base_name}.png",
             f"roadzones_{base_name}_{time.strftime('%H%M%S')}.png"]
    for out_path in names:
        try:
            if cv2.imwrite(out_path, image):
                return out_path
        except cv2.error:
            pass
    return None


def confirm_counts(counts):
    """Let the operator fix the detected counts. Vehicles hidden behind a bus can
    be missed, and one vehicle can be boxed twice. This is a manual override."""
    print("\nLook at the result image. Vehicles hidden behind buses can be missed")
    print("and a vehicle can be boxed twice. For each road, type the correct")
    print("number of vehicles, or press Enter to keep the detected one.\n")
    final = {}
    for d in DIRECTIONS:
        raw = input(f"{d} - detected {counts[d]}, correct number (Enter = keep): ").strip()
        value = counts[d]
        if raw:
            try:
                value = int(raw)
                if value < 0:
                    raise ValueError
            except ValueError:
                print(f"  not a valid number, keeping {counts[d]}")
                value = counts[d]
        final[d] = value
        if value != counts[d]:
            print(f"  -> using {value} (operator correction)")
    return final

# ======================= PART 3: SIGNAL OPTIMIZATION =======================
def ask_wait_times():
    """A photo cannot show how long vehicles have waited, so ask the user."""
    print("\nA photo cannot show how long vehicles have been waiting,")
    print(f"so enter the waiting time in seconds (Enter = {DEFAULT_WAIT}):\n")
    waits = {}
    for d in DIRECTIONS:
        raw = input(f"{d} - waiting time (seconds): ").strip()
        try:
            waits[d] = int(raw) if raw else DEFAULT_WAIT
        except ValueError:
            print(f"  not a number, using {DEFAULT_WAIT}")
            waits[d] = DEFAULT_WAIT
    return waits


def calculate_priority(state):
    """Priority = Vehicles x Wait time."""
    for d in state:
        state[d]["priority"] = state[d]["vehicles"] * state[d]["wait"]
    return state


def solve_optimization(state):
    """0-1 Integer Programming: choose exactly one direction for green."""
    problem = LpProblem("Traffic_Signal", LpMaximize)
    x = {d: LpVariable(f"x_{d}", cat="Binary") for d in DIRECTIONS}

    problem += lpSum(state[d]["priority"] * x[d] for d in DIRECTIONS)
    problem += lpSum(x[d] for d in DIRECTIONS) == 1

    problem.solve(PULP_CBC_CMD(msg=0))
    return [d for d in DIRECTIONS if x[d].value() == 1][0]


def update_state(state, winner):
    """V(t+1) = V(t) + A - D     and     W(t+1) = W(t) + T (non-green)."""
    for d in state:
        arriving = random.randint(*ARRIVING_RANGE)
        if d == winner:
            discharged = min(state[d]["vehicles"], random.randint(*DISCHARGE_RANGE))
            state[d]["wait"] = max(5, state[d]["wait"] - CYCLE_DURATION)
        else:
            discharged = 0
            state[d]["wait"] += CYCLE_DURATION
        state[d]["vehicles"] = state[d]["vehicles"] + arriving - discharged
    return state


def run_cycles(state):
    for cycle in range(1, NUM_CYCLES + 1):
        state = calculate_priority(state)
        winner = solve_optimization(state)

        print(f"\n--- Cycle {cycle} ---")
        for d in DIRECTIONS:
            print(f"{d:6s} | vehicles={state[d]['vehicles']:3d} "
                  f"wait={state[d]['wait']:3d}s "
                  f"priority={state[d]['priority']:5d}")
        print(f">>> Green signal goes to: {winner}")

        state = update_state(state, winner)


# ======================= MAIN =======================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter junction image file name or path: ")
    path = path.strip().strip('"')

    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"Could not open image: {path}")

    # Step 1: mark the four roads (or reuse saved ones)
    zones = load_or_define_zones(image, path)

    # Step 2-3: detect vehicles and give each to its road
    boxes = detect_vehicles(image)
    counts, assigned = count_by_zone(boxes, zones)
    outside = sum(1 for z in assigned if z is None)

    print("\n--- Vehicles detected from the image (approximate) ---")
    for d in DIRECTIONS:
        print(f"{d:6s}: {counts[d]}")
    print(f"TOTAL on roads: {sum(counts.values())}")
    print(f"Outside all road zones (not counted): {outside}")

    base_name = os.path.splitext(os.path.basename(path))[0]
    result = draw_result(image.copy(), boxes, assigned, zones, counts)
    out_path = save_result_image(result, base_name)
    if out_path:
        print(f"\nAnnotated image saved as: {out_path}")
        try:
            os.startfile(out_path)   # opens the image on Windows
        except Exception:
            pass
    else:
        print("\nCould not save the annotated image (is the folder write-protected?)")

    if ALLOW_MANUAL_CORRECTION:
        counts = confirm_counts(counts)
    else:
        input("\nCounts are fully automatic. Check the result image, then press Enter to continue...")

    # Step 4-6: waiting times typed by the user, then the dynamic cycles
    waits = ask_wait_times()
    state = {d: {"vehicles": counts[d], "wait": waits[d]} for d in DIRECTIONS}
    run_cycles(state)