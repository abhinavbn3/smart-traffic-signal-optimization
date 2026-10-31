# Smart Traffic Signal Optimization Project

This guide documents the project files in this folder, how the image-to-signal workflow works, the optimization model, its assumptions, and practical ways to improve it. It describes the code as it currently exists; the simulated traffic inputs mean this is a prototype and not a controller suitable for live road signals.

## 1. Project overview

The project combines computer vision with a small repeated traffic-signal simulation:

1. Load a still junction image.
2. Use a pretrained Ultralytics YOLOv8 model to detect vehicle boxes.
3. Assign detections to North, South, East, or West approaches using either hand-drawn road polygons or four automatically inferred image regions.
4. Ask the operator for each approach's current waiting time.
5. Rank approaches by `vehicle count × waiting time` and select one approach for green.
6. Update vehicle counts and waiting times using randomized arrival and discharge values, then repeat for several cycles.

The image supplies only an approximate initial count. It does not provide a live stream, vehicle arrival rate, measured queue length, signal phase state, lane geometry, pedestrian demand, or actual waiting time. Those later values are typed in or simulated.

## 2. Files in the project

| File | Purpose |
|---|---|
| `smart_signal_zones.py` | Main version. Users draw polygons around each approach once per image; vehicle detections are assigned to those polygons. Includes configurable model size, image size, confidence, NMS IoU, and an optional count-correction prompt. |
| `smart_signal_image.py` | Earlier/simpler image version. Splits the image into four regions around a configurable center point; no saved polygon editor. |
| `count_vehicles.py` | Standalone YOLO vehicle detector and total/class count tool. It does not assign vehicles to approaches or optimize signals. |
| `traffic_signal (1).py` | Standalone signal simulation. It asks for initial vehicle counts and waits, then runs the same priority-selection idea with different random traffic ranges. It asks for input at module startup. |
| `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt` | Local pretrained model weight files. The main zone script selects `yolov8m.pt`; the other image and count scripts select `yolov8s.pt`. |
| `*_zones.json` | Saved polygon coordinates for the matching image used by `smart_signal_zones.py`. |
| `Roadjunction.webp`, `detected_Roadjunction.webp` | Example source image and a detector output. |
| `Aerial_View_of_a_Bustling_City_Junction.png`, `Busy_Indian_Urban_Intersection_from_Above.png` | Additional sample source images. |
| `roadzones_*.png`, `zones_*.png` | Annotated example outputs showing detections and zone assignment. These are generated artifacts, not inputs to optimization. |

There is no dependency manifest in the folder, so package versions are not pinned.

## 3. Requirements and execution

The scripts use Python and these third-party packages:

- `opencv-python` (`cv2`) for image reading, drawing, and the polygon editor.
- `numpy` for polygon coordinate arrays and image geometry.
- `ultralytics` for YOLO inference.
- `pulp` for formulating and solving the binary optimization problem.
- A working PuLP CBC solver installation, invoked through `PULP_CBC_CMD`.

The project comments show Python 3.12 on Windows. From the project folder, run the main script with an image path:

```powershell
py -3.12 smart_signal_zones.py Roadjunction.webp
```

You can omit the path and enter it when prompted. Paths containing spaces should be quoted. The first run can take longer while the model or solver components are initialized. The `.pt` files are already present here, but if a selected model file is missing, Ultralytics may attempt to acquire it depending on the environment and package configuration.

Other entry points:

```powershell
py -3.12 smart_signal_image.py Roadjunction.webp
py -3.12 count_vehicles.py Roadjunction.webp
py -3.12 "traffic_signal (1).py"
```

## 4. Main workflow: `smart_signal_zones.py`

### 4.1 Configure behavior

The settings near the top of the file define the four directions, simulation length, randomized traffic ranges, and detection parameters:

- `CYCLE_DURATION = 30`: duration, in seconds, added to the waiting time of each non-green approach.
- `NUM_CYCLES = 6`: number of optimization decisions printed.
- `ARRIVING_RANGE = (1, 4)`: random number of arriving vehicles per direction per update, inclusive.
- `DISCHARGE_RANGE = (5, 10)`: random number of vehicles that could leave the selected approach during the update, inclusive, capped by its current count.
- `DEFAULT_WAIT = 30`: wait used when the operator presses Enter at the wait prompt.
- `MODEL_NAME = "yolov8m.pt"`: model loaded for each call to `detect_vehicles`.
- `IMG_SIZE = 960`: inference image size passed to YOLO.
- `CONFIDENCE = 0.15`: minimum detection confidence.
- `IOU_THRESHOLD = 0.5`: NMS overlap threshold.
- `ALLOW_MANUAL_CORRECTION = False`: when true, prompts for corrected counts after detections.

Vehicle class IDs are COCO IDs: car `2`, motorcycle `3`, bus `5`, and truck `7`. Other detected classes are ignored.

### 4.2 Draw or reuse road polygons

`load_or_define_zones()` looks for a sibling file named after the image with `_zones.json` appended, for example `Roadjunction_zones.json`. If found, it asks whether to reuse the polygon coordinates. Choosing to redraw opens the OpenCV editor.

For each direction in `North`, `South`, `East`, `West`, the editor accepts clicks for polygon corners:

- Left click adds a vertex.
- Enter saves that direction after at least three vertices.
- `u` or Backspace removes the latest vertex.
- `r` clears the current direction.
- `s` skips that direction (an empty polygon is saved).
- Esc cancels.

The editor displays a resized image but converts clicks back to original image coordinates before saving. The JSON polygons are therefore specific to that source image's coordinate system. If an image is resized, cropped, or replaced with another frame, the old polygon coordinates may no longer line up. The files are named from the image path, but the code does not verify that saved coordinates belong to the current image dimensions.

### 4.3 Detect and assign vehicles

`detect_vehicles(image)` loads the configured YOLO model and runs inference once on the image. It asks Ultralytics for confidence filtering, NMS, image size, and class-agnostic NMS, then keeps boxes for the four vehicle classes. Each retained box is represented as `(x1, y1, x2, y2)` in source-image pixels.

`count_by_zone()` assigns each box using its bottom-center point first. This approximates the place where the vehicle contacts the road. If that point lies outside all polygons, it tries the box center. `zone_of_point()` uses OpenCV's signed point-to-polygon distance and, when polygons overlap, chooses the containing polygon in which the point is deepest.

Boxes whose two test points fall outside every polygon are assigned `None`. They appear gray in the output and are reported as outside; they do not contribute to any direction count. Detections are counted individually, so duplicate detections and missed/occluded vehicles can affect the result.

### 4.4 Review the annotated output

`draw_result()` overlays translucent colored polygons and draws each box in the color of its assigned direction. Unassigned boxes are gray. `save_result_image()` writes `roadzones_<image basename>.png`; if that name cannot be written, it tries a time-stamped alternative. On Windows it attempts to open the output with `os.startfile()`.

With `ALLOW_MANUAL_CORRECTION = False`, the script waits for Enter after displaying the output. With it enabled, the operator can replace each approach's count with a nonnegative integer. The correction changes the initial simulation state only; it does not change the image detections or output image.

### 4.5 Collect waiting times and initialize state

The script asks for waiting time in seconds for each direction. Blank input uses 30 seconds. Non-numeric input also falls back to 30. The current implementation does not reject negative values. Initial state is stored as:

```python
{
    "North": {"vehicles": count, "wait": seconds},
    ...
}
```

## 5. Optimization logic

### 5.1 Priority score

For each approach `i`, `calculate_priority()` computes:

\[
P_i = V_i \times W_i
\]

where `V_i` is the current vehicle count and `W_i` is the current wait in seconds. A larger queue and a longer wait each increase the score linearly. The score is a heuristic urgency measure, not a measured delay or a calibrated traffic engineering objective.

### 5.2 Binary integer program

`solve_optimization()` makes one binary variable per direction:

\[
x_i \in \{0,1\}
\]

It maximizes the selected direction's priority:

\[
\max \sum_i P_i x_i
\]

subject to exactly one direction receiving green:

\[
\sum_i x_i = 1
\]

PuLP sends this model to CBC (`PULP_CBC_CMD`). In the current form there are four independent alternatives and only one constraint, so the result is simply the direction with the highest priority. A full integer-programming solver is not necessary for this specific model; `max(state, key=...)` would produce the same winner. The optimization structure becomes useful when adding constraints such as minimum/maximum green duration, incompatible movements, phase transitions, pedestrian service, or fairness limits.

Ties are not explicitly handled. The solver chooses one tied optimum, and the code searches the direction list for the variable whose value is exactly `1`. The solve status is also not checked before reading variable values.

### 5.3 State transition between cycles

After printing a cycle's scores and winner, `update_state()` independently draws an arrival count for every direction:

\[
V_i(t+1) = V_i(t) + A_i(t) - D_i(t)
\]

- For the green direction, `D_i` is a random number from 5 to 10, capped at the current vehicle count.
- For the other directions, `D_i = 0`.
- All directions receive a random `A_i` from 1 to 4.

Waiting time is updated as:

\[
W_i(t+1) =
\begin{cases}
\max(5, W_i(t)-T), & i\text{ received green} \\
W_i(t)+T, & \text{otherwise}
\end{cases}
\]

where `T` is 30 seconds. This keeps the selected approach's wait from dropping below five seconds and increases every unserved approach's wait by one cycle. The loop updates state after the final displayed cycle too, although that last updated state is not printed.

### 5.4 Important current behavior

- Because arrivals are added after a capped discharge, counts stay nonnegative if they begin nonnegative.
- If the initial vehicle count is zero, it can still build through simulated arrivals.
- Randomness is unseeded; repeated runs can produce different later cycles.
- The same arrival range is applied regardless of direction, time, lane count, or demand.
- A green is selected for one direction, but no green duration is chosen or passed to a real signal controller.
- The model treats each direction as one queue and does not represent turning movements or lane-level conflicts.

## 6. Related implementations

### `smart_signal_image.py`

This version uses a geometric rule instead of saved polygons. `get_zone()` compares the horizontal and vertical offsets of a point from `(CENTER_X, CENTER_Y)` and assigns the point to the dominant side. It is equivalent to dividing the image into four broad regions with diagonal boundaries. `count_by_zone()` classifies using the bottom-center of each box. It uses `yolov8s.pt`, and otherwise has a similar priority and cycle loop.

This approach is quick to configure but assumes the junction is near the chosen center and that image top/bottom/right/left align with the named directions. It can misclassify curved approaches or roads that occupy unusual parts of the frame.

### `count_vehicles.py`

This utility loads `yolov8s.pt`, detects the same four vehicle classes, prints a class breakdown and total, and saves an annotated image prefixed with `detected_`. It is useful for inspecting detector behavior, but it does not split counts by approach or run the signal simulation.

### `traffic_signal (1).py`

This script contains the signal-selection simulation without image processing. It requests initial counts and waits from the user, then applies the same vehicle-times-wait priority. Its arrival range is 2–10 and its discharge range is 30–45, unlike the main zone script. It clamps new vehicle counts to zero and prompts immediately when the module is executed. The main image scripts instead use arrivals 1–4 and discharge 5–10.

## 7. Detection and counting limitations

- A single still image cannot distinguish a moving vehicle from a parked vehicle or estimate how long it has waited.
- YOLO may miss small, occluded, unusually oriented, or low-contrast vehicles; it may also produce duplicate or incorrect boxes.
- The detector's class set is limited to car, motorcycle, bus, and truck. Other road users are excluded.
- Confidence and NMS thresholds trade recall against false positives and duplicate suppression. A low confidence threshold can increase both useful detections and false detections.
- The `count_vehicles.py` comment says 25%, but the configured confidence is `0.15` (15%).
- The main script uses a medium model and 960-pixel inference size; this can improve small-object detection at greater compute cost, but actual speed and accuracy depend on hardware and image quality.
- One box equals one vehicle. The pipeline does not track objects across frames, so it cannot avoid recounting a vehicle in successive images.
- Polygon quality is manual and can change the directional counts substantially. Zones should cover the intended queue areas without overlapping where possible.

## 8. Optimization limitations and improvement ideas

The present score gives a simple ordering but does not optimize network-wide delay. Some concrete next steps:

1. **Validate inputs:** require nonnegative waits/counts, validate polygon format and image dimensions, and report model/image errors clearly.
2. **Check solver results:** inspect PuLP's status and handle infeasible or failed solves. Add an explicit tie policy such as longest-wait-first or round-robin.
3. **Make runs repeatable:** accept a random seed and log the seed and configuration with each run.
4. **Use real traffic measurements:** replace random arrivals/discharges with counts from video tracking, loop detectors, or measured rates. Model arrivals by direction and time interval.
5. **Represent service realistically:** derive discharge capacity from lane count, saturation flow, green duration, and turning mix. Ensure the chosen phase can safely serve its movements.
6. **Add fairness:** impose a maximum wait or aging bonus so a direction cannot be repeatedly starved under asymmetric demand.
7. **Optimize more than one decision:** plan a sequence of phases over a horizon, include lost time and phase-change constraints, and minimize total delay/queue length rather than maximizing one product.
8. **Separate logic from I/O:** move detection, zone assignment, state updates, and optimization into testable modules; keep prompts and image windows in a command-line/UI layer.
9. **Avoid unnecessary solver overhead:** use a direct maximum for the current four-way model; retain PuLP when meaningful constraints are introduced.
10. **Pin dependencies and document hardware:** add a requirements file and specify supported Python, model weight source/version, and solver setup.

## 9. Output and data flow summary

```text
source image
   ├─ saved/drawn polygons (smart_signal_zones.py)
   │      └─ YOLO vehicle boxes → bottom-center / center point assignment → counts by approach
   └─ or image quadrants (smart_signal_image.py)
          └─ YOLO vehicle boxes → geometric assignment → counts by approach

counts + operator-entered waits
   → priority = vehicles × wait
   → choose one maximum-priority approach
   → random arrivals/discharge and wait updates
   → repeat for configured cycles
```

## 10. Suggested operating procedure

1. Choose a clear junction image and note its orientation (which edge corresponds to each direction).
2. Run `smart_signal_zones.py` and draw polygons around the relevant approach/queue areas. Review the saved JSON and annotated image.
3. Inspect every detection and zone assignment. If counts are wrong, enable `ALLOW_MANUAL_CORRECTION` for an operator-adjusted initial state.
4. Enter plausible initial waiting times; remember they are inputs, not image-derived values.
5. Treat the printed cycles as an illustrative simulation. Do not connect these recommendations directly to live signals without validated sensing, safety constraints, field calibration, and traffic authority approval.

