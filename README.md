# Smart Traffic Signal Optimization

Dynamic traffic signal control for a four-way junction. Vehicles are detected from a junction photo with YOLO, and a 0–1 Integer Programming model (PuLP with the CBC solver) chooses which direction gets the green signal in each cycle.

This is a student prototype, not a deployed traffic controller.

## How it works

1. Load a junction photo and its saved road-zone polygons.
2. Detect vehicles (cars, motorcycles, buses, trucks) with a pretrained YOLO model.
3. Assign each vehicle to the North, South, East or West road zone it stands on.
4. Enter the waiting time for each direction (press Enter for 30 s).
5. Priority of each direction = vehicles × waiting time.
6. Solve the 0–1 model so that exactly one direction gets green.
7. Update vehicles and waiting times (simulated), then repeat for several cycles.

Model: maximize Σ P(i) · x(i), subject to Σ x(i) = 1 and x(i) ∈ {0, 1}, where P(i) = V(i) × W(i).

## Setup

Python 3.12 is recommended, because some AI libraries do not yet support the newest Python versions.

```
pip install -r requirements.txt
```

PuLP is pinned to 2.9.0, because PuLP 4.x changed the API used here.

## Run

```
py -3.12 smart_signal_zones.py your_photo.jpg
```

Each photo needs a zone file named after it, for example `your_photo_zones.json`. If the file does not exist, a window opens where you click the corners of each road (Enter = done, u = undo, r = reset, s = skip). The zones are then saved for the next run.

## Files

| File | Purpose |
| --- | --- |
| `smart_signal_zones.py` | Main program: detection, road zones, optimization and cycles |
| `traffic_signal.py` | Text-only version with typed starting numbers |
| `count_vehicles.py` | Counts vehicles in a single image |
| `*_zones.json` | Road-zone polygons for the sample photos |

## Status and limitations

- Vehicle counts from an image are approximate: hidden vehicles can be missed and a vehicle can be boxed twice.
- Waiting times are entered by the operator, and arrivals and departures after the first cycle are simulated.
- The control results have not been validated on real traffic data or compared with a fixed-time baseline.

## Image credits

Add the source of each sample image here before publishing.

## Reference

Solaiappan, S. et al. (2023). Vehicular Traffic Flow Analysis and Minimize the Vehicle Queue Waiting Time Using Signal Distribution Control Algorithm. Sensors, 23(15), 6819. https://doi.org/10.3390/s23156819
