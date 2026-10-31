"""
Step 1: Count vehicles in ONE image using a pretrained YOLO model.

Usage (PowerShell):
    py -3.12 count_vehicles.py
    py -3.12 count_vehicles.py road_north.jpg

What it does:
  1. Loads a pretrained YOLO model (no training needed)
  2. Finds cars, motorcycles, buses and trucks in the image
  3. Draws a box around each one and saves the result as a new image
  4. Prints the total count
"""

import os
import sys

import cv2
from ultralytics import YOLO

# COCO class IDs that YOLO already knows -> name we show on screen
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CONFIDENCE = 0.15   # ignore detections the model is less than 25% sure about


def count_vehicles(image_path):
    """Return (total_count, breakdown_dict, annotated_image)."""
    model = YOLO("yolov8s.pt")   # small model, downloads automatically on first run

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    results = model(image, conf=CONFIDENCE, agnostic_nms=True, verbose=False)[0] 

    breakdown = {name: 0 for name in VEHICLE_CLASSES.values()}

    for box in results.boxes:
        class_id = int(box.cls[0])
        if class_id not in VEHICLE_CLASSES:
            continue   # skip people, traffic lights, etc.

        name = VEHICLE_CLASSES[class_id]
        breakdown[name] += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(image, name, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    total = sum(breakdown.values())

    cv2.putText(image, f"Vehicles: {total}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return total, breakdown, image


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter image file name or path: ")

    path = path.strip().strip('"')   # handles drag-and-drop paths with quotes

    total, breakdown, annotated = count_vehicles(path)

    print("\n--- Detection result ---")
    for name, n in breakdown.items():
        print(f"{name:10s}: {n}")
    print(f"TOTAL vehicles: {total}")

    out_path = "detected_" + os.path.basename(path)
    cv2.imwrite(out_path, annotated)
    print(f"\nAnnotated image saved as: {out_path}")
    print("Open it to check whether the boxes look right.")
