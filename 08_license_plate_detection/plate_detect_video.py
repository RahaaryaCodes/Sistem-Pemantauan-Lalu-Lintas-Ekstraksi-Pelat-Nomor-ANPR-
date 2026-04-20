from ultralytics import YOLO
import cv2
import easyocr
import re
import time
import math

# =============================
# Load models
# =============================
vehicle_model = YOLO("yolov8n.pt")     # vehicle detection + tracking
plate_model = YOLO("../best.pt")       # license plate detection (custom)
reader = easyocr.Reader(['en'], gpu=True)

TARGET_CLASSES = ["car", "motorcycle", "bus", "truck"]

vehicle_colors = {
    "car": (255, 0, 0),
    "motorcycle": (0, 255, 255),
    "bus": (0, 165, 255),
    "truck": (128, 0, 128)
}

# =============================
# Video IO
# =============================
cap = cv2.VideoCapture("../sample.mp4")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "final_vehicle_plate_inout.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (width, height)
)

# =============================
# Helpers
# =============================


def clean_plate(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


# =============================
# Memory
# =============================
plate_memory = {}
vehicle_centers = {}
counted_ids = set()

# IN / OUT
LINE_Y = 800
count_in = 0
count_out = 0

# =============================
# Main loop
# =============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # =============================
    # Vehicle detection + tracking
    # =============================
    vehicle_results = vehicle_model.track(
        frame,
        conf=0.3,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    vehicles = []

    if vehicle_results[0].boxes.id is not None:
        for box, track_id, cls in zip(
            vehicle_results[0].boxes,
            vehicle_results[0].boxes.id,
            vehicle_results[0].boxes.cls
        ):
            track_id = int(track_id)
            label = vehicle_results[0].names[int(cls)]

            if label not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1+x2)//2, (y1+y2)//2
            color = vehicle_colors[label]

            # Store trajectory (Y only)
            prev_y = vehicle_centers.get(track_id, cy)
            vehicle_centers[track_id] = cy

            # IN / OUT logic
            if track_id not in counted_ids:
                if prev_y < LINE_Y and cy >= LINE_Y:
                    count_in += 1
                    counted_ids.add(track_id)
                elif prev_y > LINE_Y and cy <= LINE_Y:
                    count_out += 1
                    counted_ids.add(track_id)

            # Draw vehicle box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label.upper()} ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            vehicles.append({
                "id": track_id,
                "center": (cx, cy),
                "label": label
            })

    # =============================
    # Plate detection + OCR
    # =============================
    plate_results = plate_model.track(
        frame,
        conf=0.15,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    if plate_results[0].boxes.id is not None:
        for box, pid in zip(plate_results[0].boxes, plate_results[0].boxes.id):
            pid = int(pid)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if pid not in plate_memory:
                plate_memory[pid] = {"text": "", "conf": 0.0}

            plate_img = frame[y1:y2, x1:x2]
            if plate_img.size > 0:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=2, fy=2)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                ocr = reader.readtext(gray)
                if ocr:
                    text, conf = ocr[0][1], ocr[0][2]
                    text = clean_plate(text)

                    if conf > plate_memory[pid]["conf"]:
                        plate_memory[pid]["text"] = text
                        plate_memory[pid]["conf"] = conf

            # Assign to nearest vehicle
            cx, cy = (x1+x2)//2, (y1+y2)//2
            nearest_vehicle = None
            min_dist = float("inf")

            for v in vehicles:
                d = distance((cx, cy), v["center"])
                if d < min_dist:
                    min_dist = d
                    nearest_vehicle = v

            # Draw plate box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            label = f"PLATE: {plate_memory[pid]['text']}"
            if nearest_vehicle:
                label += f" | VID:{nearest_vehicle['id']}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

    # =============================
    # Draw virtual line & counters
    # =============================
    cv2.line(frame, (0, LINE_Y), (width, LINE_Y), (0, 0, 255), 2)

    cv2.putText(frame, f"IN : {count_in}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {count_out}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Final Vehicle System", frame)

    if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Output saved: final_vehicle_plate_inout.avi")
