from ultralytics import YOLO
import cv2
from collections import defaultdict

# Load model
model = YOLO("yolov8n.pt")

TARGET_CLASSES = ["car", "motorcycle", "bus", "truck"]
counts = defaultdict(int)
counted_ids = set()

# Video
cap = cv2.VideoCapture("../video3.mp4")

# Counting line
LINE_Y = 400
OFFSET = 10

cv2.namedWindow("YOLO Tracking & Counting", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO + tracking (ByteTrack default)
    results = model.track(
        frame,
        conf=0.4,
        persist=True,
        verbose=False
    )

    # Draw line
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 0, 255), 2)

    if results[0].boxes.id is not None:
        for box, track_id in zip(results[0].boxes, results[0].boxes.id):
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            tid = int(track_id)

            # Draw box + ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} ID:{tid}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            # Counting logic (ID-based)
            if LINE_Y - OFFSET < cy < LINE_Y + OFFSET:
                if tid not in counted_ids:
                    counts[label] += 1
                    counted_ids.add(tid)

    # Display counts
    y = 30
    for cls in TARGET_CLASSES:
        cv2.putText(
            frame,
            f"{cls}: {counts[cls]}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        y += 30

    cv2.imshow("YOLO Tracking & Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
