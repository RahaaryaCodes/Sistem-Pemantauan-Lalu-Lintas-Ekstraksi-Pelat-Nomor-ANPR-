from ultralytics import YOLO
import cv2
from collections import defaultdict

# Load model
model = YOLO("yolov8n.pt")

TARGET_CLASSES = ["car", "motorcycle", "bus", "truck"]
counts = defaultdict(int)

# Open video
cap = cv2.VideoCapture("../video3.mp4")


# Counting line (y position)
LINE_Y = 400
offset = 10  # tolerance

# Store object centers (simple tracking-lite)
seen_centers = []

cv2.namedWindow("YOLO Vehicle Counting", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)

    # Draw counting line
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 0, 255), 2)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label not in TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Counting logic
        if LINE_Y - offset < cy < LINE_Y + offset:
            if (cx, cy, label) not in seen_centers:
                counts[label] += 1
                seen_centers.append((cx, cy, label))

    # Show counts
    y_text = 30
    for cls in TARGET_CLASSES:
        cv2.putText(
            frame,
            f"{cls}: {counts[cls]}",
            (10, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        y_text += 30

    cv2.imshow("YOLO Vehicle Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
