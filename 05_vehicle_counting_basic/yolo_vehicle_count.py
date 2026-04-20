from ultralytics import YOLO
import cv2

# Load model
model = YOLO("../yolov8n.pt")

TARGET_CLASSES = ["car", "motorcycle", "bus", "truck"]

color_map = {
    "car": (255, 0, 0),
    "motorcycle": (0, 255, 0),
    "bus": (0, 165, 255),
    "truck": (128, 0, 128)
}

cap = cv2.VideoCapture("../video3.mp4")

window_name = "Vehicle Counting (No Tracking)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)

    count = 0

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label not in TARGET_CLASSES:
            continue

        count += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        color = color_map[label]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # Show count (per frame)
    cv2.putText(
        frame,
        f"Vehicles detected: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
