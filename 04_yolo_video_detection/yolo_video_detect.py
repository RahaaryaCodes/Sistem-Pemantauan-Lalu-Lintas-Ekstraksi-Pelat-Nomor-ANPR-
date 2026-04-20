from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("../yolov8n.pt")

TARGET_CLASSES = ["car", "motorcycle", "bus", "truck"]

color_map = {
    "car": (255, 0, 0),
    "motorcycle": (0, 255, 0),
    "bus": (0, 165, 255),
    "truck": (128, 0, 128)
}

# Open video
cap = cv2.VideoCapture("../video.avi")

window_name = "YOLO Vehicle Detection"
cv2.namedWindow("YOLO Vehicle Detection", cv2.WINDOW_NORMAL)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # bisa juga 'mp4v'
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame, conf=0.4, verbose=False)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label not in TARGET_CLASSES:
            continue

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

    out.write(frame)

    cv2.imshow("YOLO Vehicle Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:  # q or ESC
        break

    # Exit if window closed (click X)
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
