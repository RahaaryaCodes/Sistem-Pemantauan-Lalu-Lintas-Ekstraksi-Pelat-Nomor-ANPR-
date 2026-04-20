from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

TARGET_CLASSES = ["car", "motorcycle", "bus", "truck"]

# Open video (0 = webcam, or path to video file)
cap = cv2.VideoCapture("../video3.mp4")

# Window config
cv2.namedWindow("YOLO Vehicle Detection", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame, imgsz=640, conf=0.4, verbose=False)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label in TARGET_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]

            color_map = {
                "car": (255, 0, 0),
                "motorcycle": (0, 255, 0),
                "bus": (0, 165, 255),
                "truck": (128, 0, 128)
            }

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

    cv2.imshow("YOLO Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
