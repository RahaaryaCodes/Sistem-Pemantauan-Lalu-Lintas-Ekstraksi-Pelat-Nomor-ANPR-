from ultralytics import YOLO
import cv2
from collections import Counter

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Classes we want
TARGET_CLASSES = ["car", "motorcycle", "bus", "truck"]

# Read image
img = cv2.imread("test4.png")

# Run detection
results = model(img, conf=0.4)

# Copy image
annotated = img.copy()

# Loop detections
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    label = model.names[cls_id]

    if label in TARGET_CLASSES:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]

        # Color per class
        color_map = {
            "car": (255, 0, 0),
            "motorcycle": (0, 255, 0),
            "bus": (0, 165, 255),
            "truck": (128, 0, 128)
        }

        color = color_map[label]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{label} {conf:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

labels = [
    model.names[int(box.cls[0])]
    for box in results[0].boxes
    if model.names[int(box.cls[0])] in TARGET_CLASSES
]

# Display
print("Vehicle count:", Counter(labels))
cv2.namedWindow("YOLO Vehicle Detection", cv2.WINDOW_NORMAL)
cv2.imshow("YOLO Vehicle Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
