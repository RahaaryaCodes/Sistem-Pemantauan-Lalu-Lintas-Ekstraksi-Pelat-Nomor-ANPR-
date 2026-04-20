from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Classes we want
TARGET_CLASSES = ["car", "motorcycle", "bus", "truck"]

# Image list
image_files = [
    "../test.jpg",
    "../test2.jpg",
    "../test3.jpg",
    "../test4.png"
]

# Color per class
color_map = {
    "car": (255, 0, 0),
    "motorcycle": (0, 255, 0),
    "bus": (0, 165, 255),
    "truck": (128, 0, 128)
}

for img_name in image_files:
    print(f"Processing: {img_name}")

    img = cv2.imread(img_name)
    if img is None:
        print(f"❌ {img_name} not found")
        continue

    # Run detection
    results = model(img, conf=0.4, verbose=False)

    annotated = img.copy()

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label not in TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]

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

    cv2.namedWindow("YOLO Vehicle Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLO Vehicle Detection", annotated)
    cv2.waitKey(0)

cv2.destroyAllWindows()
