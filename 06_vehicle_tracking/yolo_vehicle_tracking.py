from ultralytics import YOLO
import cv2

# Load model dari :contentReference[oaicite:0]{index=0}
model = YOLO("../yolov8n.pt")

TARGET_CLASSES = ["car", "motorcycle", "bus", "truck"]

color_map = {
    "car": (255, 0, 0),
    "motorcycle": (0, 255, 0),
    "bus": (0, 165, 255),
    "truck": (128, 0, 128)
}

cap = cv2.VideoCapture("../video.avi")

window_name = "Vehicle Tracking (YOLOv8 + ByteTrack)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_counting.mp4", fourcc, fps, (width, height))

counted_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        conf=0.4,
        persist=True,
        verbose=False
    )

    if results[0].boxes.id is None:
        cv2.imshow(window_name, frame)
        continue

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label not in TARGET_CLASSES:
            continue

        track_id = int(box.id[0])

        # ✅ counting hanya sekali per kendaraan
        if track_id not in counted_ids:
            counted_ids.add(track_id)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = color_map[label]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} ID:{track_id}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # ✅ tampilkan total kendaraan unik
    cv2.putText(
        frame,
        f"Total Vehicles: {len(counted_ids)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    out.write(frame)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
