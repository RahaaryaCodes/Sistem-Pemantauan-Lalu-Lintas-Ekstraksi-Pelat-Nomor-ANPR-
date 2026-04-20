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

window_name = "Direction Counting (IN / OUT)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Virtual line (Y position)
LINE_Y = 300

# Store last Y position per ID
track_history = {}

# Count results
count_in = 0
count_out = 0

# IDs already counted
counted_ids = set()


# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

# Setup writer dari :contentReference[oaicite:0]{index=0}
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_direction.mp4", fourcc, fps, (width, height))

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

    # Draw counting line
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 0, 255), 2)

    if results[0].boxes.id is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label not in TARGET_CLASSES:
                continue

            track_id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = color_map[label]

            # Center point
            center_y = int((y1 + y2) / 2)

            # Save previous position
            prev_y = track_history.get(track_id, center_y)
            track_history[track_id] = center_y

            # Direction logic (count once)
            if track_id not in counted_ids:
                if prev_y < LINE_Y and center_y >= LINE_Y:
                    count_in += 1
                    counted_ids.add(track_id)
                elif prev_y > LINE_Y and center_y <= LINE_Y:
                    count_out += 1
                    counted_ids.add(track_id)

            # Draw bbox + ID
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

            # Draw center point
            cv2.circle(frame, (int((x1+x2)/2), center_y), 4, color, -1)

    # Show counts
    cv2.putText(frame, f"IN : {count_in}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {count_out}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow(window_name, frame)

    out.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
