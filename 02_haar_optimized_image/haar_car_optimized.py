import cv2
import numpy as np

# =============================
# Load Haar Cascades
# =============================
car_cascade = cv2.CascadeClassifier("cars.xml")
motor_cascade = cv2.CascadeClassifier("two_wheeler.xml")

# =============================
# Simple NMS function
# =============================


def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    pick = []

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        overlap = (w * h) / areas[idxs[:-1]]

        idxs = np.delete(
            idxs,
            np.concatenate(([len(idxs) - 1],
                            np.where(overlap > overlapThresh)[0]))
        )

    return boxes[pick].astype("int")


# =============================
# List images
# =============================
image_files = [
    "test.jpg",
    "test2.jpg",
    "test3.jpg",
    "test4.png"
]

for img_name in image_files:
    print(f"Processing: {img_name}")

    img = cv2.imread(img_name)
    if img is None:
        print(f"❌ {img_name} not found, skip")
        continue

    # Resize for stability
    max_width = 1000
    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale)

    # =============================
    # Preprocessing
    # =============================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # =============================
    # Detection
    # =============================
    cars = car_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    motors = motor_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 40)
    )

    # =============================
    # NMS filtering
    # =============================
    cars = non_max_suppression(cars, 0.5)
    motors = non_max_suppression(motors, 0.5)

    # =============================
    # Draw results
    # =============================
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, "Car", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    for (x, y, w, h) in motors:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, "Motor", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Stage 2 - Optimized Haar + NMS", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
