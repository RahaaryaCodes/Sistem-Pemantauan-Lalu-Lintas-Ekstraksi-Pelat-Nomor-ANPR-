import cv2
import os

# Load Haar Cascade
car_cascade = cv2.CascadeClassifier("cars.xml")

# List image files (manual biar rapi sesuai request)
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

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect cars
    cars = car_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3
    )

    # Draw boxes
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Show image
    cv2.imshow("Haar Car Detection", img)

    # Press any key to go to next image
    cv2.waitKey(0)

cv2.destroyAllWindows()
