import cv2
import pytesseract
import imutils
import os

# ===============================
# KONFIGURASI
# ===============================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

IMAGE_FOLDER = "images"


def detect_plate_batch(folder_path):
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(folder_path, filename)
        print(f"Processing: {filename}")

        img = cv2.imread(image_path)
        if img is None:
            print("❌ Gagal load image")
            continue

        img = imutils.resize(img, width=600)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        plates = plate_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(100, 30)
        )

        if len(plates) == 0:
            print("⚠️ Plat tidak terdeteksi")

        for (x, y, w, h) in plates:
            roi = gray[y:y+h, x:x+w]

            roi = cv2.bilateralFilter(roi, 11, 17, 17)
            _, roi_thresh = cv2.threshold(
                roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            text = pytesseract.image_to_string(
                roi_thresh, config="--psm 7"
            ).strip()

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                img, text, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

        cv2.imshow("Haar Cascade - Batch Plate Detection", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    detect_plate_batch(IMAGE_FOLDER)
