import cv2
import supervision as sv
from ultralytics import YOLO
from datetime import datetime

model = YOLO("yolov8n.pt")
image = cv2.imread('test_2.jpg')
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# Buat nama file dengan timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
output_path = f"annotated_{timestamp}.jpg"

# Simpan gambar hasil anotasi ke file
cv2.imwrite(output_path, annotated_image)

print(f"Hasil anotasi disimpan ke: {output_path}")

# Tampilkan gambar
# Buat jendela dengan ukuran asli
# cv2.namedWindow("Annotated Image", cv2.WINDOW_NORMAL)  # Mengatur jendela agar bisa diubah ukurannya
# cv2.imshow("Annotated Image", annotated_image)
# cv2.waitKey(0)  # Tunggu hingga tombol ditekan
# cv2.destroyAllWindows()
# from ultralytics import YOLO
# import cv2

# # Load YOLOv8 model
# model = YOLO("yolov8n.pt")  # Gunakan model pre-trained atau model yang Anda miliki

# # Baca gambar
# image = cv2.imread('example_1.png')

# # Jalankan inference
# results = model(image)

# # Visualisasi hasil
# annotated_image = results[0].plot()  # Menambahkan anotasi ke gambar
# cv2.imshow("Detected Image", annotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
