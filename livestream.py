import os
import cv2
import supervision as sv
from ultralytics import YOLO
from dotenv import load_dotenv

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load .env file
load_dotenv()

# Ambil nilai dari .env
username = os.getenv("RTSP_USERNAME")
password = os.getenv("RTSP_PASSWORD")
ip = os.getenv("RTSP_IP")
port = os.getenv("RTSP_PORT")
channel = os.getenv("RTSP_CHANNEL")
subtype = os.getenv("RTSP_SUBTYPE")

# Buat RTSP URL dari variabel .env
rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"

# Buka RTSP stream
# cap = cv2.VideoCapture(rtsp_url)
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

if not cap.isOpened():
    print("Error: Tidak dapat membuka RTSP stream.")
    exit()

cv2.namedWindow("RTSP Stream with YOLO Detection", cv2.WINDOW_NORMAL)

# Loop untuk membaca frame secara real-time
while True:
    ret, frame = cap.read()
    # if not ret:
    #     print("Error: Tidak dapat membaca frame dari stream.")
    #     break

    # Jalankan YOLO inference
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Anotasi frame
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)

    # Tampilkan frame yang telah dianotasi
    cv2.imshow("RTSP Stream with YOLO Detection", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup semua jendela dan release stream
cap.release()
cv2.destroyAllWindows()
