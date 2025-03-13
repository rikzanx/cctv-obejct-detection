import cv2
import supervision as sv
from ultralytics import YOLO
from dotenv import load_dotenv

model = YOLO("yolov8n.pt")

load_dotenv()


username = os.getenv("RTSP_USERNAME")
password = os.getenv("RTSP_PASSWORD")
ip = os.getenv("RTSP_IP")
port = os.getenv("RTSP_PORT")
channel = os.getenv("RTSP_CHANNEL")
subtype = os.getenv("RTSP_SUBTYPE")

rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Tidak dapat membuka RTSP stream.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak dapat membaca frame dari stream.")
        break
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    

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
