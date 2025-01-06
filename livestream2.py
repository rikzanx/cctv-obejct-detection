import cv2
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
import os
import time
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
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Tidak dapat membuka RTSP stream.")
    exit()

# Tracking data
tracked_objects = defaultdict(lambda: 0)  # Untuk melacak objek berdasarkan ID
current_objects = set()  # Objek yang terdeteksi dalam frame saat ini
object_counter = defaultdict(lambda: 0)  # Total hitungan objek

# Threshold untuk menentukan objek baru
IOU_THRESHOLD = 0.5  # Intersection over Union minimum untuk dianggap objek yang sama
MAX_LOST_TIME = 2.0  # Waktu maksimum untuk kehilangan objek (detik)

# Timestamp terakhir deteksi
last_detected = {}

def calculate_iou(box1, box2):
    """Menghitung Intersection over Union (IoU) antara dua bounding box."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# Loop untuk membaca frame secara real-time
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak dapat membaca frame dari stream.")
        break

    # Jalankan YOLO inference
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    current_objects.clear()  # Reset objek yang terdeteksi dalam frame saat ini
    for box, cls in zip(detections.xyxy, detections.class_id):
        class_name = results.names[int(cls)]
        x1, y1, x2, y2 = map(int, box)

        # Cek apakah objek ini sudah terdeteksi sebelumnya
        is_new = True
        for obj_id, (obj_box, _) in list(tracked_objects.items()):
            if calculate_iou(obj_box, (x1, y1, x2, y2)) > IOU_THRESHOLD:
                # Update tracking data
                tracked_objects[obj_id] = ((x1, y1, x2, y2), time.time())
                is_new = False
                break

        if is_new:
            # Tambahkan objek baru ke tracking
            obj_id = len(tracked_objects) + 1
            tracked_objects[obj_id] = ((x1, y1, x2, y2), time.time())
            object_counter[class_name] += 1

        current_objects.add(class_name)

    # Hapus objek yang hilang terlalu lama
    current_time = time.time()
    tracked_objects = {
        obj_id: (box, timestamp)
        for obj_id, (box, timestamp) in tracked_objects.items()
        if current_time - timestamp < MAX_LOST_TIME
    }

    # Bersihkan terminal dan cetak hasil
    os.system('cls' if os.name == 'nt' else 'clear')  # Hapus terminal
    print("Total Deteksi Unik:")
    for obj, count in object_counter.items():
        print(f"{obj}: {count}")

    print("\nDeteksi Terkini:")
    for obj in current_objects:
        print(f"- {obj}")

    # Anotasi frame
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)

    # Tampilkan frame yang telah dianotasi
    cv2.imshow("RTSP Stream with YOLO Detection and Tracking", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup semua jendela dan release stream
cap.release()
cv2.destroyAllWindows()
