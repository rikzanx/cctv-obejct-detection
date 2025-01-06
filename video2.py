import cv2
import supervision as sv
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Buka video input
input_video_path = 'video3.dav'
output_video_path = 'output_video1.avi'

# Membuat generator untuk frame video
frames_generator = sv.get_video_frames_generator(input_video_path)

# Ambil properti video input
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frame rate
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Lebar frame
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Tinggi frame
cap.release()  # Tutup video setelah mengambil properti

# Tentukan codec dan buat VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec untuk AVI
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Proses setiap frame
for frame in frames_generator:
    # Jalankan YOLO inference
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Anotasi hasil deteksi pada frame
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)

    # Tulis frame yang telah dianotasi ke video output
    out.write(annotated_frame)

# Tutup VideoWriter
out.release()

print(f"Video output telah disimpan ke: {output_video_path}")
