import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
frames_generator = sv.get_video_frames_generator('video_1.dav')

for frame in frames_generator:

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)