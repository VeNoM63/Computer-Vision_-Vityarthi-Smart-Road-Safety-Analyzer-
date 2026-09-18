from ultralytics import YOLO
import cv2

class VehicleDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """Initializes the pretrained YOLOv8 model."""
        self.model = YOLO(model_path)
        self.vehicle_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck

    def detect(self, image, conf_threshold=0.5):
        """Detects vehicles, returns annotated frame and metrics."""
        results = self.model(image, classes=self.vehicle_classes, conf=conf_threshold, verbose=False)
        annotated_img = results[0].plot()
        
        boxes = results[0].boxes
        vehicle_count = len(boxes)
        confidences = boxes.conf.tolist() if boxes else []
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        return annotated_img, vehicle_count, avg_conf