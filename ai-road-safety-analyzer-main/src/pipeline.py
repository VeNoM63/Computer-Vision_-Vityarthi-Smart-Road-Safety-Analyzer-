from src.lane_detection import detect_lanes
from src.vehicle_detection import VehicleDetector
from src.lane_analysis import analyze_lane_position
import time

class RoadSafetyPipeline:
    def __init__(self):
        self.vehicle_detector = VehicleDetector()
        
    def process_frame(self, frame, conf_threshold=0.5):
        start_time = time.time()
        
        # Lanes & Preprocessing
        lane_annotated, lines, intermediates = detect_lanes(frame)
        
        # Vehicles
        combined_annotated, count, avg_conf = self.vehicle_detector.detect(lane_annotated, conf_threshold)
        
        # STEP 7: Final Spatial Analysis
        final_output, status, _ = analyze_lane_position(combined_annotated, lines)
        
        process_time = time.time() - start_time
        
        return {
            "final_image": final_output,
            "status": status,
            "vehicle_count": count,
            "avg_conf": avg_conf,
            "process_time": process_time,
            "intermediates": intermediates,
            "lines_detected": len(lines) if lines is not None else 0
        }