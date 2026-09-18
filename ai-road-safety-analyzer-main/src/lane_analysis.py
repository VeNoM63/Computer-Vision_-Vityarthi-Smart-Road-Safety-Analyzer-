import cv2

def analyze_lane_position(image, lines, threshold_offset=50):
    """Calculates spatial relationship between lane center and ego-vehicle proxy."""
    height, width = image.shape[:2]
    ego_center = width // 2
    
    if lines is None or len(lines) == 0:
        return image, "NO LANE DETECTED", (0, 0, 255)
    
    left_x, right_x = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1: continue
            slope = (y2 - y1) / (x2 - x1)
            if slope < -0.5:
                left_x.extend([x1, x2])
            elif slope > 0.5:
                right_x.extend([x1, x2])
                
    if left_x and right_x:
        lane_center = int((sum(left_x) / len(left_x) + sum(right_x) / len(right_x)) / 2)
        offset = abs(lane_center - ego_center)
        
        status = "LANE POSITION WARNING" if offset > threshold_offset else "SAFE / NORMAL"
        color = (0, 165, 255) if offset > threshold_offset else (0, 255, 0)
            
        cv2.circle(image, (lane_center, height - 50), 10, color, -1)
        cv2.circle(image, (ego_center, height - 50), 10, (255, 255, 255), -1)
        cv2.putText(image, f"Status: {status}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return image, status, color
    else:
        return image, "CAUTION: POOR VISIBILITY", (0, 255, 255)