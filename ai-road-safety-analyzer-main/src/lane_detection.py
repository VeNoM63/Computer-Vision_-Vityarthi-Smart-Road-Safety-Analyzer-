import cv2
import numpy as np

def region_of_interest(img, vertices):
    """Applies an image mask keeping only the region of interest."""
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image, mask

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    """Draws lines onto an image."""
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img

def detect_lanes(image):
    """
    Detects lane lines and returns the annotated image, raw lines, and intermediate CV stages.
    """
    height, width = image.shape[:2]
    
    # STEP 2: Grayscale Conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # STEP 3: Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # STEP 4: Canny Edge Detection
    edges = cv2.Canny(blur, 50, 150)
    
    # STEP 5 & 6: Region of Interest (ROI)
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2 + 50),
        (width, height)
    ]
    roi_pts = np.array([region_of_interest_vertices], np.int32)
    cropped_edges, roi_mask_visual = region_of_interest(edges, roi_pts)
    
    # Hough Line Transform (Feature extraction)
    lines = cv2.HoughLinesP(
        cropped_edges, 
        rho=2, 
        theta=np.pi/180, 
        threshold=50, 
        lines=np.array([]), 
        minLineLength=40, 
        maxLineGap=100
    )
    
    # Visualization base (Lanes + ROI outline for transparency)
    annotated_image = draw_lines(image.copy(), lines)
    cv2.polylines(annotated_image, roi_pts, isClosed=True, color=(255, 0, 255), thickness=2)
    
    intermediates = {
        "gray": gray,
        "blur": blur,
        "edges": edges,
        "roi_mask": roi_mask_visual,
        "cropped_edges": cropped_edges
    }
    
    return annotated_image, lines, intermediates