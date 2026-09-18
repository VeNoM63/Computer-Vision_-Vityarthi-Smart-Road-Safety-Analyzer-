# AI Road Safety Analyzer Using Computer Vision

## Description
An academic project developed for B.Tech Computer Science (AI & ML). This system processes road images and video frames to detect lanes, identify vehicles using pretrained YOLO models, and estimate basic lane positioning to issue visual safety warnings.

## Architecture & Computer Vision Theory
1. **Lane Detection:** Utilizes Grayscale conversion -> Gaussian Blur -> Canny Edge Detection -> ROI Masking -> Hough Line Transform.
2. **Vehicle Detection:** Implements Ultralytics YOLOv8n (nano) for real-time bounding box generation targeting MS COCO vehicle classes.
3. **Lane Position Analysis:** Estimates the ego-vehicle position (camera center) relative to the mathematical average of detected lane boundaries.

## Limitations
* Requires clear lane markings and good lighting conditions.
* Not calibrated to physical camera intrinsics; position offsets are pixel-based estimations, not metric distances.
* Ego-vehicle center proxy assumes the camera is perfectly centered on the dashboard.

## Future Improvements
* Integration of Kalman Filters for temporal lane tracking (smoothing lines between frames).
* Perspective transformation (Bird's Eye View) for accurate distance estimation.
* Custom fine-tuned YOLO model for specific regional vehicles (e.g., auto-rickshaws).
