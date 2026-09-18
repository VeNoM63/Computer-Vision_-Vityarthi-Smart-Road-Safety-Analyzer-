# **Vision-Based Lane Departure & Forward Vehicle Detection System**
An end-to-end Advanced Driver Assistance System (ADAS) pipeline engineered in Python using OpenCV and Ultralytics YOLOv8. Developed as an academic project for B.Tech Computer Science (Artificial Intelligence & Machine Learning), the system performs real-time lane boundary detection, multi-class vehicle identification, and lateral lane departure estimation from monocular dashcam video feeds to generate real-time visual safety warnings.

## **Key Features**
Monocular Lane Boundary Detection: Classical computer vision pipeline extracting left and right boundary demarcations without requiring GPU-intensive lane segmentation networks.

Real-Time Object Detection: Employs YOLOv8n optimized for low-latency edge inference across roadway agents (cars, buses, trucks, motorcycles).

Lane Departure Warning (LDW): Evaluates vehicle position relative to the lane center and triggers visual HUD warnings (LEFT DEPARTURE, RIGHT DEPARTURE, or CENTERED).

Telemetry HUD Overlay: Real-time on-screen diagnostic readouts showing lane offset values, active line vectors, vehicle bounding boxes with confidence scores, and system FPS.

## **System Architecture & Theoretical Framework**
Pipeline Flowchart
Input Frame (RGB)
|
+---> Stream A: Classical CV Pipeline
|       - Grayscale Conversion & Gaussian Smoothing
|       - Canny Edge Detection
|       - Polygonal Region of Interest (ROI) Masking
|       - Probabilistic Hough Line Transform (P-HLT)
|       - Slope Partitioning, Linear Regression & Lane Center Proxy
|
+---> Stream B: Deep Learning Pipeline
|       - Frame Normalization & Resizing
|       - YOLOv8n Forward Inference (MS COCO)
|       - Class Filtering (Vehicle Subsets: Classes 2, 3, 5, 7)
|
+---> Fusion & HUD Rendering ---> Display/Video Output

## **1. Lane Detection & Mathematical Modeling**
Pre-processing: The input frame is converted to single-channel grayscale to reduce dimensionality. A 5x5 Gaussian kernel filters out high-frequency spatial sensor noise.

Gradient Analysis: Canny edge detection calculates directional intensity gradients via Sobel operators, followed by non-maximum suppression and dual-threshold hysteresis to isolate continuous edge pixels.

Region of Interest (ROI) Dynamic Masking: A trapezoidal bitwise mask zeroes out irrelevant frame areas (sky, horizons, surrounding scenery) to isolate the forward roadway corridor.

Line Parameterization: The Probabilistic Hough Transform maps Cartesian gradient pixels (x, y) to polar parameter space (r, theta).

Slope Filtering & Extrapolation: Detected line segments are partitioned by slope m = (y2 - y1) / (x2 - x1):

Left Lane: m < -0.5

Right Lane: m > 0.5
Horizontal artifacts (road cracks, crosswalk markers) are pruned. Valid segments are fitted using first-order linear regression (y = mx + b) to project continuous lane bounds from the bottom edge to the visual horizon.

## **2. Deep Learning Vehicle Detection**
Architecture: Ultralytics YOLOv8n (nano) utilizing a CSPDarknet53 backbone with PAN-FPN feature aggregation and an anchor-free decoupled head.

Target Classes: Pretrained on MS COCO, filtered strictly for transport classes:

Class 2: car

Class 3: motorcycle

Class 5: bus

Class 7: truck

Non-Maximum Suppression (NMS): Prunes redundant bounding proposals using an IoU threshold of 0.45 and a baseline confidence threshold of 0.50.

## **3. Lateral Displacement & Departure Estimation**
Camera Center Proxy: Assumes the monocular camera is rigidly mounted along the vehicle's longitudinal centerline:
X_ego = Frame_Width / 2

Lane Center Proxy:
X_lane_center = (X_left_bottom + X_right_bottom) / 2

Offset Calculation:
Delta_X = X_ego - X_lane_center
A configurable pixel threshold determines when visual departure warnings trigger on screen.

Technical Stack
Language: Python 3.8+

Vision Core: OpenCV (cv2)

**Deep Learning Framework:** Ultralytics YOLOv8 (PyTorch backend)

**Scientific Computing:** NumPy

Acceleration: CUDA / cuDNN (Optional, for GPU acceleration)

Project Structure
lane-vehicle-detection/
|-- assets/                  # Demo clips, screenshots, HUD diagrams
|-- config/                  # Configuration files (ROI coordinates, thresholds)
|-- models/                  # Pretrained weights (yolov8n.pt)
|-- src/
|   |-- init.py
|   |-- lane_detector.py     # Canny edge and Hough transform pipeline
|   |-- vehicle_detector.py  # YOLOv8 inference wrapper
|   |-- departure_logic.py   # Lateral offset calculation and warnings
|
