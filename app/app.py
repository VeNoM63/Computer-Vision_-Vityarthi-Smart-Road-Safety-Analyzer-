import streamlit as st
import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import RoadSafetyPipeline

st.set_page_config(page_title="AI Road Safety Analyzer", layout="wide")

@st.cache_resource
def load_pipeline():
    return RoadSafetyPipeline()

pipeline = load_pipeline()

# --- HEADER ---
st.title("🛣️ AI Road Safety Analyzer using Computer Vision")
st.markdown("**Academic Project | B.Tech Computer Science (AI & ML)**")

st.markdown("### Project Objective")
st.info("To demonstrate the application of computer vision techniques for detecting road lanes and vehicles from images and video, followed by basic spatial analysis for road-safety indication.")

# --- SIDEBAR ---
st.sidebar.header("Pipeline Configuration")
conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
input_type = st.sidebar.radio("Input Source", ("Image", "Video"))

# --- IMAGE EXECUTION ---
if input_type == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload Road Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with st.spinner("Processing CV Pipeline..."):
            results = pipeline.process_frame(image, conf_threshold)
            
        st.markdown("---")
        st.markdown("### 📊 Computer Vision Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Vehicles Detected", results["vehicle_count"])
        m2.metric("Average Detection Confidence", f"{results['avg_conf']:.2f}")
        m3.metric("Lane Line Segments Detected", results["lines_detected"])
        m4.metric("Processing Time", f"{results['process_time']:.3f} s")
        
        st.markdown("---")
        st.markdown("### 🔍 Computer Vision Pipeline — Intermediate Processing Stages")
        
        # Grid Layout for clear, readable steps
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image_rgb, use_column_width=True)
            st.markdown("#### STEP 1 — Original Image")
            st.caption("Input road image used for computer vision analysis.")
            
        with col2:
            st.image(results["intermediates"]["gray"], use_column_width=True)
            st.markdown("#### STEP 2 — Grayscale Conversion")
            st.caption("Converts the RGB image to grayscale to simplify intensity-based processing.")
            
        col3, col4 = st.columns(2)
        
        with col3:
            st.image(results["intermediates"]["blur"], use_column_width=True)
            st.markdown("#### STEP 3 — Gaussian Blur")
            st.caption("Reduces image noise and smooths high-frequency details before edge detection.")
            
        with col4:
            st.image(results["intermediates"]["edges"], use_column_width=True)
            st.markdown("#### STEP 4 — Canny Edge Detection")
            st.caption("Detects significant intensity boundaries that can correspond to lane markings and object boundaries.")
            
        col5, col6 = st.columns(2)
        
        with col5:
            st.image(results["intermediates"]["roi_mask"], use_column_width=True)
            st.markdown("#### STEP 5 — Region of Interest (ROI)")
            st.caption("Restricts lane analysis to the road region where lane markings are expected.")
            
        with col6:
            st.image(results["intermediates"]["cropped_edges"], use_column_width=True)
            st.markdown("#### STEP 6 — ROI Masked Edge Map")
            st.caption("Combines edge detection with the selected road region to isolate relevant lane features.")
            
        st.markdown("---")
        
        st.image(cv2.cvtColor(results["final_image"], cv2.COLOR_BGR2RGB), use_column_width=True)
        st.markdown("#### STEP 7 — Final Detection & Analysis")
        st.caption("Combines lane detection and vehicle detection results for spatial analysis. Displays bounding boxes, ROI boundaries, and lane-center references.")

# --- VIDEO EXECUTION ---
elif input_type == "Video":
    uploaded_file = st.sidebar.file_uploader("Upload Road Video", type=['mp4', 'avi'])
    
    if uploaded_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
            
        cap = cv2.VideoCapture("temp_video.mp4")
        stframe = st.empty()
        
        st.markdown("### 📊 Dynamic Computer Vision Metrics")
        m_fps, m_veh, m_conf, m_lines = st.columns(4)
        met_fps = m_fps.empty()
        met_veh = m_veh.empty()
        met_conf = m_conf.empty()
        met_lines = m_lines.empty()
        
        stop_btn = st.sidebar.button("Stop Processing")
        
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret: break
                
            results = pipeline.process_frame(frame, conf_threshold)
            stframe.image(cv2.cvtColor(results["final_image"], cv2.COLOR_BGR2RGB), use_column_width=True)
            
            fps = 1.0 / results["process_time"] if results["process_time"] > 0 else 0
            met_fps.metric("FPS", f"{fps:.1f}")
            met_veh.metric("Vehicles Detected", results["vehicle_count"])
            met_conf.metric("Avg Confidence", f"{results['avg_conf']:.2f}")
            met_lines.metric("Lane Line Segments", results["lines_detected"])
            
        cap.release()
        os.remove("temp_video.mp4")

# --- LIMITATIONS & DISCLAIMER ---
st.markdown("---")
with st.expander("Limitations & Disclaimer", expanded=False):
    st.markdown("""
    * Lane detection can be affected by faded or occluded lane markings.
    * Performance can vary with lighting and weather conditions.
    * Vehicle detection depends on the underlying detection model.
    * The safety indication is based on visual analysis and predefined rules.
    * **This is an academic Computer Vision demonstration and not a validated ADAS.**
    """)