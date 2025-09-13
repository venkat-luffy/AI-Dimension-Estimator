import streamlit as st
import av
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from PIL import Image

# Import your custom utility files
from depth_utils import load_depth_model, get_depth_map
from dimension_utils import get_dimensions_with_depth
from hologram_utils import create_holographic_view

# --- Page and Model Setup ---
st.set_page_config(layout="wide", page_title="AI Object Dimension Estimator")
st.title("AI-Based Real-Time Object Dimension Estimator")

@st.cache_resource
def load_all_models():
    """Load all AI models once and cache them."""
    seg_model = YOLO("model/yolov8n-seg.pt")
    depth_model, depth_transform, device = load_depth_model()
    return seg_model, depth_model, depth_transform, device

seg_model, depth_model, depth_transform, device = load_all_models()

# --- Session State Initialization ---
# This is the "memory" of the app, tracking the current step of the process.
if "mode" not in st.session_state:
    st.session_state.mode = "scanning"  # Modes: scanning -> paused -> measured
    st.session_state.captured_frame = None
    st.session_state.detection_results = None
    st.session_state.selected_obj_idx = None
    st.session_state.dimensions = None

# --- Real-Time Video Frame Processing ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    This function is called for every frame from the webcam.
    It runs detection, annotates the frame, and sends it back to the browser.
    """
    img = frame.to_ndarray(format="bgr24")
    
    # Run YOLOv8 segmentation
    results = seg_model.predict(source=img, conf=0.4, verbose=False)[0]
    
    # Use YOLO's built-in plot function to draw boxes and labels
    annotated_img = results.plot()
    
    # Store the latest results in the session state so the sidebar can access it
    st.session_state.detection_results = results
    
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# --- Application Layout ---
col1, col2 = st.columns([3, 1]) # Main area for video, smaller sidebar for controls

with col1:
    st.header("Live Camera Feed")
    st.write("Point your camera at the objects you want to measure.")
    
    # The core WebRTC component that handles the live video stream
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # "Pause" button to capture the current frame for analysis
    if webrtc_ctx.video_receiver and st.session_state.mode == "scanning":
        if st.button("Pause Feed to Select & Measure", type="primary", use_container_width=True):
            try:
                latest_frame = webrtc_ctx.video_receiver.get_latest_frame()
                st.session_state.captured_frame = latest_frame.to_ndarray(format="bgr24")
                st.session_state.mode = "paused" # Change the app mode
                st.rerun()
            except Exception as e:
                st.error("Could not capture frame. Is the camera running?")

with col2:
    st.header("Controls & Results")
    
    # --- UI for 'scanning' mode ---
    if st.session_state.mode == "scanning":
        st.info("The system is currently scanning for objects in real-time.")
        if st.session_state.detection_results:
            st.subheader("Objects Detected:")
            # Display a unique list of detected object names
            detected_names = {seg_model.names[int(cls)] for cls in st.session_state.detection_results.boxes.cls}
            for name in detected_names:
                st.markdown(f"- {name.capitalize()}")
    
    # --- UI for 'paused' mode ---
    elif st.session_state.mode == "paused":
        st.info("Feed paused. Select an object from the list to measure.")
        results = st.session_state.detection_results
        object_names = [seg_model.names[int(cls)] for cls in results.boxes.cls]
        
        # Dropdown to select the specific object to measure
        selected_obj_name = st.selectbox(
            "Select Object Tag:", 
            options=list(set(object_names)), # Show unique names
        )
        
        # Find the index of the first object that matches the selected name
        st.session_state.selected_obj_idx = object_names.index(selected_obj_name)
        
        st.divider()
        st.subheader("Calibration")
        st.warning("Manual calibration is required for now.")
        pixels_per_cm_input = st.number_input("Enter a known length (cm) for a reference object in the scene.", min_value=0.1, value=10.0, key="calib_cm")
        pixels_input = st.number_input("Enter the corresponding length in pixels.", min_value=1, value=100, key="calib_px")
        
        # Measure Button
        if st.button("Measure Selected Object", use_container_width=True):
            with st.spinner("Calculating..."):
                pixels_per_cm = pixels_input / pixels_per_cm_input
                
                mask = results.masks.data[st.session_state.selected_obj_idx].cpu().numpy()
                pil_img = Image.fromarray(cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB))
                depth_map = get_depth_map(pil_img, depth_model, depth_transform, device)
                
                dims = get_dimensions_with_depth(mask, depth_map, pixels_per_cm)
                st.session_state.dimensions = dims
                st.session_state.mode = "measured"
                st.rerun()
                
        # Restart Button
        if st.button("Resume Scanning", use_container_width=True):
            st.session_state.mode = "scanning"
            st.rerun()

    # --- UI for 'measured' mode ---
    elif st.session_state.mode == "measured":
        st.success("Measurement complete!")
        dims = st.session_state.dimensions
        
        if dims:
            st.subheader("Dimension Results")
            st.metric("Width", f"{dims['width_cm']} cm")
            st.metric("Height", f"{dims['height_cm']} cm")
            st.metric("Est. Depth", f"{dims['depth_cm']} cm")
        
        # Hologram Button appears now
        if st.button("Show Holographic View", type="primary", use_container_width=True):
            with st.spinner("Generating 3D view..."):
                results = st.session_state.detection_results
                mask = results.masks.data[st.session_state.selected_obj_idx].cpu().numpy()
                pil_img = Image.fromarray(cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB))
                depth_map = get_depth_map(pil_img, depth_model, depth_transform, device)
                fig = create_holographic_view(mask, depth_map)
                st.plotly_chart(fig, use_container_width=True)
        
        # Restart Button
        if st.button("Start New Scan", use_container_width=True):
            st.session_state.mode = "scanning"
            st.rerun()
