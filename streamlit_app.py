import streamlit as st
import av
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from PIL import Image
import torch

# --- Import custom utility modules ---
import depth_utils
import dimension_utils
import hologram_utils

# --- Configuration ---
st.set_page_config(page_title="AI Dimension Estimator", layout="wide")

# --- Model Loading with Error Handling ---
@st.cache_resource
def load_all_models():
    try:
        # Try different model paths
        model_paths = ["yolov8n-seg.pt", "model/yolov8n-seg.pt", "./yolov8n-seg.pt"]
        seg_model = None
        
        for path in model_paths:
            try:
                seg_model = YOLO(path)
                st.success(f"✅ YOLOv8 model loaded from: {path}")
                break
            except Exception as e:
                st.warning(f"❌ Failed to load from {path}: {str(e)}")
                continue
        
        if seg_model is None:
            st.error("🚨 Could not load YOLOv8 model. Downloading...")
            seg_model = YOLO("yolov8n-seg.pt")  # This will auto-download
        
        # Load depth model
        depth_model, depth_transform, device = depth_utils.load_depth_model()
        st.success("✅ MiDaS depth model loaded successfully")
        
        return seg_model, depth_model, depth_transform, device
    
    except Exception as e:
        st.error(f"🚨 Error loading models: {str(e)}")
        return None, None, None, None

# Load models
seg_model, depth_model, depth_transform, device = load_all_models()

# --- Session State Initialization ---
if "mode" not in st.session_state:
    st.session_state.mode = "scanning"
    st.session_state.captured_frame = None
    st.session_state.detection_results = None
    st.session_state.selected_obj_idx = None
    st.session_state.dimensions = None
    st.session_state.processing_error = None

# --- Global variables for frame processing ---
latest_frame = None
processing_active = True

# --- Improved Frame Processing with Error Handling ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global latest_frame, processing_active
    
    try:
        if not processing_active or seg_model is None:
            return frame
            
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        latest_frame = img.copy()
        
        # Run YOLO detection
        results = seg_model.predict(source=img, conf=0.4, verbose=False, device='cpu')
        
        if results and len(results) > 0:
            result = results[0]
            
            # Draw detection results
            annotated_img = result.plot()
            
            # Store results in session state if in scanning mode
            if st.session_state.mode == "scanning":
                st.session_state.detection_results = result
            
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
        else:
            return frame
            
    except Exception as e:
        st.session_state.processing_error = str(e)
        return frame

# --- RTC Configuration ---
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# --- Main App Layout ---
st.title("🤖 AI-Based Real-Time Object Dimension Estimator")

# Check if models loaded successfully
if seg_model is None:
    st.error("❌ Models failed to load. Please check your installation and try again.")
    st.stop()

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    
    # Show processing status
    status_placeholder = st.empty()
    
    # WebRTC Streamer with improved configuration
    ctx = webrtc_streamer(
        key="ai-dimension-estimator",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "width": {"min": 640, "ideal": 1280},
                "height": {"min": 480, "ideal": 720},
                "frameRate": {"min": 10, "ideal": 30}
            },
            "audio": False
        },
        async_processing=True,
        key_suffix="main",
    )
    
    # Connection status
    if ctx.state.playing:
        status_placeholder.success("🟢 Camera Connected - Live Detection Active")
    elif ctx.state.signalling:
        status_placeholder.info("🟡 Connecting to Camera...")
    else:
        status_placeholder.warning("🔴 Camera Disconnected")
    
    # Show processing errors
    if st.session_state.processing_error:
        st.error(f"Processing Error: {st.session_state.processing_error}")

with col2:
    st.subheader("🎛️ Controls & Results")
    
    # Mode display
    st.write(f"**Current Mode:** {st.session_state.mode.upper()}")
    
    if st.session_state.mode == "scanning":
        # Show detected objects if available
        if st.session_state.detection_results is not None:
            results = st.session_state.detection_results
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                st.success("✅ Objects Detected!")
                
                # List detected objects
                detected_objects = []
                for i, cls in enumerate(results.boxes.cls):
                    class_name = seg_model.names[int(cls)]
                    confidence = results.boxes.conf[i]
                    detected_objects.append(f"{class_name} ({confidence:.2f})")
                
                st.write("**Detected Objects:**")
                for obj in detected_objects:
                    st.write(f"• {obj}")
                
                # Capture frame button
                if st.button("📸 Capture Frame for Measurement", type="primary", use_container_width=True):
                    if latest_frame is not None:
                        st.session_state.captured_frame = latest_frame.copy()
                        st.session_state.mode = "paused"
                        st.rerun()
                    else:
                        st.error("No frame available to capture")
            else:
                st.info("🔍 No objects detected. Point camera at objects...")
        else:
            st.info("🔍 Scanning for objects...")
    
    elif st.session_state.mode == "paused":
        results = st.session_state.detection_results
        
        if results and hasattr(results, 'boxes') and len(results.boxes) > 0:
            st.info("📸 Frame captured! Select an object to measure:")
            
            # Object selection
            object_names = []
            for cls in results.boxes.cls:
                object_names.append(seg_model.names[int(cls)])
            
            selected_obj_name = st.selectbox(
                "Select Object:", 
                options=list(set(object_names)),
                key="object_selector"
            )
            
            if selected_obj_name:
                st.session_state.selected_obj_idx = object_names.index(selected_obj_name)
                
                # Category determination (you can expand this)
                living_objects = ['person', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
                category = "Living" if selected_obj_name.lower() in living_objects else "Non-living"
                st.write(f"**Category:** {category}")
                
                st.divider()
                
                # Calibration section
                st.subheader("📏 Calibration")
                st.warning("⚠️ Manual calibration required for accurate measurements")
                
                ref_width_cm = st.number_input(
                    "Enter known width of reference object (cm):", 
                    min_value=0.1, 
                    value=10.0, 
                    step=0.1,
                    key="ref_width"
                )
                
                ref_width_px = st.number_input(
                    "Enter corresponding width in pixels:", 
                    min_value=1, 
                    value=100, 
                    step=1,
                    key="ref_pixels"
                )
                
                # Measurement button
                if st.button("📐 Measure Selected Object", type="primary", use_container_width=True):
                    with st.spinner("🔄 Calculating dimensions..."):
                        try:
                            pixels_per_cm = ref_width_px / ref_width_cm
                            
                            # Get mask and depth
                            if hasattr(results, 'masks') and results.masks is not None:
                                mask = results.masks.data[st.session_state.selected_obj_idx].cpu().numpy()
                                mask = (mask * 255).astype(np.uint8)
                            else:
                                # Fallback: create mask from bounding box
                                bbox = results.boxes.xyxy[st.session_state.selected_obj_idx].cpu().numpy()
                                mask = np.zeros((st.session_state.captured_frame.shape[0], st.session_state.captured_frame.shape[1]), dtype=np.uint8)
                                x1, y1, x2, y2 = bbox.astype(int)
                                mask[y1:y2, x1:x2] = 255
                            
                            # Generate depth map
                            pil_img = Image.fromarray(cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB))
                            depth_map = depth_utils.get_depth_map(pil_img, depth_model, depth_transform, device)
                            
                            # Calculate dimensions
                            dims = dimension_utils.get_dimensions_with_depth(mask, depth_map, pixels_per_cm)
                            
                            if dims:
                                st.session_state.dimensions = dims
                                st.session_state.mode = "measured"
                                st.success("✅ Measurement completed!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to calculate dimensions")
                                
                        except Exception as e:
                            st.error(f"❌ Measurement error: {str(e)}")
        
        # Back to scanning button
        if st.button("🔄 Resume Scanning", use_container_width=True):
            st.session_state.mode = "scanning"
            st.rerun()
    
    elif st.session_state.mode == "measured":
        dims = st.session_state.dimensions
        
        if dims:
            st.success("✅ Measurement Complete!")
            
            st.subheader("📊 Dimension Results")
            
            # Display measurements
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Width", f"{dims['width_cm']:.2f} cm")
                st.metric("Height", f"{dims['height_cm']:.2f} cm")
            with col_b:
                st.metric("Estimated Depth", f"{dims['depth_cm']:.2f} cm")
                st.metric("Volume", f"{dims['volume_cm3']:.2f} cm³")
            
            # Hologram button
            if st.button("🌟 Show Holographic View", type="primary", use_container_width=True):
                with st.spinner("🎭 Generating 3D hologram..."):
                    try:
                        results = st.session_state.detection_results
                        if hasattr(results, 'masks') and results.masks is not None:
                            mask = results.masks.data[st.session_state.selected_obj_idx].cpu().numpy()
                        else:
                            # Fallback mask
                            bbox = results.boxes.xyxy[st.session_state.selected_obj_idx].cpu().numpy()
                            mask = np.zeros((st.session_state.captured_frame.shape[0], st.session_state.captured_frame.shape[1]))
                            x1, y1, x2, y2 = bbox.astype(int)
                            mask[y1:y2, x1:x2] = 1
                        
                        pil_img = Image.fromarray(cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB))
                        depth_map = depth_utils.get_depth_map(pil_img, depth_model, depth_transform, device)
                        
                        fig = hologram_utils.create_holographic_view(mask, depth_map)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Hologram generation error: {str(e)}")
        
        # New scan button
        if st.button("🆕 Start New Scan", use_container_width=True):
            st.session_state.mode = "scanning"
            st.session_state.captured_frame = None
            st.session_state.detection_results = None
            st.session_state.selected_obj_idx = None
            st.session_state.dimensions = None
            processing_active = True
            st.rerun()

# --- Footer ---
st.divider()
st.markdown("**🔧 Troubleshooting Tips:**")
st.markdown("• Ensure good lighting for better detection")
st.markdown("• Point camera directly at objects")  
st.markdown("• Allow camera permissions when prompted")
st.markdown("• Refresh page if camera doesn't connect")
