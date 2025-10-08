import streamlit as st
import av
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from PIL import Image
import torch

# Import custom modules
import depth_utils
import dimension_utils
import hologram_utils

# Configuration
st.set_page_config(page_title="AI Dimension Estimator", layout="wide")

# Model Loading
@st.cache_resource
def load_all_models():
    try:
        seg_model = YOLO("yolov8n-seg.pt")
        try:
            depth_model, depth_transform, device = depth_utils.load_depth_model()
            return seg_model, depth_model, depth_transform, device, True
        except:
            return seg_model, None, None, torch.device("cpu"), False
    except:
        return None, None, None, torch.device("cpu"), False

seg_model, depth_model, depth_transform, device, depth_available = load_all_models()

# Session State
if "mode" not in st.session_state:
    st.session_state.mode = "scanning"
    st.session_state.captured_frame = None
    st.session_state.detection_results = None
    st.session_state.selected_obj_idx = None
    st.session_state.dimensions = None

# Global variables
latest_frame = None

# Frame Processing
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global latest_frame
    
    try:
        if seg_model is None:
            return frame
            
        img = frame.to_ndarray(format="bgr24")
        latest_frame = img.copy()
        
        results = seg_model.predict(source=img, conf=0.4, verbose=False, device='cpu')
        
        if results and len(results) > 0:
            result = results[0]
            annotated_img = result.plot()
            
            if st.session_state.mode == "scanning":
                st.session_state.detection_results = result
            
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
        else:
            return frame
            
    except:
        return frame

# Basic dimension calculation
def calculate_basic_dimensions(mask, pixels_per_cm):
    x, y, w, h = cv2.boundingRect(mask)
    width_cm = w / pixels_per_cm
    height_cm = h / pixels_per_cm
    estimated_depth_cm = (width_cm + height_cm) / 2
    volume_cm3 = width_cm * height_cm * estimated_depth_cm
    
    return {
        "width_cm": round(width_cm, 2),
        "height_cm": round(height_cm, 2),
        "depth_cm": round(estimated_depth_cm, 2),
        "volume_cm3": round(volume_cm3, 2)
    }

# Main App
st.title("🤖 AI Dimension Estimator")

if seg_model is None:
    st.error("❌ Failed to load models")
    st.stop()

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Camera Feed")
    
    try:
        ctx = webrtc_streamer(
            key="dimension-estimator",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    except:
        pass
    
    # File upload
    st.divider()
    uploaded_file = st.file_uploader("Or upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        if st.button("🔍 Detect Objects"):
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            results = seg_model.predict(source=img_array, conf=0.4, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                annotated_img = result.plot()
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
                
                st.session_state.captured_frame = img_array
                st.session_state.detection_results = result
                st.session_state.mode = "paused"
                st.rerun()

with col2:
    st.subheader("🎛️ Controls")
    
    if st.session_state.mode == "scanning":
        if st.session_state.detection_results is not None:
            results = st.session_state.detection_results
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                st.success("✅ Objects detected!")
                
                detected_objects = []
                for i, cls in enumerate(results.boxes.cls):
                    class_name = seg_model.names[int(cls)]
                    confidence = results.boxes.conf[i]
                    detected_objects.append(f"{class_name} ({confidence:.2f})")
                
                for obj in detected_objects:
                    st.write(f"• {obj}")
                
                if st.button("📸 Capture Frame", type="primary"):
                    if latest_frame is not None:
                        st.session_state.captured_frame = latest_frame.copy()
                        st.session_state.mode = "paused"
                        st.rerun()
            else:
                st.info("🔍 No objects detected")
        else:
            st.info("🔍 Scanning...")
    
    elif st.session_state.mode == "paused":
        results = st.session_state.detection_results
        
        if results and hasattr(results, 'boxes') and len(results.boxes) > 0:
            st.info("Select object to measure:")
            
            object_names = []
            for cls in results.boxes.cls:
                object_names.append(seg_model.names[int(cls)])
            
            selected_obj_name = st.selectbox("Object:", list(set(object_names)))
            
            if selected_obj_name:
                st.session_state.selected_obj_idx = object_names.index(selected_obj_name)
                
                living_objects = ['person', 'cat', 'dog', 'horse', 'sheep', 'cow']
                category = "Living" if selected_obj_name.lower() in living_objects else "Non-living"
                st.write(f"**Category:** {category}")
                
                st.divider()
                st.subheader("📏 Calibration")
                
                ref_width_cm = st.number_input("Known width (cm):", min_value=0.1, value=10.0, step=0.1)
                ref_width_px = st.number_input("Width in pixels:", min_value=1, value=100, step=1)
                
                if st.button("📐 Measure", type="primary"):
                    pixels_per_cm = ref_width_px / ref_width_cm
                    
                    if hasattr(results, 'masks') and results.masks is not None:
                        mask = results.masks.data[st.session_state.selected_obj_idx].cpu().numpy()
                        mask = (mask * 255).astype(np.uint8)
                    else:
                        bbox = results.boxes.xyxy[st.session_state.selected_obj_idx].cpu().numpy()
                        mask = np.zeros((st.session_state.captured_frame.shape[0], st.session_state.captured_frame.shape[1]), dtype=np.uint8)
                        x1, y1, x2, y2 = bbox.astype(int)
                        mask[y1:y2, x1:x2] = 255
                    
                    if depth_available:
                        pil_img = Image.fromarray(cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB))
                        depth_map = depth_utils.get_depth_map(pil_img, depth_model, depth_transform, device)
                        dims = dimension_utils.get_dimensions_with_depth(mask, depth_map, pixels_per_cm)
                    else:
                        dims = calculate_basic_dimensions(mask, pixels_per_cm)
                    
                    if dims:
                        st.session_state.dimensions = dims
                        st.session_state.mode = "measured"
                        st.rerun()
        
        if st.button("🔄 Resume Scanning"):
            st.session_state.mode = "scanning"
            st.rerun()
    
    elif st.session_state.mode == "measured":
        dims = st.session_state.dimensions
        
        if dims:
            st.success("✅ Measurement Complete!")
            
            st.subheader("📊 Results")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Width", f"{dims['width_cm']:.2f} cm")
                st.metric("Height", f"{dims['height_cm']:.2f} cm")
            with col_b:
                st.metric("Depth", f"{dims['depth_cm']:.2f} cm")
                st.metric("Volume", f"{dims['volume_cm3']:.2f} cm³")
            
            if depth_available and st.button("🌟 Show Hologram", type="primary"):
                results = st.session_state.detection_results
                if hasattr(results, 'masks') and results.masks is not None:
                    mask = results.masks.data[st.session_state.selected_obj_idx].cpu().numpy()
                else:
                    bbox = results.boxes.xyxy[st.session_state.selected_obj_idx].cpu().numpy()
                    mask = np.zeros((st.session_state.captured_frame.shape[0], st.session_state.captured_frame.shape[1]))
                    x1, y1, x2, y2 = bbox.astype(int)
                    mask[y1:y2, x1:x2] = 1
                
                pil_img = Image.fromarray(cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB))
                depth_map = depth_utils.get_depth_map(pil_img, depth_model, depth_transform, device)
                
                fig = hologram_utils.create_holographic_view(mask, depth_map)
                st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🆕 New Scan"):
            st.session_state.mode = "scanning"
            st.session_state.captured_frame = None
            st.session_state.detection_results = None
            st.session_state.selected_obj_idx = None
            st.session_state.dimensions = None
            st.rerun()
