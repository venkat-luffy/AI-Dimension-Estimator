import streamlit as st
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
def load_models():
    try:
        seg_model = YOLO("yolov8n-seg.pt")
        try:
            depth_model, depth_transform, device = depth_utils.load_depth_model()
            return seg_model, depth_model, depth_transform, device, True
        except:
            return seg_model, None, None, torch.device("cpu"), False
    except:
        return None, None, None, torch.device("cpu"), False

seg_model, depth_model, depth_transform, device, depth_available = load_models()

# Session State
if "mode" not in st.session_state:
    st.session_state.mode = "camera"  # camera, captured, measured
    st.session_state.captured_image = None
    st.session_state.detection_results = None
    st.session_state.selected_obj_idx = None
    st.session_state.dimensions = None

# Simple frame capture (no processing)
def video_frame_callback(frame):
    return frame  # Just return frame as-is for smooth video

# Basic dimension calculation
def calculate_dimensions(mask, pixels_per_cm):
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
st.title("📱 AI Dimension Estimator")

if seg_model is None:
    st.error("❌ Model loading failed")
    st.stop()

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.mode == "camera":
        st.subheader("📹 Camera View")
        
        # Simple camera feed without processing
        ctx = webrtc_streamer(
            key="camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Capture button
        if st.button("📸 CAPTURE PHOTO", type="primary", use_container_width=True):
            if hasattr(ctx, 'video_receiver') and ctx.video_receiver:
                try:
                    frame = ctx.video_receiver.get_frame(timeout=1)
                    if frame:
                        img = frame.to_ndarray(format="bgr24")
                        st.session_state.captured_image = img
                        st.session_state.mode = "captured"
                        st.rerun()
                except:
                    st.error("❌ Failed to capture. Try again.")
        
        # File upload alternative
        st.divider()
        uploaded_file = st.file_uploader("Or upload photo", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            st.session_state.captured_image = img_array
            st.session_state.mode = "captured"
            st.rerun()
    
    elif st.session_state.mode == "captured":
        st.subheader("📷 Captured Photo")
        
        # Show captured image
        display_img = cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2RGB)
        st.image(display_img, use_column_width=True)
        
        # Process detection button
        if st.button("🔍 DETECT OBJECTS", type="primary", use_container_width=True):
            with st.spinner("Detecting objects..."):
                results = seg_model.predict(source=st.session_state.captured_image, conf=0.4, verbose=False)
                
                if results and len(results) > 0:
                    st.session_state.detection_results = results[0]
                    # Show detection results
                    annotated_img = results[0].plot()
                    display_annotated = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    st.image(display_annotated, caption="Detection Results", use_column_width=True)
                else:
                    st.error("No objects detected!")
    
    elif st.session_state.mode == "measured":
        st.subheader("📊 Measurement Results")
        
        # Show original image
        display_img = cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2RGB)
        st.image(display_img, use_column_width=True)
        
        # Show dimensions
        if st.session_state.dimensions:
            dims = st.session_state.dimensions
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Width", f"{dims['width_cm']:.2f} cm")
                st.metric("Height", f"{dims['height_cm']:.2f} cm")
            with col_b:
                st.metric("Depth", f"{dims['depth_cm']:.2f} cm")
                st.metric("Volume", f"{dims['volume_cm3']:.2f} cm³")

with col2:
    st.subheader("🎛️ Controls")
    
    if st.session_state.mode == "camera":
        st.info("📹 Point camera at objects and click CAPTURE")
        
    elif st.session_state.mode == "captured":
        if st.session_state.detection_results:
            results = st.session_state.detection_results
            
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                st.success(f"✅ Found {len(results.boxes)} objects!")
                
                # Object selection
                object_names = []
                for cls in results.boxes.cls:
                    object_names.append(seg_model.names[int(cls)])
                
                selected_obj = st.selectbox("📦 Select Object:", list(set(object_names)))
                
                if selected_obj:
                    st.session_state.selected_obj_idx = object_names.index(selected_obj)
                    
                    # Object category
                    living_objects = ['person', 'cat', 'dog', 'horse', 'sheep', 'cow']
                    category = "🐾 Living" if selected_obj.lower() in living_objects else "📦 Non-living"
                    st.write(f"**Category:** {category}")
                    
                    st.divider()
                    
                    # Calibration
                    st.write("**📏 Calibration**")
                    ref_width_cm = st.number_input("Known width (cm):", min_value=0.1, value=10.0)
                    ref_width_px = st.number_input("Width in pixels:", min_value=1, value=100)
                    
                    # Measure button
                    if st.button("📐 MEASURE", type="primary", use_container_width=True):
                        with st.spinner("Measuring..."):
                            pixels_per_cm = ref_width_px / ref_width_cm
                            
                            # Get mask
                            if hasattr(results, 'masks') and results.masks is not None:
                                mask = results.masks.data[st.session_state.selected_obj_idx].cpu().numpy()
                                mask = (mask * 255).astype(np.uint8)
                            else:
                                bbox = results.boxes.xyxy[st.session_state.selected_obj_idx].cpu().numpy()
                                mask = np.zeros((st.session_state.captured_image.shape[0], st.session_state.captured_image.shape[1]), dtype=np.uint8)
                                x1, y1, x2, y2 = bbox.astype(int)
                                mask[y1:y2, x1:x2] = 255
                            
                            # Calculate dimensions
                            if depth_available:
                                pil_img = Image.fromarray(cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2RGB))
                                depth_map = depth_utils.get_depth_map(pil_img, depth_model, depth_transform, device)
                                dims = dimension_utils.get_dimensions_with_depth(mask, depth_map, pixels_per_cm)
                            else:
                                dims = calculate_dimensions(mask, pixels_per_cm)
                            
                            if dims:
                                st.session_state.dimensions = dims
                                st.session_state.mode = "measured"
                                st.success("✅ Measurement complete!")
                                st.rerun()
            else:
                st.warning("⚠️ No objects found in photo")
        else:
            st.info("🔍 Click DETECT OBJECTS to analyze photo")
        
        # Return button
        if st.button("🔄 TAKE NEW PHOTO", use_container_width=True):
            st.session_state.mode = "camera"
            st.session_state.captured_image = None
            st.session_state.detection_results = None
            st.session_state.selected_obj_idx = None
            st.session_state.dimensions = None
            st.rerun()
    
    elif st.session_state.mode == "measured":
        st.success("✅ Measurement Complete!")
        
        # Hologram button
        if depth_available and st.button("🌟 3D HOLOGRAM", type="primary", use_container_width=True):
            with st.spinner("Generating 3D view..."):
                results = st.session_state.detection_results
                
                if hasattr(results, 'masks') and results.masks is not None:
                    mask = results.masks.data[st.session_state.selected_obj_idx].cpu().numpy()
                else:
                    bbox = results.boxes.xyxy[st.session_state.selected_obj_idx].cpu().numpy()
                    mask = np.zeros((st.session_state.captured_image.shape[0], st.session_state.captured_image.shape[1]))
                    x1, y1, x2, y2 = bbox.astype(int)
                    mask[y1:y2, x1:x2] = 1
                
                pil_img = Image.fromarray(cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2RGB))
                depth_map = depth_utils.get_depth_map(pil_img, depth_model, depth_transform, device)
                
                fig = hologram_utils.create_holographic_view(mask, depth_map)
                st.plotly_chart(fig, use_container_width=True)
        
        # Return button
        if st.button("📷 TAKE NEW PHOTO", use_container_width=True):
            st.session_state.mode = "camera"
            st.session_state.captured_image = None
            st.session_state.detection_results = None
            st.session_state.selected_obj_idx = None
            st.session_state.dimensions = None
            st.rerun()
