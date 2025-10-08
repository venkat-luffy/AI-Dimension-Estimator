import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from PIL import Image
import torch
import time

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
    st.session_state.mode = "camera"
    st.session_state.captured_image = None
    st.session_state.detection_results = None
    st.session_state.selected_obj_idx = None
    st.session_state.dimensions = None
    st.session_state.latest_frame = None

# Global variable to store latest frame
captured_frame = None

# Frame capture callback
def video_frame_callback(frame):
    global captured_frame
    img = frame.to_ndarray(format="bgr24")
    captured_frame = img.copy()  # Store the latest frame
    return frame

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
        
        # Camera stream
        ctx = webrtc_streamer(
            key="photo_capture",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,  # Changed to False for better frame capture
        )
        
        # Capture button
        if st.button("📸 CAPTURE PHOTO", type="primary", use_container_width=True):
            if captured_frame is not None:
                st.session_state.captured_image = captured_frame.copy()
                st.session_state.mode = "captured"
                st.success("✅ Photo captured!")
                time.sleep(0.5)  # Small delay to show success message
                st.rerun()
            else:
                st.error("❌ No camera frame available. Make sure camera is connected.")
        
        # Show current frame status
        if captured_frame is not None:
            st.info("📷 Camera is ready - Click CAPTURE PHOTO")
        else:
            st.warning("🔴 Waiting for camera connection...")
        
        # File upload alternative
        st.divider()
        st.write("**OR Upload Photo:**")
        uploaded_file = st.file_uploader("Choose image file", type=['jpg', 'jpeg', 'png'])
        
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
            with st.spinner("🔍 Analyzing photo..."):
                try:
                    results = seg_model.predict(source=st.session_state.captured_image, conf=0.4, verbose=False)
                    
                    if results and len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                        st.session_state.detection_results = results[0]
                        # Show detection results
                        annotated_img = results[0].plot()
                        display_annotated = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                        st.image(display_annotated, caption="🎯 Objects Detected!", use_column_width=True)
                        st.success(f"✅ Found {len(results[0].boxes)} objects!")
                    else:
                        st.error("❌ No objects detected in photo. Try a different angle or lighting.")
                except Exception as e:
                    st.error(f"❌ Detection failed: {str(e)}")
    
    elif st.session_state.mode == "measured":
        st.subheader("📊 Measurement Results")
        
        # Show original image with detection overlay
        if st.session_state.detection_results:
            annotated_img = st.session_state.detection_results.plot()
            display_annotated = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(display_annotated, use_column_width=True)
        
        # Show dimensions
        if st.session_state.dimensions:
            dims = st.session_state.dimensions
            
            # Create measurement display
            measurement_cols = st.columns(4)
            with measurement_cols[0]:
                st.metric("📏 Width", f"{dims['width_cm']:.2f} cm")
            with measurement_cols[1]:
                st.metric("📐 Height", f"{dims['height_cm']:.2f} cm")
            with measurement_cols[2]:
                st.metric("🔺 Depth", f"{dims['depth_cm']:.2f} cm")
            with measurement_cols[3]:
                st.metric("📦 Volume", f"{dims['volume_cm3']:.2f} cm³")

with col2:
    st.subheader("🎛️ Controls")
    
    if st.session_state.mode == "camera":
        st.info("📹 **Instructions:**")
        st.write("1. Position camera to see objects clearly")
        st.write("2. Click CAPTURE PHOTO when ready")
        st.write("3. Or upload an existing photo below")
        
    elif st.session_state.mode == "captured":
        if st.session_state.detection_results:
            results = st.session_state.detection_results
            
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                st.success(f"🎯 Found {len(results.boxes)} objects!")
                
                # Object selection
                object_names = []
                object_confidences = []
                for i, cls in enumerate(results.boxes.cls):
                    name = seg_model.names[int(cls)]
                    conf = results.boxes.conf[i]
                    object_names.append(name)
                    object_confidences.append(conf)
                
                # Create dropdown options with confidence
                unique_objects = list(set(object_names))
                selected_obj = st.selectbox("📦 **Select Object to Measure:**", unique_objects)
                
                if selected_obj:
                    st.session_state.selected_obj_idx = object_names.index(selected_obj)
                    
                    # Object category
                    living_objects = ['person', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear']
                    category = "🐾 Living" if selected_obj.lower() in living_objects else "📦 Non-living"
                    st.write(f"**Category:** {category}")
                    
                    st.divider()
                    
                    # Calibration
                    st.write("**📏 Calibration Required:**")
                    st.info("💡 Measure a known object in the photo for accurate results")
                    
                    ref_width_cm = st.number_input(
                        "Known reference width (cm):", 
                        min_value=0.1, 
                        value=10.0,
                        step=0.1,
                        help="Enter the real-world width of a reference object"
                    )
                    ref_width_px = st.number_input(
                        "Reference width in pixels:", 
                        min_value=1, 
                        value=100,
                        step=1,
                        help="Count pixels for the same reference object"
                    )
                    
                    # Measure button
                    if st.button("📐 MEASURE OBJECT", type="primary", use_container_width=True):
                        with st.spinner("📐 Calculating dimensions..."):
                            try:
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
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.error("❌ Measurement calculation failed")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ No objects detected in photo")
        else:
            st.info("🔍 Click DETECT OBJECTS to analyze the photo")
        
        # Return button
        st.divider()
        if st.button("📷 TAKE NEW PHOTO", use_container_width=True):
            st.session_state.mode = "camera"
            st.session_state.captured_image = None
            st.session_state.detection_results = None
            st.session_state.selected_obj_idx = None
            st.session_state.dimensions = None
            st.rerun()
    
    elif st.session_state.mode == "measured":
        st.success("✅ **Measurement Complete!**")
        
        # Show selected object info
        if st.session_state.selected_obj_idx is not None and st.session_state.detection_results:
            results = st.session_state.detection_results
            object_names = [seg_model.names[int(cls)] for cls in results.boxes.cls]
            selected_name = object_names[st.session_state.selected_obj_idx]
            confidence = results.boxes.conf[st.session_state.selected_obj_idx]
            st.write(f"**Object:** {selected_name}")
            st.write(f"**Confidence:** {confidence:.1%}")
        
        st.divider()
        
        # Hologram button
        if depth_available:
            if st.button("🌟 SHOW 3D HOLOGRAM", type="primary", use_container_width=True):
                with st.spinner("🎭 Generating 3D holographic view..."):
                    try:
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
                        
                    except Exception as e:
                        st.error(f"❌ Hologram error: {str(e)}")
        else:
            st.info("🌟 3D Hologram requires depth model (currently unavailable)")
        
        # Return button
        st.divider()
        if st.button("📷 TAKE NEW PHOTO", use_container_width=True):
            st.session_state.mode = "camera"
            st.session_state.captured_image = None
            st.session_state.detection_results = None
            st.session_state.selected_obj_idx = None
            st.session_state.dimensions = None
            st.rerun()
