import streamlit as st

# Handle OpenCV import issues
try:
    import cv2
    opencv_available = True
except ImportError as e:
    st.error(f"❌ OpenCV import failed: {str(e)}")
    st.info("🔧 This is a Streamlit Cloud compatibility issue. The app will work with basic functionality.")
    opencv_available = False

try:
    import numpy as np
    from ultralytics import YOLO
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    from PIL import Image
    import torch
    import time
except ImportError as e:
    st.error(f"❌ Critical import error: {str(e)}")
    st.stop()

# Import custom modules with error handling
try:
    import depth_utils
    import dimension_utils
    import hologram_utils
    custom_modules_available = True
except ImportError as e:
    st.warning(f"⚠️ Custom modules import issue: {str(e)}")
    custom_modules_available = False

# Configuration
st.set_page_config(page_title="AI Dimension Estimator", layout="wide")

# Model Loading
@st.cache_resource
def load_models():
    try:
        seg_model = YOLO("yolov8n-seg.pt")
        if custom_modules_available:
            try:
                depth_model, depth_transform, device = depth_utils.load_depth_model()
                return seg_model, depth_model, depth_transform, device, True
            except:
                return seg_model, None, None, torch.device("cpu"), False
        else:
            return seg_model, None, None, torch.device("cpu"), False
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None, None, torch.device("cpu"), False

seg_model, depth_model, depth_transform, device, depth_available = load_models()

# Session State
if "mode" not in st.session_state:
    st.session_state.mode = "upload"  # Start with upload mode if OpenCV fails
    st.session_state.captured_image = None
    st.session_state.detection_results = None
    st.session_state.selected_obj_idx = None
    st.session_state.dimensions = None

# Basic dimension calculation (fallback when depth not available)
def calculate_basic_dimensions(bbox, pixels_per_cm):
    """Calculate dimensions from bounding box when OpenCV is not available"""
    x1, y1, x2, y2 = bbox
    width_px = x2 - x1
    height_px = y2 - y1
    
    width_cm = width_px / pixels_per_cm
    height_cm = height_px / pixels_per_cm
    estimated_depth_cm = (width_cm + height_cm) / 2  # Rough estimate
    volume_cm3 = width_cm * height_cm * estimated_depth_cm
    
    return {
        "width_cm": round(width_cm, 2),
        "height_cm": round(height_cm, 2),
        "depth_cm": round(estimated_depth_cm, 2),
        "volume_cm3": round(volume_cm3, 2)
    }

# Advanced dimension calculation (when OpenCV is available)
def calculate_dimensions_with_opencv(mask, pixels_per_cm):
    if not opencv_available:
        return None
    
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

# Show compatibility status
if not opencv_available:
    st.warning("⚠️ Running in compatibility mode (OpenCV unavailable)")

if seg_model is None:
    st.error("❌ Model loading failed")
    st.stop()

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.mode == "upload" or not opencv_available:
        st.subheader("📁 Upload Photo")
        
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to array
            img_array = np.array(image)
            if len(img_array.shape) == 3 and opencv_available:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            st.session_state.captured_image = img_array
            
            # Auto-detect when image is uploaded
            if st.button("🔍 DETECT OBJECTS", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyzing photo..."):
                    try:
                        # Use RGB format if OpenCV not available
                        if opencv_available:
                            source_img = img_array
                        else:
                            source_img = np.array(image)  # Keep RGB format
                        
                        results = seg_model.predict(source=source_img, conf=0.4, verbose=False)
                        
                        if results and len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                            st.session_state.detection_results = results[0]
                            st.session_state.mode = "detected"
                            
                            # Show detection results
                            annotated_img = results[0].plot()
                            if opencv_available:
                                display_annotated = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                            else:
                                display_annotated = annotated_img
                            
                            st.image(display_annotated, caption="🎯 Objects Detected!", use_column_width=True)
                            st.success(f"✅ Found {len(results[0].boxes)} objects!")
                            st.rerun()
                        else:
                            st.error("❌ No objects detected in photo.")
                    except Exception as e:
                        st.error(f"❌ Detection failed: {str(e)}")
    
    elif st.session_state.mode == "detected":
        st.subheader("🎯 Detection Results")
        
        if st.session_state.detection_results:
            # Show detection results
            annotated_img = st.session_state.detection_results.plot()
            if opencv_available:
                display_annotated = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            else:
                display_annotated = annotated_img
            st.image(display_annotated, use_column_width=True)
    
    elif st.session_state.mode == "measured":
        st.subheader("📊 Measurement Results")
        
        # Show results image
        if st.session_state.detection_results:
            annotated_img = st.session_state.detection_results.plot()
            if opencv_available:
                display_annotated = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            else:
                display_annotated = annotated_img
            st.image(display_annotated, use_column_width=True)
        
        # Show dimensions
        if st.session_state.dimensions:
            dims = st.session_state.dimensions
            
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
    
    if st.session_state.mode == "upload":
        st.info("📁 **Instructions:**")
        st.write("1. Upload a clear photo with objects")
        st.write("2. Click DETECT OBJECTS")
        st.write("3. Select object to measure")
        
    elif st.session_state.mode == "detected":
        if st.session_state.detection_results:
            results = st.session_state.detection_results
            
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                st.success(f"🎯 Found {len(results.boxes)} objects!")
                
                # Object selection
                object_names = []
                for cls in results.boxes.cls:
                    object_names.append(seg_model.names[int(cls)])
                
                unique_objects = list(set(object_names))
                selected_obj = st.selectbox("📦 **Select Object:**", unique_objects)
                
                if selected_obj:
                    st.session_state.selected_obj_idx = object_names.index(selected_obj)
                    
                    # Object category
                    living_objects = ['person', 'cat', 'dog', 'horse', 'sheep', 'cow']
                    category = "🐾 Living" if selected_obj.lower() in living_objects else "📦 Non-living"
                    st.write(f"**Category:** {category}")
                    
                    st.divider()
                    
                    # Calibration
                    st.write("**📏 Calibration:**")
                    ref_width_cm = st.number_input("Known width (cm):", min_value=0.1, value=10.0, step=0.1)
                    ref_width_px = st.number_input("Width in pixels:", min_value=1, value=100, step=1)
                    
                    # Measure button
                    if st.button("📐 MEASURE", type="primary", use_container_width=True):
                        with st.spinner("📐 Measuring..."):
                            try:
                                pixels_per_cm = ref_width_px / ref_width_cm
                                
                                # Calculate dimensions based on available methods
                                if opencv_available and hasattr(results, 'masks') and results.masks is not None:
                                    # Use mask-based calculation
                                    mask = results.masks.data[st.session_state.selected_obj_idx].cpu().numpy()
                                    mask = (mask * 255).astype(np.uint8)
                                    dims = calculate_dimensions_with_opencv(mask, pixels_per_cm)
                                else:
                                    # Use bounding box calculation
                                    bbox = results.boxes.xyxy[st.session_state.selected_obj_idx].cpu().numpy()
                                    dims = calculate_basic_dimensions(bbox, pixels_per_cm)
                                
                                if dims:
                                    st.session_state.dimensions = dims
                                    st.session_state.mode = "measured"
                                    st.success("✅ Measurement complete!")
                                    st.rerun()
                                else:
                                    st.error("❌ Measurement failed")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
        
        # Back button
        st.divider()
        if st.button("📁 UPLOAD NEW PHOTO", use_container_width=True):
            st.session_state.mode = "upload"
            st.session_state.captured_image = None
            st.session_state.detection_results = None
            st.session_state.selected_obj_idx = None
            st.session_state.dimensions = None
            st.rerun()
    
    elif st.session_state.mode == "measured":
        st.success("✅ **Complete!**")
        
        # Object info
        if st.session_state.selected_obj_idx is not None and st.session_state.detection_results:
            results = st.session_state.detection_results
            object_names = [seg_model.names[int(cls)] for cls in results.boxes.cls]
            selected_name = object_names[st.session_state.selected_obj_idx]
            confidence = results.boxes.conf[st.session_state.selected_obj_idx]
            st.write(f"**Object:** {selected_name}")
            st.write(f"**Confidence:** {confidence:.1%}")
        
        st.divider()
        
        # Hologram button (if available)
        if depth_available and custom_modules_available:
            if st.button("🌟 3D HOLOGRAM", type="primary", use_container_width=True):
                with st.spinner("🎭 Generating..."):
                    try:
                        results = st.session_state.detection_results
                        
                        if opencv_available and hasattr(results, 'masks') and results.masks is not None:
                            mask = results.masks.data[st.session_state.selected_obj_idx].cpu().numpy()
                        else:
                            # Create simple mask from bounding box
                            bbox = results.boxes.xyxy[st.session_state.selected_obj_idx].cpu().numpy()
                            mask = np.zeros((st.session_state.captured_image.shape[0], st.session_state.captured_image.shape[1]))
                            x1, y1, x2, y2 = bbox.astype(int)
                            mask[y1:y2, x1:x2] = 1
                        
                        if opencv_available:
                            pil_img = Image.fromarray(cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2RGB))
                        else:
                            pil_img = Image.fromarray(st.session_state.captured_image)
                        
                        depth_map = depth_utils.get_depth_map(pil_img, depth_model, depth_transform, device)
                        fig = hologram_utils.create_holographic_view(mask, depth_map)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Hologram error: {str(e)}")
        else:
            st.info("🌟 3D Hologram unavailable (requires depth model)")
        
        # Back button
        st.divider()
        if st.button("📁 UPLOAD NEW PHOTO", use_container_width=True):
            st.session_state.mode = "upload"
            st.session_state.captured_image = None
            st.session_state.detection_results = None
            st.session_state.selected_obj_idx = None
            st.session_state.dimensions = None
            st.rerun()

# Footer
st.divider()
st.info("💡 **Status:** " + 
        f"Object Detection: {'✅' if seg_model else '❌'} | " +
        f"OpenCV: {'✅' if opencv_available else '❌'} | " +
        f"Depth: {'✅' if depth_available else '❌'}")
