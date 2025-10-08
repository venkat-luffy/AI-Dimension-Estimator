import streamlit as st

# Handle import errors gracefully
try:
    import cv2
except ImportError:
    st.error("❌ OpenCV import failed. Installing...")
    st.code("pip install opencv-python-headless==4.8.0.76")
    st.stop()

try:
    import av
    import numpy as np
    from ultralytics import YOLO
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    from PIL import Image
    import torch
    import logging
    import time
except ImportError as e:
    st.error(f"❌ Import error: {str(e)}")
    st.info("Please check your requirements.txt file")
    st.stop()

# --- Import custom utility modules ---
try:
    import depth_utils
    import dimension_utils
    import hologram_utils
except ImportError as e:
    st.error(f"❌ Custom module import error: {str(e)}")
    st.info("Please ensure all utility files are present in your repository")
    st.stop()

# --- Configuration ---
st.set_page_config(page_title="AI Dimension Estimator", layout="wide")

# --- Suppress warnings ---
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# --- Model Loading with Error Handling ---
@st.cache_resource
def load_all_models():
    models_loaded = {"yolo": False, "depth": False}
    seg_model = None
    depth_model = None
    depth_transform = None
    device = torch.device("cpu")
    
    try:
        # Load YOLOv8 model
        st.info("🔄 Loading YOLOv8 model...")
        seg_model = YOLO("yolov8n-seg.pt")
        models_loaded["yolo"] = True
        st.success("✅ YOLOv8 model loaded successfully")
        
        # Load depth model with error handling
        st.info("🔄 Loading MiDaS depth model...")
        try:
            depth_model, depth_transform, device = depth_utils.load_depth_model()
            models_loaded["depth"] = True
            st.success("✅ MiDaS depth model loaded successfully")
        except Exception as e:
            st.warning(f"⚠️ Depth model failed: {str(e)}")
            st.info("💡 App will work with basic measurements only")
        
        return seg_model, depth_model, depth_transform, device, models_loaded
    
    except Exception as e:
        st.error(f"🚨 Error loading models: {str(e)}")
        return None, None, None, device, {"yolo": False, "depth": False}

# Load models
seg_model, depth_model, depth_transform, device, models_loaded = load_all_models()

# --- Session State ---
if "mode" not in st.session_state:
    st.session_state.mode = "scanning"
    st.session_state.captured_frame = None
    st.session_state.detection_results = None
    st.session_state.selected_obj_idx = None
    st.session_state.dimensions = None
    st.session_state.processing_error = None

# --- Global variables ---
latest_frame = None
processing_active = True

# --- Frame Processing Function ---
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
            
            # Store results in session state
            if st.session_state.mode == "scanning":
                st.session_state.detection_results = result
            
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
        else:
            return frame
            
    except Exception as e:
        st.session_state.processing_error = str(e)
        return frame

# --- Basic dimension calculation without depth ---
def calculate_basic_dimensions(mask, pixels_per_cm):
    """Calculate basic dimensions using only bounding box"""
    try:
        x, y, w, h = cv2.boundingRect(mask)
        width_cm = w / pixels_per_cm
        height_cm = h / pixels_per_cm
        
        # Estimate depth as average of width and height
        estimated_depth_cm = (width_cm + height_cm) / 2
        volume_cm3 = width_cm * height_cm * estimated_depth_cm
        
        return {
            "width_cm": round(width_cm, 2),
            "height_cm": round(height_cm, 2),
            "depth_cm": round(estimated_depth_cm, 2),
            "volume_cm3": round(volume_cm3, 2)
        }
    except Exception as e:
        st.error(f"Error in basic calculation: {str(e)}")
        return None

# --- Main App ---
st.title("🤖 AI-Based Real-Time Object Dimension Estimator")

# Check if basic functionality works
if not models_loaded["yolo"]:
    st.error("❌ Core models failed to load. Please try refreshing the page.")
    st.info("🔧 **Troubleshooting:**")
    st.info("1. Check if all dependencies are installed")
    st.info("2. Refresh the browser page")
    st.info("3. Check Streamlit Cloud logs for detailed errors")
    st.stop()

# Show model status
st.info("🎯 **Model Status:**")
st.write(f"• Object Detection: {'✅ Working' if models_loaded['yolo'] else '❌ Failed'}")
st.write(f"• Depth Estimation: {'✅ Working' if models_loaded['depth'] else '⚠️ Basic mode'}")

# --- Main Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    
    # Show processing status
    status_placeholder = st.empty()
    
    try:
        # WebRTC Streamer
        ctx = webrtc_streamer(
            key="ai-dimension-estimator",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Connection status
        if hasattr(ctx, 'state'):
            if ctx.state.playing:
                status_placeholder.success("🟢 Camera Connected - Live Detection Active")
            elif ctx.state.signalling:
                status_placeholder.info("🟡 Connecting to Camera...")
            else:
                status_placeholder.warning("🔴 Camera Disconnected - Click START to begin")
        else:
            status_placeholder.info("📷 Ready to connect camera")
            
    except Exception as e:
        st.error(f"❌ WebRTC Error: {str(e)}")
        st.info("💡 **Alternative Solution:** Use the file upload option below")
    
    # File upload alternative
    st.divider()
    st.subheader("📁 Alternative: Upload Image")
    uploaded_file = st.file_uploader("Upload an image for object detection", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Run Detection", type="primary"):
            with st.spinner("Processing..."):
                try:
                    # Convert to numpy array
                    img_array = np.array(image)
                    if len(img_array.shape) == 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # Run detection
                    results = seg_model.predict(source=img_array, conf=0.4, verbose=False)
                    
                    if results and len(results) > 0:
                        result = results[0]
                        annotated_img = result.plot()
                        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), 
                                caption="Detection Results", use_column_width=True)
                        
                        # Store results for measurement
                        st.session_state.captured_frame = img_array
                        st.session_state.detection_results = result
                        st.session_state.mode = "paused"
                        
                        st.success("✅ Detection complete! Check the controls panel →")
                        st.rerun()
                    else:
                        st.warning("No detection results")
                        
                except Exception as e:
                    st.error(f"Detection error: {str(e)}")

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
                
                # Category determination
                living_objects = ['person', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
                category = "Living" if selected_obj_name.lower() in living_objects else "Non-living"
                st.write(f"**Category:** {category}")
                
                st.divider()
                
                # Calibration section
                st.subheader("📏 Calibration")
                if not models_loaded["depth"]:
                    st.warning("⚠️ Using basic 2D measurements (no depth model)")
                else:
                    st.info("ℹ️ Manual calibration required for accurate measurements")
                
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
                            
                            # Get mask
                            if hasattr(results, 'masks') and results.masks is not None:
                                mask = results.masks.data[st.session_state.selected_obj_idx].cpu().numpy()
                                mask = (mask * 255).astype(np.uint8)
                            else:
                                # Fallback: create mask from bounding box
                                bbox = results.boxes.xyxy[st.session_state.selected_obj_idx].cpu().numpy()
                                mask = np.zeros((st.session_state.captured_frame.shape[0], st.session_state.captured_frame.shape[1]), dtype=np.uint8)
                                x1, y1, x2, y2 = bbox.astype(int)
                                mask[y1:y2, x1:x2] = 255
                            
                            # Calculate dimensions
                            if models_loaded["depth"]:
                                # Use depth estimation if available
                                pil_img = Image.fromarray(cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB))
                                depth_map = depth_utils.get_depth_map(pil_img, depth_model, depth_transform, device)
                                dims = dimension_utils.get_dimensions_with_depth(mask, depth_map, pixels_per_cm)
                            else:
                                # Use basic calculation
                                dims = calculate_basic_dimensions(mask, pixels_per_cm)
                            
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
            
            # Show measurement type
            if not models_loaded["depth"]:
                st.info("📏 Basic 2D measurements (estimated depth)")
            else:
                st.info("📐 Advanced measurements with depth estimation")
            
            st.subheader("📊 Dimension Results")
            
            # Display measurements
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Width", f"{dims['width_cm']:.2f} cm")
                st.metric("Height", f"{dims['height_cm']:.2f} cm")
            with col_b:
                st.metric("Estimated Depth", f"{dims['depth_cm']:.2f} cm")
                st.metric("Volume", f"{dims['volume_cm3']:.2f} cm³")
            
            # Hologram button (only if depth model is available)
            if models_loaded["depth"]:
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
            else:
                st.info("🌟 Holographic view requires depth model (currently unavailable)")
        
        # New scan button
        if st.button("🆕 Start New Scan", use_container_width=True):
            st.session_state.mode = "scanning"
            st.session_state.captured_frame = None
            st.session_state.detection_results = None
            st.session_state.selected_obj_idx = None
            st.session_state.dimensions = None
            processing_active = True
            st.rerun()

# Show processing errors
if st.session_state.processing_error:
    st.error(f"Processing Error: {st.session_state.processing_error}")

# --- Footer ---
st.divider()
st.markdown("**🔧 Status & Tips:**")
st.markdown(f"• **YOLOv8 Detection:** {'✅ Active' if models_loaded['yolo'] else '❌ Failed'}")
st.markdown(f"• **Depth Estimation:** {'✅ Active' if models_loaded['depth'] else '⚠️ Basic mode only'}")
st.markdown("• Ensure good lighting for better detection")
st.markdown("• Point camera directly at objects")
st.markdown("• **Camera Issues?** Try the image upload option")
st.markdown("• Allow camera permissions when prompted")

# --- Debug info for development ---
if st.checkbox("🔍 Show Debug Info"):
    st.json({
        "Mode": st.session_state.mode,
        "Models Loaded": models_loaded,
        "Has Detection Results": st.session_state.detection_results is not None,
        "Has Captured Frame": st.session_state.captured_frame is not None,
        "Has Dimensions": st.session_state.dimensions is not None
    })
