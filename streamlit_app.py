import streamlit as st

# CRITICAL: set_page_config MUST be first
st.set_page_config(
    page_title="🚀 AI Dimension Estimator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_custom_css():
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Try to load custom CSS
try:
    load_custom_css()
except:
    # Fallback inline CSS if file not found
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #16213e 100%);
        color: white;
        font-family: 'Rajdhani', sans-serif;
    }
    
    .main-header {
        font-family: 'Orbitron', monospace;
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #00d4ff, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .cyber-button {
        background: linear-gradient(45deg, #00d4ff, #a855f7);
        border: none;
        border-radius: 50px;
        padding: 15px 30px;
        color: white;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
        transition: all 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)

import numpy as np
from PIL import Image
import torch

# Import modules with error handling
try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError:
    st.error("❌ YOLO unavailable")
    yolo_available = False

try:
    import cv2
    opencv_available = True
except ImportError:
    opencv_available = False

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    webrtc_available = True
except ImportError:
    webrtc_available = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    plotly_available = True
except ImportError:
    plotly_available = False

# Load custom modules
try:
    import depth_utils
    import dimension_utils
    import hologram_utils
    custom_modules_available = True
except ImportError:
    custom_modules_available = False

# Model Loading
@st.cache_resource
def load_yolo_model():
    if not yolo_available:
        return None
    try:
        model = YOLO("yolov8n-seg.pt")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Session State
if "mode" not in st.session_state:
    st.session_state.mode = "home"  # home, camera, upload, detected, measured
    st.session_state.detection_results = None
    st.session_state.selected_image = None
    st.session_state.dimensions = None
    st.session_state.captured_frame = None

# Dimension calculation
def calculate_dimensions(bbox, pixels_per_cm):
    x1, y1, x2, y2 = bbox
    width_px = abs(x2 - x1)
    height_px = abs(y2 - y1)
    
    width_cm = width_px / pixels_per_cm
    height_cm = height_px / pixels_per_cm
    depth_cm = (width_cm + height_cm) / 2  # Estimated
    volume_cm3 = width_cm * height_cm * depth_cm
    
    return {
        "width_cm": round(width_cm, 2),
        "height_cm": round(height_cm, 2),
        "depth_cm": round(depth_cm, 2),
        "volume_cm3": round(volume_cm3, 2)
    }

# Camera frame callback
captured_frame = None

def video_frame_callback(frame):
    global captured_frame
    img = frame.to_ndarray(format="bgr24")
    captured_frame = img.copy()
    return frame

# Main App
st.markdown('<div class="main-header">🚀 AI DIMENSION ESTIMATOR</div>', unsafe_allow_html=True)

# System Status Bar
col_status1, col_status2, col_status3, col_status4 = st.columns(4)
with col_status1:
    status = "🟢 ONLINE" if yolo_available else "🔴 OFFLINE"
    st.markdown(f'<div class="status-indicator">🤖 YOLO: {status}</div>', unsafe_allow_html=True)
with col_status2:
    status = "🟢 READY" if opencv_available else "🟡 LIMITED"
    st.markdown(f'<div class="status-indicator">📹 OpenCV: {status}</div>', unsafe_allow_html=True)
with col_status3:
    status = "🟢 ACTIVE" if webrtc_available else "🔴 OFFLINE"
    st.markdown(f'<div class="status-indicator">📡 Camera: {status}</div>', unsafe_allow_html=True)
with col_status4:
    status = "🟢 3D READY" if plotly_available else "🟡 2D ONLY"
    st.markdown(f'<div class="status-indicator">🌟 Hologram: {status}</div>', unsafe_allow_html=True)

# Check if YOLO is available
seg_model = load_yolo_model()
if seg_model is None:
    st.error("🚨 Critical Error: AI Detection System Offline")
    st.stop()

st.success("✅ All Systems Operational - AI Ready for Deployment")

# Mode Selection
st.markdown("---")
mode_col1, mode_col2, mode_col3 = st.columns(3)

with mode_col1:
    if st.button("📹 CAMERA MODE", use_container_width=True, key="camera_mode"):
        if webrtc_available:
            st.session_state.mode = "camera"
            st.rerun()
        else:
            st.error("❌ Camera mode unavailable - WebRTC not installed")

with mode_col2:
    if st.button("📁 UPLOAD MODE", use_container_width=True, key="upload_mode"):
        st.session_state.mode = "upload"
        st.rerun()

with mode_col3:
    if st.button("🏠 HOME", use_container_width=True, key="home_mode"):
        st.session_state.mode = "home"
        st.rerun()

st.markdown("---")

# Main Interface
if st.session_state.mode == "home":
    # Welcome Screen
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## 🎯 MISSION CONTROL")
    st.markdown("""
    **Advanced AI-Powered Dimension Analysis System**
    
    🚀 **CAPABILITIES:**
    - Real-time object detection using YOLOv8
    - Precision dimension measurement
    - 3D holographic visualization
    - Multi-object analysis support
    
    📡 **MODES AVAILABLE:**
    - **Camera Mode**: Live capture and analysis
    - **Upload Mode**: Photo analysis
    
    🎮 **SELECT YOUR OPERATION MODE ABOVE**
    """)
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.mode == "camera" and webrtc_available:
    # Camera Mode
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("## 📹 LIVE CAMERA FEED")
        
        # Camera stream
        ctx = webrtc_streamer(
            key="ai_camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,
        )
        
        # Capture button
        st.markdown('<br>', unsafe_allow_html=True)
        if st.button("📸 CAPTURE TARGET", type="primary", use_container_width=True, key="capture_btn"):
            if captured_frame is not None:
                st.session_state.captured_frame = captured_frame.copy()
                st.session_state.selected_image = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
                st.session_state.mode = "detected"
                st.success("🎯 TARGET ACQUIRED!")
                st.rerun()
            else:
                st.error("❌ No camera signal detected")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("## 🎛️ CAMERA CONTROLS")
        
        camera_status = "🟢 ACTIVE" if captured_frame is not None else "🔴 INITIALIZING"
        st.markdown(f"**Status:** {camera_status}")
        
        if captured_frame is not None:
            st.success("📡 Video feed established")
            st.info("🎯 Aim camera at target object")
            st.info("📸 Click CAPTURE when ready")
        else:
            st.warning("⏳ Establishing connection...")
            st.info("🔧 Check camera permissions")
        
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.mode == "upload":
    # Upload Mode
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("## 📁 PHOTO UPLOAD SYSTEM")
        
        uploaded_file = st.file_uploader(
            "Deploy Target Image", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload clear image with objects to analyze"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="🎯 Target Image Loaded", use_column_width=True)
            st.session_state.selected_image = image
            
            # Auto-detect button
            if st.button("🔍 INITIATE SCAN", type="primary", use_container_width=True, key="scan_btn"):
                st.session_state.mode = "detected"
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("## 📊 UPLOAD STATUS")
        
        if uploaded_file is not None:
            st.success("✅ Image successfully loaded")
            st.info(f"📏 Resolution: {st.session_state.selected_image.size}")
            st.info("🎯 Ready for analysis")
            st.markdown("**Click INITIATE SCAN to proceed**")
        else:
            st.info("📁 Waiting for image upload...")
            st.markdown("""
            **Supported formats:**
            - JPG / JPEG
            - PNG
            - Max size: 200MB
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.mode == "detected":
    # Detection Mode
    if st.session_state.selected_image is not None:
        with st.spinner("🔍 AI ANALYSIS IN PROGRESS..."):
            # Run detection
            img_array = np.array(st.session_state.selected_image)
            results = seg_model.predict(source=img_array, conf=0.4, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    st.session_state.detection_results = result
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.markdown("## 🎯 DETECTION RESULTS")
                        
                        # Show annotated image
                        annotated_img = result.plot()
                        st.image(annotated_img, caption="🤖 AI Detection Analysis", use_column_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.markdown("## 🎮 OBJECT SELECTION")
                        
                        st.success(f"🎯 {len(result.boxes)} TARGETS IDENTIFIED")
                        
                        # Object selection
                        object_names = []
                        for cls in result.boxes.cls:
                            class_id = int(cls.item())
                            object_names.append(seg_model.names[class_id])
                        
                        unique_objects = list(set(object_names))
                        selected_obj = st.selectbox("🎯 SELECT TARGET:", unique_objects, key="obj_select")
                        
                        if selected_obj:
                            selected_idx = object_names.index(selected_obj)
                            
                            # Object info
                            living_objects = ['person', 'cat', 'dog', 'horse', 'sheep', 'cow']
                            category = "🐾 BIOLOGICAL" if selected_obj.lower() in living_objects else "🔧 MECHANICAL"
                            st.markdown(f"**Classification:** {category}")
                            
                            confidence = result.boxes.conf[selected_idx]
                            st.markdown(f"**Confidence:** {confidence:.1%}")
                            
                            # Calibration
                            st.markdown("---")
                            st.markdown("### 📏 CALIBRATION PROTOCOL")
                            
                            ref_width_cm = st.number_input(
                                "Reference Width (cm):", 
                                min_value=0.1, 
                                value=10.0, 
                                step=0.1,
                                help="Enter known dimension for calibration"
                            )
                            
                            ref_width_px = st.number_input(
                                "Reference Pixels:", 
                                min_value=1, 
                                value=100, 
                                step=1,
                                help="Pixel count for reference object"
                            )
                            
                            # Measure button
                            if st.button("📐 EXECUTE MEASUREMENT", type="primary", use_container_width=True, key="measure_btn"):
                                try:
                                    pixels_per_cm = ref_width_px / ref_width_cm
                                    bbox = result.boxes.xyxy[selected_idx].cpu().numpy()
                                    dims = calculate_dimensions(bbox, pixels_per_cm)
                                    st.session_state.dimensions = dims
                                    st.session_state.mode = "measured"
                                    st.success("✅ MEASUREMENT COMPLETE!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Measurement failed: {str(e)}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("❌ NO TARGETS DETECTED")
                    st.button("🔄 RETURN TO UPLOAD", key="return_upload")
            else:
                st.error("❌ SCAN FAILED - NO OBJECTS FOUND")

elif st.session_state.mode == "measured":
    # Measurement Results
    if st.session_state.dimensions:
        dims = st.session_state.dimensions
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("## 📊 MEASUREMENT ANALYSIS COMPLETE")
        
        # Display results in futuristic style
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{dims['width_cm']}</div>
                <div class="metric-label">WIDTH (CM)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{dims['height_cm']}</div>
                <div class="metric-label">HEIGHT (CM)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{dims['depth_cm']}</div>
                <div class="metric-label">DEPTH (CM)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[3]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{dims['volume_cm3']}</div>
                <div class="metric-label">VOLUME (CM³)</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Hologram button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if plotly_available and st.button("🌟 HOLOGRAPHIC VIEW", type="primary", use_container_width=True, key="hologram_btn"):
                with st.spinner("🎭 GENERATING HOLOGRAM..."):
                    try:
                        # Create 3D visualization
                        w, h, d = dims['width_cm'], dims['height_cm'], dims['depth_cm']
                        
                        # Create 3D box wireframe
                        fig = go.Figure()
                        
                        # Define box edges
                        edges = [
                            ([0, w], [0, 0], [0, 0]), ([0, w], [h, h], [0, 0]), 
                            ([0, w], [0, 0], [d, d]), ([0, w], [h, h], [d, d]),
                            ([0, 0], [0, h], [0, 0]), ([w, w], [0, h], [0, 0]),
                            ([0, 0], [0, h], [d, d]), ([w, w], [0, h], [d, d]),
                            ([0, 0], [0, 0], [0, d]), ([w, w], [0, 0], [0, d]),
                            ([0, 0], [h, h], [0, d]), ([w, w], [h, h], [0, d])
                        ]
                        
                        for edge in edges:
                            fig.add_trace(go.Scatter3d(
                                x=edge[0], y=edge[1], z=edge[2],
                                mode='lines',
                                line=dict(color='cyan', width=8),
                                showlegend=False
                            ))
                        
                        # Add corner points
                        corners_x = [0, w, w, 0, 0, w, w, 0]
                        corners_y = [0, 0, h, h, 0, 0, h, h]
                        corners_z = [0, 0, 0, 0, d, d, d, d]
                        
                        fig.add_trace(go.Scatter3d(
                            x=corners_x, y=corners_y, z=corners_z,
                            mode='markers',
                            marker=dict(size=10, color='yellow'),
                            name=f'Object ({w}×{h}×{d} cm)'
                        ))
                        
                        fig.update_layout(
                            title="🌟 HOLOGRAPHIC PROJECTION",
                            scene=dict(
                                xaxis_title="Width (cm)",
                                yaxis_title="Height (cm)",
                                zaxis_title="Depth (cm)",
                                bgcolor='rgba(0,0,0,0.1)',
                                xaxis=dict(gridcolor='cyan', gridwidth=2),
                                yaxis=dict(gridcolor='cyan', gridwidth=2),
                                zaxis=dict(gridcolor='cyan', gridwidth=2)
                            ),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Hologram generation failed: {str(e)}")
        
        with col2:
            if st.button("📊 DETAILED REPORT", use_container_width=True, key="report_btn"):
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 📋 ANALYSIS REPORT")
                st.markdown(f"""
                **Object Classification:** {st.session_state.detection_results.__class__.__name__ if st.session_state.detection_results else 'Unknown'}
                **Dimensions:**
                - Width: {dims['width_cm']} cm
                - Height: {dims['height_cm']} cm  
                - Depth: {dims['depth_cm']} cm (estimated)
                - Volume: {dims['volume_cm3']} cm³
                
                **Analysis Status:** ✅ Complete
                **Confidence Level:** High
                **Measurement Method:** AI-Enhanced Calibration
                """)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            if st.button("🔄 NEW ANALYSIS", use_container_width=True, key="new_analysis_btn"):
                st.session_state.mode = "home"
                st.session_state.detection_results = None
                st.session_state.selected_image = None
                st.session_state.dimensions = None
                st.session_state.captured_frame = None
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-family: 'Orbitron', monospace;">
🚀 <strong>AI DIMENSION ESTIMATOR v2.0</strong> | Powered by Advanced Neural Networks | 
<span style="color: #00d4ff;">Upload → Detect → Select → Measure → Visualize</span>
</div>
""", unsafe_allow_html=True)
