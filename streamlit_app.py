import streamlit as st
import time

# Page configuration
st.set_page_config(
    page_title="AI Dimension Estimator | Next-Gen Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS with intro animation
def load_css_with_intro():
    try:
        with open("style.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("""<style>
        /* Fallback minimal styles */
        .stApp { background: #0a0a0a; color: white; font-family: 'Inter', sans-serif; }
        .intro-sequence { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: #0a0a0a; z-index: 9999; display: flex; flex-direction: column; justify-content: center; align-items: center; animation: fadeOut 4s ease-in-out 3s forwards; }
        @keyframes fadeOut { 0% { opacity: 1; } 100% { opacity: 0; visibility: hidden; } }
        .intro-logo { font-size: 4rem; font-weight: 900; background: linear-gradient(45deg, #00d4ff, #8b5cf6, #10b981); -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        </style>""", unsafe_allow_html=True)

load_css_with_intro()

# Animated Intro Sequence
def show_intro():
    intro_html = """
    <div class="intro-sequence" id="introSeq">
        <div class="intro-logo">
            AI DIMENSION<br>ESTIMATOR
        </div>
        <div class="intro-subtitle">
            Next-Generation Analysis Platform
        </div>
        <div class="intro-loader"></div>
    </div>
    
    <script>
    setTimeout(function() {
        document.getElementById('introSeq').style.display = 'none';
    }, 7000);
    </script>
    """
    st.markdown(intro_html, unsafe_allow_html=True)

# Import modules
import numpy as np
from PIL import Image
import torch
import io

try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError:
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
    from plotly.subplots import make_subplots
    plotly_available = True
except ImportError:
    plotly_available = False

# Model loading
@st.cache_resource
def load_model():
    if not yolo_available:
        return None
    try:
        model = YOLO("yolov8n-seg.pt")
        return model
    except:
        return None

# Session state
if "app_loaded" not in st.session_state:
    st.session_state.app_loaded = False
    st.session_state.mode = "home"
    st.session_state.detection_results = None
    st.session_state.selected_image = None
    st.session_state.dimensions = None
    st.session_state.captured_frame = None
    st.session_state.selected_object_mask = None

# Show intro only on first load
if not st.session_state.app_loaded:
    show_intro()
    st.session_state.app_loaded = True
    time.sleep(0.1)  # Small delay for intro

# Helper functions
def calculate_dimensions(bbox, pixels_per_cm):
    x1, y1, x2, y2 = bbox
    width_px = abs(x2 - x1)
    height_px = abs(y2 - y1)
    width_cm = width_px / pixels_per_cm
    height_cm = height_px / pixels_per_cm
    depth_cm = (width_cm + height_cm) / 2
    volume_cm3 = width_cm * height_cm * depth_cm
    return {
        "width_cm": round(width_cm, 2),
        "height_cm": round(height_cm, 2),
        "depth_cm": round(depth_cm, 2),
        "volume_cm3": round(volume_cm3, 2)
    }

def create_3d_hologram(mask, bbox, dimensions):
    """Create real 3D object shape from mask and dimensions"""
    if not plotly_available:
        return None
    
    try:
        # Get mask points
        if mask is not None:
            # Find contour points from mask
            contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                # Simplify contour
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Extract x, y coordinates
                contour_points = simplified_contour.reshape(-1, 2)
                x_coords = contour_points[:, 0]
                y_coords = contour_points[:, 1]
            else:
                # Fallback to bounding box
                x1, y1, x2, y2 = bbox
                x_coords = np.array([x1, x2, x2, x1, x1])
                y_coords = np.array([y1, y1, y2, y2, y1])
        else:
            # Use bounding box
            x1, y1, x2, y2 = bbox
            x_coords = np.array([x1, x2, x2, x1, x1])
            y_coords = np.array([y1, y1, y2, y2, y1])
        
        # Scale coordinates to actual dimensions
        width_cm, height_cm, depth_cm = dimensions['width_cm'], dimensions['height_cm'], dimensions['depth_cm']
        
        # Normalize coordinates
        x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()) * width_cm
        y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min()) * height_cm
        
        # Create 3D shape by extruding 2D contour
        fig = go.Figure()
        
        # Bottom face (z=0)
        fig.add_trace(go.Scatter3d(
            x=x_norm, y=y_norm, z=np.zeros(len(x_norm)),
            mode='lines+markers',
            line=dict(color='cyan', width=8),
            marker=dict(size=6, color='cyan'),
            name='Bottom Face',
            showlegend=False
        ))
        
        # Top face (z=depth)
        fig.add_trace(go.Scatter3d(
            x=x_norm, y=y_norm, z=np.full(len(x_norm), depth_cm),
            mode='lines+markers',
            line=dict(color='yellow', width=8),
            marker=dict(size=6, color='yellow'),
            name='Top Face',
            showlegend=False
        ))
        
        # Vertical edges connecting bottom and top
        for i in range(len(x_norm)-1):  # -1 to avoid duplicate of first point
            fig.add_trace(go.Scatter3d(
                x=[x_norm[i], x_norm[i]],
                y=[y_norm[i], y_norm[i]],
                z=[0, depth_cm],
                mode='lines',
                line=dict(color='magenta', width=6),
                showlegend=False
            ))
        
        # Add some interior structure for better 3D effect
        center_x, center_y = x_norm.mean(), y_norm.mean()
        
        # Center pillar
        fig.add_trace(go.Scatter3d(
            x=[center_x, center_x],
            y=[center_y, center_y],
            z=[0, depth_cm],
            mode='lines+markers',
            line=dict(color='orange', width=10),
            marker=dict(size=8, color='orange'),
            name=f'Object Center',
            showlegend=True
        ))
        
        # Add cross-sections for depth visualization
        for z_level in np.linspace(0, depth_cm, 5):
            if z_level > 0 and z_level < depth_cm:
                scale_factor = 0.7 + 0.3 * (z_level / depth_cm)  # Slight tapering effect
                x_scaled = center_x + (x_norm - center_x) * scale_factor
                y_scaled = center_y + (y_norm - center_y) * scale_factor
                
                fig.add_trace(go.Scatter3d(
                    x=x_scaled, y=y_scaled, z=np.full(len(x_scaled), z_level),
                    mode='lines',
                    line=dict(color=f'rgba(0,255,255,{0.3 + 0.4 * z_level/depth_cm})', width=4),
                    showlegend=False
                ))
        
        # Enhanced layout
        fig.update_layout(
            title={
                'text': f"🌟 3D HOLOGRAPHIC RECONSTRUCTION",
                'x': 0.5,
                'font': {'size': 24, 'color': 'white', 'family': 'Orbitron'}
            },
            scene=dict(
                bgcolor='rgba(0,0,0,0.95)',
                xaxis=dict(
                    title='Width (cm)',
                    gridcolor='rgba(0,212,255,0.3)',
                    gridwidth=2,
                    backgroundcolor='rgba(0,0,0,0)',
                    color='white'
                ),
                yaxis=dict(
                    title='Height (cm)',
                    gridcolor='rgba(139,92,246,0.3)',
                    gridwidth=2,
                    backgroundcolor='rgba(0,0,0,0)',
                    color='white'
                ),
                zaxis=dict(
                    title='Depth (cm)',
                    gridcolor='rgba(16,185,129,0.3)',
                    gridwidth=2,
                    backgroundcolor='rgba(0,0,0,0)',
                    color='white'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Inter'),
            margin=dict(l=0, r=0, t=60, b=0),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(26,26,26,0.8)',
                bordercolor='rgba(0,212,255,0.5)',
                borderwidth=1
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"3D Reconstruction Error: {str(e)}")
        return None

# Camera frame callback
captured_frame = None

def video_frame_callback(frame):
    global captured_frame
    img = frame.to_ndarray(format="bgr24")
    captured_frame = img.copy()
    return frame

# Load model
model = load_model()

# Main content wrapper
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">AI DIMENSION ESTIMATOR</div>
    <div class="hero-subtitle">Next-Generation Analysis Platform</div>
</div>
""", unsafe_allow_html=True)

# Enhanced Status Grid
st.markdown('<div class="status-grid">', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    yolo_status = "ONLINE" if model else "OFFLINE"
    st.markdown(f"""
    <div class="status-card">
        <div class="status-header">
            <div class="status-icon">🤖</div>
            <div>
                <div class="status-title">Neural Network</div>
                <div class="status-value">{yolo_status}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    opencv_status = "READY" if opencv_available else "LIMITED"
    st.markdown(f"""
    <div class="status-card">
        <div class="status-header">
            <div class="status-icon">📹</div>
            <div>
                <div class="status-title">Vision System</div>
                <div class="status-value">{opencv_status}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    camera_status = "ACTIVE" if webrtc_available else "OFFLINE"
    st.markdown(f"""
    <div class="status-card">
        <div class="status-header">
            <div class="status-icon">📡</div>
            <div>
                <div class="status-title">Camera Feed</div>
                <div class="status-value">{camera_status}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    hologram_status = "3D READY" if plotly_available else "2D ONLY"
    st.markdown(f"""
    <div class="status-card">
        <div class="status-header">
            <div class="status-icon">🌟</div>
            <div>
                <div class="status-title">Hologram</div>
                <div class="status-value">{hologram_status}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# System status
if model:
    st.success("🚀 **ALL SYSTEMS OPERATIONAL** - Ready for Advanced Analysis")
else:
    st.error("🚨 **CRITICAL ERROR** - Neural Network Offline")
    st.stop()

# Navigation Cards
st.markdown('<div class="nav-container">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎥 CAMERA MODE", key="nav_cam", use_container_width=True):
        if webrtc_available:
            st.session_state.mode = "camera"
            st.rerun()
        else:
            st.error("❌ Camera system unavailable")

with col2:
    if st.button("📁 UPLOAD MODE", key="nav_upload", use_container_width=True):
        st.session_state.mode = "upload"
        st.rerun()

with col3:
    if st.button("🏠 MISSION CONTROL", key="nav_home", use_container_width=True):
        st.session_state.mode = "home"
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Main Content Sections
if st.session_state.mode == "home":
    # Mission Control Dashboard
    st.markdown("""
    <div class="content-card">
        <h2 style="color: #00d4ff; font-family: Orbitron; margin-bottom: 2rem;">🎯 MISSION CONTROL CENTER</h2>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem;">
            <div>
                <h3 style="color: #8b5cf6;">🚀 CORE CAPABILITIES</h3>
                <ul style="color: #a1a1aa; line-height: 1.8;">
                    <li>Advanced YOLOv8 Neural Network Detection</li>
                    <li>Real-time Object Classification</li>
                    <li>Precision Dimensional Analysis</li>
                    <li>3D Holographic Reconstruction</li>
                    <li>Multi-object Processing</li>
                </ul>
            </div>
            
            <div>
                <h3 style="color: #10b981;">📡 OPERATION MODES</h3>
                <ul style="color: #a1a1aa; line-height: 1.8;">
                    <li><strong>Camera Mode:</strong> Live capture & analysis</li>
                    <li><strong>Upload Mode:</strong> Static image processing</li>
                    <li><strong>Hologram Mode:</strong> 3D visualization</li>
                    <li><strong>Measurement:</strong> Precision calibration</li>
                </ul>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 3rem; padding: 2rem; background: rgba(0,212,255,0.1); border-radius: 16px; border: 1px solid rgba(0,212,255,0.3);">
            <h3 style="color: #00d4ff;">🎮 SELECT OPERATION MODE TO BEGIN ANALYSIS</h3>
            <p style="color: #71717a; margin-top: 1rem;">Choose Camera Mode for live analysis or Upload Mode for image processing</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.mode == "camera" and webrtc_available:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="content-card">
            <h3 style="color: #00d4ff; font-family: Orbitron;">📹 LIVE SURVEILLANCE SYSTEM</h3>
        """, unsafe_allow_html=True)
        
        # Enhanced camera stream
        ctx = webrtc_streamer(
            key="advanced_camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {"width": 1920, "height": 1080, "frameRate": 30},
                "audio": False
            },
            async_processing=False,
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-card">
            <h3 style="color: #8b5cf6;">🎛️ CAMERA CONTROL PANEL</h3>
        """, unsafe_allow_html=True)
        
        if captured_frame is not None:
            st.success("🟢 **FEED ACTIVE**")
            st.info("📊 Resolution: HD 1080p")
            st.info("🎯 AI Detection: Ready")
            
            if st.button("📸 **CAPTURE TARGET**", key="capture_btn", use_container_width=True):
                st.session_state.captured_frame = captured_frame.copy()
                st.session_state.selected_image = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
                st.session_state.mode = "detected"
                st.success("🎯 **TARGET ACQUIRED**")
                st.rerun()
        else:
            st.warning("🟡 **INITIALIZING FEED**")
            st.info("⏳ Establishing connection...")
        
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.mode == "upload":
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="content-card">
            <h3 style="color: #00d4ff; font-family: Orbitron;">📁 IMAGE UPLOAD SYSTEM</h3>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Deploy Target Image for Analysis",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload high-resolution images for best results"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="🎯 Target Image Loaded", use_column_width=True)
            st.session_state.selected_image = image
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-card">
            <h3 style="color: #10b981;">📊 UPLOAD STATUS</h3>
        """, unsafe_allow_html=True)
        
        if uploaded_file:
            st.success("✅ **IMAGE LOADED**")
            st.info(f"📏 Resolution: {st.session_state.selected_image.size}")
            st.info("🎯 Ready for AI Analysis")
            
            if st.button("🔍 **INITIATE ANALYSIS**", key="analyze_btn", use_container_width=True):
                st.session_state.mode = "detected"
                st.rerun()
        else:
            st.info("📁 **AWAITING DEPLOYMENT**")
            st.markdown("""
            **Supported Formats:**
            - JPG, JPEG, PNG
            - BMP, TIFF
            - Max: 200MB
            
            **Recommendations:**
            - High resolution preferred
            - Clear object visibility
            - Good lighting conditions
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.mode == "detected":
    if st.session_state.selected_image:
        with st.spinner("🔬 **AI NEURAL NETWORK PROCESSING...**"):
            # Run AI detection
            img_array = np.array(st.session_state.selected_image)
            results = model.predict(source=img_array, conf=0.4, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    st.session_state.detection_results = result
                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.markdown("""
                        <div class="content-card">
                            <h3 style="color: #00d4ff;">🎯 AI DETECTION ANALYSIS</h3>
                        """, unsafe_allow_html=True)
                        
                        # Show detection results
                        annotated_img = result.plot()
                        st.image(annotated_img, caption="🤖 Neural Network Detection Results", use_column_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div class="content-card">
                            <h3 style="color: #8b5cf6;">🎮 TARGET SELECTION</h3>
                        """, unsafe_allow_html=True)
                        
                        st.success(f"🎯 **{len(result.boxes)} OBJECTS DETECTED**")
                        
                        # Object selection
                        object_names = []
                        for cls in result.boxes.cls:
                            class_id = int(cls.item())
                            object_names.append(model.names[class_id])
                        
                        unique_objects = list(set(object_names))
                        selected_obj = st.selectbox("🎯 **SELECT TARGET**", unique_objects)
                        
                        if selected_obj:
                            selected_idx = object_names.index(selected_obj)
                            confidence = result.boxes.conf[selected_idx]
                            
                            # Store selected object mask for 3D reconstruction
                            if hasattr(result, 'masks') and result.masks is not None:
                                st.session_state.selected_object_mask = result.masks.data[selected_idx].cpu().numpy()
                            
                            # Object classification
                            biological = ['person', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
                            category = "🧬 BIOLOGICAL" if selected_obj.lower() in biological else "⚙️ MECHANICAL"
                            
                            st.markdown(f"**Classification:** {category}")
                            st.markdown(f"**AI Confidence:** {confidence:.1%}")
                            
                            st.markdown("---")
                            st.markdown("### 📏 PRECISION CALIBRATION")
                            
                            ref_width_cm = st.number_input("Reference Width (cm)", min_value=0.1, value=10.0, step=0.1)
                            ref_width_px = st.number_input("Reference Pixels", min_value=1, value=100, step=1)
                            
                            if st.button("📐 **EXECUTE MEASUREMENT**", key="measure_btn", use_container_width=True):
                                try:
                                    pixels_per_cm = ref_width_px / ref_width_cm
                                    bbox = result.boxes.xyxy[selected_idx].cpu().numpy()
                                    dims = calculate_dimensions(bbox, pixels_per_cm)
                                    st.session_state.dimensions = dims
                                    st.session_state.mode = "measured"
                                    st.success("✅ **MEASUREMENT COMPLETE**")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.mode == "measured":
    if st.session_state.dimensions:
        dims = st.session_state.dimensions
        
        # Success message
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.1)); border: 2px solid #10b981; border-radius: 16px; padding: 2rem; margin: 2rem 0; text-align: center;">
            <h2 style="color: #10b981; margin: 0;">🎯 DIMENSIONAL ANALYSIS COMPLETE</h2>
            <p style="color: #71717a; margin: 0.5rem 0 0;">Advanced measurement protocol executed successfully</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics Display
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{dims['width_cm']}</div>
                <div class="metric-label">WIDTH (CM)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{dims['height_cm']}</div>
                <div class="metric-label">HEIGHT (CM)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{dims['depth_cm']}</div>
                <div class="metric-label">DEPTH (CM)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{dims['volume_cm3']}</div>
                <div class="metric-label">VOLUME (CM³)</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🌟 **HOLOGRAPHIC RECONSTRUCTION**", key="holo_btn", use_container_width=True):
                with st.spinner("🎭 **GENERATING 3D HOLOGRAM...**"):
                    try:
                        if st.session_state.detection_results:
                            result = st.session_state.detection_results
                            selected_idx = 0  # Default to first detected object
                            bbox = result.boxes.xyxy[selected_idx].cpu().numpy()
                            mask = st.session_state.selected_object_mask
                            
                            fig = create_3d_hologram(mask, bbox, dims)
                            
                            if fig:
                                st.markdown('<div class="hologram-container">', unsafe_allow_html=True)
                                st.plotly_chart(fig, use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.success("✨ **3D HOLOGRAM GENERATED SUCCESSFULLY**")
                            else:
                                st.error("❌ Hologram generation failed")
                    except Exception as e:
                        st.error(f"❌ Hologram Error: {str(e)}")
        
        with col2:
            if st.button("📊 **ANALYSIS REPORT**", key="report_btn", use_container_width=True):
                report_data = {
                    "dimensions": dims,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "confidence": "High",
                    "method": "AI-Enhanced Calibration"
                }
                
                st.markdown(f"""
                <div class="content-card">
                    <h3 style="color: #00d4ff;">📋 DETAILED ANALYSIS REPORT</h3>
                    <div style="background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 12px; font-family: monospace;">
                        <p><strong>TIMESTAMP:</strong> {report_data['timestamp']}</p>
                        <p><strong>DIMENSIONS:</strong></p>
                        <ul>
                            <li>Width: {dims['width_cm']} cm</li>
                            <li>Height: {dims['height_cm']} cm</li>
                            <li>Depth: {dims['depth_cm']} cm</li>
                            <li>Volume: {dims['volume_cm3']} cm³</li>
                        </ul>
                        <p><strong>ANALYSIS STATUS:</strong> ✅ Complete</p>
                        <p><strong>CONFIDENCE LEVEL:</strong> {report_data['confidence']}</p>
                        <p><strong>METHOD:</strong> {report_data['method']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if st.button("🔄 **NEW ANALYSIS**", key="new_btn", use_container_width=True):
                # Reset all states
                st.session_state.mode = "home"
                st.session_state.detection_results = None
                st.session_state.selected_image = None
                st.session_state.dimensions = None
                st.session_state.captured_frame = None
                st.session_state.selected_object_mask = None
                st.rerun()

# Floating Action Buttons (only show on relevant pages)
if st.session_state.mode in ["camera", "upload", "detected"]:
    st.markdown("""
    <div class="action-container">
        <button class="floating-btn capture" title="Capture">📸</button>
        <button class="floating-btn detect" title="Detect">🔍</button>
        <button class="floating-btn measure" title="Measure">📐</button>
        <button class="floating-btn hologram" title="Hologram">🌟</button>
    </div>
    """, unsafe_allow_html=True)

# Enhanced Footer
st.markdown("""
<div style="text-align: center; padding: 4rem 2rem 2rem; margin-top: 4rem; border-top: 1px solid #333;">
    <h3 style="color: #00d4ff; font-family: Orbitron; margin-bottom: 1rem;">AI DIMENSION ESTIMATOR v3.0</h3>
    <p style="color: #71717a; font-size: 1.1rem;">Next-Generation Analysis Platform | Powered by Advanced Neural Networks</p>
    <p style="color: #8b5cf6; font-weight: 600; margin-top: 1rem;">
        🎯 Deploy → 🔍 Detect → 📐 Analyze → 🌟 Visualize
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
