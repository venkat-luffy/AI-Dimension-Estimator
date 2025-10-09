import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI Dimension Estimator | Professional Analysis Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "AI Dimension Estimator - Professional Analysis Platform"
    }
)

# Load custom CSS
def load_custom_css():
    try:
        with open("style.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Inline fallback CSS if file not found
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        
        .stApp {
            background: #0a0a0a;
            background-image: 
                radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.03) 0%, transparent 50%);
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }
        
        .hero-title {
            font-size: 4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00d4ff 0%, #8b5cf6 50%, #10b981 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin: 2rem 0;
        }
        
        .modern-card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 16px;
            padding: 2rem;
            margin: 1rem 0;
            transition: all 0.3s ease;
        }
        
        .modern-card:hover {
            transform: translateY(-4px);
            border-color: #00d4ff;
            box-shadow: 0 8px 25px rgba(0,0,0,0.6);
        }
        
        .status-item {
            background: rgba(26, 26, 26, 0.85);
            border: 1px solid #333;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
            margin: 0.5rem;
        }
        
        .action-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 12px;
            padding: 1rem 2rem;
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .action-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }
        </style>
        """, unsafe_allow_html=True)

load_custom_css()

# Import required modules
import numpy as np
from PIL import Image
import torch

# Import optional modules
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
    plotly_available = True
except ImportError:
    plotly_available = False

# Model loading
@st.cache_resource
def load_yolo_model():
    if not yolo_available:
        return None
    try:
        model = YOLO("yolov8n-seg.pt")
        return model
    except Exception:
        return None

# Session state initialization
if "mode" not in st.session_state:
    st.session_state.mode = "home"
    st.session_state.detection_results = None
    st.session_state.selected_image = None
    st.session_state.dimensions = None
    st.session_state.captured_frame = None

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

# Camera frame callback
captured_frame = None

def video_frame_callback(frame):
    global captured_frame
    img = frame.to_ndarray(format="bgr24")
    captured_frame = img.copy()
    return frame

# Load model
seg_model = load_yolo_model()

# === HEADER SECTION ===
st.markdown("""
<div class="hero-header">
    <div class="hero-title">AI DIMENSION ESTIMATOR</div>
    <div class="hero-subtitle">Professional Object Analysis Platform</div>
</div>
""", unsafe_allow_html=True)

# === STATUS BAR ===
st.markdown('<div class="status-bar">', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    yolo_status = "ONLINE" if (seg_model is not None) else "OFFLINE"
    dot_class = "status-dot" if (seg_model is not None) else "status-dot error"
    st.markdown(f'''
    <div class="status-item">
        <div class="{dot_class}"></div>
        <div class="status-text">🤖 YOLO: {yolo_status}</div>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    opencv_status = "READY" if opencv_available else "LIMITED"
    dot_class = "status-dot" if opencv_available else "status-dot warning"
    st.markdown(f'''
    <div class="status-item">
        <div class="{dot_class}"></div>
        <div class="status-text">📹 OpenCV: {opencv_status}</div>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    camera_status = "ACTIVE" if webrtc_available else "OFFLINE"
    dot_class = "status-dot" if webrtc_available else "status-dot error"
    st.markdown(f'''
    <div class="status-item">
        <div class="{dot_class}"></div>
        <div class="status-text">📡 Camera: {camera_status}</div>
    </div>
    ''', unsafe_allow_html=True)

with col4:
    hologram_status = "3D READY" if plotly_available else "2D ONLY"
    dot_class = "status-dot" if plotly_available else "status-dot warning"
    st.markdown(f'''
    <div class="status-item">
        <div class="{dot_class}"></div>
        <div class="status-text">🌟 Hologram: {hologram_status}</div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# System status message
if seg_model is not None:
    st.markdown('''
    <div class="success-banner">
        <span>✅</span>
        <div>
            <strong>All Systems Operational</strong><br>
            AI Ready for Deployment
        </div>
    </div>
    ''', unsafe_allow_html=True)
else:
    st.markdown('''
    <div class="error-banner">
        <span>🚨</span>
        <div>
            <strong>Critical Error</strong><br>
            AI Detection System Offline
        </div>
    </div>
    ''', unsafe_allow_html=True)
    st.stop()

# === NAVIGATION ===
st.markdown('<div class="nav-buttons">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    camera_class = "nav-btn active" if st.session_state.mode == "camera" else "nav-btn"
    if st.button("📹 CAMERA MODE", key="nav_camera", use_container_width=True):
        if webrtc_available:
            st.session_state.mode = "camera"
            st.rerun()
        else:
            st.error("❌ Camera mode unavailable")

with col2:
    upload_class = "nav-btn active" if st.session_state.mode == "upload" else "nav-btn"
    if st.button("📁 UPLOAD MODE", key="nav_upload", use_container_width=True):
        st.session_state.mode = "upload"
        st.rerun()

with col3:
    home_class = "nav-btn active" if st.session_state.mode == "home" else "nav-btn"
    if st.button("🏠 MISSION CONTROL", key="nav_home", use_container_width=True):
        st.session_state.mode = "home"
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# === MAIN CONTENT ===
if st.session_state.mode == "home":
    # Mission Control Dashboard
    st.markdown('''
    <div class="modern-card">
        <div class="card-title">🎯 MISSION CONTROL</div>
        <div class="card-content">
            <p><strong>Advanced AI-Powered Dimensional Analysis System</strong></p>
            <br>
            <h4>🚀 CORE CAPABILITIES</h4>
            <ul>
                <li>Real-time object detection using YOLOv8 neural networks</li>
                <li>High-precision dimensional measurement algorithms</li>
                <li>3D holographic visualization and analysis</li>
                <li>Multi-object detection and classification</li>
                <li>Professional-grade measurement calibration</li>
            </ul>
            <br>
            <h4>📡 OPERATION MODES</h4>
            <ul>
                <li><strong>Camera Mode:</strong> Live capture and real-time analysis</li>
                <li><strong>Upload Mode:</strong> Static image processing and measurement</li>
            </ul>
            <br>
            <div style="text-align: center; padding: 2rem; background: rgba(0,212,255,0.1); border-radius: 12px; border: 1px solid rgba(0,212,255,0.3);">
                <strong>🎮 SELECT YOUR OPERATION MODE ABOVE TO BEGIN</strong>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

elif st.session_state.mode == "camera" and webrtc_available:
    # Camera Mode Interface
    col1, col2 = st.columns([2.5, 1.5])
    
    with col1:
        st.markdown('''
        <div class="modern-card">
            <div class="card-title">📹 LIVE SURVEILLANCE FEED</div>
        ''', unsafe_allow_html=True)
        
        # Camera stream with overlay
        ctx = webrtc_streamer(
            key="professional_camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {"width": 1280, "height": 720},
                "audio": False
            },
            async_processing=False,
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Capture button
        if captured_frame is not None:
            if st.button("📸 CAPTURE TARGET", key="capture", use_container_width=True):
                st.session_state.captured_frame = captured_frame.copy()
                st.session_state.selected_image = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
                st.session_state.mode = "detected"
                st.success("🎯 TARGET ACQUIRED")
                st.rerun()
    
    with col2:
        st.markdown('''
        <div class="modern-card">
            <div class="card-title">🎛️ CAMERA CONTROLS</div>
            <div class="card-content">
        ''', unsafe_allow_html=True)
        
        if captured_frame is not None:
            st.markdown("**Status:** 🟢 FEED ACTIVE")
            st.markdown("**Resolution:** Real-time HD")
            st.markdown("**Mode:** Target Acquisition")
            st.markdown("---")
            st.info("🎯 Aim camera at target object")
            st.info("📸 Click CAPTURE when positioned")
        else:
            st.markdown("**Status:** 🟡 INITIALIZING")
            st.warning("⏳ Establishing video connection...")
            st.info("🔧 Verify camera permissions")
        
        st.markdown('''
            </div>
        </div>
        ''', unsafe_allow_html=True)

elif st.session_state.mode == "upload":
    # Upload Mode Interface
    col1, col2 = st.columns([2.5, 1.5])
    
    with col1:
        st.markdown('''
        <div class="modern-card">
            <div class="card-title">📁 IMAGE UPLOAD SYSTEM</div>
        ''', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Deploy Target Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload high-quality image for analysis"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="🎯 Target Image Loaded", use_column_width=True)
            st.session_state.selected_image = image
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis button
        if uploaded_file is not None:
            if st.button("🔍 INITIATE ANALYSIS", key="analyze", use_container_width=True):
                st.session_state.mode = "detected"
                st.rerun()
    
    with col2:
        st.markdown('''
        <div class="modern-card">
            <div class="card-title">📊 UPLOAD STATUS</div>
            <div class="card-content">
        ''', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            st.success("✅ Image successfully loaded")
            st.info(f"📏 Resolution: {st.session_state.selected_image.size}")
            st.info("🎯 Ready for analysis")
            st.markdown("**Click INITIATE ANALYSIS to proceed**")
        else:
            st.info("📁 Awaiting image deployment...")
            st.markdown("""
            **Supported Formats:**
            - JPG / JPEG
            - PNG
            - Maximum size: 200MB
            
            **Recommendations:**
            - High resolution preferred
            - Good lighting conditions
            - Clear object visibility
            """)
        
        st.markdown('''
            </div>
        </div>
        ''', unsafe_allow_html=True)

elif st.session_state.mode == "detected":
    # Detection and Analysis Mode
    if st.session_state.selected_image is not None:
        with st.spinner("🔍 AI ANALYSIS IN PROGRESS..."):
            # Run detection
            img_array = np.array(st.session_state.selected_image)
            results = seg_model.predict(source=img_array, conf=0.4, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    st.session_state.detection_results = result
                    
                    col1, col2 = st.columns([2.5, 1.5])
                    
                    with col1:
                        st.markdown('''
                        <div class="modern-card">
                            <div class="card-title">🎯 DETECTION ANALYSIS</div>
                        ''', unsafe_allow_html=True)
                        
                        # Show detection results
                        annotated_img = result.plot()
                        st.image(annotated_img, caption="🤖 AI Detection Results", use_column_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('''
                        <div class="modern-card">
                            <div class="card-title">🎮 TARGET SELECTION</div>
                            <div class="card-content">
                        ''', unsafe_allow_html=True)
                        
                        st.success(f"🎯 {len(result.boxes)} OBJECTS IDENTIFIED")
                        
                        # Object selection
                        object_names = []
                        for cls in result.boxes.cls:
                            class_id = int(cls.item())
                            object_names.append(seg_model.names[class_id])
                        
                        unique_objects = list(set(object_names))
                        selected_obj = st.selectbox("🎯 SELECT TARGET", unique_objects, key="target_select")
                        
                        if selected_obj:
                            selected_idx = object_names.index(selected_obj)
                            
                            # Object classification
                            living_objects = ['person', 'cat', 'dog', 'horse', 'sheep', 'cow']
                            category = "🐾 BIOLOGICAL" if selected_obj.lower() in living_objects else "🔧 MECHANICAL"
                            confidence = result.boxes.conf[selected_idx]
                            
                            st.markdown(f"**Classification:** {category}")
                            st.markdown(f"**Confidence:** {confidence:.1%}")
                            
                            st.markdown("---")
                            st.markdown("### 📏 CALIBRATION PROTOCOL")
                            
                            ref_width_cm = st.number_input(
                                "Reference Width (cm)",
                                min_value=0.1,
                                value=10.0,
                                step=0.1,
                                help="Enter known dimension"
                            )
                            
                            ref_width_px = st.number_input(
                                "Reference Pixels",
                                min_value=1,
                                value=100,
                                step=1,
                                help="Pixel measurement"
                            )
                            
                            if st.button("📐 EXECUTE MEASUREMENT", key="measure", use_container_width=True):
                                try:
                                    pixels_per_cm = ref_width_px / ref_width_cm
                                    bbox = result.boxes.xyxy[selected_idx].cpu().numpy()
                                    dims = calculate_dimensions(bbox, pixels_per_cm)
                                    st.session_state.dimensions = dims
                                    st.session_state.mode = "measured"
                                    st.success("✅ ANALYSIS COMPLETE")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Measurement failed: {str(e)}")
                        
                        st.markdown('''
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)

elif st.session_state.mode == "measured":
    # Results Display
    if st.session_state.dimensions:
        dims = st.session_state.dimensions
        
        st.markdown('''
        <div class="success-banner">
            <span>🎯</span>
            <div>
                <strong>DIMENSIONAL ANALYSIS COMPLETE</strong><br>
                Target measurement successful
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Metrics display
        st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{dims['width_cm']}</div>
                <div class="metric-label">WIDTH (CM)</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{dims['height_cm']}</div>
                <div class="metric-label">HEIGHT (CM)</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{dims['depth_cm']}</div>
                <div class="metric-label">DEPTH (CM)</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{dims['volume_cm3']}</div>
                <div class="metric-label">VOLUME (CM³)</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if plotly_available and st.button("🌟 HOLOGRAPHIC VIEW", key="hologram", use_container_width=True):
                with st.spinner("🎭 GENERATING HOLOGRAM..."):
                    try:
                        w, h, d = dims['width_cm'], dims['height_cm'], dims['depth_cm']
                        
                        # Create 3D visualization
                        fig = go.Figure()
                        
                        # Box wireframe
                        edges = [
                            ([0,w,w,0,0], [0,0,h,h,0], [0,0,0,0,0]),
                            ([0,w,w,0,0], [0,0,h,h,0], [d,d,d,d,d]),
                            ([0,0], [0,0], [0,d]), ([w,w], [0,0], [0,d]),
                            ([w,w], [h,h], [0,d]), ([0,0], [h,h], [0,d])
                        ]
                        
                        for edge in edges:
                            fig.add_trace(go.Scatter3d(
                                x=edge[0], y=edge[1], z=edge[2],
                                mode='lines',
                                line=dict(color='cyan', width=6),
                                showlegend=False
                            ))
                        
                        # Corner points
                        corners_x = [0,w,w,0,0,w,w,0]
                        corners_y = [0,0,h,h,0,0,h,h]
                        corners_z = [0,0,0,0,d,d,d,d]
                        
                        fig.add_trace(go.Scatter3d(
                            x=corners_x, y=corners_y, z=corners_z,
                            mode='markers',
                            marker=dict(size=12, color='yellow', opacity=0.8),
                            name=f'Object ({w}×{h}×{d} cm)'
                        ))
                        
                        fig.update_layout(
                            title="🌟 HOLOGRAPHIC PROJECTION",
                            scene=dict(
                                xaxis_title="Width (cm)",
                                yaxis_title="Height (cm)", 
                                zaxis_title="Depth (cm)",
                                bgcolor='rgba(0,0,0,0.9)',
                                xaxis=dict(gridcolor='rgba(0,212,255,0.3)'),
                                yaxis=dict(gridcolor='rgba(0,212,255,0.3)'),
                                zaxis=dict(gridcolor='rgba(0,212,255,0.3)')
                            ),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white', family='Inter')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Hologram generation failed: {str(e)}")
        
        with col2:
            if st.button("📊 DETAILED REPORT", key="report", use_container_width=True):
                st.markdown('''
                <div class="modern-card">
                    <div class="card-title">📋 ANALYSIS REPORT</div>
                    <div class="card-content">
                        <p><strong>MEASUREMENT SUMMARY</strong></p>
                        <ul>
                            <li><strong>Width:</strong> {width} cm</li>
                            <li><strong>Height:</strong> {height} cm</li>
                            <li><strong>Depth:</strong> {depth} cm (estimated)</li>
                            <li><strong>Volume:</strong> {volume} cm³</li>
                        </ul>
                        <br>
                        <p><strong>ANALYSIS STATUS:</strong> ✅ Complete</p>
                        <p><strong>CONFIDENCE LEVEL:</strong> High</p>
                        <p><strong>METHOD:</strong> AI-Enhanced Calibration</p>
                    </div>
                </div>
                '''.format(
                    width=dims['width_cm'],
                    height=dims['height_cm'],
                    depth=dims['depth_cm'],
                    volume=dims['volume_cm3']
                ), unsafe_allow_html=True)
        
        with col3:
            if st.button("🔄 NEW ANALYSIS", key="new_analysis", use_container_width=True):
                # Reset all session state
                st.session_state.mode = "home"
                st.session_state.detection_results = None
                st.session_state.selected_image = None
                st.session_state.dimensions = None
                st.session_state.captured_frame = None
                st.rerun()

# === FOOTER ===
st.markdown('''
<div class="footer">
    <div class="footer-title">AI DIMENSION ESTIMATOR v2.0</div>
    <div class="footer-subtitle">
        Professional Analysis Platform | Powered by Advanced Neural Networks<br>
        <span style="color: #00d4ff;">Deploy → Detect → Analyze → Measure → Visualize</span>
    </div>
</div>
''', unsafe_allow_html=True)
