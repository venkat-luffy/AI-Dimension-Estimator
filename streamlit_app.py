import streamlit as st
import time

# Page configuration
st.set_page_config(
    page_title="AI Dimension Estimator | Next-Gen Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simplified CSS - Fixed Content Display
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Orbitron:wght@400;700;900&display=swap');
    
    :root {
        --primary-bg: #0a0a0a;
        --card-bg: #1a1a1a;
        --glass-bg: rgba(22, 22, 22, 0.85);
        --border-primary: #333333;
        --neon-blue: #00d4ff;
        --neon-purple: #8b5cf6;
        --neon-green: #10b981;
        --text-primary: #ffffff;
        --text-secondary: #a1a1aa;
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-neon: linear-gradient(45deg, #00d4ff, #8b5cf6, #10b981);
    }
    
    .stApp {
        background: var(--primary-bg);
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(0, 212, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.08) 0%, transparent 50%);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    /* Intro Animation - Simplified */
    .intro-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: var(--primary-bg);
        z-index: 9999;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        opacity: 1;
        transition: opacity 1s ease-out;
    }
    
    .intro-overlay.fade-out {
        opacity: 0;
        pointer-events: none;
    }
    
    .intro-logo {
        font-size: 4rem;
        font-weight: 900;
        font-family: 'Orbitron', monospace;
        background: var(--gradient-neon);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: logoGlow 2s ease-in-out infinite alternate;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    @keyframes logoGlow {
        0% { filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.5)); }
        100% { filter: drop-shadow(0 0 40px rgba(139, 92, 246, 0.8)); }
    }
    
    .intro-subtitle {
        font-size: 1.2rem;
        color: var(--neon-blue);
        font-weight: 300;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .loading-bar {
        width: 200px;
        height: 4px;
        background: var(--border-primary);
        border-radius: 2px;
        overflow: hidden;
        position: relative;
    }
    
    .loading-progress {
        height: 100%;
        background: var(--gradient-neon);
        width: 0%;
        transition: width 2s ease-out;
        border-radius: 2px;
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 4rem 0 2rem;
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        border-radius: 0 0 20px 20px;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: clamp(2.5rem, 6vw, 4rem);
        font-weight: 900;
        font-family: 'Orbitron', monospace;
        background: var(--gradient-neon);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        animation: titlePulse 3s ease-in-out infinite;
    }
    
    @keyframes titlePulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: var(--text-secondary);
        font-weight: 300;
    }
    
    /* Status Cards */
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    .status-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-primary);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .status-card:hover {
        transform: translateY(-5px);
        border-color: var(--neon-blue);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
    }
    
    .status-header {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .status-icon {
        width: 40px;
        height: 40px;
        background: var(--gradient-primary);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }
    
    .status-title {
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .status-value {
        font-weight: 700;
        color: var(--neon-blue);
        font-family: 'Orbitron', monospace;
    }
    
    /* Navigation Buttons */
    .nav-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 2rem;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    .nav-btn {
        background: var(--card-bg);
        border: 1px solid var(--border-primary);
        border-radius: 16px;
        padding: 2rem;
        color: var(--text-primary);
        text-decoration: none;
        transition: all 0.4s ease;
        cursor: pointer;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .nav-btn:hover {
        transform: translateY(-8px);
        border-color: var(--neon-blue);
        box-shadow: 0 15px 35px rgba(0, 212, 255, 0.4);
    }
    
    .nav-btn::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.1), transparent);
        transition: left 0.6s ease;
    }
    
    .nav-btn:hover::before {
        left: 100%;
    }
    
    .nav-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .nav-title {
        font-size: 1.2rem;
        font-weight: 700;
        font-family: 'Orbitron', monospace;
        margin-bottom: 0.5rem;
    }
    
    .nav-description {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Content Cards */
    .content-card {
        background: var(--card-bg);
        border: 1px solid var(--border-primary);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .content-card:hover {
        transform: translateY(-3px);
        border-color: var(--neon-blue);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
    }
    
    /* Action Buttons */
    .action-btn {
        background: var(--gradient-primary);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        width: 100%;
    }
    
    .action-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
    }
    
    .action-btn.capture {
        background: linear-gradient(135deg, #f59e0b, #f97316);
    }
    
    .action-btn.detect {
        background: linear-gradient(135deg, #10b981, #059669);
    }
    
    .action-btn.measure {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
    }
    
    .action-btn.hologram {
        background: linear-gradient(135deg, #06b6d4, #0891b2);
    }
    
    /* Metrics */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: var(--glass-bg);
        border: 1px solid var(--border-primary);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: var(--gradient-neon);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: var(--neon-blue);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--neon-blue);
        font-family: 'Orbitron', monospace;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Success/Error Messages */
    .success-msg {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.1));
        border: 1px solid var(--neon-green);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: var(--neon-green);
    }
    
    .error-msg {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1));
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #ef4444;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .hero-title { font-size: 2.5rem; }
        .intro-logo { font-size: 2.5rem; }
        .status-grid, .nav-container { padding: 1rem; }
        .content-card { padding: 1.5rem; margin: 0.5rem; }
        .metrics-grid { grid-template-columns: repeat(2, 1fr); }
    }
    
    @media (max-width: 480px) {
        .metrics-grid { grid-template-columns: 1fr; }
        .nav-container { grid-template-columns: 1fr; }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Streamlit element fixes */
    .stSelectbox > div > div > div {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-primary) !important;
        color: var(--text-primary) !important;
    }
    
    .stNumberInput > div > div > div {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-primary) !important;
        color: var(--text-primary) !important;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# Import modules
import numpy as np
from PIL import Image
import torch

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
def load_model():
    if not yolo_available:
        return None
    try:
        model = YOLO("yolov8n-seg.pt")
        return model
    except:
        return None

# Session state initialization
if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = False
    st.session_state.mode = "home"
    st.session_state.detection_results = None
    st.session_state.selected_image = None
    st.session_state.dimensions = None
    st.session_state.captured_frame = None

# Show intro animation
if not st.session_state.intro_shown:
    intro_placeholder = st.empty()
    with intro_placeholder.container():
        st.markdown("""
        <div class="intro-overlay" id="intro-overlay">
            <div class="intro-logo">
                AI DIMENSION<br>ESTIMATOR
            </div>
            <div class="intro-subtitle">
                Next-Generation Analysis Platform
            </div>
            <div class="loading-bar">
                <div class="loading-progress" id="loading-progress"></div>
            </div>
        </div>
        
        <script>
        // Start loading animation
        setTimeout(function() {
            document.getElementById('loading-progress').style.width = '100%';
        }, 500);
        
        // Fade out intro
        setTimeout(function() {
            var intro = document.getElementById('intro-overlay');
            if (intro) {
                intro.classList.add('fade-out');
                setTimeout(function() {
                    intro.style.display = 'none';
                }, 1000);
            }
        }, 3000);
        </script>
        """, unsafe_allow_html=True)
    
    # Wait for intro to finish
    time.sleep(4)
    intro_placeholder.empty()
    st.session_state.intro_shown = True
    st.rerun()

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

def create_enhanced_3d_hologram(bbox, dimensions):
    """Create enhanced 3D hologram visualization"""
    if not plotly_available:
        return None
    
    try:
        w, h, d = dimensions['width_cm'], dimensions['height_cm'], dimensions['depth_cm']
        
        fig = go.Figure()
        
        # Create 3D box with proper edges
        # Define all vertices of the box
        vertices_x = [0, w, w, 0, 0, w, w, 0]
        vertices_y = [0, 0, h, h, 0, 0, h, h]
        vertices_z = [0, 0, 0, 0, d, d, d, d]
        
        # Define edges of the box
        edges = [
            # Bottom face edges
            [0, 1], [1, 2], [2, 3], [3, 0],
            # Top face edges  
            [4, 5], [5, 6], [6, 7], [7, 4],
            # Vertical edges
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        # Draw edges
        for edge in edges:
            start, end = edge
            fig.add_trace(go.Scatter3d(
                x=[vertices_x[start], vertices_x[end]],
                y=[vertices_y[start], vertices_y[end]],
                z=[vertices_z[start], vertices_z[end]],
                mode='lines',
                line=dict(color='cyan', width=8),
                showlegend=False
            ))
        
        # Add corner points
        fig.add_trace(go.Scatter3d(
            x=vertices_x,
            y=vertices_y,
            z=vertices_z,
            mode='markers',
            marker=dict(size=10, color='yellow', opacity=0.8),
            name=f'Object ({w}×{h}×{d} cm)',
            showlegend=True
        ))
        
        # Add center point for reference
        center_x, center_y, center_z = w/2, h/2, d/2
        fig.add_trace(go.Scatter3d(
            x=[center_x],
            y=[center_y], 
            z=[center_z],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Center Point',
            showlegend=True
        ))
        
        # Enhanced layout
        fig.update_layout(
            title={
                'text': "🌟 3D HOLOGRAPHIC VISUALIZATION",
                'x': 0.5,
                'font': {'size': 20, 'color': 'white'}
            },
            scene=dict(
                bgcolor='rgba(10,10,10,0.95)',
                xaxis=dict(
                    title='Width (cm)',
                    gridcolor='rgba(0,212,255,0.3)',
                    color='white',
                    backgroundcolor='rgba(0,0,0,0)'
                ),
                yaxis=dict(
                    title='Height (cm)',
                    gridcolor='rgba(139,92,246,0.3)',
                    color='white',
                    backgroundcolor='rgba(0,0,0,0)'
                ),
                zaxis=dict(
                    title='Depth (cm)',
                    gridcolor='rgba(16,185,129,0.3)',
                    color='white',
                    backgroundcolor='rgba(0,0,0,0)'
                )
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"3D Generation Error: {str(e)}")
        return None

# Camera callback
captured_frame = None

def video_frame_callback(frame):
    global captured_frame
    img = frame.to_ndarray(format="bgr24")
    captured_frame = img.copy()
    return frame

# Load model
model = load_model()

# === MAIN CONTENT ===
# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">AI DIMENSION ESTIMATOR</div>
    <div class="hero-subtitle">Next-Generation Analysis Platform</div>
</div>
""", unsafe_allow_html=True)

# Status Grid
st.markdown('<div class="status-grid">', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    yolo_status = "ONLINE" if model else "OFFLINE"
    st.markdown(f"""
    <div class="status-card">
        <div class="status-header">
            <div class="status-icon">🤖</div>
            <div>
                <div class="status-title">YOLO AI</div>
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
                <div class="status-title">OpenCV</div>
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
                <div class="status-title">Camera</div>
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

# System Status
if model:
    st.markdown('<div class="success-msg">🚀 <strong>ALL SYSTEMS OPERATIONAL</strong> - Ready for Analysis</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="error-msg">🚨 <strong>SYSTEM ERROR</strong> - AI Network Offline</div>', unsafe_allow_html=True)
    st.stop()

# Navigation
st.markdown('<div class="nav-container">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    camera_html = """
    <div class="nav-btn" onclick="this.style.backgroundColor='#1a1a2e'">
        <div class="nav-icon">📹</div>
        <div class="nav-title">CAMERA MODE</div>
        <div class="nav-description">Live video capture and real-time analysis</div>
    </div>
    """
    st.markdown(camera_html, unsafe_allow_html=True)
    if st.button("", key="camera_nav", help="Camera Mode"):
        if webrtc_available:
            st.session_state.mode = "camera"
            st.rerun()
        else:
            st.error("❌ Camera unavailable")

with col2:
    upload_html = """
    <div class="nav-btn" onclick="this.style.backgroundColor='#1a1a2e'">
        <div class="nav-icon">📁</div>
        <div class="nav-title">UPLOAD MODE</div>
        <div class="nav-description">Static image processing and analysis</div>
    </div>
    """
    st.markdown(upload_html, unsafe_allow_html=True)
    if st.button("", key="upload_nav", help="Upload Mode"):
        st.session_state.mode = "upload"
        st.rerun()

with col3:
    home_html = """
    <div class="nav-btn" onclick="this.style.backgroundColor='#1a1a2e'">
        <div class="nav-icon">🏠</div>
        <div class="nav-title">MISSION CONTROL</div>
        <div class="nav-description">System overview and status</div>
    </div>
    """
    st.markdown(home_html, unsafe_allow_html=True)
    if st.button("", key="home_nav", help="Home"):
        st.session_state.mode = "home"
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Main Content Based on Mode
if st.session_state.mode == "home":
    st.markdown("""
    <div class="content-card">
        <h2 style="color: #00d4ff; font-family: 'Orbitron'; margin-bottom: 2rem;">🎯 MISSION CONTROL CENTER</h2>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; margin-bottom: 2rem;">
            <div>
                <h3 style="color: #8b5cf6; margin-bottom: 1rem;">🚀 CORE CAPABILITIES</h3>
                <ul style="color: #a1a1aa; line-height: 1.8;">
                    <li>Advanced YOLOv8 Neural Network</li>
                    <li>Real-time Object Detection</li>
                    <li>Precision Measurement Analysis</li>
                    <li>3D Holographic Visualization</li>
                    <li>Multi-object Processing Support</li>
                </ul>
            </div>
            
            <div>
                <h3 style="color: #10b981; margin-bottom: 1rem;">📡 OPERATION MODES</h3>
                <ul style="color: #a1a1aa; line-height: 1.8;">
                    <li><strong>Camera Mode:</strong> Live capture & analysis</li>
                    <li><strong>Upload Mode:</strong> Static image processing</li>
                    <li><strong>Measurement:</strong> Precision calibration</li>
                    <li><strong>3D Hologram:</strong> Advanced visualization</li>
                </ul>
            </div>
        </div>
        
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(139,92,246,0.1)); border-radius: 16px; border: 1px solid rgba(0,212,255,0.3);">
            <h3 style="color: #00d4ff; margin-bottom: 1rem;">🎮 SELECT OPERATION MODE TO BEGIN</h3>
            <p style="color: #71717a;">Choose Camera Mode for live analysis or Upload Mode for image processing</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.mode == "camera" and webrtc_available:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""<div class="content-card"><h3 style="color: #00d4ff;">📹 LIVE CAMERA FEED</h3>""", unsafe_allow_html=True)
        
        ctx = webrtc_streamer(
            key="camera_stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""<div class="content-card"><h3 style="color: #8b5cf6;">🎛️ CONTROLS</h3>""", unsafe_allow_html=True)
        
        if captured_frame is not None:
            st.success("🟢 **CAMERA ACTIVE**")
            st.info("📊 Feed: Real-time HD")
            
            if st.button("📸 **CAPTURE TARGET**", key="capture_btn"):
                st.session_state.captured_frame = captured_frame.copy()
                st.session_state.selected_image = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
                st.session_state.mode = "detected"
                st.success("🎯 **TARGET ACQUIRED**")
                st.rerun()
        else:
            st.warning("🟡 **CONNECTING...**")
            st.info("⏳ Initializing camera feed")
        
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.mode == "upload":
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""<div class="content-card"><h3 style="color: #00d4ff;">📁 IMAGE UPLOAD</h3>""", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Deploy Target Image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="🎯 Target Loaded", use_column_width=True)
            st.session_state.selected_image = image
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""<div class="content-card"><h3 style="color: #10b981;">📊 STATUS</h3>""", unsafe_allow_html=True)
        
        if uploaded_file:
            st.success("✅ **IMAGE LOADED**")
            st.info(f"📏 Size: {st.session_state.selected_image.size}")
            
            if st.button("🔍 **START ANALYSIS**", key="analyze_btn"):
                st.session_state.mode = "detected"
                st.rerun()
        else:
            st.info("📁 **READY FOR UPLOAD**")
            st.markdown("**Formats:** JPG, PNG<br>**Max Size:** 200MB", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.mode == "detected":
    if st.session_state.selected_image:
        with st.spinner("🔬 **AI PROCESSING...**"):
            img_array = np.array(st.session_state.selected_image)
            results = model.predict(source=img_array, conf=0.4, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    st.session_state.detection_results = result
                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.markdown("""<div class="content-card"><h3 style="color: #00d4ff;">🎯 DETECTION RESULTS</h3>""", unsafe_allow_html=True)
                        
                        annotated_img = result.plot()
                        st.image(annotated_img, caption="🤖 AI Analysis", use_column_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""<div class="content-card"><h3 style="color: #8b5cf6;">🎮 OBJECT SELECTION</h3>""", unsafe_allow_html=True)
                        
                        st.success(f"🎯 **{len(result.boxes)} OBJECTS FOUND**")
                        
                        object_names = [model.names[int(cls.item())] for cls in result.boxes.cls]
                        unique_objects = list(set(object_names))
                        selected_obj = st.selectbox("🎯 **SELECT TARGET**", unique_objects)
                        
                        if selected_obj:
                            selected_idx = object_names.index(selected_obj)
                            confidence = result.boxes.conf[selected_idx]
                            
                            biological = ['person', 'cat', 'dog', 'horse', 'sheep', 'cow']
                            category = "🧬 BIOLOGICAL" if selected_obj.lower() in biological else "⚙️ MECHANICAL"
                            
                            st.markdown(f"**Type:** {category}")
                            st.markdown(f"**Confidence:** {confidence:.1%}")
                            
                            st.markdown("---")
                            st.markdown("### 📏 CALIBRATION")
                            
                            ref_width_cm = st.number_input("Reference Width (cm)", min_value=0.1, value=10.0)
                            ref_width_px = st.number_input("Reference Pixels", min_value=1, value=100)
                            
                            if st.button("📐 **MEASURE**", key="measure_btn"):
                                pixels_per_cm = ref_width_px / ref_width_cm
                                bbox = result.boxes.xyxy[selected_idx].cpu().numpy()
                                dims = calculate_dimensions(bbox, pixels_per_cm)
                                st.session_state.dimensions = dims
                                st.session_state.mode = "measured"
                                st.success("✅ **COMPLETE**")
                                st.rerun()
                        
                        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.mode == "measured":
    if st.session_state.dimensions:
        dims = st.session_state.dimensions
        
        st.markdown("""<div class="success-msg">🎯 <strong>MEASUREMENT ANALYSIS COMPLETE</strong></div>""", unsafe_allow_html=True)
        
        # Metrics
        st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{dims['width_cm']}</div><div class="metric-label">WIDTH (CM)</div></div>""", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{dims['height_cm']}</div><div class="metric-label">HEIGHT (CM)</div></div>""", unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{dims['depth_cm']}</div><div class="metric-label">DEPTH (CM)</div></div>""", unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{dims['volume_cm3']}</div><div class="metric-label">VOLUME (CM³)</div></div>""", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🌟 **3D HOLOGRAM**", key="hologram_btn"):
                with st.spinner("🎭 **GENERATING...**"):
                    if st.session_state.detection_results:
                        result = st.session_state.detection_results
                        bbox = result.boxes.xyxy[0].cpu().numpy()
                        fig = create_enhanced_3d_hologram(bbox, dims)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            st.success("✨ **HOLOGRAM COMPLETE**")
        
        with col2:
            if st.button("📊 **REPORT**", key="report_btn"):
                st.markdown(f"""
                <div class="content-card">
                    <h3 style="color: #00d4ff;">📋 ANALYSIS REPORT</h3>
                    <div style="background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 12px;">
                        <p><strong>DIMENSIONS:</strong></p>
                        <ul>
                            <li>Width: {dims['width_cm']} cm</li>
                            <li>Height: {dims['height_cm']} cm</li>
                            <li>Depth: {dims['depth_cm']} cm</li>
                            <li>Volume: {dims['volume_cm3']} cm³</li>
                        </ul>
                        <p><strong>STATUS:</strong> ✅ Complete</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if st.button("🔄 **NEW ANALYSIS**", key="new_btn"):
                st.session_state.mode = "home"
                st.session_state.detection_results = None
                st.session_state.selected_image = None
                st.session_state.dimensions = None
                st.session_state.captured_frame = None
                st.rerun()

# Footer
st.markdown("""
<div style="text-align: center; padding: 3rem 2rem 2rem; margin-top: 3rem; border-top: 1px solid #333;">
    <h3 style="color: #00d4ff; font-family: 'Orbitron';">AI DIMENSION ESTIMATOR v3.0</h3>
    <p style="color: #71717a;">Next-Generation Analysis Platform</p>
    <p style="color: #8b5cf6; font-weight: 600;">🎯 Deploy → 🔍 Detect → 📐 Measure → 🌟 Visualize</p>
</div>
""", unsafe_allow_html=True)
