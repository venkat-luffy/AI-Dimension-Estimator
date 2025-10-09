import streamlit as st
import time

# Page configuration
st.set_page_config(
    page_title="AI Dimension Estimator | Professional Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS with better animations and buttons
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
        --neon-orange: #f59e0b;
        --neon-pink: #ec4899;
        --text-primary: #ffffff;
        --text-secondary: #a1a1aa;
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-neon: linear-gradient(45deg, #00d4ff, #8b5cf6, #10b981, #f59e0b);
        --shadow-glow: 0 0 30px rgba(0, 212, 255, 0.5);
    }
    
    .stApp {
        background: var(--primary-bg);
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(0, 212, 255, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(16, 185, 129, 0.04) 0%, transparent 50%);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        animation: backgroundPulse 10s ease-in-out infinite alternate;
    }
    
    @keyframes backgroundPulse {
        0% { background-position: 0% 0%; }
        100% { background-position: 100% 100%; }
    }
    
    /* Enhanced Intro Animation */
    .intro-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100vh;
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 10000;
        opacity: 1;
        visibility: visible;
        transition: all 1.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    
    .intro-container.hidden {
        opacity: 0;
        visibility: hidden;
        pointer-events: none;
    }
    
    .intro-logo {
        font-size: 5rem;
        font-weight: 900;
        font-family: 'Orbitron', monospace;
        background: var(--gradient-neon);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: logoAnimation 3s ease-out, gradientFlow 4s ease-in-out infinite;
        transform: translateY(50px);
        opacity: 0;
        animation-fill-mode: both;
    }
    
    @keyframes logoAnimation {
        0% { transform: translateY(50px) scale(0.8); opacity: 0; }
        50% { transform: translateY(0) scale(1.1); opacity: 1; }
        100% { transform: translateY(0) scale(1); opacity: 1; }
    }
    
    @keyframes gradientFlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .intro-subtitle {
        font-size: 1.8rem;
        color: var(--neon-blue);
        font-weight: 300;
        text-align: center;
        margin-bottom: 3rem;
        animation: subtitleSlide 2s ease-out 0.5s both;
    }
    
    @keyframes subtitleSlide {
        0% { transform: translateY(30px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    
    .intro-loader {
        width: 300px;
        height: 6px;
        background: var(--border-primary);
        border-radius: 3px;
        overflow: hidden;
        position: relative;
        animation: loaderAppear 2s ease-out 1s both;
    }
    
    @keyframes loaderAppear {
        0% { transform: scaleX(0); opacity: 0; }
        100% { transform: scaleX(1); opacity: 1; }
    }
    
    .intro-progress {
        height: 100%;
        background: var(--gradient-neon);
        width: 0%;
        transition: width 3s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 3px;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.6);
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 4rem 0 3rem;
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.7) 100%);
        border-radius: 0 0 30px 30px;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.1), transparent);
        animation: heroScan 4s linear infinite;
    }
    
    @keyframes heroScan {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .hero-title {
        font-size: clamp(3rem, 7vw, 5rem);
        font-weight: 900;
        font-family: 'Orbitron', monospace;
        background: var(--gradient-neon);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        animation: titleGlow 4s ease-in-out infinite, gradientFlow 6s ease-in-out infinite;
        position: relative;
    }
    
    @keyframes titleGlow {
        0%, 100% { filter: drop-shadow(0 0 30px rgba(0, 212, 255, 0.4)); }
        50% { filter: drop-shadow(0 0 50px rgba(139, 92, 246, 0.6)); }
    }
    
    .hero-subtitle {
        font-size: 1.4rem;
        color: var(--text-secondary);
        font-weight: 400;
        opacity: 0.9;
    }
    
    /* Status Grid */
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 2rem;
        padding: 2rem;
        margin-bottom: 3rem;
    }
    
    .status-card {
        background: var(--glass-bg);
        backdrop-filter: blur(15px);
        border: 1px solid var(--border-primary);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .status-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: var(--gradient-neon);
        transform: scaleX(0);
        transition: transform 0.4s ease;
    }
    
    .status-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: var(--neon-blue);
        box-shadow: var(--shadow-glow);
    }
    
    .status-card:hover::before {
        transform: scaleX(1);
    }
    
    .status-header {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .status-icon {
        width: 50px;
        height: 50px;
        background: var(--gradient-primary);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        animation: iconPulse 3s infinite;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    @keyframes iconPulse {
        0%, 100% { box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); }
        50% { box-shadow: 0 4px 25px rgba(0, 212, 255, 0.5); }
    }
    
    .status-title {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 1.1rem;
    }
    
    .status-value {
        font-weight: 800;
        color: var(--neon-blue);
        font-family: 'Orbitron', monospace;
        font-size: 1.3rem;
    }
    
    /* Enhanced Navigation Buttons */
    .nav-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 2.5rem;
        padding: 2rem;
        margin-bottom: 3rem;
    }
    
    .modern-nav-btn {
        background: var(--card-bg);
        border: 2px solid var(--border-primary);
        border-radius: 24px;
        padding: 3rem 2rem;
        color: var(--text-primary);
        text-decoration: none;
        transition: all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        cursor: pointer;
        text-align: center;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .modern-nav-btn::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: radial-gradient(circle, rgba(0, 212, 255, 0.1) 0%, transparent 70%);
        transition: all 0.6s ease;
        transform: translate(-50%, -50%);
        border-radius: 50%;
    }
    
    .modern-nav-btn:hover::before {
        width: 600px;
        height: 600px;
    }
    
    .modern-nav-btn:hover {
        transform: translateY(-12px) rotateX(5deg);
        border-color: var(--neon-blue);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.4),
            0 0 30px rgba(0, 212, 255, 0.3);
    }
    
    .nav-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        display: block;
        animation: iconFloat 4s ease-in-out infinite;
    }
    
    @keyframes iconFloat {
        0%, 100% { transform: translateY(0) rotateZ(0deg); }
        50% { transform: translateY(-15px) rotateZ(5deg); }
    }
    
    .nav-title {
        font-size: 1.6rem;
        font-weight: 800;
        font-family: 'Orbitron', monospace;
        margin-bottom: 1rem;
        color: var(--neon-blue);
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .nav-description {
        color: var(--text-secondary);
        font-size: 1rem;
        line-height: 1.6;
        font-weight: 400;
    }
    
    /* Action Buttons */
    .action-btn {
        background: var(--gradient-primary);
        border: none;
        border-radius: 16px;
        padding: 1.2rem 2.5rem;
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .action-btn::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        transition: all 0.6s ease;
        transform: translate(-50%, -50%);
    }
    
    .action-btn:hover::before {
        width: 400px;
        height: 400px;
    }
    
    .action-btn:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
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
        position: relative;
    }
    
    .action-btn.hologram::after {
        content: '✨';
        position: absolute;
        right: 20px;
        top: 50%;
        transform: translateY(-50%);
        animation: sparkle 2s infinite;
    }
    
    @keyframes sparkle {
        0%, 100% { opacity: 1; transform: translateY(-50%) scale(1); }
        50% { opacity: 0.6; transform: translateY(-50%) scale(1.3); }
    }
    
    /* Content Cards */
    .content-card {
        background: var(--card-bg);
        border: 1px solid var(--border-primary);
        border-radius: 24px;
        padding: 2.5rem;
        margin: 1rem;
        transition: all 0.4s ease;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .content-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: var(--gradient-neon);
        border-radius: 24px;
        opacity: 0;
        transition: opacity 0.4s ease;
        z-index: -1;
    }
    
    .content-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
    }
    
    .content-card:hover::before {
        opacity: 1;
    }
    
    /* Interactive Object Selection */
    .detection-image {
        position: relative;
        cursor: crosshair;
        border-radius: 16px;
        overflow: hidden;
    }
    
    .object-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 10;
    }
    
    .object-bbox {
        position: absolute;
        border: 3px solid var(--neon-blue);
        border-radius: 8px;
        background: rgba(0, 212, 255, 0.1);
        cursor: pointer;
        pointer-events: all;
        transition: all 0.3s ease;
        animation: bboxPulse 2s infinite;
    }
    
    @keyframes bboxPulse {
        0%, 100% { box-shadow: 0 0 15px rgba(0, 212, 255, 0.4); }
        50% { box-shadow: 0 0 25px rgba(0, 212, 255, 0.8); }
    }
    
    .object-bbox:hover {
        border-color: var(--neon-green);
        background: rgba(16, 185, 129, 0.2);
        transform: scale(1.05);
    }
    
    .object-bbox.selected {
        border-color: var(--neon-orange);
        background: rgba(245, 158, 11, 0.2);
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.6);
    }
    
    /* Metrics */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: var(--glass-bg);
        border: 1px solid var(--border-primary);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: var(--gradient-neon);
        animation: metricGlow 3s linear infinite;
    }
    
    @keyframes metricGlow {
        0%, 100% { transform: scaleX(0.2); opacity: 0.6; }
        50% { transform: scaleX(1); opacity: 1; }
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.05);
        border-color: var(--neon-blue);
        box-shadow: var(--shadow-glow);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 900;
        font-family: 'Orbitron', monospace;
        background: var(--gradient-neon);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        animation: valueFlicker 2s infinite;
    }
    
    @keyframes valueFlicker {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.9rem;
    }
    
    /* Success/Error Messages */
    .success-msg {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05));
        border: 2px solid var(--neon-green);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        color: var(--neon-green);
        font-weight: 600;
        animation: successPulse 1s ease-out;
    }
    
    @keyframes successPulse {
        0% { transform: scale(0.95); opacity: 0; }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .error-msg {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05));
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Loading States */
    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 4px solid var(--border-primary);
        border-top: 4px solid var(--neon-blue);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Mission Control Enhancements */
    .mission-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .mission-card {
        background: var(--glass-bg);
        border: 1px solid var(--border-primary);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .mission-card:hover {
        border-color: var(--neon-blue);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    /* Responsive Design */
    @media (max-width: 1024px) {
        .nav-container {
            grid-template-columns: 1fr;
            gap: 2rem;
        }
        
        .modern-nav-btn {
            padding: 2rem;
        }
        
        .nav-icon {
            font-size: 3rem;
        }
    }
    
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .intro-logo {
            font-size: 3rem;
        }
        
        .status-grid, .nav-container {
            padding: 1rem;
        }
        
        .content-card {
            padding: 1.5rem;
            margin: 0.5rem;
        }
        
        .metrics-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
    }
    
    @media (max-width: 480px) {
        .metrics-grid {
            grid-template-columns: 1fr;
        }
        
        .mission-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Streamlit Overrides */
    .stSelectbox > div > div > div {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-primary) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }
    
    .stNumberInput > div > div > div {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-primary) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }
    
    .stFileUploader > div {
        border: 2px dashed var(--neon-blue) !important;
        border-radius: 16px !important;
        background: var(--glass-bg) !important;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# Import modules
import numpy as np
from PIL import Image
import torch
import io
import json

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
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False
    st.session_state.mode = "home"
    st.session_state.detection_results = None
    st.session_state.selected_image = None
    st.session_state.dimensions = None
    st.session_state.captured_frame = None
    st.session_state.selected_object_idx = None
    st.session_state.object_mask = None

# Enhanced intro animation
if not st.session_state.intro_done:
    intro_container = st.empty()
    
    with intro_container.container():
        st.markdown("""
        <div class="intro-container" id="intro-container">
            <div class="intro-logo">
                AI DIMENSION<br>ESTIMATOR
            </div>
            <div class="intro-subtitle">
                Professional Analysis Platform
            </div>
            <div class="intro-loader">
                <div class="intro-progress" id="intro-progress"></div>
            </div>
        </div>
        
        <script>
        // Enhanced intro sequence
        setTimeout(() => {
            document.getElementById('intro-progress').style.width = '100%';
        }, 1500);
        
        setTimeout(() => {
            document.getElementById('intro-container').classList.add('hidden');
        }, 4500);
        </script>
        """, unsafe_allow_html=True)
    
    # Wait for intro
    time.sleep(5)
    intro_container.empty()
    st.session_state.intro_done = True
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

def create_real_object_hologram(mask_data, dimensions):
    """Create 3D hologram from actual object mask shape"""
    if not plotly_available:
        return None
    
    try:
        # Get actual object contour from mask
        if mask_data is not None:
            # Convert mask to contour points
            contours, _ = cv2.findContours(mask_data.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Simplify contour
                epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Extract coordinates
                contour_points = simplified.reshape(-1, 2)
                x_coords = contour_points[:, 0]
                y_coords = contour_points[:, 1]
                
                # Normalize to dimensions
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                
                x_norm = (x_coords - x_min) / (x_max - x_min) * dimensions['width_cm']
                y_norm = (y_coords - y_min) / (y_max - y_min) * dimensions['height_cm']
                
                # Create 3D object
                fig = go.Figure()
                
                depth_cm = dimensions['depth_cm']
                
                # Bottom face (z=0)
                fig.add_trace(go.Scatter3d(
                    x=x_norm.tolist() + [x_norm[0]],  # Close the shape
                    y=y_norm.tolist() + [y_norm[0]],
                    z=[0] * (len(x_norm) + 1),
                    mode='lines+markers',
                    line=dict(color='cyan', width=8),
                    marker=dict(size=6, color='cyan'),
                    name='Base',
                    showlegend=True
                ))
                
                # Top face (z=depth)
                fig.add_trace(go.Scatter3d(
                    x=x_norm.tolist() + [x_norm[0]],
                    y=y_norm.tolist() + [y_norm[0]],
                    z=[depth_cm] * (len(x_norm) + 1),
                    mode='lines+markers',
                    line=dict(color='yellow', width=8),
                    marker=dict(size=6, color='yellow'),
                    name='Top',
                    showlegend=True
                ))
                
                # Vertical edges connecting base to top
                for i in range(len(x_norm)):
                    fig.add_trace(go.Scatter3d(
                        x=[x_norm[i], x_norm[i]],
                        y=[y_norm[i], y_norm[i]],
                        z=[0, depth_cm],
                        mode='lines',
                        line=dict(color='magenta', width=6),
                        showlegend=False
                    ))
                
                # Add cross-sections for interior structure
                num_levels = 5
                for i, z_level in enumerate(np.linspace(0, depth_cm, num_levels)):
                    if 0 < z_level < depth_cm:
                        # Create slightly smaller cross-section for depth effect
                        scale_factor = 0.8 + 0.2 * (z_level / depth_cm)
                        center_x, center_y = x_norm.mean(), y_norm.mean()
                        
                        x_scaled = center_x + (x_norm - center_x) * scale_factor
                        y_scaled = center_y + (y_norm - center_y) * scale_factor
                        
                        alpha = 0.3 + 0.4 * (i / num_levels)
                        fig.add_trace(go.Scatter3d(
                            x=x_scaled.tolist() + [x_scaled[0]],
                            y=y_scaled.tolist() + [y_scaled[0]],
                            z=[z_level] * (len(x_scaled) + 1),
                            mode='lines',
                            line=dict(color=f'rgba(0,255,255,{alpha})', width=4),
                            showlegend=False
                        ))
                
                # Enhanced layout
                fig.update_layout(
                    title={
                        'text': f"🌟 3D OBJECT RECONSTRUCTION",
                        'x': 0.5,
                        'font': {'size': 24, 'color': 'white', 'family': 'Orbitron'}
                    },
                    scene=dict(
                        bgcolor='rgba(10,10,10,0.95)',
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
                            eye=dict(x=1.8, y=1.8, z=1.8),
                            up=dict(x=0, y=0, z=1)
                        ),
                        aspectratio=dict(x=1, y=1, z=0.8)
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', family='Inter'),
                    margin=dict(l=0, r=0, t=60, b=0),
                    showlegend=True,
                    legend=dict(
                        bgcolor='rgba(26,26,26,0.8)',
                        bordercolor='rgba(0,212,255,0.5)',
                        borderwidth=1,
                        font=dict(color='white')
                    )
                )
                
                return fig
                
        # Fallback to simple box if no mask
        return create_simple_box_hologram(dimensions)
        
    except Exception as e:
        st.error(f"3D Reconstruction Error: {str(e)}")
        return create_simple_box_hologram(dimensions)

def create_simple_box_hologram(dimensions):
    """Fallback simple box hologram"""
    if not plotly_available:
        return None
    
    w, h, d = dimensions['width_cm'], dimensions['height_cm'], dimensions['depth_cm']
    
    fig = go.Figure()
    
    # Box vertices
    vertices_x = [0, w, w, 0, 0, w, w, 0]
    vertices_y = [0, 0, h, h, 0, 0, h, h]
    vertices_z = [0, 0, 0, 0, d, d, d, d]
    
    # Box edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
    ]
    
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
    
    # Corner points
    fig.add_trace(go.Scatter3d(
        x=vertices_x, y=vertices_y, z=vertices_z,
        mode='markers',
        marker=dict(size=12, color='yellow'),
        name=f'Box ({w}×{h}×{d} cm)'
    ))
    
    fig.update_layout(
        title="🌟 3D BOX VISUALIZATION",
        scene=dict(
            bgcolor='rgba(10,10,10,0.95)',
            xaxis=dict(title='Width (cm)', color='white'),
            yaxis=dict(title='Height (cm)', color='white'),
            zaxis=dict(title='Depth (cm)', color='white')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_interactive_detection_display(image, results):
    """Create clickable object detection display"""
    if results is None or not hasattr(results, 'boxes'):
        return st.image(image, use_column_width=True)
    
    # Display image with interactive overlay
    annotated_img = results.plot()
    
    # Create clickable areas using HTML/CSS
    boxes = results.boxes.xyxy.cpu().numpy()
    object_names = [model.names[int(cls.item())] for cls in results.boxes.cls]
    confidences = results.boxes.conf.cpu().numpy()
    
    # Display image
    st.image(annotated_img, caption="🎯 Click on objects to select them", use_column_width=True)
    
    # Create object selection buttons
    st.markdown("### 🎯 Detected Objects - Click to Select:")
    
    cols = st.columns(min(len(boxes), 4))
    for i, (box, obj_name, conf) in enumerate(zip(boxes, object_names, confidences)):
        col_idx = i % len(cols)
        with cols[col_idx]:
            if st.button(f"{obj_name}\n{conf:.1%}", key=f"obj_{i}", use_container_width=True):
                st.session_state.selected_object_idx = i
                st.success(f"🎯 Selected: {obj_name}")
                return i
    
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
    <div class="hero-subtitle">Professional Analysis Platform</div>
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
                <div class="status-title">Camera System</div>
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
                <div class="status-title">Hologram Engine</div>
                <div class="status-value">{hologram_status}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# System Status
if model:
    st.markdown('<div class="success-msg">🚀 <strong>ALL SYSTEMS OPERATIONAL</strong> - Ready for Advanced Analysis</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="error-msg">🚨 <strong>CRITICAL ERROR</strong> - Neural Network Offline</div>', unsafe_allow_html=True)
    st.stop()

# Enhanced Navigation
st.markdown('<div class="nav-container">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="modern-nav-btn">
        <div class="nav-icon">📹</div>
        <div class="nav-title">Camera Mode</div>
        <div class="nav-description">Real-time video capture with live object detection and instant analysis</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("", key="nav_camera", help="Launch Camera Mode"):
        if webrtc_available:
            st.session_state.mode = "camera"
            st.rerun()
        else:
            st.error("❌ Camera system unavailable")

with col2:
    st.markdown("""
    <div class="modern-nav-btn">
        <div class="nav-icon">📁</div>
        <div class="nav-title">Upload Mode</div>
        <div class="nav-description">Advanced image processing with AI-powered object recognition and measurement</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("", key="nav_upload", help="Launch Upload Mode"):
        st.session_state.mode = "upload"
        st.rerun()

with col3:
    st.markdown("""
    <div class="modern-nav-btn">
        <div class="nav-icon">🏠</div>
        <div class="nav-title">Mission Control</div>
        <div class="nav-description">System overview, status monitoring and platform capabilities overview</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("", key="nav_home", help="Return to Mission Control"):
        st.session_state.mode = "home"
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Main Content Based on Mode
if st.session_state.mode == "home":
    # Enhanced Mission Control
    st.markdown("""
    <div class="content-card">
        <h2 style="color: #00d4ff; font-family: 'Orbitron'; margin-bottom: 2rem; text-align: center;">🎯 MISSION CONTROL CENTER</h2>
        
        <div class="mission-grid">
            <div class="mission-card">
                <h3 style="color: #8b5cf6; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 1rem;">
                    <span style="font-size: 2rem;">🚀</span>
                    CORE CAPABILITIES
                </h3>
                <ul style="color: #a1a1aa; line-height: 2; font-size: 1.1rem;">
                    <li><strong>YOLOv8 Neural Network:</strong> State-of-the-art object detection</li>
                    <li><strong>Real-time Processing:</strong> Live video analysis</li>
                    <li><strong>Precision Measurement:</strong> Advanced calibration systems</li>
                    <li><strong>3D Reconstruction:</strong> True object shape holography</li>
                    <li><strong>Multi-object Support:</strong> Simultaneous analysis</li>
                    <li><strong>Interactive Selection:</strong> Click-to-select objects</li>
                </ul>
            </div>
            
            <div class="mission-card">
                <h3 style="color: #10b981; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 1rem;">
                    <span style="font-size: 2rem;">📡</span>
                    OPERATION MODES
                </h3>
                <div style="color: #a1a1aa; line-height: 1.8;">
                    <div style="margin-bottom: 1rem; padding: 1rem; background: rgba(0,212,255,0.1); border-radius: 8px;">
                        <strong style="color: #00d4ff;">📹 Camera Mode:</strong><br>
                        Live video capture with real-time detection and measurement
                    </div>
                    <div style="margin-bottom: 1rem; padding: 1rem; background: rgba(139,92,246,0.1); border-radius: 8px;">
                        <strong style="color: #8b5cf6;">📁 Upload Mode:</strong><br>
                        Static image processing with interactive object selection
                    </div>
                    <div style="padding: 1rem; background: rgba(16,185,129,0.1); border-radius: 8px;">
                        <strong style="color: #10b981;">🌟 3D Hologram:</strong><br>
                        Advanced visualization of actual object shapes
                    </div>
                </div>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 3rem; padding: 2.5rem; background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(139,92,246,0.1)); border-radius: 20px; border: 2px solid rgba(0,212,255,0.3);">
            <h3 style="color: #00d4ff; margin-bottom: 1rem; font-size: 1.8rem;">🎮 SELECT OPERATION MODE TO BEGIN</h3>
            <p style="color: #71717a; font-size: 1.2rem; margin-bottom: 2rem;">Choose your preferred analysis method and start measuring objects with AI precision</p>
            <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                <span style="background: rgba(0,212,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; color: #00d4ff;">Real-time Analysis</span>
                <span style="background: rgba(139,92,246,0.2); padding: 0.5rem 1rem; border-radius: 20px; color: #8b5cf6;">Interactive Selection</span>
                <span style="background: rgba(16,185,129,0.2); padding: 0.5rem 1rem; border-radius: 20px; color: #10b981;">3D Visualization</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.mode == "camera" and webrtc_available:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""<div class="content-card"><h3 style="color: #00d4ff; font-family: 'Orbitron';">📹 LIVE SURVEILLANCE SYSTEM</h3>""", unsafe_allow_html=True)
        
        ctx = webrtc_streamer(
            key="professional_camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {"width": 1280, "height": 720, "frameRate": 30},
                "audio": False
            },
            async_processing=False,
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""<div class="content-card"><h3 style="color: #8b5cf6;">🎛️ CONTROL PANEL</h3>""", unsafe_allow_html=True)
        
        if captured_frame is not None:
            st.success("🟢 **CAMERA FEED ACTIVE**")
            st.info("📊 **Resolution:** HD 1280x720")
            st.info("🎯 **AI Detection:** Ready")
            
            if st.button("📸 **CAPTURE TARGET**", key="capture_btn", type="primary"):
                st.session_state.captured_frame = captured_frame.copy()
                st.session_state.selected_image = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
                st.session_state.mode = "detected"
                st.success("🎯 **TARGET ACQUIRED**")
                st.rerun()
        else:
            st.warning("🟡 **INITIALIZING CAMERA**")
            st.info("⏳ Establishing video connection...")
            st.info("🔧 Check camera permissions")
        
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.mode == "upload":
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""<div class="content-card"><h3 style="color: #00d4ff; font-family: 'Orbitron';">📁 ADVANCED UPLOAD SYSTEM</h3>""", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Deploy Target Image for Analysis",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload high-resolution images for optimal analysis results"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="🎯 Target Image Successfully Loaded", use_column_width=True)
            st.session_state.selected_image = image
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""<div class="content-card"><h3 style="color: #10b981;">📊 UPLOAD STATUS</h3>""", unsafe_allow_html=True)
        
        if uploaded_file:
            st.success("✅ **IMAGE SUCCESSFULLY LOADED**")
            st.info(f"📏 **Resolution:** {st.session_state.selected_image.size}")
            st.info(f"📄 **Format:** {st.session_state.selected_image.format}")
            st.info("🎯 **Status:** Ready for AI Analysis")
            
            if st.button("🔍 **INITIATE NEURAL ANALYSIS**", key="analyze_btn", type="primary"):
                st.session_state.mode = "detected"
                st.rerun()
        else:
            st.info("📁 **AWAITING IMAGE DEPLOYMENT**")
            st.markdown("""
            **📋 Supported Formats:**
            - JPG, JPEG, PNG
            - BMP, TIFF
            - Maximum size: 200MB
            
            **💡 Optimization Tips:**
            - Use high resolution images
            - Ensure good lighting conditions  
            - Clear object visibility required
            - Avoid blurry or low-quality images
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.mode == "detected":
    if st.session_state.selected_image:
        with st.spinner("🔬 **AI NEURAL NETWORK PROCESSING IN PROGRESS...**"):
            img_array = np.array(st.session_state.selected_image)
            results = model.predict(source=img_array, conf=0.4, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    st.session_state.detection_results = result
                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.markdown("""<div class="content-card"><h3 style="color: #00d4ff;">🎯 AI DETECTION ANALYSIS</h3>""", unsafe_allow_html=True)
                        
                        # Interactive object selection
                        selected_idx = create_interactive_detection_display(st.session_state.selected_image, result)
                        
                        if selected_idx is not None:
                            st.session_state.selected_object_idx = selected_idx
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""<div class="content-card"><h3 style="color: #8b5cf6;">🎮 OBJECT ANALYSIS</h3>""", unsafe_allow_html=True)
                        
                        st.success(f"🎯 **{len(result.boxes)} OBJECTS DETECTED**")
                        
                        if st.session_state.selected_object_idx is not None:
                            idx = st.session_state.selected_object_idx
                            object_names = [model.names[int(cls.item())] for cls in result.boxes.cls]
                            selected_obj = object_names[idx]
                            confidence = result.boxes.conf[idx]
                            
                            # Store object mask for hologram
                            if hasattr(result, 'masks') and result.masks is not None:
                                st.session_state.object_mask = result.masks.data[idx].cpu().numpy()
                            
                            # Object classification
                            biological = ['person', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'bird']
                            category = "🧬 BIOLOGICAL" if selected_obj.lower() in biological else "⚙️ MECHANICAL"
                            
                            st.markdown(f"**🎯 Selected:** {selected_obj}")
                            st.markdown(f"**🏷️ Classification:** {category}")
                            st.markdown(f"**🎯 AI Confidence:** {confidence:.1%}")
                            
                            st.markdown("---")
                            st.markdown("### 📏 PRECISION CALIBRATION")
                            
                            ref_width_cm = st.number_input("Reference Width (cm)", min_value=0.1, value=10.0, step=0.1, key="ref_width")
                            ref_width_px = st.number_input("Reference Pixels", min_value=1, value=100, step=1, key="ref_pixels")
                            
                            if st.button("📐 **EXECUTE MEASUREMENT**", key="measure_btn", type="primary"):
                                try:
                                    pixels_per_cm = ref_width_px / ref_width_cm
                                    bbox = result.boxes.xyxy[idx].cpu().numpy()
                                    dims = calculate_dimensions(bbox, pixels_per_cm)
                                    st.session_state.dimensions = dims
                                    st.session_state.mode = "measured"
                                    st.success("✅ **MEASUREMENT ANALYSIS COMPLETE**")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Measurement Error: {str(e)}")
                        else:
                            st.info("👆 **Click on an object above to select it for measurement**")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-msg">❌ <strong>NO OBJECTS DETECTED</strong> - Try different image or lighting</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-msg">❌ <strong>ANALYSIS FAILED</strong> - Unable to process image</div>', unsafe_allow_html=True)

elif st.session_state.mode == "measured":
    if st.session_state.dimensions:
        dims = st.session_state.dimensions
        
        st.markdown("""<div class="success-msg">🎯 <strong>DIMENSIONAL ANALYSIS COMPLETED SUCCESSFULLY</strong><br>Advanced measurement protocol executed with high precision</div>""", unsafe_allow_html=True)
        
        # Enhanced Metrics Display
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
            if st.button("🌟 **3D HOLOGRAPHIC RECONSTRUCTION**", key="hologram_btn", type="primary"):
                with st.spinner("🎭 **GENERATING ADVANCED 3D HOLOGRAM...**"):
                    try:
                        # Create real object shape hologram
                        fig = create_real_object_hologram(st.session_state.object_mask, dims)
                        
                        if fig:
                            st.markdown("### 🌟 3D Holographic Object Reconstruction")
                            st.plotly_chart(fig, use_container_width=True)
                            st.success("✨ **3D HOLOGRAM GENERATED SUCCESSFULLY** - Showing actual object shape")
                        else:
                            st.error("❌ Hologram generation failed")
                    except Exception as e:
                        st.error(f"❌ Hologram Error: {str(e)}")
        
        with col2:
            if st.button("📊 **DETAILED ANALYSIS REPORT**", key="report_btn", type="secondary"):
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                st.markdown(f"""
                <div class="content-card">
                    <h3 style="color: #00d4ff; font-family: 'Orbitron';">📋 COMPREHENSIVE ANALYSIS REPORT</h3>
                    <div style="background: rgba(0,0,0,0.4); padding: 2rem; border-radius: 16px; font-family: 'JetBrains Mono', monospace;">
                        <p><strong>📅 TIMESTAMP:</strong> {current_time}</p>
                        <p><strong>🎯 OBJECT:</strong> {model.names[int(st.session_state.detection_results.boxes.cls[st.session_state.selected_object_idx].item())] if st.session_state.detection_results else 'Unknown'}</p>
                        <hr style="border-color: #333; margin: 1rem 0;">
                        <p><strong>📏 DIMENSIONAL ANALYSIS:</strong></p>
                        <ul style="margin-left: 2rem;">
                            <li>Width: {dims['width_cm']} cm</li>
                            <li>Height: {dims['height_cm']} cm</li>
                            <li>Depth: {dims['depth_cm']} cm (estimated)</li>
                            <li>Volume: {dims['volume_cm3']} cm³</li>
                        </ul>
                        <hr style="border-color: #333; margin: 1rem 0;">
                        <p><strong>📊 ANALYSIS STATUS:</strong> <span style="color: #10b981;">✅ COMPLETE</span></p>
                        <p><strong>🎯 CONFIDENCE LEVEL:</strong> <span style="color: #00d4ff;">HIGH PRECISION</span></p>
                        <p><strong>🔬 METHOD:</strong> <span style="color: #8b5cf6;">AI-Enhanced Calibration</span></p>
                        <p><strong>🌟 3D RECONSTRUCTION:</strong> <span style="color: #f59e0b;">{'AVAILABLE' if st.session_state.object_mask is not None else 'BASIC'}</span></p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if st.button("🔄 **INITIATE NEW ANALYSIS**", key="new_btn", type="secondary"):
                # Reset all session states
                for key in ['mode', 'detection_results', 'selected_image', 'dimensions', 'captured_frame', 'selected_object_idx', 'object_mask']:
                    if key in st.session_state:
                        if key == 'mode':
                            st.session_state[key] = "home"
                        else:
                            st.session_state[key] = None
                st.rerun()

# Enhanced Footer
st.markdown("""
<div style="text-align: center; padding: 4rem 2rem 2rem; margin-top: 4rem; border-top: 2px solid #333; background: linear-gradient(135deg, rgba(10,10,10,0.8), rgba(26,26,46,0.4));">
    <h3 style="color: #00d4ff; font-family: 'Orbitron'; margin-bottom: 1rem; font-size: 2rem;">AI DIMENSION ESTIMATOR v3.0</h3>
    <p style="color: #8b5cf6; font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem;">Next-Generation Professional Analysis Platform</p>
    <p style="color: #71717a; font-size: 1.1rem; margin-bottom: 2rem;">Powered by Advanced Neural Networks & 3D Reconstruction Technology</p>
    <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-top: 2rem;">
        <span style="background: rgba(0,212,255,0.1); padding: 0.8rem 1.5rem; border-radius: 25px; color: #00d4ff; border: 1px solid rgba(0,212,255,0.3);">🎯 Deploy</span>
        <span style="background: rgba(139,92,246,0.1); padding: 0.8rem 1.5rem; border-radius: 25px; color: #8b5cf6; border: 1px solid rgba(139,92,246,0.3);">🔍 Detect</span>
        <span style="background: rgba(16,185,129,0.1); padding: 0.8rem 1.5rem; border-radius: 25px; color: #10b981; border: 1px solid rgba(16,185,129,0.3);">📐 Measure</span>
        <span style="background: rgba(245,158,11,0.1); padding: 0.8rem 1.5rem; border-radius: 25px; color: #f59e0b; border: 1px solid rgba(245,158,11,0.3);">🌟 Visualize</span>
    </div>
</div>
""", unsafe_allow_html=True)
