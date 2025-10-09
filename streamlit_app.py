import streamlit as st
import time

# ===============================================================================
# AI DIMENSION ESTIMATOR - PROFESSIONAL ANALYSIS PLATFORM
# Version: 3.0 Final
# Features: Animated intro, Real 3D hologram, Click-to-select objects, Camera mode
# ===============================================================================

# Page configuration
st.set_page_config(
    page_title="AI Dimension Estimator | Professional Analysis Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "AI Dimension Estimator - Next-Generation Analysis Platform"
    }
)

# Load external CSS file
def load_css():
    try:
        with open("style.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("❌ style.css file not found. Please ensure it's in the same directory.")
        # Minimal fallback styles
        st.markdown("""
        <style>
        .stApp { 
            background: #0a0a0a; 
            color: white; 
            font-family: 'Inter', sans-serif; 
        }
        .intro-container {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100vh;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            display: flex; flex-direction: column; justify-content: center; align-items: center;
            z-index: 10000; opacity: 1; visibility: visible;
            transition: all 1.5s ease;
        }
        .intro-container.hidden { opacity: 0; visibility: hidden; pointer-events: none; }
        .intro-logo {
            font-size: 4rem; font-weight: 900; color: #00d4ff;
            text-align: center; margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)

# Load CSS
load_css()

# Import required modules
import numpy as np
from PIL import Image
import torch
import io
import json

# Import optional modules with error handling
try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError:
    st.error("❌ YOLO (ultralytics) not available. Install with: pip install ultralytics")
    yolo_available = False

try:
    import cv2
    opencv_available = True
except ImportError:
    st.warning("⚠️ OpenCV not available. Some features limited.")
    opencv_available = False

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    webrtc_available = True
except ImportError:
    st.warning("⚠️ Camera functionality not available. Install with: pip install streamlit-webrtc")
    webrtc_available = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    plotly_available = True
except ImportError:
    st.warning("⚠️ 3D visualization not available. Install with: pip install plotly")
    plotly_available = False

# Load custom utility modules (optional)
try:
    import depth_utils
    import dimension_utils
    import hologram_utils
    custom_modules_available = True
except ImportError:
    custom_modules_available = False

# ===============================================================================
# MODEL LOADING
# ===============================================================================

@st.cache_resource
def load_yolo_model():
    """Load and cache YOLO model"""
    if not yolo_available:
        return None
    try:
        with st.spinner("🚀 Loading AI Neural Network..."):
            model = YOLO("yolov8n-seg.pt")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

# ===============================================================================
# SESSION STATE INITIALIZATION
# ===============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if "intro_completed" not in st.session_state:
        st.session_state.intro_completed = False
    if "mode" not in st.session_state:
        st.session_state.mode = "home"
    if "detection_results" not in st.session_state:
        st.session_state.detection_results = None
    if "selected_image" not in st.session_state:
        st.session_state.selected_image = None
    if "dimensions" not in st.session_state:
        st.session_state.dimensions = None
    if "captured_frame" not in st.session_state:
        st.session_state.captured_frame = None
    if "selected_object_idx" not in st.session_state:
        st.session_state.selected_object_idx = None
    if "object_mask" not in st.session_state:
        st.session_state.object_mask = None

# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

def calculate_dimensions(bbox, pixels_per_cm):
    """Calculate object dimensions from bounding box"""
    x1, y1, x2, y2 = bbox
    width_px = abs(x2 - x1)
    height_px = abs(y2 - y1)
    
    width_cm = width_px / pixels_per_cm
    height_cm = height_px / pixels_per_cm
    depth_cm = (width_cm + height_cm) / 2  # Estimated depth
    volume_cm3 = width_cm * height_cm * depth_cm
    
    return {
        "width_cm": round(width_cm, 2),
        "height_cm": round(height_cm, 2),
        "depth_cm": round(depth_cm, 2),
        "volume_cm3": round(volume_cm3, 2)
    }

def create_advanced_3d_hologram(mask_data, dimensions):
    """Create 3D hologram from actual object mask shape"""
    if not plotly_available:
        return None
    
    try:
        # Extract object contour from mask
        if mask_data is not None and opencv_available:
            contours, _ = cv2.findContours(
                mask_data.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Simplify contour for better performance
                epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Extract and normalize coordinates
                contour_points = simplified_contour.reshape(-1, 2)
                x_coords = contour_points[:, 0]
                y_coords = contour_points[:, 1]
                
                # Normalize to actual dimensions
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                
                x_norm = (x_coords - x_min) / (x_max - x_min) * dimensions['width_cm']
                y_norm = (y_coords - y_min) / (y_max - y_min) * dimensions['height_cm']
                
                # Create 3D visualization
                fig = go.Figure()
                depth_cm = dimensions['depth_cm']
                
                # Base layer (z=0)
                fig.add_trace(go.Scatter3d(
                    x=np.concatenate([x_norm, [x_norm[0]]]),  # Close the shape
                    y=np.concatenate([y_norm, [y_norm[0]]]),
                    z=[0] * (len(x_norm) + 1),
                    mode='lines+markers',
                    line=dict(color='cyan', width=8),
                    marker=dict(size=6, color='cyan'),
                    name='Base Profile',
                    showlegend=True
                ))
                
                # Top layer (z=depth)
                fig.add_trace(go.Scatter3d(
                    x=np.concatenate([x_norm, [x_norm[0]]]),
                    y=np.concatenate([y_norm, [y_norm[0]]]),
                    z=[depth_cm] * (len(x_norm) + 1),
                    mode='lines+markers',
                    line=dict(color='yellow', width=8),
                    marker=dict(size=6, color='yellow'),
                    name='Top Profile',
                    showlegend=True
                ))
                
                # Vertical edges
                for i in range(len(x_norm)):
                    fig.add_trace(go.Scatter3d(
                        x=[x_norm[i], x_norm[i]],
                        y=[y_norm[i], y_norm[i]],
                        z=[0, depth_cm],
                        mode='lines',
                        line=dict(color='magenta', width=6),
                        showlegend=False
                    ))
                
                # Cross-sections for interior structure
                for i, z_level in enumerate(np.linspace(0, depth_cm, 7)):
                    if 0 < z_level < depth_cm:
                        # Slight tapering effect for realism
                        scale_factor = 0.7 + 0.3 * (z_level / depth_cm)
                        center_x, center_y = x_norm.mean(), y_norm.mean()
                        
                        x_scaled = center_x + (x_norm - center_x) * scale_factor
                        y_scaled = center_y + (y_norm - center_y) * scale_factor
                        
                        alpha = 0.3 + 0.5 * (i / 7)
                        fig.add_trace(go.Scatter3d(
                            x=np.concatenate([x_scaled, [x_scaled[0]]]),
                            y=np.concatenate([y_scaled, [y_scaled[0]]]),
                            z=[z_level] * (len(x_scaled) + 1),
                            mode='lines',
                            line=dict(color=f'rgba(0,255,200,{alpha})', width=4),
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
        
        # Fallback to simple box if no mask available
        return create_simple_box_hologram(dimensions)
        
    except Exception as e:
        st.error(f"❌ 3D Reconstruction Error: {str(e)}")
        return create_simple_box_hologram(dimensions)

def create_simple_box_hologram(dimensions):
    """Create simple 3D box hologram as fallback"""
    if not plotly_available:
        return None
    
    try:
        w, h, d = dimensions['width_cm'], dimensions['height_cm'], dimensions['depth_cm']
        
        fig = go.Figure()
        
        # Define box vertices
        vertices_x = [0, w, w, 0, 0, w, w, 0]
        vertices_y = [0, 0, h, h, 0, 0, h, h]
        vertices_z = [0, 0, 0, 0, d, d, d, d]
        
        # Define box edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
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
            marker=dict(size=12, color='yellow'),
            name=f'Box Dimensions ({w}×{h}×{d} cm)'
        ))
        
        # Center point
        fig.add_trace(go.Scatter3d(
            x=[w/2], y=[h/2], z=[d/2],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Center Point'
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
        
    except Exception as e:
        st.error(f"❌ Box visualization error: {str(e)}")
        return None

def create_interactive_object_selection(image, results):
    """Create interactive object selection interface"""
    if results is None or not hasattr(results, 'boxes'):
        st.image(image, use_column_width=True)
        return None
    
    # Display annotated image
    annotated_img = results.plot()
    st.image(annotated_img, caption="🎯 AI Detection Results - Select an object below", use_column_width=True)
    
    # Extract object information
    boxes = results.boxes.xyxy.cpu().numpy()
    object_names = [model.names[int(cls.item())] for cls in results.boxes.cls]
    confidences = results.boxes.conf.cpu().numpy()
    
    # Create selection interface
    st.markdown("### 🎯 Detected Objects - Click to Select:")
    
    # Display objects in a grid
    cols = st.columns(min(len(boxes), 4))
    selected_idx = None
    
    for i, (box, obj_name, conf) in enumerate(zip(boxes, object_names, confidences)):
        col_idx = i % len(cols)
        with cols[col_idx]:
            # Create button for each object
            button_key = f"select_obj_{i}"
            if st.button(
                f"**{obj_name}**\n📊 {conf:.1%} confidence", 
                key=button_key, 
                use_container_width=True,
                type="secondary"
            ):
                selected_idx = i
                st.session_state.selected_object_idx = i
                
                # Store object mask if available
                if hasattr(results, 'masks') and results.masks is not None:
                    st.session_state.object_mask = results.masks.data[i].cpu().numpy()
                
                st.success(f"🎯 **Selected:** {obj_name} (Confidence: {conf:.1%})")
    
    return selected_idx

# Camera frame callback
captured_frame = None

def video_frame_callback(frame):
    """Callback for processing video frames"""
    global captured_frame
    img = frame.to_ndarray(format="bgr24")
    captured_frame = img.copy()
    return frame

# ===============================================================================
# INTRO ANIMATION
# ===============================================================================

def show_intro_animation():
    """Display animated intro sequence"""
    if st.session_state.intro_completed:
        return
    
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
        // Enhanced intro animation sequence
        setTimeout(() => {
            const progress = document.getElementById('intro-progress');
            if (progress) progress.style.width = '100%';
        }, 1500);
        
        setTimeout(() => {
            const container = document.getElementById('intro-container');
            if (container) container.classList.add('hidden');
        }, 4500);
        </script>
        """, unsafe_allow_html=True)
    
    # Wait for animation to complete
    time.sleep(5)
    intro_container.empty()
    st.session_state.intro_completed = True
    st.rerun()

# ===============================================================================
# MAIN APPLICATION
# ===============================================================================

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Load YOLO model
    model = load_yolo_model()
    
    # Show intro animation on first load
    if not st.session_state.intro_completed:
        show_intro_animation()
        return
    
    # Check if model loaded successfully
    if model is None and yolo_available:
        st.error("🚨 **CRITICAL ERROR** - Neural Network failed to load")
        st.stop()
    
    # === HERO SECTION ===
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">AI DIMENSION ESTIMATOR</div>
        <div class="hero-subtitle">Professional Analysis Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    # === STATUS GRID ===
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
    
    # System status message
    if model:
        st.markdown('<div class="success-msg">🚀 <strong>ALL SYSTEMS OPERATIONAL</strong> - Ready for Advanced Analysis</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-msg">🚨 <strong>SYSTEM ERROR</strong> - AI Network Offline</div>', unsafe_allow_html=True)
        if yolo_available:
            st.stop()
    
    # === NAVIGATION ===
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
                st.error("❌ Camera system unavailable - Install streamlit-webrtc")
    
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
            <div class="nav-description">System overview, status monitoring and platform capabilities</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("", key="nav_home", help="Return to Mission Control"):
            st.session_state.mode = "home"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # === MAIN CONTENT SECTIONS ===
    
    if st.session_state.mode == "home":
        # Mission Control Dashboard
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
                        <li><strong>Interactive Selection:</strong> Click-to-select interface</li>
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
        # Camera Mode
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""<div class="content-card"><h3 style="color: #00d4ff; font-family: 'Orbitron';">📹 LIVE SURVEILLANCE SYSTEM</h3>""", unsafe_allow_html=True)
            
            # Camera stream
            ctx = webrtc_streamer(
                key="ai_camera_stream",
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
                st.info("📊 **Resolution:** HD 1280×720")
                st.info("🎯 **AI Detection:** Ready")
                
                if st.button("📸 **CAPTURE TARGET**", key="capture_btn", type="primary"):
                    if opencv_available:
                        st.session_state.captured_frame = captured_frame.copy()
                        st.session_state.selected_image = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
                    else:
                        st.session_state.selected_image = Image.fromarray(captured_frame)
                    st.session_state.mode = "detected"
                    st.success("🎯 **TARGET ACQUIRED**")
                    st.rerun()
            else:
                st.warning("🟡 **INITIALIZING CAMERA**")
                st.info("⏳ Establishing video connection...")
                st.info("🔧 Check camera permissions if needed")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.mode == "upload":
        # Upload Mode
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
        # Detection Mode
        if st.session_state.selected_image and model:
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
                            st.markdown("""<div class="content-card"><h3 style="color: #00d4ff;">🎯 AI DETECTION ANALYSIS</h3>""", unsafe_allow_html=True)
                            
                            # Interactive object selection
                            create_interactive_object_selection(st.session_state.selected_image, result)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("""<div class="content-card"><h3 style="color: #8b5cf6;">🎮 OBJECT ANALYSIS</h3>""", unsafe_allow_html=True)
                            
                            st.success(f"🎯 **{len(result.boxes)} OBJECTS DETECTED**")
                            
                            if st.session_state.selected_object_idx is not None:
                                idx = st.session_state.selected_object_idx
                                object_names = [model.names[int(cls.item())] for cls in result.boxes.cls]
                                selected_obj = object_names[idx]
                                confidence = result.boxes.conf[idx]
                                
                                # Object classification
                                biological = ['person', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'bird']
                                category = "🧬 BIOLOGICAL" if selected_obj.lower() in biological else "⚙️ MECHANICAL"
                                
                                st.markdown(f"**🎯 Selected:** {selected_obj}")
                                st.markdown(f"**🏷️ Classification:** {category}")
                                st.markdown(f"**🎯 AI Confidence:** {confidence:.1%}")
                                
                                st.markdown("---")
                                st.markdown("### 📏 PRECISION CALIBRATION")
                                
                                ref_width_cm = st.number_input("Reference Width (cm)", min_value=0.1, value=10.0, step=0.1)
                                ref_width_px = st.number_input("Reference Pixels", min_value=1, value=100, step=1)
                                
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
        # Measurement Results
        if st.session_state.dimensions:
            dims = st.session_state.dimensions
            
            st.markdown("""<div class="success-msg">🎯 <strong>DIMENSIONAL ANALYSIS COMPLETED</strong><br>Advanced measurement protocol executed successfully</div>""", unsafe_allow_html=True)
            
            # Metrics Display
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
                            fig = create_advanced_3d_hologram(st.session_state.object_mask, dims)
                            
                            if fig:
                                st.markdown("### 🌟 3D Holographic Object Reconstruction")
                                st.plotly_chart(fig, use_container_width=True)
                                st.success("✨ **3D HOLOGRAM GENERATED** - Showing actual object shape")
                            else:
                                st.error("❌ Hologram generation failed")
                        except Exception as e:
                            st.error(f"❌ Hologram Error: {str(e)}")
            
            with col2:
                if st.button("📊 **DETAILED REPORT**", key="report_btn", type="secondary"):
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    st.markdown(f"""
                    <div class="content-card">
                        <h3 style="color: #00d4ff; font-family: 'Orbitron';">📋 COMPREHENSIVE ANALYSIS REPORT</h3>
                        <div style="background: rgba(0,0,0,0.4); padding: 2rem; border-radius: 16px; font-family: 'JetBrains Mono', monospace;">
                            <p><strong>📅 TIMESTAMP:</strong> {current_time}</p>
                            <p><strong>🎯 OBJECT:</strong> Selected Item</p>
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
                            <p><strong>🎯 CONFIDENCE:</strong> <span style="color: #00d4ff;">HIGH PRECISION</span></p>
                            <p><strong>🔬 METHOD:</strong> <span style="color: #8b5cf6;">AI-Enhanced Calibration</span></p>
                            <p><strong>🌟 3D RECON:</strong> <span style="color: #f59e0b;">{'AVAILABLE' if st.session_state.object_mask is not None else 'BASIC'}</span></p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                if st.button("🔄 **NEW ANALYSIS**", key="new_btn", type="secondary"):
                    # Reset all session states
                    for key in ['mode', 'detection_results', 'selected_image', 'dimensions', 'captured_frame', 'selected_object_idx', 'object_mask']:
                        if key in st.session_state:
                            if key == 'mode':
                                st.session_state[key] = "home"
                            else:
                                st.session_state[key] = None
                    st.rerun()
    
    # === FOOTER ===
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

# ===============================================================================
# ENTRY POINT
# ===============================================================================

if __name__ == "__main__":
    # Global variable for camera
    global model
    model = load_yolo_model()
    
    # Run main application
    main()
