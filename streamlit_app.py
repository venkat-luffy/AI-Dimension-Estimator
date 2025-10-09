import streamlit as st

# CRITICAL: set_page_config MUST be the absolute first Streamlit command
st.set_page_config(page_title="AI Dimension Estimator", layout="wide")

import sys

# Handle NumPy import
try:
    import numpy as np
    numpy_available = True
except ImportError as e:
    st.error(f"❌ Critical NumPy error: {str(e)}")
    st.error("🚨 This is a system-level issue. Please restart the app.")
    st.stop()

# Handle other imports
try:
    from PIL import Image
    import torch
    pil_available = True
except ImportError as e:
    st.error(f"❌ PIL/Torch import error: {str(e)}")
    st.stop()

try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError as e:
    st.warning(f"⚠️ YOLO unavailable: {str(e)}")
    yolo_available = False

try:
    import cv2
    opencv_available = True
except ImportError as e:
    st.warning(f"⚠️ OpenCV unavailable: {str(e)}")
    opencv_available = False

try:
    import plotly.graph_objects as go
    plotly_available = True
except ImportError as e:
    st.warning(f"⚠️ Plotly unavailable: {str(e)}")
    plotly_available = False

# Model Loading
@st.cache_resource
def load_yolo_model():
    if not yolo_available:
        return None
    try:
        model = YOLO("yolov8n-seg.pt")
        return model
    except Exception as e:
        st.error(f"❌ YOLO loading failed: {str(e)}")
        return None

# Session State
if "detection_results" not in st.session_state:
    st.session_state.detection_results = None
    st.session_state.selected_image = None
    st.session_state.dimensions = None

# Simple dimension calculation
def calculate_simple_dimensions(bbox, pixels_per_cm):
    """Calculate dimensions from bounding box"""
    x1, y1, x2, y2 = bbox
    width_px = abs(x2 - x1)
    height_px = abs(y2 - y1)
    
    width_cm = width_px / pixels_per_cm
    height_cm = height_px / pixels_per_cm
    # Simple depth estimate
    depth_cm = (width_cm + height_cm) / 2
    volume_cm3 = width_cm * height_cm * depth_cm
    
    return {
        "width_cm": round(width_cm, 2),
        "height_cm": round(height_cm, 2),
        "depth_cm": round(depth_cm, 2),
        "volume_cm3": round(volume_cm3, 2)
    }

# Main App
st.title("📱 AI Dimension Estimator")

# System Status (now after page config)
st.info(f"**System Status:** NumPy: {'✅' if numpy_available else '❌'} | " +
        f"YOLO: {'✅' if yolo_available else '❌'} | " +
        f"OpenCV: {'✅' if opencv_available else '❌'}")

# Check critical components
if not numpy_available:
    st.error("❌ NumPy is required. Please restart the application.")
    st.stop()

if not yolo_available:
    st.error("❌ YOLO is required for object detection.")
    st.stop()

# Load model
seg_model = load_yolo_model()
if seg_model is None:
    st.error("❌ Failed to load YOLO model")
    st.stop()

st.success("✅ All systems ready!")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📁 Upload Image")
    
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.session_state.selected_image = image
        
        # Detection button
        if st.button("🔍 DETECT OBJECTS", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                try:
                    # Convert image for YOLO
                    img_array = np.array(image)
                    
                    # Run detection
                    results = seg_model.predict(source=img_array, conf=0.4, verbose=False)
                    
                    if results and len(results) > 0:
                        result = results[0]
                        if hasattr(result, 'boxes') and len(result.boxes) > 0:
                            st.session_state.detection_results = result
                            
                            # Show annotated image
                            try:
                                annotated_img = result.plot()
                                st.image(annotated_img, caption="Detection Results", use_column_width=True)
                                st.success(f"✅ Found {len(result.boxes)} objects!")
                            except:
                                st.success(f"✅ Found {len(result.boxes)} objects!")
                        else:
                            st.warning("No objects detected")
                    else:
                        st.warning("No detection results")
                        
                except Exception as e:
                    st.error(f"Detection failed: {str(e)}")

with col2:
    st.subheader("🎛️ Controls")
    
    if st.session_state.detection_results is not None:
        results = st.session_state.detection_results
        
        if hasattr(results, 'boxes') and len(results.boxes) > 0:
            # Object selection
            object_names = []
            for cls in results.boxes.cls:
                try:
                    class_id = int(cls.item())  # Convert tensor to int
                    object_names.append(seg_model.names[class_id])
                except:
                    object_names.append(f"object_{len(object_names)}")
            
            # Remove duplicates while preserving order
            unique_objects = []
            seen = set()
            for obj in object_names:
                if obj not in seen:
                    unique_objects.append(obj)
                    seen.add(obj)
            
            if unique_objects:
                selected_obj = st.selectbox("📦 Select Object:", unique_objects)
                
                if selected_obj:
                    # Find index of selected object
                    try:
                        selected_idx = object_names.index(selected_obj)
                        
                        # Object category
                        living_objects = ['person', 'cat', 'dog', 'horse', 'sheep', 'cow']
                        category = "🐾 Living" if selected_obj.lower() in living_objects else "📦 Non-living"
                        st.write(f"**Category:** {category}")
                        
                        st.divider()
                        
                        # Calibration
                        st.write("**📏 Calibration**")
                        ref_width_cm = st.number_input("Known width (cm):", min_value=0.1, value=10.0, step=0.1)
                        ref_width_px = st.number_input("Width in pixels:", min_value=1, value=100, step=1)
                        
                        # Measure button
                        if st.button("📐 MEASURE", type="primary", use_container_width=True):
                            try:
                                pixels_per_cm = ref_width_px / ref_width_cm
                                
                                # Get bounding box
                                bbox = results.boxes.xyxy[selected_idx].cpu().numpy()
                                
                                # Calculate dimensions
                                dims = calculate_simple_dimensions(bbox, pixels_per_cm)
                                st.session_state.dimensions = dims
                                
                                st.success("✅ Measurement complete!")
                                
                                # Show results
                                st.subheader("📊 Results")
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Width", f"{dims['width_cm']:.2f} cm")
                                    st.metric("Height", f"{dims['height_cm']:.2f} cm")
                                with col_b:
                                    st.metric("Depth", f"{dims['depth_cm']:.2f} cm")
                                    st.metric("Volume", f"{dims['volume_cm3']:.2f} cm³")
                                
                                # Basic 3D visualization
                                if plotly_available:
                                    if st.button("🌟 SHOW 3D VIEW", type="secondary"):
                                        try:
                                            import plotly.graph_objects as go
                                            
                                            # Create simple 3D box representation
                                            w, h, d = dims['width_cm'], dims['height_cm'], dims['depth_cm']
                                            
                                            # Define box vertices
                                            x = [0, w, w, 0, 0, w, w, 0]
                                            y = [0, 0, h, h, 0, 0, h, h]
                                            z = [0, 0, 0, 0, d, d, d, d]
                                            
                                            # Create 3D scatter plot
                                            fig = go.Figure(data=go.Scatter3d(
                                                x=x, y=y, z=z,
                                                mode='markers+lines',
                                                marker=dict(size=8, color='cyan'),
                                                name=f'{selected_obj} ({w}×{h}×{d} cm)'
                                            ))
                                            
                                            fig.update_layout(
                                                title=f"3D View: {selected_obj}",
                                                scene=dict(
                                                    xaxis_title="Width (cm)",
                                                    yaxis_title="Height (cm)",
                                                    zaxis_title="Depth (cm)"
                                                )
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                        except Exception as e:
                                            st.error(f"3D view error: {str(e)}")
                                
                            except Exception as e:
                                st.error(f"Measurement error: {str(e)}")
                    
                    except Exception as e:
                        st.error(f"Selection error: {str(e)}")
        else:
            st.info("No objects detected in the image")
    else:
        st.info("Upload an image and click DETECT OBJECTS to begin")
    
    # Reset button
    if st.session_state.detection_results is not None:
        st.divider()
        if st.button("🔄 RESET", use_container_width=True):
            st.session_state.detection_results = None
            st.session_state.selected_image = None
            st.session_state.dimensions = None
            st.rerun()

# Footer
st.divider()
st.markdown("**📱 AI Dimension Estimator** - Upload → Detect → Select → Measure")
