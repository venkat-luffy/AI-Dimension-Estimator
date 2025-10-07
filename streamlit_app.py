import streamlit as st
import av
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from PIL import Image

# --- Import custom utility modules ---
import depth_utils
import dimension_utils
import hologram_utils

# --- Model Loading ---
@st.cache_resource
def load_all_models():
    seg_model = YOLO("model/yolov8n-seg.pt")
    depth_model, depth_transform, device = depth_utils.load_depth_model()
    return seg_model, depth_model, depth_transform, device

seg_model, depth_model, depth_transform, device = load_all_models()

# --- Session State Initialization ---
if "mode" not in st.session_state:
    st.session_state.mode = "scanning"
    st.session_state.captured_frame = None
    st.session_state.detection_results = None
    st.session_state.selected_obj_idx = None
    st.session_state.dimensions = None

# --- Frame Processing ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    results = seg_model.predict(source=img, conf=0.4, verbose=False)[0]
    annotated_img = results.plot()
    if st.session_state.mode == "scanning":
        st.session_state.detection_results = results
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# --- Layout ---
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Live Camera Feed")
    ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_receiver and st.session_state.mode == "scanning":
        if st.button("Pause Feed to Select & Measure", use_container_width=True):
            try:
                latest_frame = ctx.video_receiver.get_latest_frame()
                paused_img = latest_frame.to_ndarray(format="bgr24")
                paused_results = seg_model.predict(source=paused_img, conf=0.4, verbose=False)[0]
                st.session_state.captured_frame = paused_img
                st.session_state.detection_results = paused_results
                st.session_state.mode = "paused"
                st.rerun()
            except Exception as e:
                st.error(f"Could not capture frame: {e}")

with col2:
    st.header("Controls & Results")
    if st.session_state.mode == "scanning":
        if st.session_state.detection_results:
            st.subheader("Objects Detected:")
            detected_names = {seg_model.names[int(cls)] for cls in st.session_state.detection_results.boxes.cls}
            for name in detected_names:
                st.markdown(f"- {name.capitalize()}")

    elif st.session_state.mode == "paused":
        results = st.session_state.detection_results
        if not results or len(results.boxes) == 0:
            st.warning("No objects were detected in the captured frame.")
            if st.button("Resume Scanning"):
                st.session_state.mode = "scanning"
                st.rerun()
        else:
            st.info("Feed paused. Select an object from the list to measure.")
            object_names = [seg_model.names[int(cls)] for cls in results.boxes.cls]
            selected_obj_name = st.selectbox("Select Object Tag:", options=list(set(object_names)))
            st.session_state.selected_obj_idx = object_names.index(selected_obj_name)

            st.divider()
            st.subheader("Calibration")
            st.warning("Manual calibration is required for now.")
            ref_width_cm = st.number_input("Enter a known real-world width (cm) for a reference object:", min_value=0.1, value=8.56, key="calib_cm")
            ref_width_px = st.number_input("Enter the corresponding width in pixels:", min_value=1, value=300, key="calib_px")

            if st.button("Measure Selected Object", use_container_width=True):
                with st.spinner("Calculating..."):
                    pixels_per_cm = ref_width_px / ref_width_cm
                    mask = results.masks.data[st.session_state.selected_obj_idx].cpu().numpy()
                    pil_img = Image.fromarray(cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB))
                    depth_map = depth_utils.get_depth_map(pil_img, depth_model, depth_transform, device)
                    dims = dimension_utils.get_dimensions_with_depth(mask, depth_map, pixels_per_cm)
                    st.session_state.dimensions = dims
                    st.session_state.mode = "measured"
                    st.rerun()

            if st.button("Resume Scanning"):
                st.session_state.mode = "scanning"
                st.rerun()

    elif st.session_state.mode == "measured":
        st.success("Measurement complete!")
        dims = st.session_state.dimensions
        if dims:
            st.subheader("Dimension Results")
            st.metric("Width", f"{dims['width_cm']:.2f} cm")
            st.metric("Height", f"{dims['height_cm']:.2f} cm")
            st.metric("Est. Depth", f"{dims['depth_cm']:.2f} cm")

            if st.button("Show Holographic View", type="primary", use_container_width=True):
                with st.spinner("Generating 3D view..."):
                    results = st.session_state.detection_results
                    mask = results.masks.data[st.session_state.selected_obj_idx].cpu().numpy()
                    pil_img = Image.fromarray(cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB))
                    depth_map = depth_utils.get_depth_map(pil_img, depth_model, depth_transform, device)
                    fig = hologram_utils.create_holographic_view(mask, depth_map)
                    st.plotly_chart(fig, use_container_width=True)

        if st.button("Start New Scan"):
            st.session_state.mode = "scanning"
            st.rerun()
