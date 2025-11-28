import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# Page config
st.set_page_config(
    page_title="Radar-Camera Fusion Object Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Radar-Camera Fusion Object Detection System")

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    model_type = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt"])
    
    st.info("Note: Radar panel is currently a placeholder as per requirements.")
    simulate_radar = st.checkbox("Simulate Radar Data", value=False)

# Load model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

try:
    model = load_model(model_type)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Layout: 2 Columns
col1, col2 = st.columns(2)

with col1:
    st.header("Camera Feed (Live Detection)")
    camera_placeholder = st.empty()
    run_detection = st.checkbox("Start Camera Detection")

with col2:
    st.header("Radar Data")
    radar_placeholder = st.empty()
    if not simulate_radar:
        radar_placeholder.markdown(
            """
            <div style="
                border: 2px dashed #444; 
                border-radius: 10px; 
                height: 400px; 
                display: flex; 
                align-items: center; 
                justify-content: center; 
                background-color: #1E1E1E;
                color: #888;
            ">
                <h3>Radar Data Placeholder</h3>
                <p>(No Radar Input Connected)</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Main Detection Loop
if run_detection:
    cap = cv2.VideoCapture(0)
    
    # Initialize Modules
    from radar_module import MockRadarDetector
    from fusion_module import FusionEngine
    
    radar = MockRadarDetector()
    fusion = FusionEngine()
    
    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        stop_button = st.button("Stop Detection")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame")
                break
            
            # Run inference
            results = model(frame, conf=confidence_threshold)
            camera_detections = results[0].boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]
            
            # Radar Simulation
            radar_detections = []
            if simulate_radar:
                radar_detections = radar.detect(camera_objects=camera_detections)
            
            # Fusion
            fused_objects = fusion.fuse(camera_detections, radar_detections)
            
            # Visualize results on the frame
            # We draw our own boxes to include radar info
            annotated_frame = frame.copy()
            
            for obj in fused_objects:
                bbox = obj['bbox'].astype(int)
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                label = f"{model.names[obj['class_id']]} {obj['conf']:.2f}"
                if obj['radar_data']:
                    # Append Radar Info
                    r_data = obj['radar_data']
                    label += f" | R:{r_data['range']:.1f}m V:{r_data['velocity']:.1f}m/s"
                    # Draw a different color for fused objects
                    cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 165, 255), 2) # Orange
                
                cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert BGR to RGB for Streamlit
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display
            camera_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            
            # Update Radar Panel (Only if simulation is on, but keep it minimal/text based as requested)
            if simulate_radar:
                with radar_placeholder.container():
                     st.write(f"**Radar Objects Detected:** {len(radar_detections)}")
                     if radar_detections:
                         st.dataframe(radar_detections)
            else:
                 radar_placeholder.empty()
                 radar_placeholder.markdown(
                    """
                    <div style="
                        border: 2px dashed #444; 
                        border-radius: 10px; 
                        height: 400px; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        background-color: #1E1E1E;
                        color: #888;
                    ">
                        <h3>Radar Data Placeholder</h3>
                        <p>(No Radar Input Connected)</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
        cap.release()
else:
    camera_placeholder.info("Click 'Start Camera Detection' to begin.")
