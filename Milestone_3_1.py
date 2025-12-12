import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Autonomous Car Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS for Professional Dark UI ---
st.markdown("""
<style>
    /* Global Settings */
    .stApp {
        background-color: #0E1117; /* Very Dark Blue-Grey */
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #E0E0E0 !important; /* Off-White for readability */
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161B22; /* Slightly lighter dark */
        border-right: 1px solid #30363D;
    }
    
    /* Stats Cards (KPIs) Styling */
    .metric-card {
        background-color: #21262D; /* Card Background */
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 14px;
        color: #8B949E; /* Muted Grey */
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #58A6FF; /* Tech Blue */
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #238636; /* GitHub Green */
        color: white;
        border: none;
        border-radius: 5px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }

</style>
""", unsafe_allow_html=True)

# --- 3. Sidebar (Controls) ---
st.sidebar.title("🛠 Control Panel")
st.sidebar.markdown("---")

# Model Settings
st.sidebar.subheader("Model Configuration")
model_path = st.sidebar.text_input("Model Path:", "yolov8n.pt")
try:
    model = YOLO(model_path)
    st.sidebar.success("Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"Error: {e}")

confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")

# Source Settings
st.sidebar.subheader("Input Source")
source_type = st.sidebar.radio("Select Source:", ["Image", "Video", "Webcam"])

# --- 4. Main Dashboard Layout ---

st.title("🚗 Autonomous Car Vision System")
st.markdown("Real-time object detection powered by YOLOv8")

# Create two columns: Left for Display, Right for Stats
col1, col2 = st.columns([0.75, 0.25])

# Placeholder for Stats (We initialize them to keep structure)
with col2:
    st.markdown("### 📊 Live Statistics")
    
    # Placeholders for dynamic updates using the CSS Card style
    kpi1_placeholder = st.empty()
    kpi2_placeholder = st.empty()
    kpi3_placeholder = st.empty()
    
    st.markdown("---")
    st.markdown("### 📝 Detected Classes")
    detected_objects_placeholder = st.empty()

# --- Helper Function to Render Cards ---
def update_kpi(placeholder, title, value, color="#58A6FF"):
    placeholder.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color: {color}">{value}</div>
        </div>
    """, unsafe_allow_html=True)

# Initial Stats Render
update_kpi(kpi1_placeholder, "FPS", "0", "#58A6FF")
update_kpi(kpi2_placeholder, "Objects", "0", "#A371F7") # Purple
update_kpi(kpi3_placeholder, "System Status", "IDLE", "#8B949E") # Grey

# --- 5. Processing Logic ---

def process_frame(frame):
    """Core function to process a single frame"""
    start_time = time.time()
    
    # YOLO Prediction
    results = model.predict(frame, conf=confidence)
    res_plotted = results[0].plot()
    
    # Calculate Stats
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    object_count = len(results[0].boxes)
    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
    
    return res_plotted, fps, object_count, detected_classes

def video_handler(cap):
    st_frame = st.empty()
    stop_button = st.button("🔴 Stop Processing")
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("Video stream ended.")
            break
            
        # Process
        processed_frame, fps, obj_count, classes = process_frame(frame)
        
        # Update UI Stats
        update_kpi(kpi1_placeholder, "FPS", f"{int(fps)}", "#58A6FF")       # Blue for FPS
        update_kpi(kpi2_placeholder, "Objects Detected", f"{obj_count}", "#A371F7") # Purple for Count
        update_kpi(kpi3_placeholder, "System Status", "ACTIVE", "#3FB950")  # Green for Status
        
        # Update Class List
        unique_objects = {obj: classes.count(obj) for obj in set(classes)}
        detected_objects_placeholder.json(unique_objects)

        # Display Image
        st_frame.image(processed_frame, channels="BGR", use_container_width=True)
    
    cap.release()
    update_kpi(kpi3_placeholder, "System Status", "STOPPED", "#F85149")

# --- 6. Execution based on Source ---

if source_type == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1.image(image, caption='Original Input', use_container_width=True)
        
        if st.sidebar.button('🚀 Start Detection'):
            img_array = np.array(image)
            processed_frame, _, obj_count, classes = process_frame(img_array)
            
            col1.image(processed_frame, caption='Processed Output', use_container_width=True)
            
            # Update Stats
            update_kpi(kpi2_placeholder, "Objects Detected", f"{obj_count}", "#A371F7")
            update_kpi(kpi3_placeholder, "System Status", "COMPLETED", "#3FB950")
            
            unique_objects = {obj: classes.count(obj) for obj in set(classes)}
            detected_objects_placeholder.write(unique_objects)

elif source_type == "Video":
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    if uploaded_file:
        tfile = open("temp_video.mp4", "wb") 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture("temp_video.mp4")
        
        with col1:
            video_handler(cap)

elif source_type == "Webcam":
    if st.sidebar.button("🎥 Start Webcam"):
        cap = cv2.VideoCapture(0)
        with col1:
            video_handler(cap)