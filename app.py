

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle
from pathlib import Path
import time

from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
from PIL import Image

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Face Attendance System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #145a8d;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("attendance_data")
EMBEDDINGS_FILE = DATA_DIR / "users.pkl"
ATTENDANCE_FILE = DATA_DIR / "attendance.csv"

RECOGNITION_THRESHOLD = 0.6
MIN_FACE_SIZE = 80

# Create data directory
DATA_DIR.mkdir(exist_ok=True)

# =============================================================================
# INITIALIZE SESSION STATE
# =============================================================================
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.face_detector = None
    st.session_state.face_recognizer = None

# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource
def load_models():
    """Load face detection and recognition models"""
    try:
        face_detector = MTCNN()
        face_recognizer = FaceNet()
        return face_detector, face_recognizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Load models
if not st.session_state.models_loaded:
    with st.spinner("Loading AI models... This may take a moment."):
        face_detector, face_recognizer = load_models()
        if face_detector and face_recognizer:
            st.session_state.face_detector = face_detector
            st.session_state.face_recognizer = face_recognizer
            st.session_state.models_loaded = True

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def detect_face(image):
    """Detect and return the largest face in the image"""
    detections = st.session_state.face_detector.detect_faces(image)
    
    if not detections:
        return None
    
    # Get largest face
    detection = max(detections, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = detection['box']
    
    # Check minimum size
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return None
    
    # Extract face with padding
    padding = int(0.2 * max(w, h))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)
    
    face = image[y1:y2, x1:x2]
    
    # Resize to 160x160 (required by FaceNet)
    face = cv2.resize(face, (160, 160))
    
    return face, (x, y, w, h)

def get_embedding(face_image):
    """Get face embedding vector"""
    # Convert to RGB
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_rgb = np.expand_dims(face_rgb, axis=0)
    
    # Get embedding
    embedding = st.session_state.face_recognizer.embeddings(face_rgb)[0]
    
    # Normalize
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding

def load_users():
    """Load registered users from file"""
    if EMBEDDINGS_FILE.exists():
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_users(users):
    """Save registered users to file"""
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(users, f)

def init_attendance_file():
    """Initialize attendance CSV file"""
    if not ATTENDANCE_FILE.exists():
        df = pd.DataFrame(columns=['User', 'Date', 'Punch-in', 'Punch-out'])
        df.to_csv(ATTENDANCE_FILE, index=False)

def recognize_face(face_image):
    """Recognize a face and return username"""
    users = load_users()
    
    if not users:
        return None
    
    # Get embedding for the face
    embedding = get_embedding(face_image)
    
    # Find best match
    best_match = None
    best_similarity = 0
    
    for username, stored_embedding in users.items():
        similarity = 1 - cosine(embedding, stored_embedding)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = username
    
    # Check threshold
    if best_similarity >= (1 - RECOGNITION_THRESHOLD):
        return best_match, best_similarity
    
    return None

def mark_attendance(username, confidence):
    """Mark attendance for a user"""
    init_attendance_file()
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Load attendance
    df = pd.read_csv(ATTENDANCE_FILE)
    
    # Check today's record
    today = df[(df['User'] == username) & (df['Date'] == current_date)]
    
    if today.empty:
        # First time today - Punch in
        new_row = pd.DataFrame([{
            'User': username,
            'Date': current_date,
            'Punch-in': current_time,
            'Punch-out': ''
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        
        return "punch-in", current_time
        
    else:
        # Check if already punched out
        last = today.iloc[-1]
        
        if pd.isna(last['Punch-out']) or last['Punch-out'] == '':
            # Punch out
            idx = df[(df['User'] == username) & 
                    (df['Date'] == current_date) &
                    (df['Punch-out'].isna() | (df['Punch-out'] == ''))].index[-1]
            
            df.at[idx, 'Punch-out'] = current_time
            df.to_csv(ATTENDANCE_FILE, index=False)
            
            return "punch-out", current_time
        else:
            # New punch-in
            new_row = pd.DataFrame([{
                'User': username,
                'Date': current_date,
                'Punch-in': current_time,
                'Punch-out': ''
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(ATTENDANCE_FILE, index=False)
            
            return "punch-in-again", current_time

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">👤 Face Attendance System</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["📸 Mark Attendance", "➕ Register User", "📊 View Attendance", "👥 Registered Users"]
    )
    
    # Check if models are loaded
    if not st.session_state.models_loaded:
        st.error("⚠️ Models failed to load. Please refresh the page.")
        return
    
    # Page routing
    if page == "📸 Mark Attendance":
        mark_attendance_page()
    elif page == "➕ Register User":
        register_user_page()
    elif page == "📊 View Attendance":
        view_attendance_page()
    elif page == "👥 Registered Users":
        list_users_page()

# =============================================================================
# PAGE FUNCTIONS
# =============================================================================

def mark_attendance_page():
    st.header("📸 Mark Attendance")
    st.markdown("Capture your face to mark attendance (punch-in/punch-out)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Camera input
        img_file = st.camera_input("Position your face in the camera")
        
        if img_file is not None:
            # Read image
            bytes_data = img_file.getvalue()
            image = Image.open(img_file)
            img_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Detect face
            with st.spinner("Detecting face..."):
                result = detect_face(img_bgr)
            
            if result:
                face, (x, y, w, h) = result
                
                # Draw rectangle on image
                display_img = img_array.copy()
                cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                with st.spinner("Recognizing..."):
                    recognition = recognize_face(face)
                
                if recognition:
                    username, confidence = recognition
                    
                    # Mark attendance
                    action, time_stamp = mark_attendance(username, confidence)
                    
                    # Display success message
                    if action == "punch-in":
                        st.markdown(
                            f'<div class="success-box">✅ <strong>PUNCH-IN</strong><br>'
                            f'Welcome, {username}!<br>'
                            f'Time: {time_stamp}<br>'
                            f'Confidence: {confidence:.1%}</div>',
                            unsafe_allow_html=True
                        )
                    elif action == "punch-out":
                        st.markdown(
                            f'<div class="success-box">✅ <strong>PUNCH-OUT</strong><br>'
                            f'Goodbye, {username}!<br>'
                            f'Time: {time_stamp}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="success-box">✅ <strong>PUNCH-IN (RETURN)</strong><br>'
                            f'Welcome back, {username}!<br>'
                            f'Time: {time_stamp}</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Display image with box
                    st.image(display_img, caption="Detected Face", use_container_width=True)
                    
                else:
                    st.markdown(
                        '<div class="error-box">❌ <strong>Face not recognized</strong><br>'
                        'Please register first or try again with better lighting.</div>',
                        unsafe_allow_html=True
                    )
                    st.image(display_img, caption="Unknown Face", use_container_width=True)
            else:
                st.markdown(
                    '<div class="error-box">❌ <strong>No face detected</strong><br>'
                    'Please ensure your face is clearly visible and well-lit.</div>',
                    unsafe_allow_html=True
                )
    
    with col2:
        st.markdown("### 📋 Instructions")
        st.info(
            "1. Position your face in the camera\n"
            "2. Click the camera button\n"
            "3. Wait for recognition\n"
            "4. First capture = Punch-in\n"
            "5. Second capture = Punch-out"
        )
        
        # Show registered users count
        users = load_users()
        st.metric("Registered Users", len(users))

def register_user_page():
    st.header("➕ Register New User")
    st.markdown("Register a new face for attendance tracking")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Username input
        username = st.text_input("Enter username:", placeholder="e.g., john_doe")
        
        if username:
            # Check if user already exists
            users = load_users()
            if username in users:
                st.warning(f"⚠️ User '{username}' already exists!")
            else:
                # Camera input
                img_file = st.camera_input("Capture your face")
                
                if img_file is not None:
                    # Read image
                    bytes_data = img_file.getvalue()
                    image = Image.open(img_file)
                    img_array = np.array(image)
                    
                    # Convert RGB to BGR for OpenCV
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # Detect face
                    with st.spinner("Detecting face..."):
                        result = detect_face(img_bgr)
                    
                    if result:
                        face, (x, y, w, h) = result
                        
                        # Draw rectangle
                        display_img = img_array.copy()
                        cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        
                        st.image(display_img, caption="Detected Face", use_container_width=True)
                        
                        # Register button
                        if st.button("✅ Confirm Registration", type="primary"):
                            with st.spinner("Registering..."):
                                # Get embedding
                                embedding = get_embedding(face)
                                
                                # Save user
                                users[username] = embedding
                                save_users(users)
                                
                                st.markdown(
                                    f'<div class="success-box">✅ <strong>Registration Successful!</strong><br>'
                                    f'User "{username}" has been registered.</div>',
                                    unsafe_allow_html=True
                                )
                                time.sleep(2)
                                st.rerun()
                    else:
                        st.markdown(
                            '<div class="error-box">❌ <strong>No face detected</strong><br>'
                            'Please ensure your face is clearly visible and well-lit.</div>',
                            unsafe_allow_html=True
                        )
    
    with col2:
        st.markdown("### 📋 Instructions")
        st.info(
            "1. Enter your username\n"
            "2. Position your face clearly\n"
            "3. Ensure good lighting\n"
            "4. Click camera button\n"
            "5. Confirm registration"
        )
        
        st.markdown("### ⚠️ Tips")
        st.warning(
            "• Use a unique username\n"
            "• Face the camera directly\n"
            "• Remove sunglasses/masks\n"
            "• Ensure good lighting"
        )

def view_attendance_page():
    st.header("📊 View Attendance Records")
    
    init_attendance_file()
    
    # Date selector
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_date = st.date_input("Select date:", datetime.now())
    
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    with col3:
        filter_option = st.selectbox("Filter:", ["All", "Punch-in Only", "Punch-out Only"])
    
    # Load attendance
    df = pd.read_csv(ATTENDANCE_FILE)
    
    if not df.empty:
        # Filter by date
        date_str = selected_date.strftime("%Y-%m-%d")
        filtered_df = df[df['Date'] == date_str].copy()
        
        if not filtered_df.empty:
            # Apply additional filters
            if filter_option == "Punch-in Only":
                filtered_df = filtered_df[filtered_df['Punch-in'].notna()]
            elif filter_option == "Punch-out Only":
                filtered_df = filtered_df[filtered_df['Punch-out'].notna()]
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_entries = len(filtered_df)
                st.metric("Total Entries", total_entries)
            
            with col2:
                unique_users = filtered_df['User'].nunique()
                st.metric("Unique Users", unique_users)
            
            with col3:
                completed = len(filtered_df[filtered_df['Punch-out'].notna() & (filtered_df['Punch-out'] != '')])
                st.metric("Completed Records", completed)
            
            # Display table
            st.markdown(f"### Attendance for {date_str}")
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"attendance_{date_str}.csv",
                mime="text/csv"
            )
        else:
            st.info(f"ℹ️ No attendance records found for {date_str}")
    else:
        st.info("ℹ️ No attendance records available")
    
    # Show all-time statistics
    if not df.empty:
        st.markdown("---")
        st.markdown("### 📈 All-Time Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_records = len(df)
            st.metric("Total Records", total_records)
        
        with col2:
            unique_dates = df['Date'].nunique()
            st.metric("Active Days", unique_dates)
        
        with col3:
            all_users = df['User'].nunique()
            st.metric("All Users", all_users)
        
        with col4:
            completed_all = len(df[df['Punch-out'].notna() & (df['Punch-out'] != '')])
            st.metric("Completed", completed_all)

def list_users_page():
    st.header("👥 Registered Users")
    
    users = load_users()
    
    if users:
        st.success(f"✅ Total registered users: **{len(users)}**")
        
        # Create a dataframe
        user_df = pd.DataFrame({
            'No.': range(1, len(users) + 1),
            'Username': list(users.keys())
        })
        
        # Display in a nice table
        st.dataframe(
            user_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Delete user section
        st.markdown("---")
        st.markdown("### 🗑️ Delete User")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_to_delete = st.selectbox(
                "Select user to delete:",
                options=[""] + list(users.keys())
            )
        
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if user_to_delete and st.button("🗑️ Delete", type="primary"):
                del users[user_to_delete]
                save_users(users)
                st.success(f"✅ User '{user_to_delete}' deleted!")
                time.sleep(1)
                st.rerun()
    else:
        st.info("ℹ️ No users registered yet. Please register users first!")
        
        if st.button("➕ Go to Registration"):
            st.session_state.page = "Register User"
            st.rerun()

# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    main()