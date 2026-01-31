# Face Authentication Attendance System

A real-time face recognition-based attendance system using MTCNN for face detection and FaceNet for embedding-based recognition. Features automatic punch-in/punch-out logging via webcam with lighting normalization and modular architecture.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [System Overview](#system-overview)
- [Model and Approach](#model-and-approach)
  - [Face Detection: MTCNN](#face-detection-mtcnn)
  - [Face Recognition: FaceNet](#face-recognition-facenet)
- [Face Embeddings Explained](#face-embeddings-explained)
- [Training Process](#training-process)
- [Accuracy Expectations](#accuracy-expectations)
- [Known Failure Cases](#known-failure-cases)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Demo

- **Local demo**: Run the system on your laptop using a webcam
- **Hosted demo** *(optional)*: Can be extended with Flask/Streamlit for web deployment

## Features

✅ **User Registration** - Register users with face embeddings  
✅ **Real-time Recognition** - Instant punch-in/punch-out detection  
✅ **Attendance Logging** - Automatic CSV-based attendance records  
✅ **Visual Feedback** - Bounding boxes with confidence scores  
✅ **Duplicate Prevention** - Cooldown logic to prevent duplicate entries  
✅ **Lighting Normalization** - Basic preprocessing for varied lighting conditions  
✅ **Modular Design** - Clean, maintainable codebase

## System Overview

```
Webcam frame
     ↓
MTCNN → Detect faces and facial landmarks
     ↓
Align and crop faces
     ↓
FaceNet → Generate 512-D embeddings
     ↓
Compare embeddings with registered users
     ↓
Mark attendance in CSV
```

## Model and Approach

### Face Detection: MTCNN

**Multi-task Cascaded Convolutional Networks (MTCNN)** handles face detection with the following advantages:

- Detects faces and landmarks in a single step
- Provides aligned faces → improves embedding quality
- Robust to moderate variations in scale, position, and lighting
- Lightweight and suitable for real-time webcam applications

**Why not YOLO?**
- YOLO detects faces as objects but doesn't provide facial landmarks
- Requires additional alignment steps for FaceNet
- MTCNN is specifically optimized for face recognition pipelines

## Training Process

The system uses a **pretrained FaceNet model** (no new CNN training required):

1. **For new users**: Embeddings are computed from captured images and saved
2. **Recognition** relies on the **metric learning principle**:
   - Faces from the **same person** → embeddings cluster together
   - Faces from **different people** → embeddings are far apart

Accuracy depends on:
- Face alignment quality (MTCNN performance)
- Lighting conditions
- Proper threshold calibration (`RECOGNITION_THRESHOLD = 0.6`)
- Distance from camera
- Face image quality

## Known Failure Cases

⚠️ **Poor lighting** (too dim, backlit, or harsh shadows)  
⚠️ **Extreme head angles** or partial occlusion (masks, hands, hair)  
⚠️ **Multiple faces** (currently only largest face is recognized)  
⚠️ **Spoofing attacks** (photos or videos of a person)  
⚠️ **Small faces** (faces smaller than `MIN_FACE_SIZE` are ignored)

**Note**: Basic preprocessing like normalization and histogram equalization is applied to improve robustness.

## Installation

### Prerequisites

- Python 3.7 or higher
- Webcam/camera device
- Operating System: Windows, macOS, or Linux

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Face-Attendance-System.git
cd Face-Attendance-System

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
opencv-python
numpy
pandas
mtcnn
keras-facenet
scipy
```

## Usage

### Starting the System

```bash
python main.py
```

### Menu Options

When you run the system, you'll see the following options:

```
1. Register new user
2. Start attendance system (webcam)
3. View today's attendance
4. List registered users
5. Exit
```

### Controls

**During Registration:**
- Press **'c'** → Capture face for registration
- Press **'q'** → Quit registration process

**During Attendance:**
- Press **'q'** → Quit attendance mode

### Workflow

1. **Register Users**: Select option 1 and capture face images
2. **Start Attendance**: Select option 2 to begin real-time recognition
3. **View Logs**: Select option 3 to see today's attendance records


## Configuration

Key parameters can be adjusted in `main.py` or configuration file:

```python
RECOGNITION_THRESHOLD = 0.6   # Similarity threshold (0.0 - 1.0)
MIN_FACE_SIZE = 40           # Minimum face size in pixels
COOLDOWN_SECONDS = 300       # Time between duplicate entries (5 min)
```

## Future Improvements

- [ ] Detect and log multiple users simultaneously
- [ ] Implement basic liveness detection (blink/head movement)
- [ ] Advanced lighting normalization for extreme conditions
- [ ] GUI for desktop application (Tkinter/PyQt)
- [ ] Web deployment (Flask/Streamlit)
- [ ] Export attendance to Excel/PDF
- [ ] Email notifications for attendance
- [ ] Admin dashboard with analytics
- [ ] Mobile app integration
- [ ] Cloud storage for embeddings
