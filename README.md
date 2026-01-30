#Face Authentication Attendance System

A face recognition-based attendance system using MTCNN for face detection and FaceNet for embedding-based recognition. Supports real-time punch-in/punch-out logging via webcam with basic lighting normalization and modular code.

Table of Contents

Demo

Features

System Overview

Model and Approach

Face Embeddings

Training Process

Accuracy Expectations

Known Failure Cases

Installation

Usage

Future Improvements

Demo

Local demo: Run the system on your laptop using a webcam.

Hosted demo (optional): Can be extended with Flask/Streamlit for web deployment.

Features

Register users with face embeddings

Real-time recognition for punch-in/punch-out

Attendance logs in CSV format

Visual feedback: bounding boxes & confidence scores

Cooldown logic to prevent duplicate entries

System Overview
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

Model and Approach
Face Detection: MTCNN

Detects faces and landmarks in a single step

Provides aligned faces → improves embedding quality

Robust to moderate variations in scale, position, and lighting

Lightweight, suitable for real-time webcam applications

Why not YOLO?

YOLO detects faces as objects but doesn’t provide landmarks

Requires additional alignment for FaceNet

MTCNN is optimized for face recognition pipelines

Face Recognition: FaceNet

Converts each face into a 512-D embedding vector

Embeddings are normalized to unit length

Uses cosine similarity to identify users

Threshold-based recognition → decides Punch-in / Punch-out

Face Embeddings Explained

Each face is represented by 512 numbers → a “fingerprint” of the face

Why 512?

Captures subtle facial features: eyes, nose, mouth, jawline, contours

Balances accuracy vs. memory & computation

Used in Inception-ResNetV1 backbone for robust, multi-scale features

Cosine similarity compares embeddings:

High similarity → same person

Low similarity → different person

Normalized embeddings lie on a unit hypersphere for reliable comparison

Training Process

Uses pretrained FaceNet model (no new CNN training required)

For new users: embeddings are computed and saved

Recognition relies on metric learning principle:

Faces from the same person → embeddings close together

Faces from different people → embeddings far apart

Accuracy Expectations

Depends on:

Face alignment (MTCNN)

Lighting conditions

Proper threshold (RECOGNITION_THRESHOLD = 0.6)

Typical accuracy: 95–98% for frontal, well-lit faces

Known Failure Cases

Poor lighting (too dim or backlit)

Extreme head angles or partial occlusion

Multiple faces (currently only largest face is recognized)

Spoofing attacks (photo/video)

Faces smaller than MIN_FACE_SIZE ignored

Note: Basic pre-processing like normalization and histogram equalization is applied to improve robustness.

Installation
git clone https://github.com/<username>/Face-Attendance-System.git
cd Face-Attendance-System
pip install -r requirements.txt


Dependencies:

opencv-python

numpy

pandas

mtcnn

keras_facenet

scipy

Usage
python main.py


Menu Options:

Register new user

Start attendance system (webcam)

View today’s attendance

List registered users

Exit

Controls:

Press 'c' → capture face during registration

Press 'q' → quit webcam feed

Future Improvements

Detect and log multiple users simultaneously

Implement basic liveness detection (blink/head movement)

Better lighting normalization for extreme conditions

GUI for desktop or web deployment
