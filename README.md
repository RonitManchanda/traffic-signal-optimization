# Traffic Signal Optimization with Computer Vision & Machine Learning

## Overview

This project focuses on building an intelligent traffic signal optimization system using computer vision and machine learning.

The system is developed in two major phases:

1. Learning OpenCV, vehicle detection, and multi-object tracking  
2. Using extracted traffic data to train machine learning models for signal optimization  

The goal is to go from raw traffic video → structured traffic data → optimized signal decisions.

---
## Project Details Portfolio
Portfolio Project Detail Page [Traffic Signal Optimization](https://ronitmanchanda.vercel.app/projects/traffic-signal-optimization).
## Project Goals

- Detect vehicles from intersection footage  
- Track each vehicle with a consistent ID  
- Classify movement by lane and direction  
- Count turning and straight-going vehicles accurately  
- Extract useful traffic features from video  
- Train a machine learning model for signal optimization  
- Move toward adaptive traffic signal control  

---

## Project Architecture

Traffic Video Input  
↓  
Vehicle Detection (YOLO)  
↓  
Multi-Object Tracking (BoT-SORT / ByteTrack)  
↓  
Lane + Direction Classification  
↓  
Traffic Metrics Extraction  
↓  
Machine Learning Model  
↓  
Signal Timing Optimization  

---

## Phase 1: OpenCV and Vehicle Tracking

### Objective

The first phase focuses on building a reliable computer vision system to understand traffic flow from video.

This includes:
- learning OpenCV fundamentals  
- detecting vehicles  
- tracking vehicles across frames  
- reducing ID switching  
- classifying direction and lanes  
- generating accurate traffic counts  

---

## Tools and Technologies

### Computer Vision and Tracking
- Python  
- OpenCV  
- Ultralytics YOLO  
- BoT-SORT  
- ByteTrack  
- NumPy  

### Future Machine Learning
- Pandas  
- Scikit-learn  
- PyTorch or TensorFlow  

---

## Core Components

### 1. Video Processing (OpenCV)

Used to:
- process video frame-by-frame  
- draw overlays and lane zones  
- visualize tracking IDs  
- debug motion and direction  

---

### 2. Vehicle Detection

YOLO is used to detect:
- cars  
- trucks  
- buses  
- motorcycles  

Key considerations:
- confidence threshold tuning  
- filtering weak detections  
- reducing noise  

---

### 3. Multi-Object Tracking

BoT-SORT / ByteTrack is used to:
- assign unique IDs to vehicles  
- maintain IDs across frames  
- recover IDs after occlusion  
- reduce duplicate tracks  

---

### 4. Tracker Tuning

Extensive testing was performed across parameters:

- track_high_thresh  
- track_low_thresh  
- new_track_thresh  
- track_buffer  
- match_thresh  

Metrics tracked:
- re-identification events  
- number of unique IDs  
- short tracks  
- average track length  

Results:
- reduced ID switching  
- fewer duplicate vehicles  
- longer, more stable tracks  

---

### 5. Lane Classification

Custom polygon regions define:
- turning lanes  
- straight lanes  

Vehicles are assigned lanes based on position and movement.

---

### 6. Direction Detection

Uses vector-based motion instead of simple x/y checks.

Example:

exit_vec = (-X_AXIS[0], -X_AXIS[1])

This allows:
- accurate direction detection with angled cameras  
- classification of turning vs straight vehicles  

---

### 7. Turn Validation

Movement is split into phases:
- approach phase  
- exit phase  

This ensures:
- correct turn detection  
- fewer false classifications  

---

### 8. Occlusion Handling

Handles real-world issues:
- vehicles overlapping  
- trucks blocking cars  
- temporary detection loss  

---

## Extracted Traffic Metrics

- total vehicle count  
- unique tracked vehicles  
- lane-based counts  
- turn vs straight ratios  
- track duration  
- traffic flow patterns  

These metrics will be used for machine learning.

---

## Phase 2: Machine Learning Optimization

### Objective

Use traffic data from tracking to optimize signal timing.

---

## Input Features

- vehicle count per lane  
- turn vs straight ratios  
- arrival rate  
- queue length  
- congestion level  
- movement speed  

---

## ML Approaches

### Supervised Learning
- predict optimal signal timing  

### Reinforcement Learning (future)
- learn optimal signal control policies  

### Hybrid Approach
- combine rules with ML  

---

## Example Output

Lane A: 30 seconds green  
Lane B: 20 seconds green  
Turn lane: 15 seconds green  

Dynamically adjusted based on traffic.

---

## Challenges

### Tracking Issues
- ID switching  
- occlusions  
- detection inconsistency  

Solutions:
- increased match_thresh  
- adjusted track_buffer  
- tuned new_track_thresh  

---

### Direction and Lane Issues
- camera angle distortion  
- lane overlap  

Solutions:
- vector-based motion analysis  
- polygon lane segmentation  

---

### Performance Issues
- video lag  
- processing overhead  

Solutions:
- optimized thresholds  
- efficient frame handling  

---

## Future Improvements

- add ReID model for stronger identity tracking  
- improve lane detection with segmentation  
- build real-time analytics dashboard  
- implement reinforcement learning  
- deploy as live system  

---

## Repository Structure

traffic-signal-optimization/
│
├── tracking/
│   ├── detection.py
│   ├── tracking.py
│   ├── lane_logic.py
│   ├── direction.py
│   └── tuning/
│
├── ml/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── optimization/
│
├── assets/
│   └── videos/
│
├── temp_botsort_configs/
├── botsort_yaml_tuning_details/
├── botsort_yaml_tuning_summary.csv
└── README.md

---

## Why This Project Matters

Traffic congestion is a real-world problem.

This project demonstrates:
- computer vision in real environments  
- multi-object tracking under uncertainty  
- data-driven optimization  
- applied machine learning  

---

## Key Takeaway

This project is about turning raw visual data into structured insights and then into intelligent decision-making systems.

---

## Status

Active development  

Current focus:
- improving tracking stability  
- reducing ID fragmentation  
- refining lane classification  
- preparing data for machine learning  

---

## Author

Ronit Manchanda  

GitHub: https://github.com/RonitManchanda  
Portfolio: https://ronitmanchanda.vercel.app  