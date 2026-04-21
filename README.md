# Traffic Signal Optimization System

## Overview
This project analyzes traffic video using computer vision to detect vehicles, track movement, and optimize signal timing.

## Features
- Vehicle detection using YOLO
- Multi-object tracking (BoT-SORT)
- Lane-based direction classification
- Turning vs straight movement detection
- Traffic flow analysis

## Tech Stack
- Python
- OpenCV
- YOLO
- NumPy

## How It Works
1. Detect vehicles in each frame
2. Track vehicles across frames
3. Determine direction using coordinate motion
4. Classify lane and turning behavior
5. Aggregate traffic flow data

## Future Improvements
- Real-time signal control
- ML-based optimization
- Multi-intersection coordination