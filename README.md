# AI-Powered-Smart-City-Traffic-Management-System
AI-Powered Smart City Traffic Management System

An AI-based traffic signal control system that uses YOLOv8 object detection, ByteTrack multi-object tracking, and Q-Learning (Reinforcement Learning) to dynamically manage traffic signals at a 4-way intersection based on real-time vehicle and pedestrian density.

🚦 Overview

Traditional traffic signals run on fixed timers, regardless of how much traffic is actually present. This project builds a smarter alternative:

A video feed of a 4-way intersection is split into 4 virtual lanes (quadrants).
YOLOv8 detects and tracks vehicles (car, motorcycle, bus, truck) and pedestrians in each lane.
A Q-Learning agent observes lane congestion and decides which lane should get the green signal next.
A signal controller applies the decision with proper minimum/maximum green time and yellow-light transitions.
Everything is visualized live on the video with an on-screen analytics dashboard.
✨ Features
Real-time vehicle & pedestrian detection using YOLOv8
Multi-object tracking with ByteTrack (persistent IDs across frames)
4-lane congestion analysis (LOW / MEDIUM / HIGH) based on live counts
Reinforcement Learning traffic signal control (Q-Learning) that adapts to traffic patterns over time
Persistent Q-table — the agent's learning is saved (q_table.pkl) and reused on the next run
Vehicle flow rate tracking (vehicles/min) per lane over a rolling time window
Average waiting time estimation per lane
Live annotated dashboard overlay: signal lights, lane density heatmap, and a stats panel
Optional output video export with all overlays baked in
🛠️ Tech Stack
Component	Technology
Object Detection	YOLOv8 (Ultralytics)
Object Tracking	ByteTrack
Decision Making	Q-Learning (custom RL agent)
Video Processing	OpenCV
Language	Python 3
Data Handling	NumPy, pickle
