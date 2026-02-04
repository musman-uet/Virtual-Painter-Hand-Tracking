# Virtual Painter - Hand Tracking

A virtual painting application that lets you draw in the air using hand gestures. Built with MediaPipe and OpenCV.

## Features
- Draw with your index finger
- Change colors (Red, Blue, Green)
- Eraser tool
- Adjustable brush size

## Demo
[Add your video link here or upload demo.mp4]

## Installation

1. Clone the repository:
```
git clone https://github.com/YOUR_USERNAME/Virtual-Painter-Hand-Tracking.git
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Download `hand_landmarker.task` from [Google MediaPipe](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) and place it in the root directory.

4. Run the app:
```
python painter.py
```

## Controls
- **Two fingers up (✌️):** Selection mode - change colors and brush size
- **One finger up (☝️):** Drawing mode
- **Press 'q':** Quit the application

## Technologies Used
- Python
- OpenCV
- MediaPipe
- NumPy
