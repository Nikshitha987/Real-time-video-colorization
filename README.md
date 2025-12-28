# Real-Time Video Colorization

## Overview
This project implements **real-time video colorization**, allowing grayscale videos or webcam streams to be colorized on the fly using deep learning models.  
It leverages **OpenCV** and **pretrained Caffe models** to generate realistic colors for black-and-white videos.

The project includes a **GUI** to display live colorized video and allows users to switch between multiple colorization models.

---

## Features
- Colorize grayscale videos in real time  
- Support for webcam input  
- Multiple pretrained colorization models  
- GUI to view and switch between models dynamically  
- High-quality colorization using deep learning  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Nikshitha987/Real-time-video-colorization.git
cd Real-time-video-colorization
Create a Python virtual environment (recommended):

bash
Copy code
python -m venv venv
Activate the virtual environment:

Windows (PowerShell):

bash
Copy code
venv\Scripts\Activate.ps1
Windows (CMD):

bash
Copy code
venv\Scripts\activate
Linux / macOS:

bash
Copy code
source venv/bin/activate
Install required packages:

bash
Copy code
pip install -r requirements.txt
Typical dependencies: opencv-python, numpy, Pillow, etc.

Usage
1. Colorize a Video File
bash
Copy code
python app.py --video path/to/grayscale_video.mp4
2. Real-Time Webcam Colorization
bash
Copy code
python app.py --webcam
A GUI window will open displaying the colorized video.

Users can switch between models if multiple are available.

Press q to quit the application.

Pretrained Models
Important: Large model files are not included due to GitHub size limits.

Required files (place inside the models/ folder):

colorization_deploy_v2.prototxt

colorization_release_v2.caffemodel (download from OpenCV)

pts_in_hull.npy

Directory structure:

Copy code
models/
├── colorization_deploy_v2.prototxt
├── colorization_release_v2.caffemodel
└── pts_in_hull.npy
Project Structure
perl
Copy code
Real-time-video-colorization/
│
├── models/                 # Pretrained Caffe models (download separately)
├── app.py                  # Main real-time colorization script
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files and folders
└── README.md               # Project documentation
