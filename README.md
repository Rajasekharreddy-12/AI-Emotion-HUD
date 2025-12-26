# AI-Emotion-HUD
An immersive, full-screen Artificial Intelligence interface that performs real-time facial expression analysis. This project leverages DeepFace for neural emotion recognition and OpenCV for a high-performance, multithreaded Graphical User Interface.
🚀 Key Features
Multithreaded AI Core: Separates heavy DeepFace processing from the camera feed to maintain a silky smooth 60 FPS experience.

Immersive Full-Screen HUD: A borderless, cinematic interface designed to look like a sci-fi terminal.

Emoji Integration: Dynamically renders high-resolution Unicode emojis corresponding to your current mood.

"Liquid" UI Animation: Uses linear interpolation (Lerp) to ensure emotion bars glide smoothly rather than jumping.

Face Tracking & Scanning: A dynamic focus-box follows the subject with a digital "scanning line" effect.

🛠 Tech Stack
Python 3.x

OpenCV: Computer Vision and UI rendering.

DeepFace: State-of-the-art VGG-Face based emotion recognition.

Pillow (PIL): Used for rendering high-quality Unicode emojis and custom fonts.

Threading: Ensures zero-lag between the AI analysis and the video display.

📋 Prerequisites
Before running the project, ensure you have a webcam connected and the following libraries installed:

Bash

pip install opencv-python deepface tf-keras pillow
Note: On first run, DeepFace will download the pre-trained weights for the emotion model (approx. 150MB).

💻 How to Run
Clone the Repository:

Bash

git clone https://github.com/YOUR_USERNAME/AI-Emotion-HUD.git
cd AI-Emotion-HUD
Execute the Script:

Bash

python main.py
Controls:

Press 'q' to exit the full-screen interface.

🧠 How it Works
The system utilizes a dual-layer approach:

The Analysis Layer: A background thread captures frames and passes them to a Convolutional Neural Network (CNN) to calculate probability scores for 5 core emotions: Angry, Happy, Sad, Surprise, Neutral.

The Rendering Layer: The main thread handles the camera feed and overlays the HUD. It takes the "raw" scores from the AI and smooths them out using a smoothing constant to create a polished user experience.
