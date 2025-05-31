# 🔥 Fire Detection & Characteristics using Computer Vision

This project implements a real-time **fire detection system** using the **YOLO model** and **OpenCV**. It also demonstrates image processing techniques like grayscale conversion, Sobel edge detection, and thresholding to highlight different characteristics of video frames. The program processes a video file selected by the user and visually displays the detection and image transformation results in real-time.

## 📌 Features

- 📁 **File Picker**: GUI to select video file input using `tkinter`.
- 🧠 **Fire Detection**: Uses a pretrained YOLO model (`fire.pt`) for detecting fire in video frames.
- 🎞️ **Real-Time Processing**:
  - Frame resizing
  - Grayscale conversion
  - Sobel edge detection (gradient-based edge finding)
  - Binary thresholding
- 🖼️ **Annotated Output**: Displays bounding boxes and confidence scores on fire detection.
- 📊 **Multi-View Display**: Combines original, grayscale, Sobel edge, and thresholded views into a single 4-panel window.

## 🧰 Technologies Used

- `Python 3.x`
- `OpenCV`
- `cvzone`
- `tkinter` (for file selection dialog)
- `Ultralytics YOLOv8`
- `Math` module

## 🧠 How It Works

1. **Load video** through a GUI file picker.
2. **Read each frame** from the video.
3. Apply the following filters on each frame:
   - **Grayscale**: Converts the BGR image to 1-channel gray.
   - **Sobel Edge Detection**: Computes edges using horizontal and vertical gradients.
   - **Thresholding**: Converts grayscale image to binary (black & white).
4. **YOLO Inference**:
   - Detects fire regions in the frame.
   - Draws bounding boxes with class name and confidence percentage.
5. **Display Output**:
   - Four views combined: Original + Grayscale + Sobel + Threshold.
   - Press `'q'` to exit the window.

## 🖼️ Sample Output

> The window will display:
- Original frame with detection box
- Grayscale version
- Sobel edge-detected frame
- Thresholded binary frame

All in a 2x2 stacked format.

## 📁 Project Structure
├── fire.pt # Pretrained YOLO model for fire detection
├── fire.py # Main Python script
├── README.md # Project description file


## ▶️ How to Run

1. Make sure you have Python 3.x installed.
2. Install the required libraries:

```bash
pip install opencv-python cvzone ultralytics

Now you can run it.
