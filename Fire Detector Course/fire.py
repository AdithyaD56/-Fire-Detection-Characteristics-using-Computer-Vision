from ultralytics import YOLO
import cv2
import cvzone
import math
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

video_path = filedialog.askopenfilename(title="Select a Video File",
                                        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])

if not video_path:
    print("No video selected. Exiting.")
    exit()

model = YOLO('fire.pt')
classnames = ['fire']

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video has ended or cannot be read.")
        break

    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cvzone.putTextRect(gray, 'Grayscale', (10, 25), scale=1.5, thickness=2)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  
    sobel_edge = cv2.magnitude(sobel_x, sobel_y)  
    sobel_edge = cv2.convertScaleAbs(sobel_edge)  
    sobel_edge_3channel = cv2.cvtColor(sobel_edge, cv2.COLOR_GRAY2BGR)  
    cvzone.putTextRect(sobel_edge_3channel, 'Sobel Edge Detection', (10, 25), scale=1.5, thickness=2)

    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)  
    cvzone.putTextRect(thresholded, 'Thresholding', (10, 25), scale=1.5, thickness=2)

    results = model(frame, stream=True)

    for info in results:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            class_id = int(box.cls[0])
            if confidence > 50:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[class_id]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)

    gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  
    top_row = cv2.hconcat([frame, gray_3channel])  
    bottom_row = cv2.hconcat([sobel_edge_3channel, thresholded])  
    stacked_frame = cv2.vconcat([top_row, bottom_row]) 

    output_display = cv2.resize(stacked_frame, (1280, 720))

    cv2.imshow('Fire Detection & Characteristics using Computer vision. ', output_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()