import cv2
from ultralytics import YOLO
from cv2 import getTickCount, getTickFrequency

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    loop_start = getTickCount()
    success, frame = cap.read()

    if success:
        results = model.predict(source=frame,conf=0.65) 
    annotated_frame = results[0].plot()

    loop_time = getTickCount() - loop_start
    total_time = loop_time / (getTickFrequency())
    FPS = int(1 / total_time)

    #text
    fps_text = f"FPS: {FPS:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 255) 
    text_position = (10, 30)  

    cv2.putText(annotated_frame, fps_text, text_position, font, font_scale, text_color, font_thickness)
    cv2.imshow('img', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()


