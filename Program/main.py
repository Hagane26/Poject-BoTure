from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolo11n-pose.pt')
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    w,h = frame.shape[:2]

    result = model(frame)

    for keypoints in result[0].keypoints.data:
        keypoints = keypoints.cpu().numpy()
        x,y, confidance = keypoints[0]

        if confidance > 0.7:
            cv2.circle(frame,(int(x),int(y)),radius=5,color=(255,0,0),thickness=1)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

cap.release()
cv2.destroyAllWindows()