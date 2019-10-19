import numpy as np 
import cv2, math, os, sys

cap = cv2.VideoCapture(cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

while(True):
    ret, frame = cap.read()

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    