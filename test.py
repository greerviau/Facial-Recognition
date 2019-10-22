import cv2, os, time
import numpy as np 
from facial_recognition import FacialRecognition

if __name__ == '__main__':
    fr = FacialRecognition()
    cap = cv2.VideoCapture(cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    while(True):
        ret, frame = cap.read()

        frame, names, faces_frame = fr.find_faces(frame, return_face_frame=True)

        if faces_frame is not None:
            cv2.imshow('Faces', faces_frame)
        cv2.imshow('Facial Recognition',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()