import numpy as np 
import cv2, math, os, sys
import classify
import sys
import cv2
import preprocess
import time

cap = cv2.VideoCapture(cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

classifier = classify.Classify()
preprocessor = preprocess.PreProcessor()

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if not os.path.exists('valid/jackie'):
    os.makedirs('valid/jackie')
#count = 0
i = 0
while(True):
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(160, 160), flags = cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        img_slice = frame[y:y+h, x:x+w]
        img_slice = cv2.resize(img_slice, (160,160), interpolation = cv2.INTER_AREA)
        #cv2.imwrite('valid/jackie/jackie_{:06d}.png'.format(count), img_slice)
        #count+=1
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
        #cv2.imwrite('opencv'+str(i)+'.png', frame)
        aligned_image = preprocessor.align(img_slice)
        if aligned_image is not None:
            startx, starty = x, y
            endx, endy = x+w, y+h
            cv2.rectangle(frame, (startx, starty), (endx, endy), (0, 255, 0), 5)
            name = classifier.predict(aligned_image)
            font = cv2.FONT_HERSHEY_SIMPLEX 
            cv2.putText(frame, name, (startx,starty-20), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    i+=1
    
cap.release()
cv2.destroyAllWindows()