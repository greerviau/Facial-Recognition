import cv2, os, time
import numpy as np 

cap = cv2.VideoCapture(cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

subject = 'greer'
save_path = 'valid/'+subject
delay = 1

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if not os.path.exists(save_path):
    os.makedirs(save_path)

i = 0
while(True):
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100), flags = cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        img_slice = frame[y:y+h, x:x+w]
        img_slice = cv2.resize(img_slice, (160,160), interpolation = cv2.INTER_AREA)
        cv2.imwrite(save_path+'/greer_{:06d}.png'.format(i), img_slice)
        i+=1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(delay)

cap.release()
cv2.destroyAllWindows()