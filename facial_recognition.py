import numpy as np 
import cv2, math, os, sys
import classify
import sys
import cv2
import preprocess
import time
import cache

class FacialRecognition():

    def __init__(self):

        self.CONF = 0.8

        self.classifier = classify.Classify()
        self.preprocessor = preprocess.PreProcessor()
        self.class_names = self.classifier.get_class_names()
        self.cache = cache.Cache(max_size=10)

        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def find_faces(self, img, return_face_frame=True):

        frame = img

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(160, 160), flags = cv2.CASCADE_SCALE_IMAGE)

        faces_frame = None
        names = []
        for (x, y, w, h) in faces:

            img_slice = frame[y:y+h, x:x+w]
            img_slice = cv2.resize(img_slice, (160,160), interpolation = cv2.INTER_AREA)
            aligned_image = self.preprocessor.align(img_slice)

            if aligned_image is not None:

                if return_face_frame:
                    if faces_frame is None:
                        faces_frame = aligned_image
                    else:
                        faces_frame = np.concatenate((faces_frame,aligned_image), axis=1)

                startx, starty = x, y
                endx, endy = x+w, y+h

                cv2.rectangle(frame, (startx, starty), (endx, endy), (0, 255, 0), 5)

                pred = self.classifier.predict(aligned_image)
                print(pred[0])

                self.cache.add(pred[0])
                pred = self.cache.mean()
                best_class_index = np.argmax(pred)
                best_class_probability = pred[best_class_index]
                
                name = 'Unknown Face'
                if best_class_probability > self.CONF:
                    name = self.class_names[best_class_index]

                names.append(name)

                font = cv2.FONT_HERSHEY_SIMPLEX 
                cv2.putText(frame, name, (startx,starty-20), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        return frame, names, faces_frame     
                