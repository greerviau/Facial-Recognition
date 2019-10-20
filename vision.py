import numpy as np 
import cv2, math, os, sys
import classify
import sys
import cv2
import preprocess
import time
import cache

class Vision():

    def __init__(self):

        self.cap = cv2.VideoCapture(cv2.CAP_DSHOW)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        self.CONF = 0.8

        self.classifier = classify.Classify()
        self.preprocessor = preprocess.PreProcessor()
        self.class_names = self.classifier.get_class_names()
        self.cache = cache.Cache(max_size=10)

        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def check_camera(self, show_frame = False):
        ret, frame = self.cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(160, 160), flags = cv2.CASCADE_SCALE_IMAGE)

        show_faces = None
        name = None
        for (x, y, w, h) in faces:
            img_slice = frame[y:y+h, x:x+w]
            img_slice = cv2.resize(img_slice, (160,160), interpolation = cv2.INTER_AREA)
            aligned_image = self.preprocessor.align(img_slice)
            if aligned_image is not None:
                if show_faces is None:
                    show_faces = aligned_image
                else:
                    show_faces = np.concatenate((show_faces,aligned_image), axis=1)
                startx, starty = x, y
                endx, endy = x+w, y+h
                cv2.rectangle(frame, (startx, starty), (endx, endy), (0, 255, 0), 5)
                pred = self.classifier.predict(aligned_image)
                self.cache.add(pred)
                pred = self.cache.mean()
                best_class_indices = np.argmax(pred, axis=1)
                best_class_probabilities = pred[np.arange(len(best_class_indices)), best_class_indices]
                name = 'Unknown Face'
                if best_class_probabilities[0] > self.CONF:
                    name = self.class_names[best_class_indices[0]]

                font = cv2.FONT_HERSHEY_SIMPLEX 
                cv2.putText(frame, name, (startx,starty-20), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                print('\rRunning Avg Conf: ',np.argmax(self.cache.mean(), axis=1),end='')

        if show_frame:
            if show_faces is not None:
                cv2.imshow('faces', show_faces)
            cv2.imshow('frame',frame)
        
        if name is not None:
            return True, name 
        else:
            return False, name

    def end(self):
        self.cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    vis = Vision()
    while(True):
        face_detected, name = vis.check_camera(show_frame=True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vis.end()
    
    '''
    while(True):
        ret, frame = self.cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(160, 160), flags = cv2.CASCADE_SCALE_IMAGE)

        show_faces = None
        for (x, y, w, h) in faces:
            img_slice = frame[y:y+h, x:x+w]
            img_slice = cv2.resize(img_slice, (160,160), interpolation = cv2.INTER_AREA)
            aligned_image = self.preprocessor.align(img_slice)
            if aligned_image is not None:
                if show_faces is None:
                    show_faces = aligned_image
                else:
                    show_faces = np.concatenate((show_faces,aligned_image), axis=1)
                startx, starty = x, y
                endx, endy = x+w, y+h
                cv2.rectangle(frame, (startx, starty), (endx, endy), (0, 255, 0), 5)
                pred = self.classifier.predict(aligned_image)
                self.cache.add(pred)
                pred = self.cache.mean()
                best_class_indices = np.argmax(pred, axis=1)
                best_class_probabilities = pred[np.arange(len(best_class_indices)), best_class_indices]
                name = 'Unknown Face'
                if best_class_probabilities[0] > self.CONF:
                    name = self.class_names[best_class_indices[0]]

                font = cv2.FONT_HERSHEY_SIMPLEX 
                cv2.putText(frame, name, (startx,starty-20), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        if show_faces is not None:
            cv2.imshow('faces', show_faces)
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    '''