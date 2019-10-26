# Facial Recognition

## Overview
A Facial Recognition pipeline using a pretrained Tensorflow implementation of Facenet.
This pipeline uses a Harr Cascade Classifier to first detect all the faces in a scene.
It then runs each of the faces through Facenet which computes a 128 dimensional feature map for each face.
Then an SVC (Support Vector Classifier) is used to predict a probability of the face belonging to each class.
The class with the highest prediction confidence that is also at least 80% is determined to be the correct classification.
If none of the predictions achieve 80% confidence then the face is classified as "Unknown".

This project was made durring HackUMass 2019 as part of a security system my team and I were developing. This was my contribution.

## Installation
Git clone and ```cd``` into the directory
```
git clone https://github.com/greerviau/Facial-Recognition.git && cd Facial-Recognition
```
Download the pretrained weights [here](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit) and extract them to ```facenet/src/models/pretrained_model```


## Usage
First you have to collect images of the faces you want to detect.
1. Run: ```collect_photos.py <name-of-subject> <number-of-photos>``` and use it to take pictures of a subjects face.
2. Collect photos with different facial expressions and angles.
3. Repeat this for all the subjects whos faces you want to detect.

Next the images need to be aligned. Run:
```
python3 facenet/src/align_dataset_mtcnn.py "path/to/faces" "path/to/aligned_faces" --image_size 160
```
* path/to/faces = path to the main folder where you saved your folder seperated faces
* path/to/aligned_faces = path to the folder to save your aligned faces

Once your images are aligned you need to train the SVC (Support Vector Classifer) on your dataset. Run:
```
python3 facenet/src/classifier.py TRAIN "path/to/aligned_faces" "facenet/src/models/pretrained_model" "facenet/src/sv_classifier.pkl"
```

Finally run ```test.py``` to test that it is working.

## Pipeline
To use this as a pipeline you will need to import ```facial_recognition.py```
The facial recognition can be used by creating a FacialRecognition object an calling ```find_faces()``` as shown bellow.
```
import cv2
from facial_recognition import FacialRecognition

fr = FacialRecognition()

image = cv2.imread('path/to/image')

image, names, faces_frame = fr.find_faces(image, return_face_frame = True)

if faces_frame is not None:
	cv2.imshow('Faces', faces_frame)
cv2.imshow('Facial Recognition',frame)

cv2.waitKey(0)

cv2.destroyAllWindows()
```

## References 
* Pretrained Facenet model https://github.com/davidsandberg/facenet
* Implementing Facenet https://medium.com/@athul929/building-a-facial-recognition-system-with-facenet-b9c249c2388a
