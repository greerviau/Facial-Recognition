# Facial Recognition

## Overview
A Facial Recognition pipeline using a tensorflow implementation of Facenet.
This implementation uses a Harr Cascade Classifier to first detect all the faces in a scene.
It then runs each of those faces through a pretrained tensorflow implementation of facenet which computes the feature map representation of each face.
Then an SVC (Support Vector Classifier) is used to classify the 128 dimensional feature maps.

## Installation
Git clone and ```cd``` into the directory
```
git clone https://github.com/greerviau/Facial-Recognition.git && cd Facial-Recognition
```

## Usage
First you have to collect images of the faces you want to detect.
1. Run: ```collect_image.py``` and use it to take pictures of a subjects face.
   * Make sure you adjust the ```save_path``` variable for each subject so they are saved in individualy labeled folders.
2. Collect atleast 20 photos (preferably more) with different facial expressions and angles.
3. Repeat this for all the subjects whos faces you want to detect.

Next the images need to be aligned. Run:
```
python3 facenet/src/align_dataset_mtcnn.py "path/to/faces" "path/to/aligned_faces" --image_size 160
'''
path/to/faces = path to the main folder where you saved your faces
path/to/aligned_faces = path to the folder to save your aligned faces

Once your images are aligned you need to train the SVC (Support Vector Classifer) on your dataset. Run:
'''
python3 facenet/src/classifier.py TRAIN "path/to/aligned_faces" "facenet/src/models/pretrained_model" "facenet/src/svc_classifer.pkl"
'''

Now run '''test.py''' to test that it is working.

## Pipeline
To use this as a pipeline you will need to import '''facial_recognition.py'''
Follow the example bellow
'''
import cv2
import facial_recognition as fr

image = cv2.imread('path/to/image')

image, names, faces_frame = fr.find_faces(image, return_face_frame = True)

if faces_frame is not None:
	cv2.imshow('Faces', faces_frame)
cv2.imshow('Facial Recognition',frame)

cv2.waitKey(0)

cv2.destroyAllWindows()
'''

