from __future__ import division, print_function, unicode_literals
from learningrecognizer.preprocessing import globalVariables

import cv2
import pickle
import numpy as np
from skimage import io


import learningrecognizer.preprocessing.ImageClass as IC
import learningrecognizer.preprocessing.FaceDetector as FD
from learningrecognizer import encodings


with open(globalVariables.SVM_Path, "rb") as f:
    model = pickle.load(f)


fname = "10.pgm"

image = io.imread(fname)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

img = IC.Image("", fname, image)
detector = FD.FaceDetector(None, img)

detector.setDetectedFaces()
print(detector.detected_faces)


for i, face_rect in enumerate(detector.detected_faces):

    pose_landmarks = detector.getPosePrediction(face_rect)
    face_encodings = encodings.extract(detector.Image, pose_landmarks)

    print("Type of face_encodings", type(face_encodings))
    X = np.ravel(face_encodings)
    print("Type of X : ", type(X))
    #print("X : ", X)

    X = X.reshape(1, -1)

    predictedLabels = model.predict(X)
    print(predictedLabels)

print("Akhuu")

