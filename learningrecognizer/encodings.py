from __future__ import division, print_function, unicode_literals
from learningrecognizer.preprocessing import globalVariables

import csv
import dlib

face_encoder = dlib.face_recognition_model_v1(globalVariables.resnet_Path)

counter = 0

def extract(Image, pose_landmarks):

    global face_encodings
    global counter
    try:
        face_encodings = face_encoder.compute_face_descriptor(Image.img, pose_landmarks, 1)
        #print(face_encodings)

    except IndexError:
        counter += 1
        print("unable to find encodings in ---- " +Image.fname)

    return face_encodings


def exportToExcel(listOfEncodings):

    print(len(listOfEncodings))
    with open(globalVariables.dataFile_Path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(listOfEncodings)


print("Counter : ", counter)
print("Akhiiiii")
