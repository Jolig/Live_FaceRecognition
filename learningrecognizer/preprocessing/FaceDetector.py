from __future__ import division, print_function, unicode_literals

from learningrecognizer.preprocessing import globalVariables
from learningrecognizer.preprocessing import ImageClass

import cv2
import glob
import os

import numpy as np

import dlib
#import openface

from skimage import io
from learningrecognizer import encodings


mainDir = "orl_faces/"
detectedImagesPath = "detected_faces/"

n = len(glob.glob(mainDir + "*"))
print(n)

listOfEncodings = []

predictor_model = globalVariables.prediction_Path

face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
#face_aligner = openface.AlignDlib(predictor_model)


class FaceDetector:

    def __init__(self, detected_faces, Image):
        self.detected_faces = detected_faces
        self.Image = Image

    def setDetectedFaces(self):
        self.detected_faces = face_detector(self.Image.img, 1)

    def setImage(self, Image):
        self.Image = Image

    def getDetectedFaces(self):
        return self.detected_faces

    def getImage(self):
        return self.Image

    def getPosePrediction(self, face_rect):
        return face_pose_predictor(self.Image.img, face_rect)


def detection():

    for dirName, subdirList, fileList in os.walk(mainDir):
        classLabel = "".join(dirName.rsplit(mainDir))

        for fname in fileList:
            # print(fname)
            image = io.imread(dirName + "/" + fname)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            img = ImageClass.Image(classLabel, fname, image)
            fd = FaceDetector(None, img)
            fd.setDetectedFaces()
            posePrediction(fd)


def posePrediction(fd):

    for i, face_rect in enumerate(fd.detected_faces):
        crop = fd.Image.img[face_rect.top(): face_rect.bottom(), face_rect.left(): face_rect.right()]
        # type(crop) ndarray
        # dlib.rectangle

        classLabel = fd.Image.classLabel
        fname = fd.Image.fname

        pose_landmarks = fd.getPosePrediction(face_rect)
        

        #alignedFace = face_aligner.align(150, img.image, face_rect,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        newfname = fname[0: fname.find(".")]
        newName = classLabel + "_" + newfname + "_{}".format(i)

        fd.Image.fname = newName

        # cv2.imwrite(detectedImagesPath + newName, alignedFace)

        face_encodings = encodings.extract(fd.Image, pose_landmarks)

        if face_encodings is not None:
            appended_data = np.append(face_encodings, classLabel)
            listOfEncodings.append(appended_data)

def main():

    heading = list(range(0, 128))
    heading.append("classLabel")
    listOfEncodings.append(heading)
    detection()
    encodings.exportToExcel(listOfEncodings)

if __name__ == '__main__':
    main()
