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

cap = cv2.VideoCapture(0)
process_this_frame = True

global face_locations
global predictedLabels
global face_encodings

while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = []
    predictedLabels = []

    # Only process every other frame of video to save time
    if process_this_frame:

        img = IC.Image("", "rgb_small_frame", rgb_small_frame)
        detector = FD.FaceDetector(None, img)

        detector.setDetectedFaces()
        print(detector.detected_faces)

        for i, face_rect in enumerate(detector.detected_faces):

            curr_tuple = (face_rect.top(), face_rect.right(), face_rect.bottom(), face_rect.left())
            face_locations.append(curr_tuple)
            pose_landmarks = detector.getPosePrediction(face_rect)
            face_encodings = encodings.extract(detector.Image, pose_landmarks)

            X = np.ravel(face_encodings)
            X = X.reshape(1, -1)

            predictedLabels.append(model.predict(X))

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, predictedLabels):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        print("locations", face_locations)
        #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1, cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()