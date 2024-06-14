import os
import pickle

import cv2
import cvzone
import face_recognition
import numpy as np

# code for defining your webcam
cap = cv2.VideoCapture(0)
# defining the size of the webcam window
cap.set(3, 640)
cap.set(4, 480)

# importing graphics and modes
imgBackground = cv2.imread('Files/Resources/background.png')

# importing mode images
folderModePath = 'Files/Resources/Modes'
modePathList = os.listdir(folderModePath)
# creating a list to hold all modes
imgModeList = []
for path in modePathList:
    # to append the img in the modes file in the list
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
# print(len(imgModeList))

# load the encoding file
print("Loading Encoded file ...")

file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
# print(studentIds)
print("Encoded file load ...")

# loop for keeping camera on
while True:
    success, img = cap.read()
    # scaling  image down to reduce computational power
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # encodings for the new images
    # telling the location of the face in the img , extract it and encode it
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    # to relay the webcam into the image background
    # the parameters specify the height and width of the relayed cam and img background
    imgBackground[162:162 + 480, 55:55 + 640] = img
    # the mode images are also relayed into the img background using indexes
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[3]

    # comparing face to encodings generated one after the other
    # zip allows you to use the two parameters in the bracket else a new for loop should be created
    for encodeFace, faceLocation in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        # the lower the distance the better the match
        face_dis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print("matches", matches)
        # print("distance", face_dis)

        # the select the specific index we chose the face distance with the minimum value
        matchIndex = np.argmin(face_dis)
        print("matchIndex", matchIndex)

        if matches[matchIndex]:
            # print("known face detected")
            # print(studentIds[matchIndex])
            # to put the target box on face
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

    # to display your webcam on screen
    # cv2.imshow("Webcam", img)
    # to display the image background on screen
    cv2.imshow("Face Attendance", imgBackground)

    cv2.waitKey(1)
