import os
import pickle

import cv2
import face_recognition

# importing the student  images
folderPath = 'Files/Images'
PathList = os.listdir(folderPath)
print(PathList)
imgList = []
studentIds = []
for path in PathList:
    # to append the img in the modes file in the list
    imgList.append(cv2.imread(os.path.join(folderPath, path)))

    # this removes the png from the id
    # print(os.path.splitext(path)[0])
    studentIds.append(os.path.splitext(path)[0])
print(studentIds)


def findEncodings(imgList):
    encodeList = []
    for img in imgList:
        # converting images to color appropriate for cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # finding the encodings
        encode = face_recognition.face_encodings(img)[0]
        # loop through all images and save them
        encodeList.append(encode)

    return encodeList


print("Encoding started")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding complete")

file = open("Encode file.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("file saved")
