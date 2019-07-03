# face_detection
import cv2

import numpy as np

face_cascade = cv2.CascadeClassifier('/home/radhika/Documents/Python_files/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('/home/radhika/Documents/Python_files/opencv-master/data/haarcascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    cv2.imshow('cap',img)
    k = cv2.waitKey(0)
    if k==255:
       cv2.destroyAllWindows ()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh)in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    cv2.imshow('img',img)
    k = cv2.waitKey(33)
    print(k)
    if k == (83)  : # ESC
        break
print(gray)
cap.release()
cv2.destroyAllWindows()
