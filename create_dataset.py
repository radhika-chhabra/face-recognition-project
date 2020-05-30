#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
image_eyes = []
path = 'E:\\SIProject\\dataset'

while True:
     ret, img = cap.read()
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


     face = face_cascade.detectMultiScale(gray, 1.3, 5)
     for (x,y,w,h) in face:
         cv2.rectangle(img, (x, y) ,(x+w, y+h), (0,255,0), 2)
         image_eyes.append(img[y:(y+h), x:(x+w)])

     cv2.imshow("Face", img)
     for i, x in enumerate(image_eyes):
         cv2.imwrite(os.path.join(path , "face-" + str(i) + ".jpg"), x)
        
     k = cv2.waitKey(30) & 0xff

     if k== ord('q'):
         break
cap.release() 
cv2.destroyAllWindows()





