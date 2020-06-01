import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

name = input("Enter your name : ")
cap = cv2.VideoCapture(0)
image_eyes = []
path = 'E:\\SIProject\\dataset'

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in face:
        image_eyes.append(img[y:(y+h), x:(x+w)])
        print(" Images Captured - ", len(image_eyes), end="\r")

    cv2.imshow("Face", img)
    
    for i, x in enumerate(image_eyes):
        cv2.imwrite(os.path.join(path , name +"-" + str(i)+ ".jpg"), x)
        
    k = cv2.waitKey(30) & 0xff

    if k== ord('q'):
        break
        
cap.release() 
cv2.destroyAllWindows()
