#!/usr/bin/env python
# coding: utf-8

# In[150]:


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


# ## KNN

# In[151]:


#knn ..calculating euclidian distance
def dist(x1,y1):
    return np.sqrt(sum((x1-y1)**2))

def knn(x,y,train,k=5):
    value=[]
    
    num=x.shape[0]
    
    for i in range(num):
        dis=dist(train,x[i])
        value.append((dis,y[i]))
    
    
    value=sorted(value)
    
    value=np.array(value[:k])
    
    getUnique=np.unique(value[:,1],return_counts=True)
    result=getUnique[0][getUnique[1].argmax()]
    
    return result


# ## Reading the Dataset (Faces)

# In[152]:


path = 'E:\\SIProject\\dataset\\'

face_data = [] 
labels  = []

for fx in os.listdir(path):
    #print(path+fx)
    image=cv2.imread(path+fx) 
    image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (50,50)) # resize 50,50 shape
    face_data.append(image)
    

    label=os.path.basename(fx).split('.')
    label=label[0].split('-')
    labels.append(label[0])





face_data = np.concatenate(face_data, axis=0).reshape(-1, 50*50*3)

labels = np.array(labels)

plt.imshow(face_data[19].reshape(50,50,3))


cap=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            test_face=img[y:(y+h), x:(x+w)]
            test_face = cv2.resize(test_face, (50,50))
           
            test_face = test_face.reshape(50*50*3,)
            name=knn(face_data,labels,test_face)


            cv2.rectangle(img, (x, y) ,(x+w, y+h), (0,255,0), 2)
            cv2.putText(img, name,(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2, cv2.LINE_AA)
        
        k = cv2.waitKey(30) & 0xff

        if k== ord('q'):
             break

        cv2.imshow("Recog", img)
        
cap.release() 
cv2.destroyAllWindows()


