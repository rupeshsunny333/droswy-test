import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time



#added new comment
#jai pubg

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('./haar_cascade_files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('./haar_cascade_files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('./haar_cascade_files/haarcascade_righteye_2splits.xml')


import tensorflow as tf
IMG_SIZE = 145
def prepare(filepath, face_cas="./haarcascade_frontalface_default.xml"):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_array = img_array / 255
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
model = tf.keras.models.load_model("./drowiness_new6.h5")


lbl=['Close','Open']

#model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )
    saved1,saved2=0,0
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
##        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
##        r_eye = cv2.resize(r_eye,(24,24))
##        r_eye= r_eye/255
##        r_eye=  r_eye.reshape(24,24,-1)
##        r_eye = np.expand_dims(r_eye,axis=0)
        cv2.imwrite('rightEye.jpg',r_eye)
        try:
            prediction = model.predict([prepare("./rightEye.jpg")])
        except:
            print("djfddfbhbdkfjhsdjknf")
        saved1=np.argmax(prediction)
        if(saved1==3):
            lbl='Open' 
        if(saved1==2):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
##        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
##        l_eye = cv2.resize(l_eye,(24,24))
##        l_eye= l_eye/255
##        l_eye=l_eye.reshape(24,24,-1)
##        l_eye = np.expand_dims(l_eye,axis=0)
        cv2.imwrite('leftEye.jpg',l_eye)
        prediction = model.predict([prepare("./leftEye.jpg")])
        saved2=np.argmax(prediction)
        if(saved2==3):
            lbl='Open' 
        if(saved2==2):
            lbl='Closed'
        break
    print(saved1,saved2)
    if(saved1==2 or saved2==2):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite('image.jpg',frame)
        try:
            sound.play()
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
