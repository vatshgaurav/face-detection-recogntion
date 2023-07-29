# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 23:46:13 2020

@author: VATSH
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 23:04:26 2020

@author: VATSH
"""
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
os.chdir("C:/Users/GAURAV/Desktop/FDRP")


def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    #cam = cv2.VideoCapture(0)
    im = cv2.imread('timg1.jpg',0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
        #ret, im =cam.read()
    #gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray=im
    
    faces=faceCascade.detectMultiScale(gray, 1.2,5)    
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
        if(conf < 50):
            ts = time.time()      
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            aa=df.loc[df['Id'] == Id]['Name'].values
            tt=str(Id)+"-"+aa
            attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
            
        else:
            Id='Unknown'                
            tt=str(Id)  
        if(conf > 75):
            noOfFile=len(os.listdir("ImagesUnknown"))+1
            cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
        cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
    attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
    cv2.imshow('im',im) 
    if(cv2.waitKey(0)==ord('q')):
        
        ts = time.time()      
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        Hour,Minute,Second=timeStamp.split(":")
        fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
        attendance.to_csv(fileName,index=False)
        #cam.release()
        cv2.destroyAllWindows()
        print(attendance)
    
    
TrackImages()
   
