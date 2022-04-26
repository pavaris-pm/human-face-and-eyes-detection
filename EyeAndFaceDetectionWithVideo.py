#test an eye detection using video by using cv2 and haarCascadeClassifier

import cv2
import datetime

#for testing with Video as a mp4 type
url_video = "D:/CoderCamp 2021 Event Dvent Programing/2022-01-12 22.02.25 EDP WEEK3 Choice Selector/Event Driven Programming week 3_ Choice_Selector_program.mp4"
captureImg = cv2.VideoCapture(0)
eye_cascade = cv2.CascadeClassifier("D:/opencv_project/detection/haarcascade_eye_tree_eyeglasses.xml")
face_cascade = cv2.CascadeClassifier("D:/opencv_project/detection/haarcascade_frontalface_default.xml")

while(captureImg.isOpened()):
    date = str(datetime.datetime.now())
    check, frame = captureImg.read()
    #frame = cv2.resize(frame, (800,600)) // for resize the frame in case it is too big
    eye_scaleFactor = 1.2; eye_minNeighbor = 3
    face_scaleFactor = 1.25; face_minNeighbor = 6
    #convert the frame to gray scale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_detect = eye_cascade.detectMultiScale(frame_gray, eye_scaleFactor, eye_minNeighbor)
    face_detect = face_cascade.detectMultiScale(frame_gray, face_scaleFactor, face_minNeighbor)

    if(check == True):
        #for eye detection part
        for (x,y,w,h) in eye_detect:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=5)
            cv2.putText(frame, "Eyes", (x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0),cv2.LINE_4)

        #for face detection part
        for (x2,y2,w2,h2) in face_detect:
            cv2.rectangle(frame, (x2,y2), (x2+w2, y2+h2), (255,0,0), thickness=5)
            cv2.putText(frame, "Face", (x2-10,y2-10),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,0,0),cv2.LINE_4)
        cv2.putText(frame, "Timestamp", (30,40),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),cv2.LINE_4)            
        cv2.putText(frame, date, (30,70),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),cv2.LINE_4)            
        cv2.imshow("Eye detection with Video", frame)   
        if(cv2.waitKey(1) & 0xFF == ord('e')):
            break
    else:
        break

cv2.imshow("Eye detection with Video", frame)
captureImg.release()
cv2.destroyAllWindows()
