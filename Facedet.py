import cv2
import matplotlib
import numpy

facecascade = cv2.CascadeClassifier(r"C:\Users\Admin\Desktop\Opencv\haarcascade_frontalface_default.xml")
eyecascade = cv2.CascadeClassifier(r'C:\Users\Admin\Desktop\Opencv\haarcascade_eye.xml')

def detect(gray,frame):
    faces = facecascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h , x:x+w]
        roi_color = frame[y:y+h , x:x+w]
        eyes = eyecascade.detectMultiScale(roi_gray,1.1,3)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    return frame

video_capture = cv2.VideoCapture(0)
while True:
    _,frame = video_capture.read()
    if frame is not None:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        canvas = detect(gray,frame)
        cv2.imshow('Video',canvas)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
