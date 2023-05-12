# FACE-RECOGNITION-USING-OPEN-CV
import os;
import cv2 as cv;
import numpy as np 
def data(directory):
    faces=[] 
    faceid=[] 
    for path,subdirnames,filenames in os.walk(directory): 
        for filename in filenames:
            if filename.startswith("."):
                continue 
                name=os.path.basename(path) 
                imgpath=os.path.join(path,filename) 
                test=cv.imread(imgpath)
                if test is None: 
                    continue 
        facesrec,gray=faceDetect(test) 
        if len(facesrec)!=1: 
            continue 
            (x,y,w,h)=facesrec[0] 
            grayface=gray[y:y+w,x:x+h] 
            faces.append(grayface) 
            faceid.append(int(name)) 
            return faces,faceid
        def faceDetect(test):
            gray=cv.cvtColor(test,cv.COLOR_BGR2GRAY) 
            haar=cv.CascadeClassifier("opencv_haarcascade_frontalface_default.xml at master · opencv_opencv · GitHub1") 
            faces=haar.detectMultiScale(gray,scaleFactor=1.30,minNeighbors=5) 
            return faces,gray 
        def Recognizer(): 
            face_recog=cv.face.LBPHFaceRecognizer_create() 
            faces,faceid=data("\\dataset") 
            face_recog.train(faces,np.array(faceid)) 
            return face_recog 
        while(1): 
            vcap=cv.VideoCapture(0, cv.CAP_DSHOW) 
            haar=cv.CascadeClassifier("opencv_haarcascade_frontalface_default.xml at master · opencv_opencv · GitHub1") 
            ret,frame=vcap.read() 
            togray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
            faces = haar.detectMultiScale(frame, 1.3, 5) 
            names={0:"Vignesh"} 
        for (x,y,w,h) in faces: 
            gray = togray[y:y+h, x:x+h] 
            face_recog=Recognizer() 
            label,confidence=face_recog.predict(gray) 
            if confidence<90:
                print(names[label]," ",confidence) 
                cv.rectangle(frame,(x,y),(x+h,y+h),(255,0,0)) 
                cv.putText(frame,names[label],(x+3,y+3),cv.FONT_HERSHEY_PLAIN,1.0,(255,0,0))
                cv.imshow("face_Recog",frame)
                k=cv.waitKey(0) 
                vcap.release() 
                cv.destroyAllWindows()
