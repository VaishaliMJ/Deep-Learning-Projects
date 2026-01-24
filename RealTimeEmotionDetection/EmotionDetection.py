"""---------------------------------------------------------------------------------------------------------------
                Real Time Emotion Detection(CNN+OpenCV)
                            Vaishali Jorwekar
-------------------------------------------------------------------------------------------------------------------
Problem statement:Developed a real-time facial emotion recognition system using convolutional Neural Networks (CNN)
                  integrated with OpenCV for live video 
-------------------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports and constants
#####################################################################################################
import cv2,os
import cv2.data
import numpy as np
import keras
ARTIFACT_DIR="artifacts_emotion"

BEST_MODEL=os.path.join(ARTIFACT_DIR,"emotion_cnn.h5")
EMOTION_CLASSES=["angry","disgusted","fearful","happy","neutral","sad","suprised"]
#####################################################################################################

cap=cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    img_resized=cv2.resize(frame,(500,500))
    if not ret:
        break
    faceDetector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    grayFrame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    
    NoFaces=faceDetector.detectMultiScale(grayFrame,scaleFactor=1.2,minNeighbors=3)
    
    for (x,y,w,h) in NoFaces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        r_grayFrame =grayFrame[y:y+h,x:x+w]
        croppedImg=np.expand_dims(np.expand_dims(cv2.resize(r_grayFrame,(48,48)),-1),0)
        
        model=keras.models.load_model(BEST_MODEL)
        emotionPredict=model.predict(croppedImg)
        idx=int(np.argmax(emotionPredict))
        cv2.putText(frame,EMOTION_CLASSES[idx],(x+5,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        
    # Display the frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
