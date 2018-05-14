# -*- coding: utf-8 -*-
"""
VGIS Group 843
Rita and Atanas
"""


import cv2
import numpy as np
import helper
#from scipy.spatial import distance

#video = 'Caviar\Fainting\Rest_FallOnFloor.mp4'
#video = 'Caviar\Fighting\Fight_OneManDown.mp4'
#video = 'Caviar\Fighting\Fight_RunAway1.mp4'
#video = 'Caviar\Left_bags\LeftBag.mp4'
##video = 'Caviar\Left_bags\LeftBag_PickedUp.mp4'
video = 'Caviar\Left_bags\LeftBox.mp4'
#video = 'Caviar\Tracking\Meet_WalkSplit.mp4'
#video = 'Caviar\Tracking\Meet_WalkTogether2.mp4'
#video = 'Videos\EnterExitCrossingPaths2cor.mp4'
#video = 'Videos\EnterExitCrossingPaths1cor.mp4'
#video = 'Videos\OneLeaveShop1cor.mp4'
#video = 'Videos\OneShopOneWait1cor.mp4'


cap = cv2.VideoCapture(video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


backgroundImg = helper.getBackground(video)

nr_frame = 0
predictions_list = []
kalman_filters_list = []

while cap.isOpened():
    
    nr_frame += 1
    ret, frm = cap.read()
    if not ret:
        break
    if cv2.waitKey(25) & 0xFF==ord('q'):
        break
    
    gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # detection of active pixels through difference between the background and
    # the current frame
    frameDiff = cv2.absdiff(backgroundImg, gray)
    
    thresh = cv2.threshold(frameDiff, 30, 255, cv2.THRESH_BINARY)[1]
                
    # removal of noise with median filter
    median = cv2.medianBlur(thresh, 15)
    
    frame = cv2.cvtColor(median, cv2.COLOR_GRAY2BGR)
    
    # find contours in image
    img, contours, hierarchy = cv2.findContours(median.copy(), cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    
    
    # split wide blobs that might have more than one person merged together
    #candidates = helper.split_wide_blobs(contours, frame)
    
    candidates = []
    # merge blobs of same person, splitted in the segmentation
    if len(contours) > 0:
        candidates = helper.blob_fusion(contours, frame)
        
    people = np.empty((len(candidates), 4), int)    
    for c in range(len(candidates)):
        x = candidates[c][0]
        y = candidates[c][1]
        w = candidates[c][2] - x
        h = candidates[c][3] - y
        people[c] = x, y, w, h
        #cv2.rectangle(frame, (int(x),int(y)),(int(x+w),int(y+h)), (0,255,255), 2)     
    
    # track people with kalman filter
    helper.kalman_filter_tracking(people, predictions_list, kalman_filters_list,
                                  fps, frame)
    
    
    # every half second extract features
    if not nr_frame % int(fps/2):
        helper.extract_features(predictions_list, fps)
        
    
    # categorize blobs
    for detection in predictions_list:
        
        if detection[1] == 0:
            vel_thresh = 10
            
            x = detection[2][-1][0]
            y = detection[2][-1][1]
            vel = detection[4]
            category = detection[6]
            
            cv2.putText(frame, category+" vel-"+str(vel), 
                        (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
            
            if category == "Unknown" and vel < vel_thresh:
                detection[6] = "Object"
                detection[8] = False
            elif (category == "Unknown" or category == "Object") and vel > vel_thresh:
                detection[6] = "Person"
                detection[8] = False
            elif category == "Person" and vel < vel_thresh:
                detection[6] = "Still Person"
                detection[8] = False
            elif category == "Still Person" and vel > vel_thresh:
                detection[6] = "Person"
                detection[8] = False
    
    
    # find abandoned luggage and flag people
    helper.detect_abandoned_objects(predictions_list, fps, frame)
                    
        
    for p in predictions_list:
        if p[1] == 0:
            x = p[2][-1][0]
            y = p[2][-1][1]
            w = p[3][-1][0]
            h = p[3][-1][1]
            if p[8]:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)  
            else:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)   
    
    
    display = np.hstack((frm, frame))
    cv2.imshow("Frame", display)
    
    
cap.release()
cv2.destroyAllWindows()

