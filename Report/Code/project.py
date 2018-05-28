# -*- coding: utf-8 -*-
"""
VGIS Group 843
Rita and Atanas
"""


import cv2
import numpy as np
import helper
import cluster
#from scipy.spatial import distance



#video = 'Caviar\Fainting\Rest_FallOnFloor.mpg'
#video = 'Caviar\Fighting\Fight_OneManDown.mpg'
#video = 'Caviar\Fighting\Fight_RunAway1.mpg'
#video = 'Caviar\Left_bags\LeftBag.mpg'
#video = 'Caviar\Left_bags\LeftBag_PickedUp.mpg'
#video = 'Caviar\Left_bags\LeftBox.mpg'
#video = 'Caviar\Tracking\Meet_WalkSplit.mpg'
video = 'Caviar\Walking\Walk3.mpg'
#video = 'Caviar\Tracking\Meet_WalkTogether2.mpg'
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

#file = open('testfile7.txt', 'w')
data = cluster.data
scaler = cluster.scaler
clf = cluster.clf
labels = cluster.labels_binary
print(data[0])
print(labels[0])

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

            to_write = (x, y, vel)
            sample = np.array([x, y])
            sample = sample.reshape(1, -1)
            sample = scaler.transform(sample)

            if clf.predict(sample) == 1:
                cv2.circle(frame, (x, y), 12, [0, 255, 255], 2)
                print('Suspicious at frame %d' % nr_frame)
            #file.write(str(to_write)+'\n')

            cv2.putText(frame, category+" vel-"+str(vel), 
                        (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
            
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
    
#file.close()
cap.release()
cv2.destroyAllWindows()

