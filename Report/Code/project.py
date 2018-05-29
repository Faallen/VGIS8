# -*- coding: utf-8 -*-
"""
VGIS Group 843
Rita and Atanas

Semester project
"""

import cv2
import numpy as np
import helper

# Videos
#video = 'Caviar\Fainting\Rest_FallOnFloor.mpg'
video = 'Caviar\Fighting\Fight_OneManDown.mpg'
#video = 'Caviar\Fighting\Fight_RunAway1.mpg'
#video = 'Caviar\Walking\Walk2.mpg'
#video = 'Caviar\Left_bags\LeftBag.mp4'
##video = 'Caviar\Left_bags\LeftBag_PickedUp.mp4'
#video = 'Caviar\Left_bags\LeftBox.mp4'
#video = 'Caviar\Tracking\Meet_WalkSplit.mp4'
#video = 'Caviar\Tracking\Meet_WalkTogether2.mp4'
#video = 'Caviar\Loitering\Browse_WhileWaiting2.mp4'



cap = cv2.VideoCapture(video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# get background image obtained through temporal median filter
backgroundImg = helper.get_background(video)

nr_frame = 0
predictions_list = []
kalman_filters_list = []
heights_list = [1]
blob_id = [0]
vel_data = dict()
pause = False
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
    
    kernel_eli = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    closing = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel_eli)
    
    frame = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
    
    frame_tracks = frm.copy()
    
    # find contours in image
    img, contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    # merge blobs of same person, split in the segmentation
    if len(contours) > 0:
        candidates = helper.blob_fusion(contours, heights_list)
        
        
    people = np.empty((len(candidates), 4), int) 
    for c in range(len(candidates)):
        x = candidates[c][0]
        y = candidates[c][1]
        w = candidates[c][2] - x
        h = candidates[c][3] - y
        people[c] = x, y, w, h
    
    
    # track people with kalman filter
    helper.kalman_filter_tracking(people, predictions_list, kalman_filters_list,
                                  fps, blob_id)
    
    
    # draw tracks on frame
    helper.draw_tracks(frame_tracks, predictions_list)
    
    
    # handle occlusions
    helper.handle_occlusion(frm, predictions_list, fps, kalman_filters_list)
    
    
    # every half second extract features
    if not nr_frame % int(fps/2):
        helper.extract_features(predictions_list, fps)
        
        
    for p in predictions_list:
        if p[14] != "Fainting" and p[1] == 0 and (p[6] == "Person" or p[6] == "Still Person"):
            h = p[3][-1][1]
            heights_list.append(h)
        
        
    # categorize blobs
    for detection in predictions_list:
        
        if detection[1] == 0:
            vel_thresh = 5
            x = detection[2][-1][0]
            y = detection[2][-1][1]
            vel = detection[4]
            category = detection[6]
            if detection[10] in vel_data:
                if len(vel_data[detection[10]]) > 5:
                    vel_data[detection[10]].pop(0)
                else:
                    vel_data[detection[10]].append([nr_frame, vel])
            else:
                vel_data[detection[10]] = [[nr_frame, vel]]

            if len(vel_data[detection[10]]) > 5:
                if helper.detect_running(vel_data[detection[10]], 23, 3):
                    print('Suspicious Running')
                    pause = True
                    cv2.circle(frame, (x, y), 12, (0,255,255), 2)
                    cv2.circle(frame_tracks, (x, y), 12, (0,255,255), 2)

            cv2.putText(frame, "-"+category+" vel-"+str(vel), 
                        (x+15,y-5), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
            
            if category == "Unknown" and vel < vel_thresh:
                detection[6] = "Object"
                detection[8] = False
                detection[14] = ""
            elif (category == "Unknown" or category == "Object") and vel > vel_thresh:
                detection[6] = "Person"
                detection[8] = False
                detection[14] = ""
            elif category == "Person" and vel < vel_thresh:
                detection[6] = "Still Person"
            elif category == "Still Person" and vel > vel_thresh:
                detection[6] = "Person"
    
    
    # detect fainting or falling down
    helper.detect_fainting(predictions_list, heights_list)
    
    
    # find abandoned luggage and flag people
    helper.detect_abandoned_objects(predictions_list, fps)    
    
    
    # detect people loitering
    helper.detect_loitering(predictions_list, fps)
    
    
    cv2.putText(frame,"frame "+str(nr_frame), (15,20), cv2.FONT_HERSHEY_PLAIN, 1, 
                (255,255,255), 1)
    
    
    top_string = ""
    for p in predictions_list:
        if p[1] == 0:
            string = str(p[10])
            for b in p[11]:
                string += "," + str(b)
          
            cv2.putText(frame, string, (p[2][-1][0],p[2][-1][1]-5), 
                        cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
            cv2.putText(frame_tracks, string, (p[2][-1][0],p[2][-1][1]-5), 
                        cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
    
            x = p[2][-1][0]
            y = p[2][-1][1]
            w = p[3][-1][0]
            h = p[3][-1][1]
            if p[8]:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)  
                cv2.rectangle(frame_tracks, (x,y), (x+w,y+h), (0,0,255), 2)  
                if p[14] != "":
                    top_string += ("-" + p[14])
            else:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)   
                cv2.rectangle(frame_tracks, (x,y), (x+w,y+h), (0,255,0), 2)
    
    
    if top_string == "":
        color = (0,255,0)
    else:
        color = (0,0,255)
    cv2.rectangle(frame_tracks, (0,0), (width,20), color, -1)
    cv2.putText(frame_tracks, top_string, (0,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    

    display = np.hstack((frame_tracks,frame))
    cv2.imshow("Frame", display)
    if pause:
        cv2.waitKey(0)
        pause=False
    
    
cap.release()
cv2.destroyAllWindows()


