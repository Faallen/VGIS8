# -*- coding: utf-8 -*-
"""
VGIS Group 843
Rita and Atanas

Helper methods
"""

import cv2
import numpy as np
from scipy.spatial import distance
from random import randint
import math


'''    
Get the background of the video obtained through temporal median filter
input: video feed
output: background image
'''    
def get_background(video0):
    
    cap = cv2.VideoCapture(video0)
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    int_frm = int(fps/2.0)
    nr_frames = int(n_frames/int_frm)
    
    total_frames = np.zeros((nr_frames, height, width), dtype=np.uint8)
    median = np.zeros((height, width), dtype=np.uint8)
    
    i = 0
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        if not ret:
            break
              
        # add frame to array every half second
        if i%int_frm == 0 and i/int_frm < len(total_frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (15, 15), 0) 
            total_frames[int(i/int_frm)] = gray
            
        i+=1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # for each pixel get the median value
    for j in range(len(total_frames[0])):
        for k in range(len(total_frames[0,0])):        
            e = sorted(total_frames[:,j,k])
            median[j,k] = e[int(len(e)/2.0)]
    
    cv2.imwrite('median.png', median)
    
    return median   



'''
Merge blobs of the same person, that became separated after segmentation
input: list of contours, list of heights
output: fused blobs
'''
def blob_fusion(contours, heights_list):
    
    sum1 = 0
    candidates = np.zeros((len(contours),5))
    idx = 0
    for c in contours:
        if cv2.contourArea(c) < 50:
            continue
        (x1, y1, w, h) = cv2.boundingRect(c)
        x2 = x1 + w
        y2 = y1 + h        
        sum1 += w*h
        
        candidates[idx] = [x1, y1, x2, y2, 1]
        idx += 1
    
    candidates = candidates[:idx]
    
    if len(candidates) == 0:
        return []
    
    l1 = sum1/len(candidates)
    l1 = l1-l1/10
    heights_list = sorted(heights_list)
    m_val = heights_list[int(len(heights_list)/2)]
    l2 = m_val/2
    
    n = len(candidates)
    n_new = n
    iterate = True
    
    while iterate:
        
        n = n_new
        merged = False
        candidates_new = np.zeros((max(1,len(candidates)-1),5))
        
        for i in range(len(candidates)):
            for j in range(len(candidates)):
                if i != j:
                    (x1i, y1i, x2i, y2i, b) = candidates[i]
                    (x1j, y1j, x2j, y2j, b) = candidates[j]
                    
                    # fusion condition 1
                    if ((x2i-x1i)*(y2i-y1i) < l1) or ((x2j-x1j)*(y2j-y1j) < l1):
                            
                        # fusion condition 2
                        if (abs(y2i-y1j) < l2) or (abs(y2j-y1i) < l2):
                        
                            # fusion condition 3
                            if (x1i > x1j and not (x1i+10) > x2j) or (
                                    x1j > x1i and not (x1j+10) > x2i):
                                x1_new = min(x1i, x2i, x1j, x2j)
                                y1_new = min(y1i, y2i, y1j, y2j)
                                x2_new = max(x1i, x2i, x1j, x2j)
                                y2_new = max(y1i, y2i, y1j, y2j)
                                
                                candidates[i][4] = 0
                                candidates[j][4] = 0
                                merged = True
                                break
            if merged:
                break
        
        if merged:
            idx = 0
            for c in candidates:
                if c[4] == 1:
                    candidates_new[idx] = c
                    idx += 1
            
            for i in range(len(candidates_new)):
                if candidates_new[i][4] == 0:
                    candidates_new[i] = [x1_new, y1_new, x2_new, y2_new, 1]
                    break
            
            candidates = candidates_new    
              
        n_new = len(candidates)        
        iterate = n != n_new
    
    return candidates



'''
Create a new Kalman filter
input: list of kalman filters, predictions list, a detected person and blob id
'''
def new_kalman_filter(kalman_filters_list, predictions_list, person, blob_id):
    loc_x, loc_y, w, h = person
    # KalmanFilter(number_of_dynamic_parameters, number_of_measurement_parameters)
    # dynamic parameters: x-position, y-position, x-velocity and y-velocity
    # measurements: x- and y-positions for each frame
    kalman = cv2.KalmanFilter(4, 2) 
    # matrix H relates the state xk to the measurement zk
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    # matrix A is the transition matrix that relates the state x at the previous 
    # time step k-1 to the state at the current step k
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],
                                       np.float32)
    # process noise covariance Q
    kalman.processNoiseCov = np.array([[0.002,0,0,0],[0,0.002,0,0],[0,0,0.003,0],
                                       [0,0,0,0.003]], np.float32)
    # measurement noise covariance R
    kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 0.00002
    
    # prediction of the next step from the previous state
    kalman.predict()
    # correction of the state using a new observed measurement
    measurement = np.array([[np.float32(loc_x)], [np.float32(loc_y)]])
    kalman.correct(measurement)
    kalman_filters_list.append(kalman)
    # Predictions list
    # Index 0 - indicates whether a new point has already been added to a kalman
    # filter in a frame, this way blobs close to each other are not added to 
    # the same list
    # Index 1 - number of frames on which the kalman filter has not detected 
    # new measurements
    # Index 2 - list of predictions of the Kalman filter (The first values in 
    # the list are the measuremets because the prediction is (0,0))
    # Index 3 - list of width and height of blob
    # Index 4 - velocity of the blob
    # Index 5 - direction of the blob
    # Index 6 - category (unknown, person, object, still person)
    # Index 7 - track color
    # Index 8 - flag for suspicious behavior
    # Index 9 - flag for possible suspicious behavior
    # Index 10 - ID of track
    # Index 11 - list of people merged in the blob
    # Index 12 - flag that indicates if blob merged with another blob and was lost
    # Index 13 - list of histograms that categorize people merged together
    # Index 14 - string with description of suspicious behavior
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    blob_id[0] = blob_id[0] + 1
    predictions_list.append([[True], 0, [[int(loc_x), int(loc_y)]], [[w, h]], 
                             0, [0,0], "Unknown", [r,g,b], False, False, 
                             blob_id[0], [], False, [], ""])



'''
Tracking object with Kalman Filter
input: list of people detected in the frame, list of predictions, list of
kalman filters, frame rate of video and blob id
'''
def kalman_filter_tracking(people, predictions_list, kalman_filters_list, 
                           fps, blob_id):

    for person in people:
        loc_x, loc_y, w, h = person
       
        previous_locations = np.zeros((len(predictions_list),2), np.int16)
        for p in range(len(predictions_list)):
            previous_locations[p] = predictions_list[p][2][-1]
        
        distances = np.zeros(len(previous_locations))
        for pl in range(len(previous_locations)):
            distances[pl] = abs(distance.euclidean(previous_locations[pl], 
                     (loc_x, loc_y)))
       
        sorted_indexes = distances.argsort()
        added = False
        for idx_dist in sorted_indexes:
            # adds point to a Kalman filter if the distance is less than
            # 30 and a point has not already been added to that list,
            # does so in order of minimum distance to points in lists
            if distances[idx_dist] < 30 and not predictions_list[idx_dist][0]:
                kalman = kalman_filters_list[idx_dist]
                prediction = kalman.predict()
                measurement = np.array([[np.float32(loc_x)], [np.float32(loc_y)]])
                kalman.correct(measurement)
                predictions_list[idx_dist][0] = True
                predictions_list[idx_dist][1] = 0
                predictions_list[idx_dist][2].append([int(prediction[0]), 
                                int(prediction[1])])
                predictions_list[idx_dist][3].append([w, h])
                predictions_list[idx_dist][12] = False
                added = True
                break
       
        # for a detection not assigned to a Kalman filter a new Kalman 
        # filter is created
        if not added:  
            new_kalman_filter(kalman_filters_list, predictions_list, person, 
                              blob_id)
       
    
    idx_list = []
    # Kalman filters with no assigned detection are continued based on 
    # predicted new positions until 5 frames in a row
    for pred in range(len(predictions_list)):
        if not predictions_list[pred][0]:                
            if predictions_list[pred][1] < 6:
                kalman = kalman_filters_list[pred]
                prediction = kalman.predict()
                predictions_list[pred][2].append([int(prediction[0]),int(prediction[1])])
                predictions_list[pred][3].append([predictions_list[pred][3][-1][0], 
                                 predictions_list[pred][3][-1][1]])
            predictions_list[pred][1] += 1
        
        # delete detection from predictions_list if the detection was not merged
        # with another one and it has been lost for more than 3 seconds; or if 
        # the detection was merged but the lifespan of detection was only 1 second
        # thus being ignored; or if the detection was not merged and the period 
        # of detection was smaller than 1 sec and the detection has been lost for
        # a few frame; and if the detection is not an object classified as suspicious
        if ((predictions_list[pred][1] > fps*3 and not predictions_list[pred][12
            ]) or (predictions_list[pred][12] and (len(predictions_list[pred][2
            ]) < 1*fps)) or (not predictions_list[pred][12] and len(predictions_list[pred][
            2]) < fps and predictions_list[pred][1] > 5) and (not (predictions_list[pred
            ][6] == "Object" and predictions_list[pred][8]) )):
    
            idx_list.append(pred)
     
    idx_list.reverse()
    for idx in idx_list:
        del(predictions_list[idx])
        del(kalman_filters_list[idx])            

    for i in range(len(predictions_list)):
        predictions_list[i][0] = False

    

'''
Draw predicted tracks on frame
input: frame and predictions list
'''
def draw_tracks(frame, predictions_list):
    c = 0
    for p in predictions_list:
        c += 1
        for i in range(len(p[2])-1): 
            x1 = p[2][i][0]
            y1 = p[2][i][1]
            w1 = p[3][i][0]
            h1 = p[3][i][1]
            x2 = p[2][i+1][0]
            y2 = p[2][i+1][1]
            w2 = p[3][i+1][0]
            h2 = p[3][i+1][1]
            
            cv2.line(frame, (int(x1+w1/2),int(y1+h1)), (int(x2+w2/2),int(y2+h2)), 
                     (p[7][0],p[7][1],p[7][2]),2)



'''
Detect when people are occluded and track them once the occlusion ends
input: current frame, list of predictions, frame rate and list of kalman filters 
'''
def handle_occlusion(frame, predictions_list, fps, kalman_filters_list):

    # if detection is lost checks if the blob in the previous frame intersects
    # almost completely with onother in the current frame, thus checking if
    # blobs merged
    for lost_p in predictions_list:
        # if the detection has not been lost for long, and its detection lasted  
        # at least a second, and the blob is still not classified as merged
        if 0 < lost_p[1] < fps and len(lost_p[2]) > fps and not lost_p[12]: 
            
            for candidate_p in predictions_list:
                # if the predictions are not the same, and the second blob being
                # evaluated has not stopped being tracked, and the lost blob is 
                # not already flagged as having merged  with the candidate
                if lost_p == candidate_p or candidate_p[1] != 0 or len(
                        candidate_p[2]) < 2 or lost_p[10] in lost_p[11]:
                    continue
                
                x1_lost_p = lost_p[2][-1][0]
                y1_lost_p = lost_p[2][-1][1]
                x2_lost_p = lost_p[3][-1][0] + x1_lost_p
                y2_lost_p = lost_p[3][-1][1] + y1_lost_p
                
                x1_cand_p = candidate_p[2][-1][0]
                y1_cand_p = candidate_p[2][-1][1]
                x2_cand_p = candidate_p[3][-1][0] + x1_cand_p
                y2_cand_p = candidate_p[3][-1][1] + y1_cand_p
              
                # calculate area of intersection between the lost blob and the
                # candidate for having merged with it
                intersection = max(0, min(x2_lost_p,x2_cand_p) - max(x1_lost_p,
                                   x1_cand_p)) * max(0, min(y2_lost_p,
                                   y2_cand_p) - max(y1_lost_p,y1_cand_p))
                
                area = (x2_lost_p-x1_lost_p) * (y2_lost_p-y1_lost_p)
                
                # if the intersection area is approximate to the area of the
                # lost blob
                if area-area/10 <= intersection <= area: 
                    lost_p[12] = True
                    # add the lost blob to the list of merged blobs of the candidate
                    candidate_p[11].append(lost_p[10])
                    
                    # calculate color histogram of person lost
                    mask_lost_p = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
                    mask_lost_p[y1_lost_p:y2_lost_p, x1_lost_p:x2_lost_p] = 255
                    
                    hist_lost_p = cv2.calcHist([frame], [0, 1, 2], mask_lost_p, 
                                           [8, 8, 8],[0, 256, 0, 256, 0, 256])
                    
                    # store histogram of lost blob in the candidate list for 
                    # merged blobs
                    candidate_p[13].append([lost_p[10], hist_lost_p])
                    
                    # if the own candidate's color histogram information is not
                    # already stored in its list, do so
                    add = True
                    for i in candidate_p[13]:
                        if candidate_p[10] == i[0]:
                            add = False
                    
                    if add:
                        x1_cand_p = candidate_p[2][-2][0]
                        y1_cand_p = candidate_p[2][-2][1]
                        x2_cand_p = candidate_p[3][-2][0] + x1_cand_p
                        y2_cand_p = candidate_p[3][-2][1] + y1_cand_p
                        
                        mask_cand_p = np.zeros((frame.shape[0],frame.shape[1]),np.uint8)
                        mask_cand_p[y1_cand_p:y2_cand_p, x1_cand_p:x2_cand_p] = 255
                        
                        hist_cand_p = cv2.calcHist([frame], [0, 1, 2], mask_cand_p, 
                                               [8, 8, 8],[0, 256, 0, 256, 0, 256])
                        
                        candidate_p[13].append([candidate_p[10], hist_cand_p])
                            
                               
    # if a new detection appears checks if it intersects almost completely with 
    # another one in the previous frame that had a blob merged, thus checking 
    # if occlusion has ended
    idx_p1 = -1
    del_list = []
    for new_p in predictions_list:
        idx_p1 += 1
        if len(new_p[2]) == 1:
            
            for candidate_p in predictions_list:
                # if candidate track corresponds to more than one blob
                if new_p == candidate_p or len(candidate_p[11]) == 0:
                    continue
                
                x1_new_p = new_p[2][-1][0]
                y1_new_p = new_p[2][-1][1]
                x2_new_p = new_p[3][-1][0] + x1_new_p
                y2_new_p = new_p[3][-1][1] + y1_new_p
                
                x1_cand_p = candidate_p[2][-2][0]
                y1_cand_p = candidate_p[2][-2][1]
                x2_cand_p = candidate_p[3][-2][0] + x1_cand_p
                y2_cand_p = candidate_p[3][-2][1] + y1_cand_p
                
                # calculate the area of the intersection between the new detection
                # and the candidate to be the origin merged blob in the previous
                # frame
                intersection = max(0, min(x2_new_p, x2_cand_p) - max(x1_new_p, 
                                   x1_cand_p)) * max(0, min(y2_new_p, 
                                   y2_cand_p) - max(y1_new_p, y1_cand_p))
                
                area = (x2_new_p-x1_new_p) * (y2_new_p-y1_new_p)
                
                # if the is significant intersection then occlusion has ended
                if area-area/10 <= intersection <= area:
                    
                    # calculate color histogram of new detection
                    mask_new_p = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
                    mask_new_p[y1_new_p:y2_new_p, x1_new_p:x2_new_p] = 255
                    
                    hist_new_p = cv2.calcHist([frame], [0, 1, 2], mask_new_p, 
                                           [8, 8, 8],[0, 256, 0, 256, 0, 256])
                    
                    # find which of the original blobs before occlusion 
                    # has most similar color histogram to the new detection in 
                    # order to find out which of the tracks the new detection 
                    # belongs to
                    max_similarity = 0
                    id_orig_blob = 0
                    index = 0
                    i = 0
                    for merged_p in candidate_p[13]:
                        similarity = cv2.compareHist(hist_new_p, merged_p[1], 
                                                     cv2.HISTCMP_CORREL)
                        
                        if similarity > max_similarity:
                            max_similarity = similarity
                            id_orig_blob = merged_p[0]
                            index = i
                        
                        i += 1
                            
                    # assign the found detection to the correct track
                    if id_orig_blob != 0:
                        for j in range(len(predictions_list)):
                            if predictions_list[j][10] == id_orig_blob:
                                kalman = kalman_filters_list[j]
                                measurement = np.array([[np.float32(x1_new_p)], 
                                                         [np.float32(y1_new_p)]])
                                kalman.correct(measurement)
                                predictions_list[j][1] = 0
                                predictions_list[j][2].append([x1_new_p, y1_new_p])
                                predictions_list[j][3].append([x2_new_p-x1_new_p, 
                                                y2_new_p-y1_new_p])
                                predictions_list[j][12] = False
                            break
                        
                        del(candidate_p[13][index])
                        for i in range(len(candidate_p[11])):
                            if candidate_p[11][i] == id_orig_blob:
                                break
                        del(candidate_p[11][i])
                        del_list.append(idx_p1)        
                                
    # delete new detection if the original track was found
    del_list.reverse()
    for idx in del_list:
        del(predictions_list[idx])
        del(kalman_filters_list[idx])                 
                    
                    
    # for each detection, if the detection has merged another one, checks if 
    # that one is still merged or if it has been detected again, deletes if so
    # if the merged blob is not in the predictions_list anymore deletes it
    for pred in predictions_list:
        del_list = []
        for i in range(len(pred[11])):
            found_i = False
            for p in predictions_list:
                if p[10] == pred[11][i]:
                    if not p[12]:
                        del_list.append(i)
                    found_i = True
            
            if not found_i:
                del_list.append(i)
        
        del_list.reverse()
        for j in del_list:
            del(pred[11][j])
        
        
        del_list = []
        for i in range(len(pred[13])):
            if pred[13][i][0] == pred[10]:
                continue
            found_i = False
            for p in predictions_list:
                if p[10] == pred[13][i][0]:
                    if not p[12]:
                        del_list.append(i)
                    found_i = True
            
            if not found_i:
                del_list.append(i)
                
        del_list.reverse()        
        for j in del_list:
            del(pred[13][j])



'''
Extract features: velocity and direction
input: list of predictions and frame rate
'''
def extract_features(predictions_list, fps):
    
    for p in predictions_list: 
        if len(p[2]) > fps-1:
            x0 = p[2][-int(fps/2)][0]
            y0 = p[2][-int(fps/2)][1]
            x1 = p[2][-1][0]
            y1 = p[2][-1][1]
            dx = x1 - x0
            dy = y1 - y0
            dist = np.sqrt(dx**2 + dy**2)
            velocity = int(dist/1)
            dx_norm = dx / dist
            dy_norm = dy / dist
            
            p[4] = velocity
            p[5][0] = dx_norm
            p[5][1] = dy_norm



'''
Detect fainting
input: list of predictions and heights list
'''
def detect_fainting(predictions_list, heights_list):
    
    for pred in predictions_list:
        width = pred[3][-1][0]
        height = pred[3][-1][1]
        hw_ratio = height/width
        
        heights_list = sorted(heights_list)
        median_height = heights_list[int(len(heights_list)/2)]
        # if detection is a still person, the detection has not been lost for
        # more than 5 frames and there is only one person in the blob
        if pred[1] < 6 and pred[6] == "Still Person" and len(pred[11]) == 0:
            if hw_ratio < 0.8 and width >= 9*median_height/10 and width <= 11*median_height/10:
                pred[8] = True
                pred[14] = "Fainting"
        
        # if person is no longer laying down desclassify them as suspicious
        if pred[14] == "Fainting":
            if hw_ratio > 1:
                pred[8] = False
                pred[14] = ""



'''
Detect abandoned objects and picked-up objects and flag people
input: predictions list and frame rate
'''
def detect_abandoned_objects(predictions_list, fps):
    
    t_abandoned = 1*fps
    
    for person in predictions_list:
        if person[1] > 6:
            continue
        for object_ in predictions_list:
            if object_[1] > 6 or person == object_:
                continue
            
            # if object has been detected for more than t_abandoned, and the 
            # object has not yet been flagged as suspicious, and the detection
            # time of the person is longer than the one of the object
            if object_[6] == "Object" and len(object_[2]) > t_abandoned and person[
                    6] == ("Person" or "Still Person") and object_[8
                    ] == False and len(person[2]) > len(object_[2]):
                
                # if distance between object and person is greater than the 
                # person's height
                if abs(distance.euclidean(person[2][-1], 
                                          object_[2][-1])) > person[3][-1][1]:
                    
                    # find index in list of locations of when the person might
                    # have dropped the object
                    index = len(person[2]) - len(object_[2])
                   
                    x1_obj = object_[2][0][0]
                    y1_obj = object_[2][0][1]
                    x2_obj = object_[3][0][0] + x1_obj
                    y2_obj = object_[3][0][1] + y1_obj
                    
                    # for the possible moments when the object might have been 
                    # dropped verify if at any moment the bounding boxes of the 
                    # person and object intersect if so, classify both as suspicious
                    for i in range(index-10, index+10):
                        x1_person = person[2][i][0]
                        y1_person = person[2][i][1]
                        x2_person = person[3][i][0] + x1_person
                        y2_person = person[3][i][1] + y1_person
                        
                        intersection = max(0, min(x2_obj,x2_person) - max(x1_obj, 
                                           x1_person)) * max(0, min(y2_obj, 
                                           y2_person) - max(y1_obj,y1_person))
                        
                        area = (x2_obj-x1_obj) * (y2_obj-y1_obj)
                        
                        if area-area/5 <= intersection <= area:
                            person[8] = True
                            object_[8] = True
                            object_[14] = "Abandoned object"
                
      
    for obj in predictions_list:  
        # if object is suspicious and was lost
        if obj[6] == "Object" and obj[8] and obj[1]:
            for person in predictions_list:
                if person == obj:
                    continue

                x1_obj = obj[2][-1][0]
                y1_obj = obj[2][-1][1]
                x2_obj = obj[3][-1][0] + x1_obj
                y2_obj = obj[3][-1][1] + y1_obj
                
                x1_person = person[2][-1][0]
                y1_person = person[2][-1][1]
                x2_person = person[3][-1][0] + x1_person
                y2_person = person[3][-1][1] + y1_person
                
                intersection = max(0, min(x2_obj, x2_person) - max(x1_obj, 
                                   x1_person)) * max(0, min(y2_obj,
                                   y2_person) - max(y1_obj, y1_person))
                
                area = (x2_obj-x1_obj) * (y2_obj-y1_obj)
                
                # if a person's bounding box intersects intirely with the
                # bounding box of a suspicious object, the person is classified  
                # as being possibly suspicious
                if intersection == area:
                    person[9] = True
                
                # if, after being classified as possibly suspicious the person
                # leaves the location where the object was and the detection of
                # the object is still lost it means that the person picked-up
                # the object
                if person[9]:
                    intersection = max(0, min(x2_obj, x2_person) - max(x1_obj,
                                       x1_person)) * max(0, min(y2_obj,
                                       y2_person) - max(y1_obj, y1_person))

                    if intersection == 0:
                        person[8] = True
                        obj[8] = False
                        person[14] = "Abandoned object picked-up"



'''
Detect people loitering
input: list of predictions and frame rate
'''
def detect_loitering(predictions_list, fps):
    
    t_loitering = 60*fps
    
    for detection in predictions_list:
        # if detection is still being predicted, and is not already flagged,
        # it is a person and has been detected for longer than a minute
        if (detection[1] < 6 and not detection[8]) and (detection[6
           ] == "Person" or detection[6] == "Still Person") and len(
           detection[2]) > t_loitering:
           
            detection[8] = True
            detection[14] = "Loitering"



