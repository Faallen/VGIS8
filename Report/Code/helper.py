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


'''
Create a new Kalman filter
'''
def new_kalman_filter(kalman_filters_list, predictions_list, person):
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
    # Index 1 - number of frames the kalman filter has not detected new measurements
    # Index 2 - list of predictions of the Kalman filter (The first values in 
    # the list are the measuremets because the prediction is (0,0))
    # Index 3 - list of width and height of blob
    # Index 4 - velocity of the blob
    # Index 5 - direction of the blob
    # Index 6 - cetegory (unknown, person, object, still person)
    # Index 7 - track color
    # Index 8 - flag for suspicious behavior
    # Index 9 - flag for possible suspicious behavior
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    predictions_list.append([[True], 0, [[int(loc_x), int(loc_y)]], [[w, h]], 
                             0, [0,0], "Unknown", [r,g,b], False, False])
    
    
    
'''
Draw predicted tracks on frame
'''
def draw_tracks(frame, predictions_list):
    c = 0
    for p in predictions_list:
        c += 1
        for i in range(len(p[2])-1): 
            cv2.line(frame, (p[2][i][0],p[2][i][1]), (p[2][i+1][0],p[2][i+1][1]), 
                     (p[7][0],p[7][1],p[7][2]))



'''
Extract features: velocity, direction
'''
def extract_features(predictions_list, fps):
    
    for p in predictions_list: 
        if len(p[2]) > fps-1:
            x0 = p[2][-fps][0]
            y0 = p[2][-fps][1]
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
'''
def detect_fainting(predictions_list, frame):
    
    for pred in predictions_list:
        if pred[1] < 6:
            x = pred[2][-1][0]
            y = pred[2][-1][1]
            width = pred[3][-1][0]
            height = pred[3][-1][1]
            hw_ratio = height/width
            
            if hw_ratio < 0.9:
                cv2.rectangle(frame, (x,y), (x+width,y+height), (0,0,255), 2)
                cv2.putText(frame,"Fainting", (15,20), cv2.FONT_HERSHEY_PLAIN, 1, 
                            (255,255,255), 1)
    
    
    
'''    
Get the background of the video    
'''    
def getBackground(video0):
    
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
        if i%int_frm==0 and i/int_frm<len(total_frames):
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
            median[j,k] = e[int(round(len(e)/2.0))]
    
    cv2.imwrite('median.png', median)
    
    return median    
    


'''
Merge blobs of the same person, that became separated after segmentation
'''
def blob_fusion(contours, frame):
    
    sum1 = 0
    sum2 = 0
    candidates = np.zeros((len(contours),5))
    idx = 0
    for c in contours:
        if cv2.contourArea(c) < 50:
            continue
        (x1, y1, w, h) = cv2.boundingRect(c)
        x2 = x1 + w
        y2 = y1 + h        
        sum1 += (x2-x1)*(y2-y1)
        sum2 += y2-y1
        
        candidates[idx] = [x1, y1, x2, y2, 1]
        idx += 1
    
    candidates = candidates[:idx]
    
    if len(candidates) == 0:
        return []
    
    l1 = sum1/len(candidates)
    l2 = np.sqrt(sum2/len(candidates))
    
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
                    
                    if ((x2i-x1i) * (y2i-y1i) < l1) or ((x2j-x1j) * (y2j-y1j) < l1):
                        
                        if (y1j < (y2i+l2)) or (y1i < (y2j+l2)):
                            
                            if ((x1i >= x1j) and (x1i <= x2j)) or ((x2i >= x1j) and 
                                (x2i <= x2j)):
                                
                                hi = y2i - y1i
                                hj = y2j - y1j
                                
                                # extra condition
                                if (abs(y1j-y1i) < 2*hi) or (abs(y1i-y1j) < 1.5*hj):
                                
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
Split wide blobs that might have more than one person merged together
'''
def split_wide_blobs(contours, frame):
    
    candidates = []
    for c in contours:
        #if cv2.contourArea(c) < 150:
        #    continue
        (x, y, w, h) = cv2.boundingRect(c)
        #cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
        
        # Find the convexity defects of the contour
        hull = cv2.convexHull(c, returnPoints = False)
        defects = cv2.convexityDefects(c, hull)
        
        if defects is None:
            continue
                
        # To contain more than one person, the height of the bounding box  
        # must be less than five times the width and the contour of the 
        # region must be longer than the bounding box perimeter
        if h < 5*w and cv2.arcLength(c,1) > (2*w+2*h):
            
            split = False
            for i in range(len(defects)):
                s, e, f, d = defects[i,0]
                start = tuple(c[s][0])
                end = tuple(c[e][0])
                far = tuple(c[f][0])
                
                # Find the point that is located on the upper edge of the  
                # region and has a y-value greater than both the convexity  
                # defect’s start and end point
                #if far[1] < (y+int(h/2)) and 
                if start[1] < far[1] and end[1] < far[1]:
                    #cv2.circle(frame, far, 5, [0,0,255], -1)
                    #cv2.circle(frame, start, 5, [0,255,255], -1)
                    #cv2.circle(frame, end, 5, [255,255,0], -1)
                    split = True
                    break;
          
            if split:
                person1 = np.where(c[:,0,0] > far[0])
                person1_contour = c[person1]
                person2 = np.where(c[:,0,0] < far[0])
                person2_contour = c[person2]
                
                (x1, y1, w1, h1) = cv2.boundingRect(person1_contour)
                #cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0,0,255), 2) 
                (x2, y2, w2, h2) = cv2.boundingRect(person2_contour)
                #cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0,0,255), 2)
    
                candidates.append([x1, y1, w1, h1])
                candidates.append([x2, y2, w2, h2])
        
        else:
            candidates.append([x, y, w, h])

    return candidates



'''
Tracking object with Kalman Filter
'''
def kalman_filter_tracking(people, predictions_list, kalman_filters_list, 
                           fps, frame):
    
    # Kalman Filter
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
            # Adds point to a Kalman filter if the distance is less than
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
                added = True
                break
       
        # For a detection not assigned to a Kalman filter a new Kalman 
        # filter is created
        if not added:  
            new_kalman_filter(kalman_filters_list, predictions_list, person)
       
    
    idx_list = []
    # Kalman filters with no assigned detection are continued based on 
    # predicted new positions until 5 frames in a row
    for pred in range(len(predictions_list)):
        if not predictions_list[pred][0]:
            if predictions_list[pred][1] < 7:
                kalman = kalman_filters_list[pred]
                prediction = kalman.predict()
                predictions_list[pred][2].append([int(prediction[0]),int(prediction[1])])
                predictions_list[pred][3].append([predictions_list[pred][3][-1][0], 
                                 predictions_list[pred][3][-1][1]])
            predictions_list[pred][1] += 1
        
        if predictions_list[pred][1] > fps*2 or (len(predictions_list[pred][2])<fps and predictions_list[pred][1]>5):
            
            idx_list.append(pred)
     
        
    for idx in idx_list:
        if predictions_list[idx][6] == "Object" and predictions_list[idx][8]: 
            continue
        del(predictions_list[idx])
        del(kalman_filters_list[idx])
            

    for i in range(len(predictions_list)):
        predictions_list[i][0] = False
    
    
    draw_tracks(frame, predictions_list)



'''
Detect abandoned objects and picked-up objects and flag people
'''
def detect_abandoned_objects(predictions_list, fps, frame):
    
    t_abandoned = 1*fps
    
    #if not nr_frame % int(fps/2):
    for obj1 in predictions_list:
        if obj1[1] > 6:
            continue
        for obj2 in predictions_list:
            if obj2[1] > 6 or obj1 == obj2:
                continue
            
            if obj2[6] == "Object" and len(obj2[2]) > t_abandoned and obj1[6] == (
                    "Person" or "Still Person") and obj2[8] == False:

                if abs(distance.euclidean(obj1[2][-1], obj2[2][-1])) > obj1[3][-1][1]:

                    for i in range(t_abandoned):
                        
                        if abs(distance.euclidean(obj1[2][-i], obj2[2][-i])) < (
                                obj1[3][-i][1]*2+20):
                            obj1[8] = True
                            obj2[8] = True
                
                
    for obj2 in predictions_list:  
        if obj2[6] == "Object" and obj2[8] and obj2[1]:
            for obj1 in predictions_list:
                if obj1 == obj2:
                    continue

                XA1 = obj2[2][-1][0]
                YA1 = obj2[2][-1][1]
                XA2 = obj2[3][-1][0] + XA1
                YA2 = obj2[3][-1][1] + YA1
                
                XB1 = obj1[2][-1][0]
                YB1 = obj1[2][-1][1]
                XB2 = obj1[3][-1][0] + XB1
                YB2 = obj1[3][-1][1] + YB1
                
                # Compute the area of the intersection, which is a rectangle too:
                SI = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))
                
                area = (XA2-XA1) * (YA2-YA1)
                
                cv2.rectangle(frame, (XA1,YA1), (XA2,YA2), (255,0,255), 2) 
                cv2.rectangle(frame, (XB1,YB1), (XB2,YB2), (255,0,255), 2) 
                
                if SI == area:
                    obj1[9] = True
                
                if obj1[9]:
                    SI = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))
   
                    if SI == 0:
                        obj1[8] = True
                        obj2[8] = False

    