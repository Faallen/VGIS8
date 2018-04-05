# -*- coding: utf-8 -*-

"""
Image Processing and Computer Vision Mini-Project
Group 843
"""

import cv2
import numpy as np
import helper
from scipy.spatial import distance
from random import randint
import scipy.io as sio


video1 = 'soccer_match_1.avi'
video2 = 'soccer_match_2.avi'
video3 = 'soccer_match_3.avi'
video4 = 'soccer_match_4.avi'


def detect_people(video):
    
    cap = cv2.VideoCapture(video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   
    nr_frame = -1
    
    frame_points = np.array([[510, 75], [180, 320], [440, 480], [1640, 480], 
                             [1855, 360], [1530, 105], [1280, 50], [640, 50]])
    court_points = np.array([[500, 50], [500, 450], [610, 450], [1200, 450], 
                             [1300, 450], [1300, 50], [1100, 50], [595, 50]])
    H, status = cv2.findHomography(frame_points, court_points)
    
    predictions_list = []
    kalman_filters_list = []
    predictions_data = []
    
    while(cap.isOpened()):
        
        nr_frame += 1
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        thresh = cv2.threshold(frame_gray, 140, 255, cv2.THRESH_BINARY)[1]
        
        # dilate and erode
        kernel_eli = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
        thresh2 = cv2.dilate(thresh, kernel_eli, iterations=2)
        thresh3 = cv2.erode(thresh2, kernel_eli, iterations=1)
        
        thresh_c = cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR)
            
        img, contours, hierarchy = cv2.findContours(
                thresh3.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        court = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY)[1]
        court = cv2.cvtColor(court,cv2.COLOR_GRAY2BGR)
        tracks_image = np.ones(thresh_c.shape)
        
        candidates = list()
        for c in contours:
            if cv2.contourArea(c) < 200:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(thresh_c, (x, y), (x + w, y + h), (0, 255, 255), 2)
          
            xc, yc = helper.get_court_point(H, x+int(w/2), y+int(h))
            
            if 540 < x < 1380:
                standard_height = (1/8)*y + 50
            else:
                standard_height = (1/10)*y + 30
            
            # Evaluate ratio of foreground
            wdt = max(x+int(standard_height/3),x+w)
            hgt = y+int(standard_height)
            pixel_values = thresh[y:hgt, x:wdt]
            area = pixel_values.shape[0] * pixel_values.shape[1] 
            foreground = (pixel_values == 255).sum()
            ratio = foreground/area
            if ratio < 0.15:
                continue
            
            candidates.append([x, y+h-int(standard_height), 
                               x+int(standard_height/3), y+h, ratio, c])

        blobs, cnt = helper.detect_connected_blobs(candidates, thresh_c)
        
        
        people = list()
        index = -1
        for c in cnt:
            index += 1
            if blobs[index]:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(thresh_c, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                xc, yc = helper.get_court_point(H, x+int(w/2), y+int(h))
                
                if 440 < x < 1480:
                    standard_height = (1/8)*y + 55
                else:
                    standard_height = (1/10)*y + 30
                
                # Find the convexity defects of the contour
                hull = cv2.convexHull(c, returnPoints = False)
                defects = cv2.convexityDefects(c, hull)
                
                if defects is None:
                    continue
            
                # Detect tall blobs
                if (h-standard_height) > 10:
                    cv2.rectangle(thresh_c, (x, y), (x + w, y + h), (0, 0, 255), 4)
                    
                    # Sort defects by distance/depth
                    aux = defects[:,0]
                    sorted_defects = aux[aux[:,3].argsort()[::-1]]
                    
                    split = False
                    for i in range(len(sorted_defects)):
                        s, e, f, d = sorted_defects[i]
                        start = tuple(c[s][0])
                        end = tuple(c[e][0])
                        far = tuple(c[f][0])
                        cv2.line(thresh_c, start, end, [0,255,0], 2)
                        cv2.circle(thresh_c, far, 5, [0,0,255], -1)
                        
                        # The point selected is the defect point with the largest 
                        # depth and a maximum absolute gradient of 1.5. The point
                        # should not be in the top or bottom fourth of the region.
                        gradient = helper.calc_gradient(start, end, far)
                        if gradient <= 1.5 and (y+int(h/4)) < far[1] < (y+int(3*h/4)):
                            cv2.circle(thresh_c, far, 5, [0,0,255], 7)
                            split = True
                            break
                        
                    if split:
                        person1 = np.where(c[:,0,1] > far[1])
                        person1_contour = c[person1]
                        person2 = np.where(c[:,0,1] < far[1])
                        person2_contour = c[person2]
                        
                        (x1, y1, w1, h1) = cv2.boundingRect(person1_contour)
                        cv2.rectangle(thresh_c, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 255), 2) 
                        (x2, y2, w2, h2) = cv2.boundingRect(person2_contour)
                        cv2.rectangle(thresh_c, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 255), 2)
        
                        x1c, y1c = helper.get_court_point(H, x1+int(w1/2), y1+int(h1))
                        x2c, y2c = helper.get_court_point(H, x2+int(w2/2), y2+int(h2))
                        people.append([x1c, y1c, cv2.boundingRect(person1_contour)])
                        people.append([x2c, y2c, cv2.boundingRect(person2_contour)])
                
                
                # To contain more than one person, the height of the bounding box must be 
                # less than five times the width and the contour of the region must be
                # longer than the bounding box perimeter
                elif h < 5*w and cv2.arcLength(c,1) > (2*w+2*h):
                            
                    split = False
                    for i in range(len(defects)):
                        s, e, f, d = defects[i,0]
                        start = tuple(c[s][0])
                        end = tuple(c[e][0])
                        far = tuple(c[f][0])
                        
                        # Find the point that is located on the upper edge of the region 
                        # and has a y-value greater than both the convexity defect’s 
                        # start and end point
                        if far[1] < (y+int(h/4)) and start[1] < far[1] and end[1] < far[1]:
                            cv2.circle(thresh_c, far, 5, [0,0,255], -1)
                            cv2.circle(thresh_c, start, 5, [0,255,255], -1)
                            cv2.circle(thresh_c, end, 5, [255,255,0], -1)
                            split = True
                            break;
                        
                    if split:
                        person1 = np.where(c[:,0,0] > far[0])
                        person1_contour = c[person1]
                        person2 = np.where(c[:,0,0] < far[0])
                        person2_contour = c[person2]
                        
                        (x1, y1, w1, h1) = cv2.boundingRect(person1_contour)
                        cv2.rectangle(thresh_c, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 255), 2) 
                        (x2, y2, w2, h2) = cv2.boundingRect(person2_contour)
                        cv2.rectangle(thresh_c, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 255), 2)
            
                        x1c, y1c = helper.get_court_point(H, x1+int(w1/2), y1+int(h1))
                        x2c, y2c = helper.get_court_point(H, x2+int(w2/2), y2+int(h2))
                        people.append([x1c, y1c, cv2.boundingRect(person1_contour)])
                        people.append([x2c, y2c, cv2.boundingRect(person2_contour)])
                
                else:
                    people.append([xc, yc, cv2.boundingRect(c)])
        
           
        # Kalman Filter
        for person in people:
            loc_x = person[0]
            loc_y = person[1]
            cv2.circle(court, (loc_x, loc_y), 2, (255, 255, 0), 10)
            
            if nr_frame == 0:
                helper.new_kalman_filter(kalman_filters_list, predictions_list, 
                                         loc_x, loc_y, predictions_data, person)
            
            if nr_frame > 0:
                
                # get the last locations of each target
                previous_locations = np.zeros((len(predictions_list),2), np.int16)
                for p in range(len(predictions_list)):
                    previous_locations[p] = predictions_list[p][-1]
                
                distances = np.zeros(len(previous_locations))
                for pl in range(len(previous_locations)):
                    distances[pl] = abs(distance.euclidean(previous_locations[pl], 
                             (loc_x, loc_y)))
                
                sorted_indexes = distances.argsort()
                added = False
                for idx_dist in sorted_indexes:
                    # adds point to a kalman filter if the distance is less than
                    # 30 and a point has not already been added to that list
                    # does so in order of minimum distance to points in lists
                    if distances[idx_dist] < 30 and not predictions_list[idx_dist][0]:
                        kalman = kalman_filters_list[idx_dist]
                        prediction = kalman.predict()
                        measurement = np.array([[np.float32(loc_x)], 
                                                 [np.float32(loc_y)]])
                        kalman.correct(measurement)
                        predictions_list[idx_dist].append([int(prediction[0]), 
                                        int(prediction[1])])
                        predictions_list[idx_dist][0] = True
                        predictions_list[idx_dist][1] = 0
                        predictions_data[idx_dist].append([int(prediction[0]-person[2][2]/2),
                                        int(prediction[1]-person[2][3]),person[2][2],person[2][3]])
                        added = True
                        break
                
                # for a detection not assigned to a kalman filter a new kalman 
                # filter is created
                if not added:  
                    helper.new_kalman_filter(kalman_filters_list, predictions_list,
                                             loc_x, loc_y, predictions_data, person)
            

        # kalman filters with no assigned detection are continued based on 
        # predicted new positions until 10 frames in a row
        for pred in range(len(predictions_list)):
            if not predictions_list[pred][0] and (predictions_list[pred][1] < 6):
                kalman = kalman_filters_list[pred]
                prediction = kalman.predict()
                predictions_list[pred].append([int(prediction[0]),int(prediction[1])])
                predictions_list[pred][1] += 1
                predictions_data[pred].append([int(prediction[0]-predictions_data[pred][-1][2]/2),
                                int(prediction[1]-predictions_data[pred][-1][3]),
                                predictions_data[pred][-1][2],predictions_data[pred][-1][3]])
       
        
        for i in range(len(predictions_list)):
            predictions_list[i][0] = False
        
            
        helper.paint(court, predictions_list)
        helper.draw_on_frame(frame, thresh_c, court, frame_points, court_points)
        
        both = np.vstack((frame,thresh_c))
        all_in_one = np.vstack((both,court))
        
        resized = cv2.resize(all_in_one, (int(width/2), int(height/2)*3), 
                             interpolation=cv2.INTER_AREA)
        cv2.imshow("Frame", resized)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    nr=0
    for p in predictions_list:
        if len(p) < 100:
            continue
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        nr +=1
        for i in range(len(p)-1): 
            if i<2:
                continue
            cv2.line(tracks_image, (p[i][0],p[i][1]), (p[i+1][0],p[i+1][1]), 
                     (b,g,r))
    cv2.imwrite("tracks4.jpg", tracks_image)
   
    sio.savemat('data_video4.mat', {'data_video4':predictions_data})
     
    cap.release()
    cv2.destroyAllWindows()
       


detect_people(video4)

