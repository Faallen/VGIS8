import numpy as np
import cv2
import helper
from scipy.spatial import distance
from random import randint
import scipy.io as sio

cap = cv2.VideoCapture('Walk1.mpg')
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
kernel = np.ones((4, 4), np.uint8)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_count = -1
predictions_list = []
kalman_filters_list = []
predictions_data = []

while 1:
    frame_count += 1
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cont = np.zeros((height, width, 3), np.uint8)

    thresh_c = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    #thresh_c = cv2.threshold(thresh_c, 120, 255, cv2.THRESH_BINARY)[1]
    thresh_c = cv2.cvtColor(thresh_c, cv2.COLOR_GRAY2BGR)
    court = thresh_c.copy()
    im2, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    people = list()
    tracks_image = np.ones(thresh_c.shape)

    for c in contours:
        if cv2.contourArea(c) < 120:
            continue
        x, y, w, h = cv2.boundingRect(c)

        people.append([x, y, (x, y, w, h)])
        
        cv2.rectangle(cont, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.drawContours(cont, contours, -1, (0, 255, 0), 1)
    for person in people:
        
        loc_x = person[0]
        loc_y = person[1]
        
        if frame_count == 0:
                helper.new_kalman_filter(kalman_filters_list, predictions_list, 
                                         person[0], person[1], predictions_data, person)
        if frame_count > 0:
                
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
            if not predictions_list[pred][0] and (predictions_list[pred][1] < 5):
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

        both = np.vstack((frame, thresh_c))
        all_in_one = np.vstack((both, court))
        
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
    cv2.imshow('framesub', cont)
    #cv2.imshow('frame', oframe)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
