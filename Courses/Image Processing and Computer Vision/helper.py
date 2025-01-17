# -*- coding: utf-8 -*-

"""
Image Processing and Computer Vision Mini-Project
Group 843

Helper methods
"""

import cv2
import os
import re
import numpy as np


'''
Generate video from images
'''
def generate_video(nr_video):
    dir_path = '.\\Data_set\\Images\\' + nr_video + '\\'
    fps = 33.0
    
    # Sort images by name
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
            
    images = sorted(os.listdir(dir_path), key=alphanum_key)
    
    # Determine width and height of frames
    frame = cv2.imread(dir_path+images[0])
    height, width, c = frame.shape
    
    # Define o codec
    file_name = 'soccer_match_' + nr_video + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(file_name, fourcc, fps, (width,height))   
    
    for image in images:
        frame = cv2.imread(dir_path+image)
        out.write(frame)
    
    out.release()
    cv2.destroyAllWindows()



'''
Draw lines and points on frame
'''
def draw_on_frame(grey_frame, binary_frame, court, points_frame, points_court):
    
    pts = points_frame.reshape((-1,1,2))
    cv2.polylines(grey_frame, [pts], True, (0,255,255))
    
    for i in range(len(points_frame)):
        cv2.circle(grey_frame, (points_frame[i,0],points_frame[i,1]), 2, (255,0,0), 7)
        cv2.circle(court, (points_court[i,0],points_court[i,1]), 2, (255,0,0), 5)
    
    cv2.rectangle(court, (500, 50), (1300, 450), (0, 255, 255), 2)
    



'''
Through homography, given a point in the frame, return the corresponding
point in the court
'''
def get_court_point(H, point_x, point_y):
    point_in_video = np.array([[point_x, point_y, 1]])
    matrix = np.dot(H, point_in_video.T)
    x = matrix[0] / matrix[2]
    y = matrix[1] / matrix[2]
    return x, y

        

''' 
The gradient is calculated for the line from the defect point perpendicular 
on the line between the convexity defect start and end points
'''
def calc_gradient(start, end, far):
    
    x1 = start[0]; y1 = start[1]
    x2 = end[0]; y2 = end[1]
    x3 = far[0]; y3 = far[1]
    
    k = ((y2-y1) * (x3-x1) - (x2-x1) * (y3-y1)) / ((y2-y1)**2 + (x2-x1)**2)
    x4 = x3 - k * (y2-y1)
    y4 = y3 + k * (x2-x1)
    
    gradient = (far[1]-int(y4))/(far[0]-int(x4))
    return abs(gradient)



'''
Returns list of blobs that are not connected
'''
def detect_connected_blobs(candidates, thresh_c):
    
    blobs = np.ones(len(candidates))
    
    for i in range(len(candidates)):
        for j in range(i+1,len(candidates)):
            
            XA1 = candidates[i][0]
            YA1 = candidates[i][1]
            XA2 = candidates[i][2]
            YA2 = candidates[i][3]
            XB1 = candidates[j][0]
            YB1 = candidates[j][1]
            XB2 = candidates[j][2]
            YB2 = candidates[j][3]
            
            # Area of rectangles
            SA = (XA2-XA1)*(YA2-YA1)
            SB = (XB2-XB1)*(YB2-YB1)
            
            # Area of the intersection
            SI = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))
            # Area of the union
            SU = SA + SB - SI
            # Ratio
            ratio = SI / SU
        
            if ratio > 0.45:
                if candidates[i][4] > candidates[j][4]:
                    blobs[j,0] = 0
                else:
                    blobs[i,0] = 0
    
    items = [contour[5] for contour in candidates]
    return blobs, items



'''
Draw tracks
'''
def paint(court, predictions_list):
    for p in predictions_list:
        for i in range(len(p)-1): 
            if i<2:
                continue
            cv2.line(court, (p[i][0],p[i][1]), (p[i+1][0],p[i+1][1]), 
                     (int(p[2][0]/8),0,int(p[2][1]/2)))
  


'''
Create new Kalman filter
'''
def new_kalman_filter(kalman_filters_list, predictions_list, loc_x, loc_y, 
                      predictions_data, person):
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
    # First flag in predictions_list indicates whether a new point has already
    # been added to a kalman filter in a frame, this way blobs close to each
    # other are not added to the same list. Number in second position indicates 
    # the number of frames the kalman filter has not detected new measurements.
    predictions_list.append([[True], 0, [loc_x, loc_y]])
    predictions_data.append([[int(loc_x-person[2][2]/2), int(loc_y-person[2][3]), 
                              person[2][2], person[2][3]]])
    
    