
----------
File mini_project_ipcv.py
----------

- detect_people(video)
Main function which receives as input one of the videos from the data set. It does all the processing and, in the end, produces an image with the tracks found in world coordinates (court as seen from above).


----------
File helper.py
----------

Contains a series of helper methods.

- generate_video(nr_video)
Generates a video from the images in the data set.

- draw_on_frame(grey_frame, binary_frame, court, points_frame, points_court)
Draws lines and points on frame.

- get_court_point(H, point_x, point_y)
Through homography, given a point in the frame, returns the corresponding point in the court.

- calc_gradient(start, end, far)
The gradient is calculated for the line from the defect point perpendicular on the line between the convexity defect start and end points.

- detect_connected_blobs(candidates, thresh_c)
Returns list of blobs that are not connected.

- paint(court, predictions_list)
Draws tracks on frame.

- new_kalman_filter(kalman_filters_list, predictions_list, loc_x, loc_y)
Creates new Kalman filter.

- save_results_image(predictions_list, tracks_image)
Draws and saves results image.

- draw_groud_truth_image(data_file)
Draws ground truth image from xml file.


