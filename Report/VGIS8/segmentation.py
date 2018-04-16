import numpy as np
import cv2


def _controlls(show):
    for item in show:
        cv2.imshow(item[0], item[1])

    output = False
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        output = True

    return output


cap = cv2.VideoCapture('Walk1.mpg')
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
kernel = np.ones((3, 3), np.uint8)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = -1

while cap.isOpened():
    frame_count += 1
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = fgbg.apply(gray)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown == 255] = 0
    markers = cv2.watershed(frame, markers)
    frame[markers == -1] = [255, 0, 0]
    if _controlls((('bg', sure_bg), ('fg', sure_fg), ('unknown', unknown), ('frame', frame))):
        break



cap.release()
cv2.destroyAllWindows()
