import cv2

cap = cv2.VideoCapture('vtest.avi')

for i in range(100):
    ret, frame = cap.read()

    cv2.imwrite("images/{0}.jpg".format(str(i)), frame, )
    print(i)
