import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(True):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([30, 50, 0])
    upper_blue = np.array([88, 255, 139])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    if(cv.waitKey(1) &0xFF==ord("q")) :
        break
cv.destroyAllWindows()

