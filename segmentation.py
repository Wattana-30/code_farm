import cv2
import base64
import numpy as np
from camera import getImg


def capture_img():
    return getImg()


def find_leafAreaIndex(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 252, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, thresh3 = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    number_of_black_pix = np.sum(thresh2 == 0)
    convertToCm = 0.0264583333 * number_of_black_pix
    return convertToCm


def imgToBase64(img):
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text


def startEvent():
    '''
    1.cap image
    2.convert img to str value
    3.find area index and ndvi
    4.send value to mySQl
    5.send mqtt to plc for finished
    '''
    img = capture_img()

    if img:
        base64_Str = imgToBase64(img)
        ValueofArea = find_leafAreaIndex(img)
        return base64_Str
    else:
        print('camera not open')
