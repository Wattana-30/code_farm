import cv2
import base64
import numpy as np
import camera as cm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import contour as ct
import json 


def capture_img():
    return cm.getImg()


def find_leafAreaIndex(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 252, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    number_of_black_pix = np.sum(thresh2 == 0)
    convertToCm = 0.0264583333 * number_of_black_pix
    return convertToCm


def ndvi(img):
    # Load image and convert to float - for later division
    im = cv2.imread("Image/someleaf.jpg").astype(np.float64)
    # Split into 3 channels, discarding the first and saving the second as R, third as NearIR
    _, R, NearIR = cv2.split(im)
    # Compute NDVI values for each pixel
    NDVI = (NearIR - R) / (NearIR + R + 0.001)
    plt.imsave("Image/NDVI.jpg",NDVI)
    return NDVI


def imgToBase64(img):
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text


def startEvent(farm_id, position):    
    #start
    img = capture_img()
    base64_str = imgToBase64(img)
    data = {
        "farm_id": farm_id,
        "position": position,
        "image": base64_str.decode() 
    }

    data_str = json.dumps(data)
    return data_str
    ''' 
    imgToFindValue = ct.contouring()
    base64_rgb = imgToBase64(imgToFindValue)
    ValueofArea = find_leafAreaIndex(imgToFindValue)
    NDVI = ndvi(imgToFindValue)
    base64_NDVI = imgToBase64(NDVI)
    '''
   
