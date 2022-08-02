import cv2
import base64
import camera as cm
import cv2
import json


def capture_img():
    return cm.getImg()


def imgToBase64(img):
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text


def startEvent(farm_id, position):
    # start
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
