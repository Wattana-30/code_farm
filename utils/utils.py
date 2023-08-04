from cgitb import grey
from unittest import expectedFailure
import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
from datetime import datetime
import os
from PIL import Image
import io
import json
from uuid import uuid4
import statistics as st
from ast import Num
import time
# crud
from crud import plant_features_crud
from crud import farm_crud


# schemas
from schemas import plant_features_schemas
from schemas import farm_schemas



def create_path(image_name: str, folder: str):
    return os.path.join(os.getcwd(), "images", folder, f"{image_name}.jpg")


def save_image(filename: str, base64_str: str):
    image = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(image))
    img.save(filename, 'jpeg')

def findLeafArea(image):
    #image = cv2.imread(image)
    blank_mask = np.zeros(image.shape, dtype=np.uint8)
    original = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # set lower and upper color limits
    lower_val = (0, 18, 0)
    upper_val = (60, 255, 120)
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_val, upper_val)
    # Perform morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Find contours and filter for largest contour
    # Draw largest contour onto a blank mask then bitwise-and
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    cv2.fillPoly(blank_mask, [cnts], (255, 255, 255))
    blank_mask = cv2.cvtColor(blank_mask, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("show :",blank_mask)
    # cv2.waitKey(1000)
    number_of_white_pix = np.sum(blank_mask == 255)
    convertToCm = 0.0264583333 * number_of_white_pix
    print('result :', convertToCm,'cm^2 ')
    return convertToCm


def contour(img):
    # convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # mask of green (36,0,0) ~ (70, 255,255)
    mask1 = cv2.inRange(hsv, (0, 0, 101), (30, 255, 255))

    # mask o yellow (15,0,0) ~ (36, 255, 255)
    mask2 = cv2.inRange(hsv, (0, 2, 16), (20, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    target = cv2.bitwise_and(img, img, mask=mask)
    return target


def contouring(image):
    # Read image, create blank masks, color threshold

    blank_mask = np.zeros(image.shape, dtype=np.uint8)
    original = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, (0, 0, 101), (30, 255, 255))
    # mask o yellow (15,0,0) ~ (36, 255, 255)
    mask2 = cv2.inRange(hsv, (0, 2, 16), (20, 255, 255))

    # final mask and masked
    mask = cv2.bitwise_or(mask1, mask2)

    # Perform morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours and filter for largest contour
    # Draw largest contour onto a blank mask then bitwise-and
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    cv2.fillPoly(blank_mask, [cnts], (255, 255, 255))
    blank_mask = cv2.cvtColor(blank_mask, cv2.COLOR_BGR2GRAY)

    result = cv2.bitwise_and(original, original, mask=blank_mask)

    # Crop ROI from result
    x, y, w, h = cv2.boundingRect(blank_mask)
    ROI = result[y:y+h, x:x+w]
    cv2.bitwise_not(blank_mask)
    # image in this case is your image you want to eliminate black
    result[np.where((result == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    return result


def imgToBase64(img):
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text


def stringToImage(base64_string):
    decoded_data = base64.b64decode(base64_string)
    np_data = np.frombuffer(decoded_data, np.uint8)
    return cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)



def capture_img(url: str):
    video_capture = cv2.VideoCapture(url)
    # if not video_capture.isOpened():
    #     return False
    count = 0
    while True:
        count += 1
        _, frame = video_capture.read()
        if count > 20:
            break

    video_capture.release()
    return frame


def publish_to_web(mqtt, img, NDVI, position):
    status = "end" if position == "Point_6" else "next"

    data = {
        "originalImage": img,
        "ndviImage": NDVI,
        "status": status
    }

    payload = json.dumps(data)

    mqtt.publish('plant/dashboard/image', payload)
    print('Successful')

    

def findNdviValue(NDVI):
    b, g, r = cv2.split(NDVI)
    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom==0] = 0.01
    ndvi = (r.astype(float) - b) / bottom # THIS IS THE CHANGED LINE
    
    print('min :'+str(ndvi.min()))
    print('max :'+str(ndvi.max()))
    print('mean :'+str(ndvi.mean()))

    # color_mapped_prep = ndvi.astype(np.uint8)
    # color_mapped_image = cv2.applyColorMap(color_mapped_prep, fastiecm)


    return ndvi.mean(),ndvi.std()
    
def startEvent(mqtt, farm_id, position, green_id, dt):    
    #start

    # img = capture_img('http://192.168.1.109:1234/video')
    img_noir = capture_img('http://192.168.1.103:1234/video')

    # get path image
    # rgb_path = f"{position}_{dt}_rgb".replace(":", "_").split('.')[0]
    ndvi_path = f"{position}_{dt}_ndvi".replace(":", "_").split('.')[0]
    noir_path = f"{position}_{dt}_noir".replace(":", "_").split('.')[0]

    # get full path
    # rgb_path = create_path(rgb_path, 'RGB')
    noir_path = create_path(noir_path, 'NIR')
    ndvi_path = create_path(ndvi_path, 'NDVI')
    # if(True):
    #     try:
    #         # save original image
    #         cv2.imwrite(rgb_path, img)
    #     except: 
    #         print("have no image -0")

    if(True):
        try:
            # save noir image
            cv2.imwrite(noir_path, img_noir)
        except:
            print("have no image noir camera") 

    mean_ndvi,std_ndvi, = findNdviValue(noir_path)

    plt.imsave(ndvi_path, NDVI)
    # mean_rgb,std_rgb = findRGBValue(img)
    print(mean_ndvi)
    leaf_area_index = findLeafArea(img_noir) #change from img rgb


    # pack and publish to web
    publish_to_web(mqtt, imgToBase64(img).decode(), imgToBase64(NDVI).decode(), position)
    #publish_to_web(mqtt, data['image'], data['image'], data['position'])
  

    item = plant_features_schemas.PlantFeaturesBase(
        green_id=green_id,
        plant_loc=position,
        rgb_path=1,
        std_rgb=1,
        mean_rgb=1,
        noir_path=noir_path,
        mean_ndvi=mean_ndvi,
        std_ndvi=std_ndvi,
        ndvi_path=ndvi_path,
        leaf_area_index=leaf_area_index,
        created_at=datetime.now()
    )

    plant_features_crud.create_plant_features(farm_id, item)

