from cgitb import grey
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
    blank_mask = np.zeros(image.shape, dtype=np.uint8)
    original = image.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
    number_of_white_pix = np.sum(blank_mask == 255)
    convertToCm = 0.0264583333 * number_of_white_pix
    print('result :', convertToCm,'cm^2 ')
    return convertToCm



def ndvi(color_image, noir_image):
    # extract nir, red green and blue channel  
    nir_channel = noir_image[:,:,0]/256.0  
    green_channel = noir_image[:,:,1]/256.0  
    blue_channel = noir_image[:,:,2]/256.0  
    red_channel = color_image[:,:,0]/256.0  

    # align the images  
    # Run the ECC algorithm. The results are stored in warp_matrix.  
    #   Find size of image1  
    warp_mode = cv2.MOTION_TRANSLATION  
    if warp_mode == cv2.MOTION_HOMOGRAPHY :   
        warp_matrix = np.eye(3, 3, dtype=np.float32)  
    else :  
        warp_matrix = np.eye(2, 3, dtype=np.float32)  
    number_of_iterations = 5000
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)  
    sz = color_image.shape  
    (cc, warp_matrix) = cv2.findTransformECC (color_image[:,:,1],noir_image[:,:,1],warp_matrix, warp_mode, criteria)  
    if warp_mode == cv2.MOTION_HOMOGRAPHY:  
       # Use warpPerspective for Homography   
       nir_aligned = cv2.warpPerspective (nir_channel, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)  
    else :  
        # Use warpAffine for nit_channel, Euclidean and Affine  
        nir_aligned = cv2.warpAffine(nir_channel, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);  

       # calculate ndvi  
    ndvi_image = (nir_aligned - red_channel)/(nir_aligned + red_channel)  
    ndvi_image = (ndvi_image+1)/2  
    ndvi_image = cv2.convertScaleAbs(ndvi_image*255)  
    ndvi_image = cv2.applyColorMap(ndvi_image, cv2.COLORMAP_JET)
  
    return ndvi_image  



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
        if count > 10:
            break

    cv2.waitKey(10)
    cv2.destroyAllWindows()
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
        #split just red dimention of ndvi
        ndviRed = NDVI[:,:,2]
        #convert ndvi chanel red to scale value between 0-1
        sc_ndvi = (ndviRed-ndviRed.min())/(ndviRed.max()-ndviRed.min())
        #find mean Value of ndvi
        #find standaard deviation of ndvi
        return sc_ndvi.mean(),sc_ndvi.std()


def findRGB(rgb):
        gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
        sc_gray = (gray-gray.min())/(gray.max()-gray.min())
        return sc_gray.mean(),sc_gray.std()



    
def startEvent(mqtt, farm_id, position, green_id, dt):    
    #start
    img = capture_img('http://192.168.1.109:1234/video')
    img_noir = capture_img('http://192.168.1.119:1234/video')

    # get path image
    rgb_path = f"{position}_{dt}_rgb".replace(":", "_").split('.')[0]
    ndvi_path = f"{position}_{dt}_ndvi".replace(":", "_").split('.')[0]
    noir_path = f"{position}_{dt}_noir".replace(":", "_").split('.')[0]

    # get full path
    rgb_path = create_path(rgb_path, 'RGB')
    noir_path = create_path(noir_path, 'NIR')
    ndvi_path = create_path(ndvi_path, 'NDVI')

    # save original image
    cv2.imwrite(rgb_path, img)
    # save noir image
    cv2.imwrite(noir_path, img_noir)
     

    NDVI = ndvi(img, img_noir)
    plt.imsave(ndvi_path, NDVI)
    NDVI = cv2.imread(ndvi_path)
    
    mean_ndvi,std_ndvi = findNdviValue(NDVI)
    mean_rgb,std_rgb = findRGB(img)
    print(mean_ndvi)
    print(std_ndvi)



    # pack and publish to web
    publish_to_web(mqtt, imgToBase64(img).decode(), imgToBase64(NDVI).decode(), position)
    #publish_to_web(mqtt, data['image'], data['image'], data['position'])
    leaf_area_index = findLeafArea(img)
    print(leaf_area_index)

    item = plant_features_schemas.PlantFeaturesBase(
        green_id=green_id,
        plant_loc=position,
        rgb_path=rgb_path,
        mean_rgb=mean_rgb,
        std_rgb=std_rgb,
        noir_path=noir_path,
        mean_ndvi=mean_ndvi,
        std_ndvi=std_ndvi,
        ndvi_path=ndvi_path,
        leaf_area_index=leaf_area_index,
        created_at=datetime.now()
    )

    plant_features_crud.create_plant_features(farm_id, item)

