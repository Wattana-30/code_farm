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


def find_leafAreaIndex(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh2 = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    number_of_black_pix = np.sum(thresh2 == 0)
    convertToCm = 0.0264583333 * number_of_black_pix
    return convertToCm


def ndvi(img):
    # Load image and convert to float - for later division
    im = img.astype(np.float64)
    # Split into 3 channels, discarding the first and saving the second as R, third as NearIR
    _, R, NearIR = cv2.split(im)
    # Compute NDVI values for each pixel
    NDVI = (NearIR - R) / (NearIR + R + 0.001)
    return NDVI


def imgToBase64(img):
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text


def stringToImage(base64_string):
    decoded_data = base64.b64decode(base64_string)
    np_data = np.frombuffer(decoded_data, np.uint8)
    return cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)


def contouring(rgb_path):
    # Read image, create blank masks, color threshold
    image = cv2.imread(rgb_path)
    blank_mask = np.zeros(image.shape, dtype=np.uint8)
    original = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([30, 50, 0])
    upper = np.array([88, 255, 139])
    mask = cv2.inRange(hsv, lower, upper)
    
    # Perform morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours and filter for largest contour
    # Draw largest contour onto a blank mask then bitwise-and
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    cv2.fillPoly(blank_mask, [cnts], (255,255,255))
    blank_mask = cv2.cvtColor(blank_mask, cv2.COLOR_BGR2GRAY)
    

    result = cv2.bitwise_and(original,original,mask=blank_mask)
  
    #Crop ROI from result
    x,y,w,h = cv2.boundingRect(blank_mask)
    ROI = result[y:y+h, x:x+w]
    cv2.bitwise_not(blank_mask)
    # image in this case is your image you want to eliminate black
    result[np.where((result==[0,0,0]).all(axis=2))] = [255,255,255] 
    return result


def publish_to_web(mqtt, img, NDVI, position):
    status = "next" if position == "Point_1" else "end"

    data = {
        "originalImage": img,
        "ndviImage": NDVI,
        "status": status
    }

    payload = json.dumps(data)

    mqtt.publish('plant/dashboard/image', payload)
    print('Successful')


def pack_save_insert_toweb(mqtt, payload):
    data = json.loads(payload)

    # get path image
    dt = datetime.now()
    rgb_path = f"Point_1_{dt}_rgb".replace(":", "").replace(".", "")
    ndvi_path = f"Point_1_{dt}_ndvi".replace(":", "").replace(".", "")

    # get full path
    rgb_path = create_path(rgb_path, 'RGB')
    ndvi_path = create_path(ndvi_path, 'NDVI')

    # base64 to image
    img = stringToImage(data['image'])

    # save original image
    # save_image(rgb_path, data['image'])
    cv2.imwrite(rgb_path, img)

    # find ndvi
    imgToFindValue = contouring(rgb_path)
    NDVI = ndvi(imgToFindValue)

    # save ndvi image
    # plt.imshow(NDVI)
    # plt.show()
    plt.imsave(ndvi_path, NDVI)
    NDVI = cv2.imread(ndvi_path)

    # pack and publish to web
    publish_to_web(mqtt, data['image'], imgToBase64(NDVI).decode(), data['position'])


    '''
        farm_id
        rgb_path
        ndvi_path
        leaf_area_index
        plan_loc
    '''

    farm_id = data['farm_id']
    leaf_area_index = find_leafAreaIndex(img)
    plan_loc = data['position']

    item = plant_features_schemas.PlantFeaturesBase(
        plant_loc=plan_loc,
        rgb_path=rgb_path,
        noir_path=str(uuid4()),
        ndvi_path=ndvi_path,
        leaf_area_index=leaf_area_index
    )

    plant_features_crud.create_plant_features(farm_id, item)


    # item = farm_schemas.FarmBase(farm_name='นันธิดา บ้านสวน')
    # farm_crud.create_farm(item)
