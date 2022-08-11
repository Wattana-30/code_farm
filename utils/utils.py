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


def contouring(rgb_path):
    # Read image, create blank masks, color threshold
    image = cv2.imread(rgb_path)
    blank_mask = np.zeros(image.shape, dtype=np.uint8)
    original = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([93, 132, 147])
    upper = np.array([62, 86, 98])
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


def capture_img(url: str):
    video_capture = cv2.VideoCapture(url)
    if not video_capture.isOpened():
        return False
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



def startEvent(mqtt, farm_id, position):    
    #start
    img = capture_img('http://192.168.1.102:1234/video')
    img_noir = capture_img('http://192.168.1.109:1234/video')

    # get path image
    dt = datetime.now()
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
     
    # find ndvi
    # try:
    #     NDVI = ndvi(img, img_noir)

    #     # save ndvi image
    #     # plt.imshow(NDVI)
    #     # plt.show()
    #     plt.imsave(ndvi_path, NDVI)
    #     NDVI = cv2.imread(ndvi_path)
    # except:
    #     NDVI = img

    NDVI = ndvi(img, img_noir)
    plt.imsave(ndvi_path, NDVI)
    NDVI = cv2.imread(ndvi_path)

    # pack and publish to web
    publish_to_web(mqtt, imgToBase64(img).decode(), imgToBase64(NDVI).decode(), position)
    #publish_to_web(mqtt, data['image'], data['image'], data['position'])

    '''
        farm_id
        rgb_path
        noir_path
        ndvi_path
        leaf_area_index
        plan_loc
    '''

    leaf_area_index = find_leafAreaIndex(img)

    item = plant_features_schemas.PlantFeaturesBase(
        plant_loc=position,
        rgb_path=rgb_path,
        noir_path=noir_path,
        ndvi_path=ndvi_path,
        leaf_area_index=leaf_area_index,
        created_at=datetime.now()
    )

    plant_features_crud.create_plant_features(farm_id, item)


    # item = farm_schemas.FarmBase(farm_name='นันธิดา บ้านสวน')
    # farm_crud.create_farm(item)




# 109
# sudo python /home/kor/Desktop/process/stream.py & > /home/kor/Desktop/log.txt 2>&1

# 102
# sudo python /home/kor/code_farm/process/stream.py & > /home/kor/log.txt 2>&1

'''
sudo nano /etc/xdg/lxsession/LXDE-pi/autostart

@lxpanel --profile LXDE-pi
@pcmanfm --desktop --profile LXDE-pi

@xset s off
@xset -dpms
@xset s noblank

@firefox --kiosk http://192.168.1.100:3000/
'''
