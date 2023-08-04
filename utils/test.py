import cv2
import numpy as np
import fastiecm as fastiecm 

def findNdviValue():
    cap = cv2.VideoCapture('http://192.168.1.103:1234/video')
    while True :
        _,frame = cap.read()

        b, g, r = cv2.split(NDVI)
        bottom = (r.astype(float) + b.astype(float))
        bottom[bottom==0] = 0.01
        ndvi = (r.astype(float) - b) / bottom # THIS IS THE CHANGED LINE
    
        print('min :'+str(ndvi.min()))
        print('max :'+str(ndvi.max()))
        print('mean :'+str(ndvi.mean()))

        color_mapped_prep = ndvi.astype(np.uint8)
        color_mapped_image = cv2.applyColorMap(color_mapped_prep, fastiecm)
  
        cv2.imshow("",color_mapped_image)
        if(cv2.waitKey(1)&0xff==ord("q")) :
            break
    cap.release()
    cv2.destroyAllWindows()


    return color_mapped_image


