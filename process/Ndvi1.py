import cv2
import numpy

def ndvi():
    noir_image = cv2.imread('Image/4444.jpg',cv2.IMREAD_COLOR)
    color_image = cv2.imread('Image/444444.jpg',cv2.IMREAD_COLOR)   
    red_channel = color_image[:,:,0]/256.0  
    nir_channel = noir_image[:,:,0]/256.0  
    green_channel = noir_image[:,:,1]/256.0  
    blue_channel = noir_image[:,:,2]/256.0  




    # align the images  
    # Run the ECC algorithm. The results are stored in warp_matrix.  
    #   Find size of image1  
    warp_mode = cv2.MOTION_TRANSLATION  
    if warp_mode == cv2.MOTION_HOMOGRAPHY :   
        warp_matrix = numpy.eye(3, 3, dtype=numpy.float32)  
    else :  
        warp_matrix = numpy.eye(2, 3, dtype=numpy.float32)  
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

    # calculate gndvi_image  
    gndvi_image = (nir_channel - green_channel)/(nir_channel + green_channel)  
    gndvi_image = (gndvi_image+1)/2  
    gndvi_image = cv2.convertScaleAbs(gndvi_image*255)  
    gndvi_image = cv2.applyColorMap(gndvi_image, cv2.COLORMAP_JET)  


    # calculate bndvi_image  
    bndvi_image = (nir_channel - blue_channel)/(nir_channel + blue_channel)  
    bndvi_image = (bndvi_image+1)/2  
    bndvi_image = cv2.convertScaleAbs(bndvi_image*255)  
    bndvi_image = cv2.applyColorMap(bndvi_image, cv2.COLORMAP_JET)  
    cv2.imshow("show",ndvi_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return ndvi_image



ndvi()