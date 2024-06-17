import cv2
import imutils
from skimage.transform import AffineTransform, warp

def preprocess_(img):   
    horimage = cv2.flip(img, 1)    
    crop_image = img[30:200, 30:220]
    res = cv2.resize(crop_image, (225, 225))    
    rotateimage = imutils.rotate(res, angle=360)
    rotateimage = cv2.flip(rotateimage, 1)    
    vertimage = cv2.rotate(img, cv2.ROTATE_180)
    transform = AffineTransform(translation=(-205,0))
    warp_image = warp(img,transform, mode="reflect") 
    translate = cv2.normalize(src=warp_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)    
    
    return horimage, res, rotateimage, vertimage, translate