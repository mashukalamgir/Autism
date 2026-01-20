import cv2
import warnings
warnings.filterwarnings("ignore")

ImgName = "Dataset\\validation\\Autism\\IMG_1970.jpg"
fldr = ImgName.split("\\")
img = cv2.imread(ImgName)
resizeImg = cv2.resize(img, (225, 225))
cv2.imwrite("GraphsAndImages\\Original.jpg", resizeImg)

import Preprocess
preprocess = Preprocess.preprocess_(resizeImg)
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\HorizontalIMG.jpg", preprocess[0])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\CropIMG.jpg", preprocess[1])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\RotateIMG.jpg", preprocess[2])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\VerticalIMG.jpg", preprocess[3])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\TranslateIMG.jpg", preprocess[4])

import FeatureExtraction
featExt = FeatureExtraction.feat_ext_(preprocess)
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature1.jpg", featExt[0][0])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature2.jpg", featExt[0][1])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature3.jpg", featExt[0][2])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature4.jpg", featExt[0][3])

cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature11.jpg", featExt[1][0])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature12.jpg", featExt[1][1])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature13.jpg", featExt[1][2])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature14.jpg", featExt[1][3])

cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature21.jpg", featExt[2][0])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature22.jpg", featExt[2][1])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature23.jpg", featExt[2][2])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature24.jpg", featExt[2][3])   

cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature31.jpg", featExt[3][0])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature32.jpg", featExt[3][1])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature33.jpg", featExt[3][2])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature34.jpg", featExt[3][3])

cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature41.jpg", featExt[4][0])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature42.jpg", featExt[4][1])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature43.jpg", featExt[4][2])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\FeatExt\\Feature44.jpg", featExt[4][3])
    
import FeatureFusion
featFus = FeatureFusion.feat_fusion_(featExt, preprocess)

cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\Featurefusion1.jpg", featFus[0])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\Featurefusion2.jpg", featFus[1])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\Featurefusion3.jpg", featFus[2])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\Featurefusion4.jpg", featFus[3])
cv2.imwrite("GraphsAndImages\\"+fldr[-2]+"\\Featurefusion5.jpg", featFus[4])