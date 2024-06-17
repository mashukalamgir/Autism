from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2

def feat_fusion_(FeatureMap, img):
    FusedFeature = []
    for i in range(len(FeatureMap)):
        fused = []
        for ik in range(len(FeatureMap[0])):
            Feat = (FeatureMap[i][ik] ).astype("uint8")
            image = img_to_array(Feat)
            image = np.expand_dims(image, axis=0)
            image = resnet50.preprocess_input(image)            
            fused.append(image[0].astype("uint8"))
        FusedFeature.append(fused) 
    res = []
    for k in range(len(FusedFeature)):
        out_arr = np.vstack((FusedFeature[k][0], FusedFeature[k][2]))
        out_arr1 = np.vstack((FusedFeature[k][1], FusedFeature[k][3]))
        out_arr2 = np.hstack((out_arr, out_arr1))
        out_arr2 = cv2.cvtColor(img[k], cv2.COLOR_RGB2GRAY)
        out_arr2 = cv2.cvtColor(out_arr2, cv2.COLOR_GRAY2RGB)
        res.append(out_arr2)
    return res