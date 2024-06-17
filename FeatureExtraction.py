import tensorflow as tf
import cv2
import numpy as np

def feat_ext_(imge):
    gridvalue = []
    for a in range(len(imge)):
        img = imge[a]
        
        width = img.shape[1]
    
        width_cutoff = width // 2
        left1 = img[:, :width_cutoff]
        right1 = img[:, width_cutoff:]
        img = cv2.rotate(left1, cv2.ROTATE_90_CLOCKWISE)
        width_cutoff = width // 2
        l1 = img[:, :width_cutoff]
        l2 = img[:, width_cutoff:]
        l1 = cv2.rotate(l1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        l2 = cv2.rotate(l2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.rotate(right1, cv2.ROTATE_90_CLOCKWISE)
        width_cutoff = width // 2
        r1 = img[:, :width_cutoff]
        r2 = img[:, width_cutoff:]
        r1 = cv2.rotate(r1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        r2 = cv2.rotate(r2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
        gridImg = [l2, l1, r2, r1]
        feat = []
        for i in range(len(gridImg)):            
            h = len(gridImg[i])
            w = len(gridImg[i][0])
            model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3, 3),activation ='relu', input_shape=(h,w,3))])
            layer_outputs = [layer.output for layer in model.layers]
            feature_map_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
            inputImg = np.reshape(gridImg[i], (1, h, w, 3))                                 
            feature_maps = feature_map_model.predict(inputImg)   
            feature_map = feature_maps
            size=feature_map.shape[1]
            feature_image = feature_map[0, :, :, 0]
            feature_image-= feature_image.mean()
            feature_image/= feature_image.std ()
            feature_image*=  64
            feature_image+= 128
            feature_image= np.clip(feature_image, 0, 255).astype('uint8')
            image_belt = image_belt[:, 0 * size : (0 + 110) * size] = feature_image 
            image_belt = cv2.cvtColor(image_belt, cv2.COLOR_RGB2BGR)
            image_belt = cv2.resize(image_belt, (h, w))
            feat.append(image_belt)
        gridvalue.append(feat)
    return gridvalue