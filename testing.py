import cv2
import warnings
import warnings;warnings.filterwarnings("ignore")
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

def main():

#%% Input Data

    ImgName = "Dataset/training/Normal/Img_8.jpg"
    img = cv2.imread(ImgName)
    resizeImg = cv2.resize(img, (225, 225))

#%% Preprocess Data
    
    import Preprocess
    preprocess = Preprocess.preprocess_(resizeImg)

#%% Feature Extraction
    
    import FeatureExtraction
    featExt = FeatureExtraction.feat_ext_(preprocess)

#%% Feature Fusion
        
    import FeatureFusion
    featFus = FeatureFusion.feat_fusion_(featExt, preprocess)

#%% Classification
    
    import Classifier
    Classifier.prediction(featFus[0], "Classifier")


    
if __name__ == '__main__':
    main()