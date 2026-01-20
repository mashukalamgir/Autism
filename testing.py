import cv2
import warnings
warnings.filterwarnings("ignore")

def main():

#%% Input Data

    ImgName = "Dataset\\validation\\Normal\\Img_15.jpg"
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

#%% Existing Classifiers
    
    import Existing
    Existing.ResNetprediction(featFus[0], "ResNet")
    
    Existing.AlexNetprediction(featFus[0], "AlexNet")
    
    Existing.CNNprediction(featFus[0], "CNN")
    
    Existing.DNNprediction(featFus[0], "DNN")

#%% Result
    
    import Result
    Result.plot()
    
if __name__ == '__main__':
    main()