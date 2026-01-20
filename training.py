import cv2
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np

def main():
    
#%% Input Data

    path = "Dataset\\training"
    listpath = os.listdir(path)
    data = []
    label = []
    for fname in listpath:
        fpath = path + "\\" + fname
        listfile = os.listdir(fpath)
        for file in listfile:
            if 'Thumbs.db' not in file:
                Imgname = fpath + "\\" + file
                img = cv2.imread(Imgname)
                resizeImg = cv2.resize(img, (225, 225))
                
                if fname == "Autism":
                    label.append(0)
                    label.append(0)
                    label.append(0)
                    label.append(0)
                    label.append(0)
                elif fname == "Normal":
                    label.append(1)
                    label.append(1)
                    label.append(1)
                    label.append(1)
                    label.append(1)

#%% Preprocess Data
                
                import Preprocess
                
                preprocess = Preprocess.preprocess_(resizeImg)

#%% Feature Extraction

                import FeatureExtraction
                
                featExt = FeatureExtraction.feat_ext_(preprocess)

#%% FeatureFusion
                
                import FeatureFusion
                
                featFus = FeatureFusion.feat_fusion_(featExt, preprocess)
                
                data.append(featFus[0])
                data.append(featFus[1])
                data.append(featFus[2])
                data.append(featFus[3])
                data.append(featFus[4])                        
    
    data = np.asarray(data)
    label = np.asarray(label)

#%% Training the Classifier Model
                
    import Classifier
    Classifier.classifier(data, label, "Classifier")

#%% Existing Classifiers
    
    import Existing
    obj1=Existing.exist_ResNet(data, label, "ResNet")
    obj1.train()
    
    obj2=Existing.exist_AlexNet(data, label, "AlexNet")
    obj2.train()
    
    obj3=Existing.exist_cnn(data, label, "CNN")
    obj3.train()
    
    obj4=Existing.exist_DNN(data, label, "DNN")
    obj4.train()
    
if __name__ == '__main__':
    main()