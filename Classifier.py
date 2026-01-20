from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Add
from tensorflow.keras.optimizers import Adam
import numpy as np

class classifier():
      def __init__(self, data, label, modelpath):
          self.data = data
          self.label = label
          self.model = modelpath
      def train(self):
            X_train, X_test, y_train, y_test = train_test_split(
                                            self.data, self.label, test_size=0.2, random_state=42)
            
            import Optimization
            opti = Optimization.opt()
            lr = opti[1]
            d = opti[2]/5000
            b = np.round(opti[3]*100)
            # defining model
            print("Fitting Model...........")
            model1=Sequential()
            # adding convolution layer
            model1.add(Conv2D(32,(3,3),activation='relu',input_shape=(225, 225, 3)))
            # adding pooling layer
            model1.add(MaxPool2D(2,2))     
                    
            
            model2=Sequential()
            # adding convolution layer
            model2.add(Conv2D(32,(3,3),activation='relu',input_shape=(225, 225, 3)))
            # adding pooling layer
            model2.add(MaxPool2D(2,2))
            
            
            mergedOut = Add()([model1.output,model2.output])
            mergedOut = Flatten()(mergedOut)    
            mergedOut = Dense(100,activation='relu')(mergedOut)
            mergedOut = Dense(2,activation='softmax')(mergedOut)
            
            newModel = Model([model1.input,model2.input], mergedOut)
            opt = Adam(learning_rate=lr, decay=d)
            newModel.compile(loss='sparse_categorical_crossentropy', optimizer = opt,  metrics=['accuracy'])
            print("Fitting the Model........................")
            
            # newModel.fit([X_train, X_train], y_train, epochs=100, batch_size=b)
            # newModel.save(self.modelpath) 
            
def prediction(Feature, modelpath):
    from tensorflow.keras.models import load_model
    import warnings
    warnings.filterwarnings("ignore")
    model_name = modelpath
    #loading the model    
    model = load_model(model_name)
    #defining the classes
    classes = ["Autism", "Normal"] 
    prediction=np.reshape(Feature, [1, 225, 225, 3])    
    pred = model.predict([prediction, prediction])
    pred=np.argmax(pred[0])
    if pred==0:
        print("\nPredicted class : ",classes[pred],"\n")
        return classes[pred]
    else:
        print("\nPredicted class : ",classes[pred],"\n")  
        return classes[pred]