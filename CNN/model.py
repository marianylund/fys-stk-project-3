# Contains the structure on the network, forward and backwards passes
import pathlib
from yacs.config import CfgNode as CN

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class Model():

    def __init__(self, cfg):
        print("Start creating model")        
        self.cfg = cfg
        self.choose_model()
        self.model.summary()
        print("Finished")
    
    def choose_model(self):
        print("Model type: " + self.cfg.model_type)
        if(self.cfg.model_type == "simplest"):
            self.simplest()
        else:
            raise Exception("This model type was not found: " + self.cfg.model_type) 

    def simplest(self):
        self.model = Sequential([
            Flatten(input_shape=(self.cfg.image_size, self.cfg.image_size, 1)),
            Dense(1, activation='relu'),
            Dense(10, activation='softmax'),
        ])

        self.model.compile(optimizer = Adam(), 
                           loss='categorical_crossentropy', 
                           metrics=['accuracy'])
    
