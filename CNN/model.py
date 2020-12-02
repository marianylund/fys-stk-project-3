# Contains the structure on the network, forward and backwards passes
import pathlib

from tensorflow.keras.optimizers import Adam, Adagrad, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Transfer learning:
# https://keras.io/api/applications/
from tensorflow.keras.applications import VGG16

class Model():
    def __init__(self, cfg):
        print("Start creating model")        
        self.cfg = cfg
        self.choose_optimizer()
        self.choose_model()
        self.model.summary()
        print("Finished")

    def choose_optimizer(self):
        self.lr_schedule = self.cfg.learning_rate # TODO: lr schedulers https://keras.io/api/optimizers/learning_rate_schedules/

        if self.cfg.optimizer == "sgd" or self.cfg.optimizer == "sdg":
            self.optimizer = SGD(learning_rate=self.lr_schedule)
        elif self.cfg.optimizer == "adam":
            self.optimizer = Adam(learning_rate=self.lr_schedule)
        elif self.cfg.optimizer == "adagrad":
            self.optimizer = Adagrad(learning_rate=self.lr_schedule)
        else:
            raise Exception("Could not find optimizer: " + self.cfg.optimizer) 
        
    def choose_model(self):
        print("Model type: " + self.cfg.model_type)
        if(self.cfg.model_type == "simplest"):
            self.simplest()
        elif self.cfg.model_type == "VG16_transfer_learning":
            self.VG16_transfer_learning()
        else:
            raise Exception("This model type was not found: " + self.cfg.model_type)
        self.model.compile(optimizer = self.optimizer, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy']) 

    def simplest(self):
        self.model = Sequential([
            Flatten(input_shape=(self.cfg.image_size, self.cfg.image_size, 1)),
            Dense(1, activation='relu'),
            Dense(10, activation='softmax'),
        ])    

    def VG16_transfer_learning(self):
        vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(self.cfg.image_size, self.cfg.image_size, 1))
        vgg16.trainable = False # will not retrain the weights of the pretrained model
        #vgg16.summary()
        self.model = Sequential([
            vgg16,
            Flatten(),
            Dense(1, activation='relu'),
            Dense(10, activation='softmax'),
        ])
        
    @staticmethod
    def get_model_name(k):
        return 'model_'+str(k)+'.h5'