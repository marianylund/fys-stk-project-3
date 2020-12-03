# Contains the structure on the network, forward and backwards passes
import pathlib

from keras.optimizers import Adam, Adagrad, SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D

# Transfer learning:
# https://keras.io/api/applications/
from keras.applications import MobileNetV2

class Model():
    def __init__(self, cfg):
        print("Start creating model")        
        self.cfg = cfg
        self.input_shape = (self.cfg.image_size, self.cfg.image_size, self.cfg.channels)
        self.weight_init = "glorot_normal" # TODO: choose
        self.choose_optimizer()
        self.choose_model()
        
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
        if(self.cfg.model_type == "Triple_model"):
            self.Triple_model()
        elif self.cfg.model_type == "MobileNetV2_transfer_learning":
            self.MobileNetV2_transfer_learning()
        elif self.cfg.model_type == "simple_NN":
            self.simple_NN()
        else:
            raise Exception("This model type was not found: " + self.cfg.model_type)
        self.model.compile(optimizer = self.optimizer, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        self.model.summary() 

    def simplest(self):
        self.model = Sequential([
            Flatten(input_shape=self.input_shape),
            Dense(1, activation='relu'),
            Dense(self.cfg.num_classes, activation='softmax'),
        ])

    def simple_NN(self):
        self.cfg.NN_act0 = "relu"
        self.cfg.NN_act1 = "relu"
        self.cfg.NN_act2 = "relu"
        self.cfg.NN_act3 = "relu"

        self.model = Sequential()
        self.model.add(Flatten(input_shape=self.input_shape))
        #self.model.add(Dense(1024, activation=self.cfg.NN_act, kernel_initializer = self.weight_init))
        #self.model.add(Dense(800, activation=self.cfg.NN_act, kernel_initializer = self.weight_init))
        self.model.add(Dense(512, activation=self.cfg.NN_act0, kernel_initializer = self.weight_init))
        self.model.add(Dense(400, activation=self.cfg.NN_act1, kernel_initializer = self.weight_init))
        self.model.add(Dense(200, activation=self.cfg.NN_act2, kernel_initializer = self.weight_init))
        self.model.add(Dense(50, activation=self.cfg.NN_act3, kernel_initializer = self.weight_init))
        
        self.model.add(Dense(self.cfg.num_classes, activation='softmax', kernel_initializer = self.weight_init))

    def Triple_model(self):
        self.cfg.CNN_model_l0_act = "relu"
        self.cfg.CNN_model_layers1 = 3
        self.cfg.CNN_model_l1 = 32
        self.cfg.CNN_model_l1_act = "relu"
        self.cfg.CNN_model_layers2 = 3
        self.cfg.CNN_model_l2 = 64
        self.cfg.CNN_model_l2_act = "relu"
        self.cfg.CNN_model_layers3 = 3
        self.cfg.CNN_model_l3 = 16
        self.cfg.CNN_model_l3_act = "relu"
        self.cfg.CNN_model_dropout = 0.2

        self.model = Sequential()
        self.model.add(Conv2D(self.cfg.CNN_model_l1, (7, 7), activation = self.cfg.CNN_model_l0_act, input_shape = self.input_shape, kernel_initializer = self.weight_init))
        self.model.add(MaxPooling2D(pool_size = (2,2)))

        for i in range(self.cfg.CNN_model_layers1, 1, -1):
            self.model.add(Conv2D(self.cfg.CNN_model_l1, (5, 5), activation = self.cfg.CNN_model_l1_act, kernel_initializer = self.weight_init))
            self.model.add(MaxPooling2D(pool_size = (2,2)))
        for i in range(self.cfg.CNN_model_layers2, 1, -1):
            self.model.add(Conv2D(self.cfg.CNN_model_l2, (3, 3), activation = self.cfg.CNN_model_l2_act, kernel_initializer = self.weight_init))
            self.model.add(MaxPooling2D(pool_size = (2,2)))
        for i in range(self.cfg.CNN_model_layers3, 1, -1):
            self.model.add(Conv2D(self.cfg.CNN_model_l3, (1, 1), activation = self.cfg.CNN_model_l3_act, kernel_initializer = self.weight_init))
        
        self.model.add(Dropout(self.cfg.CNN_model_dropout))

        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(1024, activation='relu', kernel_initializer = self.weight_init))
        self.model.add(Dense(512, activation='relu', kernel_initializer = self.weight_init))
        self.model.add(Dense(self.cfg.num_classes, activation='softmax', kernel_initializer = self.weight_init))

    def MobileNetV2_transfer_learning(self):
        MobileNetV2_layer = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        MobileNetV2_layer.trainable = False # will not retrain the weights of the pretrained model
        #MobileNetV2_layer.summary()
        self.model = Sequential([
            MobileNetV2_layer,
            GlobalAveragePooling2D(),
            Dense(1024, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(512, activation='relu'),
            Dense(self.cfg.num_classes, activation='softmax'),
        ])
        
    @staticmethod
    def get_model_name(k):
        return 'model_'+str(k)+'.h5'