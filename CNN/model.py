# Contains the structure on the network, forward and backwards passes
import pathlib

import tensorflow as tf
from keras.optimizers import Adam, Adagrad, SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
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
        if self.cfg.decay_rate == -1:
            lr_schedule = self.cfg.learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.cfg.learning_rate,
                                        decay_steps=self.cfg.decay_steps,
                                        decay_rate=self.cfg.decay_rate)
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
        elif self.cfg.model_type == "TripleV2":
            self.TripleV2()
        elif self.cfg.model_type == "MobileNetV2_trainable":
            self.MobileNetV2_trainable()
        elif self.cfg.model_type == "MobileNetV3_trainable":
            self.MobileNetV3_trainable()
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

        self.model = Sequential()
        self.model.add(Flatten(input_shape=self.input_shape))
        self.model.add(Dense(50, activation=self.cfg.NN_act0, kernel_initializer = self.weight_init))
        self.model.add(Dense(20, activation=self.cfg.NN_act1, kernel_initializer = self.weight_init))
        
        self.model.add(Dense(self.cfg.num_classes, activation='softmax', kernel_initializer = self.weight_init))

    def TripleV2(self):
        self.cfg.CNN_model_l0_size = 32
        self.cfg.CNN_model_l1_size = 32
        self.cfg.CNN_model_l2_size = 64
        self.cfg.CNN_model_l3_size = 16

        self.cfg.CNN_model_l1_count = 3
        self.cfg.CNN_model_l2_count = 3
        self.cfg.CNN_model_l3_count = 3

        self.cfg.CNN_model_dropout = 0.2

        self.model = Sequential()
        self.model.add(Conv2D(self.cfg.CNN_model_l0_size, (7, 7), activation = "tanh", input_shape = self.input_shape, kernel_initializer = self.weight_init))
        self.model.add(MaxPooling2D(pool_size = (2,2)))

        for i in range(self.cfg.CNN_model_l1_count, 1, -1):
            self.model.add(Conv2D(self.cfg.CNN_model_l1_size, (5, 5), activation = "tanh", kernel_initializer = self.weight_init))
            self.model.add(MaxPooling2D(pool_size = (2,2)))
        for i in range(self.cfg.CNN_model_l2_count, 1, -1):
            self.model.add(Conv2D(self.cfg.CNN_model_l2_size, (3, 3), activation = "tanh", kernel_initializer = self.weight_init))
            self.model.add(MaxPooling2D(pool_size = (2,2)))
        for i in range(self.cfg.CNN_model_l3_count, 1, -1):
            self.model.add(Conv2D(self.cfg.CNN_model_l3_size, (1, 1), activation = "tanh", kernel_initializer = self.weight_init))
            self.model.add(MaxPooling2D(pool_size = (2,2)))

        self.model.add(Dropout(self.cfg.CNN_model_dropout))
        
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(1024, activation='relu', kernel_initializer = self.weight_init))
        self.model.add(Dense(512, activation='relu', kernel_initializer = self.weight_init))
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
    
    def MobileNetV2_trainable(self):
        self.cfg.Dense_activations = "relu"
        self.cfg.MobileNet_alpha = 1.0
        self.cfg.layers_to_freeze = 100
        self.cfg.Dense0 = 224
        self.cfg.Dense1 = 50


        MobileNetV2_layer = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        print("Layers: ", len(MobileNetV2_layer.layers))
        for l in MobileNetV2_layer.layers[:self.cfg.layers_to_freeze]:
            l.trainable = False
        #MobileNetV2_layer.summary()
        self.model = Sequential([
            MobileNetV2_layer,
            GlobalAveragePooling2D(),
            Dense(self.cfg.Dense0, activation=self.cfg.Dense_activations, kernel_initializer = self.weight_init),
            Dense(self.cfg.Dense1, activation=self.cfg.Dense_activations, kernel_initializer = self.weight_init),
            Dense(self.cfg.num_classes, activation='softmax'),
        ])
    
    def MobileNetV3_trainable(self):
        self.cfg.Dense_activations = "tanh"
        self.cfg.MobileNet_alpha = 1.0
        self.cfg.dropout = 0.2
        self.cfg.Dense0 = 512
        self.cfg.Dense1 = 350

        MobileNetV2_layer = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        print("Layers: ", len(MobileNetV2_layer.layers))
        for l in MobileNetV2_layer.layers:
            #print(l)
            if not isinstance(l, BatchNormalization):
                l.trainable = False
        #MobileNetV2_layer.summary()
        self.model = Sequential([
            MobileNetV2_layer,
            Dropout(self.cfg.dropout),
            GlobalAveragePooling2D(),
            Dense(self.cfg.Dense0, activation=self.cfg.Dense_activations, kernel_initializer = self.weight_init),
            Dense(self.cfg.Dense1, activation=self.cfg.Dense_activations, kernel_initializer = self.weight_init),
            Dense(self.cfg.num_classes, activation='softmax'),
        ])
        
        """
          decay_rate:
    distribution: uniform
    min: 0.7
    max: 0.8
  decay_steps:
    distribution: int_uniform
    min: 800000
    max: 1000000"""
    @staticmethod
    def get_model_name(k):
        return 'model_'+str(k)+'.h5'