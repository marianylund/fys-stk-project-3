# Followed the Intro to Keras example: https://colab.research.google.com/drive/1pMcNYctQpRoBKD5Z0iXeFWQD8hIDgzCV#scrollTo=gUaHHRYo3cuo
from keras.callbacks import EarlyStopping

from CNN.model import Model
from CNN.dataloader import DataLoader, get_chosen_bricks_list
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt
from cv2 import cv2
import numpy as np

class Trainer():
    """Responsible for running the training, sending info to wandb and making predictions"""
    def __init__(self, cfg = None, wandb_on = True):
        if cfg == None:
            self.cfg = self.set_up_configs()
        else:
            self.cfg = cfg
        self.dl = DataLoader(self.cfg)
        self.model = Model(self.cfg).model

        early_stopping = EarlyStopping(monitor='val_loss', patience=self.cfg.early_stopping_patience, mode='auto', restore_best_weights=True)
        run_callbacks = [early_stopping]

        if wandb_on:
            wandb.init(project="project3", entity="fys-stk-gang", config = self.cfg)
            if self.cfg.name != "":
                wandb.run.name = self.cfg.name
            if self.cfg.notes != "":
                wandb.run.notes = self.cfg.notes
            logging = WandbCallback(data_type="image", generator=self.dl.validation_generator, labels=get_chosen_bricks_list())
            run_callbacks.append(logging)

        history = self.model.fit(
            self.dl.train_generator,
            steps_per_epoch=len(self.dl.train_generator),
            epochs = self.cfg.epochs,
            validation_data = self.dl.validation_generator,
            validation_steps=len(self.dl.validation_generator),
            validation_freq = 1,
            callbacks=run_callbacks
        )

        self.make_predictions(self.model, self.dl, self.cfg, wandb_on)

    def overwrite_configs(self, cfg):
        """Function to override the default configurations"""
        cfg.epochs = 30
        cfg.image_size = 224
        cfg.learning_rate = 0.011410793024097232
        cfg.decay_rate = 0.9033169911446738 # -1 to turn it off, 0.9 usually
        cfg.decay_steps = 406574
        cfg.batch_size = 38
        cfg.channels = 3
        cfg.name = "" # Here you can change the name of the run, leave empty or do not change if you want a random name
        cfg.notes = "" # A longer description of the run, like a -m commit message in git. This helps you remember what you were doing when you ran this run.
        return cfg

    def set_up_configs(self):
        """Contains the default configurations"""
        cfg = wandb.config # Config is a variable that holds and saves hyperparameters and inputs
        cfg.image_size = 224

        cfg.model_type = "MobileNetV3_trainable" # [Triple_model, MobileNetV2_transfer_learning, simple_NN, TripleV2, MobileNetV2_trainable, MobileNetV3_trainable]
        cfg.optimizer = 'adam' # [sgd, adam, adagrad]
        cfg.channels = 3 # has to be 3 for transfer learning
        cfg.dense_layer_units = 1 # for transfer learning

        cfg.CNN_model_l1_count = 2
        cfg.CNN_model_l2_count = 2
        cfg.CNN_model_l3_count = 2
        
        cfg.CNN_model_l0_size = 32
        cfg.CNN_model_l1_size = 32
        cfg.CNN_model_l2_size = 64
        cfg.CNN_model_l3_size = 16

        cfg.dropout = 0.2
        cfg.num_classes = 10 # changing it does not do much, just a value to keep track of
        cfg.learning_rate = 0.01
        cfg.decay_rate = -1 # -1 to turn it off, 0.9 usually
        cfg.decay_steps = 10000
        cfg.batch_size = 32
        cfg.epochs = 5
        cfg.name = "" 
        cfg.notes = "" 

        cfg.early_stopping_patience = 5

        return self.overwrite_configs(cfg) # has to be last

    def make_predictions(self, cnn_model:Model, dl:DataLoader, cfg, wandb_on = True):
        """Shows predicted images in w&b with their top 3 prediction names"""
        predicted_images = []
        chosen_bricks = dl.labels
        num_of_chosen_bricks = len(chosen_bricks)

        if not wandb_on:
            plt.figure(figsize=(10,10))
    
        for i in range(num_of_chosen_bricks):
            brick_name = chosen_bricks[i]
            pic = dl.get_random_test_image_by_index(i)
            img = pic.copy()

            # Get predictions for the lego brick
            prediction = cnn_model.predict(pic.reshape(1, cfg.image_size, cfg.image_size, self.cfg.channels))[0]

            # Do lots of magic so it is possible to show it in wandb
            img = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            img.astype(np.uint8)
            if self.cfg.channels == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)

            # Format predictions to string to overlay on image
            text = sorted(['{:s} : {:.1f}%'.format(chosen_bricks[k].title(), 100*v) for k,v in enumerate(prediction)], 
                key=lambda x:float(x.split(':')[1].split('%')[0]), reverse=True)[:3]

            # Add text to image -  We add the true probabilities and predicted probabilities on each of the images in the test dataset
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, 'True Name : %s' % brick_name, (10, 328), font, 0.5, (255,255,255))
            for k, t in enumerate(text):
                cv2.putText(img, t, (10, 348+k*18), font, 0.5, (255,255,255))

            if wandb_on:    
                # Add predicted image from test dataset with annotations to array
                predicted_images.append(wandb.Image(img, caption="Actual: %s" % brick_name))
            else:
                plt.subplot(2,5,i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(img, interpolation='nearest')
                plt.xlabel("Actual: %s" % brick_name)
        
        if wandb_on:
            wandb.log({"predictions": predicted_images})
        else:
            plt.show()  


    

        

