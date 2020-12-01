# Responsible for running the training, saving checkpoints and updating the graphs
# Followed the Intro to Keras example: https://colab.research.google.com/drive/1pMcNYctQpRoBKD5Z0iXeFWQD8hIDgzCV#scrollTo=gUaHHRYo3cuo
from tensorflow.keras.callbacks import EarlyStopping
from yacs.config import CfgNode as CN

from CNN.model import Model
from CNN.dataloader import DataLoader, get_chosen_bricks_list
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt
from cv2 import cv2

class Trainer():
    def __init__(self):
        self.cfg = self.set_up_configs()
        self.dl = DataLoader(self.cfg)
        self.model = Model(self.cfg).model

        wandb.init(project="project3", entity="fys-stk-gang", config = self.cfg)
        if self.cfg.name != "":
            wandb.run.name = self.cfg.name
        if self.cfg.notes != "":
            wandb.run.notes = self.cfg.notes

        checkpoint = EarlyStopping(monitor='val_accuracy', patience=self.cfg.early_stopping_patience, mode='auto', restore_best_weights=True)
        logging = WandbCallback(data_type="image", generator=self.dl.validation_generator, labels=get_chosen_bricks_list())

        history = self.model.fit(
            self.dl.train_generator,
            steps_per_epoch=len(self.dl.train_generator),
            epochs = self.cfg.epochs,
            validation_data = self.dl.validation_generator,
            validation_steps=len(self.dl.validation_generator),
            #validation_freq = 1,
            callbacks=[checkpoint, logging]
        )

        #OBS! For now it predicts randomly on train or valid data, will fix test dataset later
        self.make_predictions(self.model, self.dl, self.cfg) # comment out to disable making predications to w&b

    def overwrite_configs(self, cfg):
        """Feel free to overwrite any of the configurations"""
        cfg.epochs = 2
        cfg.name = "SuchTestVeryNice" # Here you can change the name of the run, leave empty or do not change if you want a random name
        cfg.notes = "" # A longer description of the run, like a -m commit message in git. This helps you remember what you were doing when you ran this run.
        return cfg

    def set_up_configs(self):
        """Here are all the default configurations, all of them are used somewhere else in the code"""
        cfg = wandb.config # Config is a variable that holds and saves hyperparameters and inputs
        cfg.image_size = 400

        cfg.model_type = "simplest" # [simplest]
        cfg.optimizer = 'nadam' # TODO: does not do anything yet, here just for reminder

        cfg.learning_rate = 0.01
        cfg.batch_size = 32
        cfg.epochs = 30

        cfg.early_stopping_patience = 5

        return self.overwrite_configs(cfg) # has to be last

    def make_predictions(self, cnn_model:Model, dl:DataLoader, cfg):
        """Shows predicted images in w&b with their top 3 prediction names"""
        predicted_images = []
        chosen_bricks = get_chosen_bricks_list()
        num_of_chosen_bricks = len(chosen_bricks)
        for i in range(num_of_chosen_bricks):
            brick_name = chosen_bricks[i]
            pic, img = dl.get_random_image_of_brick_reshaped(brick_name)
            # Get predictions for the lego brick
            prediction = cnn_model.predict(pic.reshape(1, cfg.image_size, cfg.image_size, 1))[0]

            # Format predictions to string to overlay on image
            text = sorted(['{:s} : {:.1f}%'.format(chosen_bricks[k].title(), 100*v) for k,v in enumerate(prediction)], 
                key=lambda x:float(x.split(':')[1].split('%')[0]), reverse=True)[:3]
            
            # Upscale image
            #img = cv2.resize(pic, (352, 352))
            
            # Add text to image -  We add the true probabilities and predicted probabilities on each of the images in the test dataset
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, 'True Name : %s' % brick_name, (10, 328), font, 0.5, (255,255,255))
            for k, t in enumerate(text):
                cv2.putText(img, t, (10, 348+k*18), font, 0.5, (255,255,255))
                
            # Add predicted image from test dataset with annotations to array
            predicted_images.append(wandb.Image(img, caption="Actual: %s" % brick_name))     
        
        wandb.log({"predictions": predicted_images})

    

        

