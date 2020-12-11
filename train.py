from CNN.trainer import Trainer
import wandb
import os

print("Staring training")

def set_config_for_simple_nn():
    cfg = wandb.config 
    cfg.image_size = 224

    cfg.model_type = "simple_NN" # [Triple_model, MobileNetV2_transfer_learning, simple_NN, TripleV2, MobileNetV2_trainable, MobileNetV3_trainable]
    cfg.optimizer = 'adam'
    cfg.channels = 1 
    cfg.name = "" 
    cfg.notes = "" 
    cfg.dropout = 0.2
    cfg.num_classes = 10 # changing it does not do much, just a value to keep track of
    cfg.learning_rate = 0.009115346102532486
    cfg.decay_rate = -1 # -1 to turn it off, 0.9 usually
    cfg.decay_steps = 10000
    cfg.batch_size = 117
    cfg.epochs = 20

    cfg.early_stopping_patience = 5
    return cfg

def set_config_for_tripple():
    cfg = wandb.config 
    cfg.image_size = 224

    cfg.model_type = "TripleV2" # [Triple_model, MobileNetV2_transfer_learning, simple_NN, TripleV2, MobileNetV2_trainable, MobileNetV3_trainable]
    cfg.optimizer = 'adam'
    cfg.channels = 1 
    cfg.name = "" 
    cfg.notes = "" 
    cfg.dropout = 0.2
    cfg.num_classes = 10 # changing it does not do much, just a value to keep track of
    cfg.learning_rate = 0.01141
    cfg.decay_rate = 0.9033 # -1 to turn it off, 0.9 usually
    cfg.decay_steps = 406574
    cfg.batch_size = 38
    cfg.epochs = 25

    cfg.early_stopping_patience = 5
    return cfg

def set_config_for_transfer_learning():
    cfg = wandb.config
    cfg.image_size = 224

    cfg.model_type = "MobileNetV2_trainable" # [Triple_model, MobileNetV2_transfer_learning, simple_NN, TripleV2, MobileNetV2_trainable, MobileNetV3_trainable]
    cfg.optimizer = 'adam' 
    cfg.channels = 3 # has to be 3 for transfer learning
    cfg.name = "" 
    cfg.notes = "" 
    cfg.Dense_activations = "relu"
    cfg.MobileNet_alpha = 1.0
    cfg.layers_to_freeze = 100
    cfg.Dense0 = 485
    cfg.Dense1 = 370

    cfg.dropout = 0.0017101185244705718
    cfg.num_classes = 10 # changing it does not do much, just a value to keep track of
    cfg.learning_rate = 0.01902516532710387
    cfg.decay_rate = 0.8140472019082363 # -1 to turn it off, 0.9 usually
    cfg.decay_steps = 826146
    cfg.batch_size = 53
    cfg.epochs = 25

    cfg.early_stopping_patience = 5
    return cfg

print('Have you logged in to wandb (1 for yes, 0 for no)?')
wandb_on = input()
while(int(wandb_on) != 0 and int(wandb_on) != 1):
    print(f'Sorry, I did not catch that. Looks like you said {wandb_on}.\nHave you logged in to wandb?\n(please, write 1 for yes, 0 for no)')
    wandb_on = input()

wandb_on = bool(int(wandb_on))

if(wandb_on):
    os.system('wandb enabled')
else:
    os.system('wandb disabled')

print("Choose option:\n1 for running training for SimpleNN model.\n2 for CNN model\n3 for transfer learning")
x = input()
while(int(x) != 1 and int(x) != 2 and int(x) != 3):
    print('Sorry, I did not catch that. \nChoose option:\n1 for running training for SimpleNN model.\n2 for CNN model\n3 for transfer learning')
    x = input()

if int(x) == 1:
    trainer = Trainer(cfg = set_config_for_simple_nn(), wandb_on=wandb_on)
elif int(x) == 2:
    trainer = Trainer(cfg = set_config_for_tripple(), wandb_on=wandb_on)
elif int(x) == 3:
    trainer = Trainer(cfg = set_config_for_transfer_learning(), wandb_on=wandb_on)
else:
    print("Something is wrong. X should be 1, 2, or 3: " + str(x))