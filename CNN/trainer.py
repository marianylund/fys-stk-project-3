# Responsible for running the training, saving checkpoints and updating the graphs

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.datasets import cifar10

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# wandb.init(project="fys-stk", sync_tensorboard=True)
# # Log metrics with wandb
# # If you are using tensorboard already, we will sync all the information from tensorboard and this isn't necessary.
# # If you are using an estimator you can add our hook.
# classifier.train(input_fn=train_input_fn, steps=100000, hooks=[WandbHook()])

# # Save model to wandb
# saver = tf.train.Saver()
# saver.save(sess, os.path.join(wandb.run.dir, "model.ckpt"))