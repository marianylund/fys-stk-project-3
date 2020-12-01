# Loads and prepares the data for training, validation and
from cv2 import cv2
from glob import glob
import numpy as np
import matplotlib.pylab as plt
import pathlib
import os, shutil
from yacs.config import CfgNode as CN


from tensorflow.keras.preprocessing import image

DATASET_PATH = pathlib.Path("dataset") # Downloaded images from https://www.kaggle.com/joosthazelzet/lego-brick-images?select=dataset
OUTPUT_PATH = pathlib.Path("DataFiles")


class DataLoader():
    """Contains the train, test and validation dataset that has been augumented and rescaled"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.labels = get_chosen_bricks_list()
        train_datagen = image.ImageDataGenerator(
                                    rescale = 1./255, 
                                    shear_range = 0.2,
                                    zoom_range = 0.2, 
                                    vertical_flip = True,
                                    horizontal_flip=True,
                                    validation_split=0.2
                                    )

        self.train_generator = train_datagen.flow_from_directory(
                                    OUTPUT_PATH,
                                    target_size = (cfg.image_size, cfg.image_size),
                                    batch_size = cfg.batch_size,
                                    class_mode = 'categorical',
                                    color_mode='grayscale',
                                    subset='training'
                                    )
        self.validation_generator = train_datagen.flow_from_directory(
                                    OUTPUT_PATH,
                                    target_size = (cfg.image_size, cfg.image_size),
                                    batch_size = cfg.batch_size,
                                    class_mode = 'categorical',
                                    color_mode='grayscale',
                                    subset='validation'
                                    )

    # TODO: change it to actually take an image from a test set
    def get_random_image_of_brick_reshaped(self, brick_name:str):
        image = get_random_image_of_brick(brick_name)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Resize image and normalize it
        pic = cv2.resize(img, (self.cfg.image_size, self.cfg.image_size)).astype('float32') / 255
        return pic, img

def print_out_shape_of_design_matrix(self):
    print("steps_per_epoch ", len(self.train_generator))
    print(" ? ", len(self.train_generator[0]))
    print("Batch_size ", len(self.train_generator[0][0]))
    print("Image width", len(self.train_generator[0][0][0]))
    print("Image height", len(self.train_generator[0][0][0][0]))
    print("Image greyscale value", len(self.train_generator[0][0][0][0][0]))

    print("Validation: ")
    print("steps_per_epoch ", len(self.validation_generator))
    print(" ? ", len(self.validation_generator[0]))
    print("Batch_size ", len(self.validation_generator[0][0]))
    print("Image width", len(self.validation_generator[0][0][0]))
    print("Image height", len(self.validation_generator[0][0][0][0]))
    print("Image greyscale value", len(self.validation_generator[0][0][0][0][0]))

def get_chosen_bricks_list():
    return ['3062 Round Brick 1x1', '4150 flat tile round 2x2', '4274 Connector Peg w Knob', '41677 lever 2M', '3005 brick 1x1', '14719 flat tile corner 2x2', '3024 plate 1x1', '4490 brick bow 1x3', '18654 beam 1M', '3004 brick 1x2']

def get_set_with_brick_names():
    all_lego_names = get_all_lego_image_names()
    lego_names_set = set()
    for name in all_lego_names:
        lego_names_set.add(name[:-9])
    return lego_names_set

def get_all_lego_image_names(path:str = DATASET_PATH):
    return sorted(os.listdir(path))

def get_random_image_of_brick(brick_name:str):
    full_path = str(OUTPUT_PATH.joinpath(brick_name)) + "/*.*"
    image = cv2.imread(np.random.choice([k for k in glob(full_path)]))
    assert (image != None).all(), "No image found with the name: " + brick_name
    return image

def get_brick_image_name(name:str, image_number:int = 0, left:bool = True):
    return f"{name} {image_number:03d}{'L' if left else 'R'}.png"

def show_image(image_name:str, read_path:str = DATASET_PATH):
    image = cv2.imread(read_path + image_name)
    plt.figure(figsize=(3,3))
    plt.title(image_name)
    plt.imshow(image)
    
def show_all_bricks_as_images():
    """
    Works best in a notebook, it will show 50 images of lego bricks in the dataset
    """
    all_legos = get_set_with_brick_names()
    for name in all_legos:
        brick_name = get_brick_image_name(name)
        show_image(brick_name)

def get_image_of_brick(brick_name:str):
    brick_image_name = get_brick_image_name(brick_name)
    brick_path = DATASET_PATH.joinpath(brick_image_name)
    image = cv2.imread(str(brick_path))
    assert (image != None).all(), "No image found with the name: " + brick_name
    return image

def show_all_chosen_bricks_as_images():
    """
    Shows all chosen legos
    """
    all_legos = get_chosen_bricks_list()
    plt.figure(figsize=(10,10))
    for i in range(len(all_legos)):
        plt.subplot(2,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        brick_name = all_legos[i]
        plt.imshow(get_image_of_brick(brick_name))
        plt.xlabel(brick_name)
    plt.show()
        
def get_image_names_for_brick(brick_name: str):
    return list(filter(lambda x: x if brick_name in x else None, get_all_lego_image_names()))

def delete_everything_in_directory(folder:str):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def copy_chosen_bricks_to_folders():
        chosen_bricks_output = OUTPUT_PATH
        chosen_bricks_output.mkdir(exist_ok=True, parents=True)
        for brick_name in get_chosen_bricks_list():
            brick_folder = chosen_bricks_output.joinpath(brick_name)
            brick_folder.mkdir(exist_ok=True, parents=True)
            for i in range(400):
                image_path = DATASET_PATH.joinpath(get_brick_image_name(brick_name, i, left = True))
                shutil.copy(image_path, brick_folder)
            for i in range(400):
                image_path = DATASET_PATH.joinpath(get_brick_image_name(brick_name, i, left = False))
                shutil.copy(image_path, brick_folder)

def print_status():
    print("Chosen bricks: ", get_chosen_bricks_list())
    show_all_chosen_bricks_as_images()
    