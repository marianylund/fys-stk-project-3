# Loads and prepares the data for training, validation and
import cv2
import matplotlib.pylab as plt
import pathlib
import os, shutil

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras import layers

DATASET_PATH = pathlib.Path("dataset") # Downloaded images from https://www.kaggle.com/joosthazelzet/lego-brick-images?select=dataset
OUTPUT_PATH = pathlib.Path("DataFiles")

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

def show_all_chosen_bricks_as_images():
    """
    Works best in a notebook, it will show 10 images, as we have chosen 10 bricks
    """
    all_legos = get_chosen_bricks_list()
    for name in all_legos:
        brick_name = get_brick_image_name(name)
        show_image(brick_name)      
        
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

if __name__ == "__main__":
    train_datagen = image.ImageDataGenerator(
                                rescale = 1./255, 
                                shear_range = 0.2,
                                zoom_range = 0.2, 
                                vertical_flip = True,
                                horizontal_flip=True,
                                validation_split=0.2
                                )
    test_dataset = image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
                                OUTPUT_PATH,
                                target_size = (400,400),
                                batch_size = 16,
                                class_mode = 'categorical',
                                color_mode='rgb',
                                subset='training'
                                )
    validation_generator = train_datagen.flow_from_directory(
                                OUTPUT_PATH,
                                target_size = (400,400),
                                batch_size = 16,
                                class_mode = 'categorical',
                                color_mode='rgb',
                                subset='validation'
                                )

