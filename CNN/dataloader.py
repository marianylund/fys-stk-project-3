# Loads and prepares the data for training, validation and
from cv2 import cv2
from glob import glob
import numpy as np
import matplotlib.pylab as plt
import pathlib
import os, shutil
from yacs.config import CfgNode as CN
from random import randint


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
                                    rotation_range = 180,
                                    #brightness_range=(0.7, 1.5), 
                                    #zoom_range = 0.2, 
                                    vertical_flip = True,
                                    horizontal_flip = True,
                                    )

        test_datagen = image.ImageDataGenerator(rescale = 1./255)

        self.train_generator = train_datagen.flow_from_directory(
                                    OUTPUT_PATH.joinpath("Train"),
                                    classes = self.labels,
                                    target_size = (cfg.image_size, cfg.image_size),
                                    batch_size = cfg.batch_size,
                                    class_mode = 'categorical',
                                    color_mode='grayscale',
                                    )

        self.validation_generator = train_datagen.flow_from_directory(
                                    OUTPUT_PATH.joinpath("Validation"),
                                    classes = self.labels,
                                    target_size = (cfg.image_size, cfg.image_size),
                                    batch_size = cfg.batch_size,
                                    class_mode = 'categorical',
                                    color_mode='grayscale',
                                    )

        self.test_generator = test_datagen.flow_from_directory(
                                    OUTPUT_PATH.joinpath("Test"),
                                    classes = self.labels,
                                    target_size = (cfg.image_size, cfg.image_size),
                                    batch_size = 1,
                                    class_mode = 'categorical',
                                    color_mode='grayscale',
                                    shuffle=False,
                                    )
        #self.show_test_images()

    def get_random_test_image(self, brick_name:str):
        class_index = next(i for i in range(len(self.labels)) if brick_name in self.labels[i])
        assert class_index != None, "Could not find class index for " + brick_name
        return self.get_random_test_image_by_index(class_index)
    
    def get_random_test_image_by_index(self, class_index:int):
        assert class_index >= 0 and class_index < self.cfg.num_classes, str(class_index) + " is out of class range"
        img = self.test_generator[class_index * 8 + randint(0, 7)][0][0] # getting random image out of our test data
        assert (img != None).all(), "Could not get img"
        return img


    def show_test_images(self):
        arr = self.test_generator

        plt.figure(figsize=(10,10))
        for i in range(0, 10):
            plt.subplot(2,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            slice_im = arr[i * 8][0][0]
            backtorgb = cv2.cvtColor(slice_im, cv2.COLOR_GRAY2RGB)
            plt.imshow(backtorgb, interpolation='nearest')
            brick_name = (arr.filepaths[i*8]).split("\\")[-1].split(".")[0]
            plt.xlabel(brick_name)
        plt.show()

def print_out_shape_of_design_matrix(self):
    print("steps_per_epoch ", len(self.train_generator))
    print(" 0 image, 1 answer: ", len(self.train_generator[0]))
    print("Batch_size ", len(self.train_generator[0][0]))
    print("Image width", len(self.train_generator[0][0][0]))
    print("Image height", len(self.train_generator[0][0][0][0]))
    print("Image greyscale value", len(self.train_generator[0][0][0][0][0]))

    print("Validation: ")
    print("steps_per_epoch ", len(self.validation_generator))
    print("0 image, 1 answer:", len(self.validation_generator[0]))
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

def copy_chosen_bricks_to_folders(test_dataset_size:float = 0.01, validation_dataset_size:float = 0.2, chosen_bricks_output:pathlib.Path = OUTPUT_PATH):
        chosen_bricks_output.mkdir(exist_ok=True, parents=True)
        total_images = 800
        number_of_test_images = int(total_images * test_dataset_size)
        number_of_valid_images = int(total_images * validation_dataset_size)
        number_of_train_images = int(total_images - (number_of_test_images + number_of_valid_images))
        
        print(f'Total images (per brick): {total_images}. Train: {number_of_train_images}, validation {number_of_valid_images}, test {number_of_test_images}')
        assert number_of_test_images%2==0 and number_of_valid_images%2==0 and number_of_valid_images%2==0, "All number og images has to even (for left and right)"
        
        print("Dataset will be generated to ", chosen_bricks_output)
        print(f"Generating train dataset. From {0} to {number_of_train_images} images")
        new_path = chosen_bricks_output.joinpath("Train")
        copy_images_for_each_brick(new_path, 0, number_of_train_images)
        if validation_dataset_size > 0.0:
            from_i = number_of_train_images
            to_i = number_of_train_images + number_of_valid_images
            print(f"Generating validation dataset. From {from_i} to {to_i} images")
            new_path = chosen_bricks_output.joinpath("Validation")
            copy_images_for_each_brick(new_path, from_i, to_i)
        
        if test_dataset_size > 0.0:
            from_i = number_of_train_images + number_of_valid_images
            to_i = number_of_train_images + number_of_valid_images + number_of_test_images
            print(f"Generating test dataset. From {from_i} to {to_i} images")
            new_path = chosen_bricks_output.joinpath("Test")
            copy_images_for_each_brick(new_path, from_i, to_i)
        
def copy_images_for_each_brick(output_path:pathlib.Path, from_image:int, to_number_of_images:int):
    output_path.mkdir(exist_ok=True, parents=True)
    for brick_name in get_chosen_bricks_list():
            brick_folder = output_path.joinpath(brick_name)
            brick_folder.mkdir(exist_ok=True, parents=True)
            copy_images(brick_folder, brick_name, int(from_image/2), int(to_number_of_images/2), left = True)
            copy_images(brick_folder, brick_name, int(from_image/2), int(to_number_of_images/2), left = False)

def copy_images(brick_folder:pathlib.Path, brick_name:str, from_image:int, to_number_of_images:int, left:bool = True):
    for i in range(from_image, to_number_of_images):
        image_path = DATASET_PATH.joinpath(get_brick_image_name(brick_name, i, left = left))
        shutil.copy(image_path, brick_folder)

def print_status():
    print("Chosen bricks: ", get_chosen_bricks_list())
    show_all_chosen_bricks_as_images()

# import wandb
# import matplotlib.pylab as plt
# if __name__ == "__main__":
#     cfg = wandb.config # Config is a variable that holds and saves hyperparameters and inputs
#     cfg.image_size = 100

#     cfg.model_type = "simplest" # [simplest]
#     cfg.optimizer = 'adam' # TODO: does not do anything yet, here just for reminder

#     cfg.learning_rate = 0.01
#     cfg.batch_size = 32
#     cfg.epochs = 30
#     cfg.num_classes = 10 # changing it does not do much
#     cfg.early_stopping_patience = 5
#     dl = DataLoader(cfg)
    
#     pic = dl.get_random_test_image_by_index(0)
#     img = pic
#     img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
#     font = cv2.FONT_HERSHEY_DUPLEX
#     cv2.putText(img, 'True Name', (10, 328), font, 0.5, (255,255,255))
#     #img = cv2.cvtColor(dl.get_random_test_image("3004"), cv2.COLOR_GRAY2RGB)
#     plt.imshow(img)
#     plt.show()

