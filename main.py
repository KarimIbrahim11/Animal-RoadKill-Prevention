import tensorflow as tf
import numpy as np
import cv2
import imghdr
import os
from helpers import *


if __name__ == '__main__':
    # Limiting the use of VRAM in GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

    # Path
    data_dir = 'D:/Coding Projects/Pycharm Projects/Datasets/animals'
    labels_pth = 'name of the animals.txt'

    # Loading Labels
    labels = load_labels(labels_pth)


    # Building Data Pipeline from DS
    data = tf.keras.preprocessing.image_dataset_from_directory(data_dir)

    # Display Images from batches to check for correct label assignment
    batch_images_plot(data, labels, scaled_yet=False)

    # Scaling Data
    data = data.map(lambda x,y: (x/255, y)) # only transforming batch[0] (x)
    print("Scaling Success?", data.as_numpy_iterator().next()[0].max()==1.0)

    # Data split
    train_size = int(len(data)*50/84)
    val_size = int(len(data)*17/84)
    test_size = int(len(data)*17/84)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)
    print("Validating Data Split Success?", len(data) == len(train)+len(test)+len(val))




