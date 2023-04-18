import tensorflow as tf
import numpy as np
import cv2
import imghdr
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
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
    # batch_images_plot(data, labels, scaled_yet=False)

    # Pre Processing Data pipeline
    # Scale
    data = data.map(lambda x, y: (x / 255, y))  # only transforming batch[0] (x)
    print("Scaling Success?", data.as_numpy_iterator().next()[0].max() == 1.0)
    # data, scaling_success = scaled(data)

    # Data split
    train_size = int(len(data)*50/84)
    val_size = int(len(data)*17/84)
    test_size = int(len(data)*17/84)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)
    print("Validating Data Split Success?", len(data) == len(train)+len(test)+len(val))

    # DL Model
    model = Sequential()

    model.add(Conv2D(16,(3,3),1,activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())

    model.add(Conv2D(32,(3,3),1,activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(16,(3,3),1,activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

    model.summary()

