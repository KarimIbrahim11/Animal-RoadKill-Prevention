import tensorflow as tf
import numpy as np
import cv2
import imghdr
import os
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # Limiting the use of VRAM in GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

    data_dir = 'D:/Coding Projects/Pycharm Projects/Datasets/animals'

    # Reading Labels from text file
    labels_pth = 'name of the animals.txt'
    labels_file = open(labels_pth, "r")
    labels = labels_file.read().splitlines()
    labels_file.close()
    print(labels)

    # Building Data Pipeline from DS
    data = tf.keras.preprocessing.image_dataset_from_directory(data_dir) # Data pipeline
    data_iterator = data.as_numpy_iterator() # Loading data generator as numpy iterator to access the data
    batch = data_iterator.next()    # load one batch of data


    # Checking correct labels assigned to each image in Data Pipeline in two different batches
    for i in range(0,2):
        fig,ax = plt.subplots(ncols=4, figsize=(20,20))
        for idx, img in enumerate(batch[0][:4]):
            ax[idx].imshow(img.astype(int))
            ax[idx].title.set_text(labels[batch[1][idx]])
        plt.show()
        batch = data_iterator.next()  # load another batch of data


