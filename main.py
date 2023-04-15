import tensorflow as tf
import cv2
import imghdr
import os

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

    # Reading Labels from text file
    labels_pth = 'data/name of the animals.txt'
    labels_file = open(labels_pth, "r")
    labels = labels_file.read().splitlines()
    labels_file.close()

    print(labels)

    # Cleaning Images
    data_dir = 'data'
    image_exts = ['jpeg', 'jpg', 'bmp', 'png']


