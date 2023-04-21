import cv2
import imghdr
import os

# Data dir
data_dir = 'D:/Coding Projects/Pycharm Projects/Datasets/animals/'

# Reading Labels from text file
labels_pth = 'name of the animals.txt'
labels_file = open(labels_pth, "r")
labels = labels_file.read().splitlines()
labels_file.close()


# Removing images less than 9 KB, with weird extensions and not-readable
image_exts = ['jpeg', 'jpg', 'bmp', 'png']
for label in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, label)):
        image_path = os.path.join(data_dir, label, image)
        # Removing images less than 9 KB
        file_stats = os.stat(image_path)
        if file_stats.st_size <= 9000:
            os.remove(image_path)
        # Making sure its readable and has one of the supported extensions
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)