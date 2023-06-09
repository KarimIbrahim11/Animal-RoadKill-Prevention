import os
import collections
from shutil import copy


# Helper method to split dataset into train and test folders
def prepare_data(filepath, src, dest):
    classes_images = collections.defaultdict(list)
    with open(filepath, 'r') as txt:
        paths = [read.strip() for read in txt.readlines()]
        for p in paths:
            food = p.split('/')
            classes_images[food[0]].append(food[1] + '.jpg')

    for food in classes_images.keys():
        # print("\nCopying images into ",food)
        if not os.path.exists(os.path.join(dest, food)):
            os.makedirs(os.path.join(dest, food))
        for i in classes_images[food]:
            copy(os.path.join(src, food, i), os.path.join(dest, food, i))
    print("Copying Done!")


# Prepare train dataset by copying images from food-101/images to food-101/train using the file train.txt

dataset_path = 'D:/Coding Projects/Pycharm Projects/Datasets/animals'

print("Creating train data...")
prepare_data(dataset_path+'/meta/train.txt', dataset_path+'/images', dataset_path+'/train')
print("Success")

# Prepare test data by copying images from food-101/images to food-101/test using the file test.txt

print("Creating test data...")
prepare_data(dataset_path+'/meta/test.txt', dataset_path+'/images', dataset_path+'/test')
print("Success..")
