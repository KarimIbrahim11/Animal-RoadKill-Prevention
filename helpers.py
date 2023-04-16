import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
# Returns array of Labels
def load_labels(pth):
    # Reading Labels from text file
    labels_pth = pth
    labels_file = open(labels_pth, "r")
    labels = labels_file.read().splitlines()
    labels_file.close()
    return labels


# Plots 4 images of 2 batches with their labels to double check correct assignment
def batch_images_plot(d, labels, scaled_yet=False):
    data_iterator = d.as_numpy_iterator()  # Loading data generator as numpy iterator to access the data
    batch = data_iterator.next()  # load one batch of data

    # Checking correct labels assigned to each image in Data Pipeline in two different batches
    for i in range(0,2):
        fig,ax = plt.subplots(ncols=4, figsize=(20,20))
        for idx, img in enumerate(batch[0][:4]):
            if not scaled_yet:
                ax[idx].imshow(img.astype(int)) # without scaling
            else:
                ax[idx].imshow(img) # with scaling
            ax[idx].title.set_text(labels[batch[1][idx]])
        plt.show()
        batch = data_iterator.next()  # load another batch of data
    return