import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
import PIL
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from sklearn.metrics import confusion_matrix




# Returns array of Labels
def load_labels(pth):
    # Reading Labels from text file
    labels_pth = pth
    labels_file = open(labels_pth, "r")
    labels = labels_file.read().splitlines()
    labels_file.close()
    return labels


# Plots 4 images of 2 batches with their labels to double checking correct assignment
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


# Helper-function for loading images
def load_images(image_paths):
    # Load the images from disk.
    images = [plt.imread(path) for path in image_paths]

    # Convert to a numpy array and return it.
    return np.asarray(images)

# Helper-function for joining a directory and list of filenames.
def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]



# Function used to plot at most 9 images in a 3x3 grid, and writing the true and predicted classes below each image.
def plot_images(images, cls_true, cls_names, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = cls_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = cls_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def predict(model, image_path):
    # Load and resize the image using PIL.
    img = PIL.Image.open(image_path)
    input_shape = model.layers[0].output_shape[0][1:3]
    img_resized = img.resize(input_shape, PIL.Image.LANCZOS)

    # Plot the image.
    # plt.imshow(img_resized)
    # plt.show()

    # Convert the PIL image to a numpy-array with the proper shape.
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    # Use the VGG16 model to make a prediction.
    # This outputs an array with 1000 numbers corresponding to
    # the classes of the ImageNet-dataset.
    pred = model.predict(img_array)

    # Decode the output of the VGG16 model.
    pred_decoded = decode_predictions(pred)[0]

    # Print the predictions.
    for code, name, score in pred_decoded:
        print("{0:>6.2%} : {1}".format(score, name))


# Helper-function for printing whether a layer in the model should be trained.
def print_layer_trainable(model):
    for layer in model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))