import tensorflow as tf
from helpers import *
from tensorflow.keras.models import Model, Sequential, model_from_json, save_model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight


def fine_tune_pipeline():
    # Model instantiation
    model = VGG16(include_top=True, weights='imagenet')

    # Input pipeline
    input_shape = model.layers[0].output_shape[0][1:3]
    print(input_shape)

    # Creating Generators
    datagen_train = ImageDataGenerator(
        rescale=1./255,
        # rotation_range=180,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # shear_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        # zoom_range=[0.9,1.5],
        # horizontal_flip=True,
        # vertical_flip=True,
        fill_mode='nearest')

    datagen_test = ImageDataGenerator(rescale=1./255)

    # Specifying batch size
    batch_size = 16

    # Save augmented images to check our parameters
    save_augmented_to_dir = False

    if save_augmented_to_dir:
        save_to_dir = 'augmented_images/'
    else:
        save_to_dir = None

    # Calling generators from directory
    generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                        target_size=input_shape,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        save_to_dir=save_to_dir)

    generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                      target_size=input_shape,
                                                      batch_size=batch_size,
                                                      shuffle=False)

    # Steps test specifies the number of steps to run before quitting (disable endless checking)
    nb_train_samples = generator_train.n
    nb_validation_samples = generator_test.n
    steps_per_epoch = nb_train_samples // batch_size
    steps_test  = nb_validation_samples // batch_size

    # Get the file-paths for all the images in the training- and test-sets
    image_paths_train = path_join(train_dir, generator_train.filenames)
    image_paths_test = path_join(test_dir, generator_test.filenames)

    # Get the class-numbers for all the images in the training- and test-sets.
    cls_train = generator_train.classes
    cls_test = generator_test.classes

    # Getting class names
    class_names = list(generator_train.class_indices.keys())

    # Get the number of classes for the dataset.
    num_classes = generator_train.num_classes

    # Load the first images from the train-set.
    images = load_images(image_paths=image_paths_train[0:9])

    # Get the true classes for those images.
    cls_true = cls_train[0:9]

    # Plot the images and labels using our helper-function above.
    # plot_images(images=images, cls_true=cls_true, cls_names=class_names, smooth=True)

    # Adding Class weights to compensate for the data imbalance
    class_weight = compute_class_weight(class_weight='balanced',
                                        classes=np.unique(cls_train),
                                        y=cls_train)
    class_weight = {i:w for i,w in enumerate(class_weight)}

    print("Class Weights:", class_weight)


    # Getting the last conv block
    transfer_layer = model.get_layer('block5_pool')
    print("Transfer Layer Output:", transfer_layer.output)

    # Dropping the classification layers on the old model
    conv_model = Model(inputs=model.input, outputs=transfer_layer.output)
    # conv_model.summary()

    # Sequential API adding a new model
    new_model = Sequential()
    new_model.add(conv_model)
    new_model.add(Flatten())
    new_model.add(Dense(1024, activation='relu'))
    new_model.add(Dropout(0.5))
    new_model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(lr=1e-5)
    loss = 'categorical_crossentropy'
    metrics = ['categorical_accuracy']

    # Showing trainable layers
    # print_layer_trainable(conv_model)




    # Freezing weights in early layers
    conv_model.trainable = True
    for layer in conv_model.layers:
        # Boolean whether this layer is trainable.
        trainable = ('block5' in layer.name or 'block4' in layer.name)

        # Set the layer's bool.
        layer.trainable = trainable

    # Model Load Weights
    new_model.load_weights('saved_models/weightsonly/vgg16ft_1_finetune.h5')

    # Compiling Model
    new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Saving model architecture to json
    # json_config = new_model.to_json()
    # saved_model = model_from_json(json_config)


    # Saving Training Progress

    filepath = "saved_models/weightsonly/finetune-weights-in-hdf5/cp-{epoch:04d}-{val_categorical_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True, monitor='val_categorical_accuracy')
    callbacks_list = [checkpoint]


    # Defining epochs
    epochs = 30


    # Training
    if class_weight is not None and len(class_weight) > 0:
        print("YES")
        history = new_model.fit(x=generator_train,
                                epochs=epochs,
                                steps_per_epoch=steps_per_epoch,
                                class_weight=class_weight,
                                validation_data=generator_test,
                                validation_steps=steps_test,
                                callbacks=callbacks_list )
    else:
        print("NO")
        history = new_model.fit(x=generator_train,
                                epochs=epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=generator_test,
                                validation_steps=steps_test,
                                callbacks=callbacks_list )

    # Saving model + weights
    new_model.save('saved_models/vgg16ft_1_finetune.h5', save_format='h5', overwrite=False)
    new_model.save_weights('saved_models/weightsonly/vgg16ft_1_finetune.h5', overwrite=False)

    # Summarize history for accuracy
    plot_training_history(history)

    # Evaluating model on test set
    result = new_model.evaluate(generator_test, steps=steps_test)
    print("Test-set classification accuracy: {0:.2%}".format(result[1]))

    # Example Errors
    example_errors(new_model, generator_test, steps_test, cls_test, image_paths_test, class_names)


if __name__ == '__main__':
    # Specifying Directories
    train_dir = 'D:/Coding Projects/Pycharm Projects/Datasets/animals/train'
    test_dir = 'D:/Coding Projects/Pycharm Projects/Datasets/animals/test'

    # Limit VRAM Usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # USE GPU
    with tf.device('/device:GPU:0'):
        fine_tune_pipeline()