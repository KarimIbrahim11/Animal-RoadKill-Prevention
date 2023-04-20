from keras.models import load_model
from helpers import load_labels, load_image, preprocess_image, plot_prediction


if __name__ == '__main__':

    # Loading Labels
    labels = load_labels('D:/Coding Projects/Pycharm Projects/Datasets/animals/meta/labels.txt')
    labels.sort()

    # Whole model
    model = load_model('saved_models/vgg16ft_2_finetune.h5')
    # # Arch only
    # model = model_from_json('json_string')
    # # Weights only
    # model.load_weights('my_model_weights.h5')

    # Load input data
    img = load_image('test_images/17.jpg')

    # Pre process data
    img = preprocess_image(img)

    # Perform inference
    pred = model.predict(img)

    # Show Prediction
    plot_prediction(img, pred, labels)





