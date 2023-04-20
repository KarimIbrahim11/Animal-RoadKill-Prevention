# Animal-RoadKill-Prevention
This is an image classification project that aims to classify animal images into their names using deep learning techniques.

## Dataset
The dataset used in this project is the: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals?resource=downloadThe dataset. It's an unbalanced dataset that has 90 animal classes with an average of 65 image per class.  Since the task is to classify animals on the road, I removed Classes of Sea animals, most birds and almost all reptiles. That resulted in 47 class found in the: labels.txt file. The dataset has been split into training, and testing sets only as it was very small. The training set contains 80% of the images, the set set contains the rest 20% of the images.

## Methodology
The project uses deep learning techniques for image classification. The model architecture used is keras's VGG16 pretrained on ImageNet. I first Transfer Learned the model(i.e: dropped the classification layers and added my own, freezing the old layers) till it reached a decent accuracy and then I unfreezed Block4 and Block 5 of the vgg16 and re-trained the last 5 layers (Fine-Tuning). The model is trained using the a learning rate of 1e-5. I used class weights to compensate for the unbalanced dataset. The training is stopped after 70 epochs or when the validation loss stops decreasing.

## Results
The model achieves an accuracy of 82% on the testing set.
