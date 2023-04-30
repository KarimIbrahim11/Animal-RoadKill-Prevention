# Animal-RoadKill-Prevention
This is an image classification project that aims to classify animal images into their names using deep learning techniques.

### Streamlit Demo Link: https://karimibrahim11-animal-roadkill-prevention-streamlit-app-7t2zxz.streamlit.app/

## Dataset
The dataset used in this project is the: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals?resource=downloadThe dataset. It's an unbalanced dataset that has 90 animal classes with an average of 65 image per class.  Since the task is to classify animals on the road, I removed Classes of Sea animals, most birds and almost all reptiles. I also removed very small images from each class, files less thank 9KB were deleted. That resulted in 47 class found in the: labels.txt file. The dataset has been split into training, and testing sets only as it was very small. The training set contains 80% of the images, the set set contains the rest 20% of the images.

## Methodology
The project uses deep learning techniques for image classification. The model architecture used is keras's VGG16 pretrained on ImageNet. I first Transfer Learned the model(i.e: dropped the classification layers and added my own, freezing the old layers) till it reached a decent accuracy and then I unfreezed Block4 and Block 5 of the vgg16 and re-trained the last 5 layers (Fine-Tuning). The model is trained using the a learning rate of 1e-5. I used class weights to compensate for the unbalanced dataset. The training is stopped after 70 epochs or when the validation loss stops decreasing.

## Results
The model achieves an accuracy of 82% on the testing set.
### - Sample Result 1
![results_1](https://user-images.githubusercontent.com/47744559/235335251-3cce6b50-2df2-489a-a130-4d6e28a76771.jpg)
### - Sample Result 2
![results_2](https://user-images.githubusercontent.com/47744559/235335255-e12a9471-2b6e-4ae7-a6c3-a18883ac0091.jpg)



## Project Structure

### Directories

- ```augmented images/```  (the path I save to the augmented image in the preprocessing step for checking them out)
- ```ds-manipulation/```   (the path I use for the dataset manipulation (i.e: cleaning and Dataset Splitting))
- ```saved models/```      (the path I use to sae the models and weights for inference)
- ```test images/```       (the path I use for storing images that I used to test inference)

### .txt Files

- ```labels.txt```       contains the labels of the 47 animal
- ```datasetsrc.txt```    contains the link of the kaggle dataset

### .py Scripts

- ```requirements.txt```  contains the packages used with their respective versions
- ```train.py```          the script used in Transfer-Learning
- ```fine_tune.py```      the script used for fine-tuning 
- ```streamlit_app.py```  the script used for the streamlit web ui service
- ```inference.py```      the script used for local inference (without streamlit_app)
- ```helpers.py```        the script used for providing helper functions to all of the aforementioned scripts


## Instructions to Use Repo and Pre-train your Model on a local Dataset

- Collect/ Download your DS .
- Use ```ds-manipulation/custom_dataset_meta.py``` script to generate ```labels.txt```, ```train.txt```, ```test.txt``` files which will then be used by the ```split_dataset.py``` .
- Use ```ds-manipulation/split_dataset.py``` to split the dataset to test and train in a directory structure that follows ```tf.Datasets``` standards .
- (optional) Use ```ds-manipulation/data_cleaning.py``` for cleaning the small files (default is < 9KB) feel free to add your own techniques or change the already existing ones .
- Depending on the size of your Dataset and the similarity of the original trained on task decide whether to transfer learn or fine-tune or both based on the      following table: 

![hackernoon](https://user-images.githubusercontent.com/47744559/235335453-44381f50-f319-4bca-88f2-3a7938aa2501.jpg)
** Image courtesy of Hackernoon : https://hackernoon.com/transfer-learning-approaches-and-empirical-observations-efeff9dfeca6

- Now, use the ```train.py``` to transfer-learn your model to the new problem. I transfer learned using a Scheduler and Early Stopping techinque for 5 epochs on the val_loss metric. The script automatically saves best the best model only in the saved_models/dir. 
- Use ```fine_tune.py``` for fine_tuning your saved model (make sure to change the directory of the load_model function to the saved model in saved_models/ dir). Recommended training on a mid size dataset (100K images) is 70 epochs. 
- Scroll the ```helpers.py``` function for useful visualization and Classification Report, Confusion Matrix Functions
- Use ```inference.py``` to use the saved model on test_images (perhaps in the test_images/ dir)
- If you wish to use ```streamlit_app.py``` to deploying your model on web. First, Upload your repo to github (It has to be public). Create an account on Streamlit and connect it to your Github. Create an app on streamlit and link it to the Repo/branch and streamlit_app.py file. Use the command ``` pip freeze > requirements.txt ``` to export your dependencies to a text file that streamlit will use to open your webapp. 
- Note: Use the decorator with the ```getModel()``` in the ```streamlit_app.py``` to prevent the model from loading every time you upload an image.



