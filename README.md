# EE541_Project23

Motive:

The goal of this project was to use deep learning techniques for the task of American sign language detection.
American sign language is a visual-gestural language used by deaf and hard- of-hearing individuals in the United States and Canada. 
It is a complex and rich language that relies on hand gestures, facial expressions, and body posture to convey meaning. 
We have built a custom CNN model to achieve this goal and also trained the non-pretrained ResNet-18 model to compare the performances of the two models.

Dataset:

The Dataset is in the form of a hdf5 file "ASLdatasetGroup23.hdf5". The training dataset is a combination of 87,000 images from kaggle and 12,300 images
from custom dataset. The test dataset is a combination of 29 images from kaggle and 1450 images from custom dataset. The images correspond to 29 classes with 26 of the 29 
classes corresponding to the letters of the english alphabet and the rest are SPACE, delete and NOTHING.

The link to the dataset is : https://drive.google.com/file/d/1VGD3k4bXlX1Q0FwXL2jt0Bj-qRPpVdsE/view?usp=sharing

Files Description:

File 1: EE541_Project_Primary_CNN_Model.ipynb - This corresponds to the first model we built during the duration of this project. This is a fairly simple model
with just 2 convolutional layers.

File 2: EE541_Project_Secondary_CNN_Model.ipynb - This corresponds to the second model we built during the duration of this project. This is an improved model
when compared to the primary model in terms of performance (accuracy and loss)

File 3: EE541_Project_Final_CNN_Model.ipynb - This corresponds to the final model we built during the duration of this project. This is the best model out of the 3
models we built. It generalizes better and also a relatively large model with 5 convolutional layers.

File 4: EE541_Project_ResNet_Model.ipynb - This corresponds to the ResNet-18 model we used to compare our model's performances. This is not a pre-trained model but
just a resnet model (untrained) which is trained on our custom dataset.


Getting Started:

1. Download the dataset from the above link mentoned into a directory.
2. Chose any of the models from the above 4 files and download into the same directory.
3. Run the program to train and test the model for the dataset.

Disclaimer: The libraries used in building the model have to be installed on your local machine.







