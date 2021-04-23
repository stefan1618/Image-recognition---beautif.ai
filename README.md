# Image recognition - beautif.ai
---


[//]: (Image references)

[ROC_curve]:Documentation/ROC_curve.jpg

## Project description
This project is primarly created for a use in an application called *Beautif.ai* - Application that is designed to edit and enhance pictures on a mobile device. 

The primary goal of the project is to build a robust classifier that will recognize 5 types of pictures : 

- Indoor selfie
- Outdoor selfie
- Indoor photo pose
- Outdoor photo pose
- Picture without people

A use case example would be to offer certain picture filters according to the picture type.

The steps for building this project are the following:

- Manually collecting data
- Preparing the custom dataset
- Building a classifier
- Training the classifier on training and validation set
- Succesfully testing the classifier with additional test set

## File description
This project includes the following files:
- requirements.txt - all of project's dependencies
- Training_Beautif.ipynb - a script for data preprocessing, training and saving the model
- Predictions_Beautif.ipynb - a script for making the predictions of the used model
- README.md - documentation and summary of the results

## Dataset info

The dataset consists of *5 000 images*, or around 1 000 per category. First, the data are pre-processed by converting the images to *RGB color space*, *image normalization to 0 - 1 range* is applied and  the images are resized to *224x224 pixels*. Next, the dataset is divided into *80% training* and *30% validation data*. Independent dataset of 50 images was provided and processed in the same way in order to test the model. Additionally we used data augmentation while training the models.





#### Dataset distribution and Train-Test split

| Category |Train set| Validation set|  Total No. of pictures |
|-------|---------|---------------|------------------|
|Indoor selfie| | |1.000|
|Outdoor selfie| | |1.000|
|Indoor posing| | |1.000|
|Outdoor posing|||1.000|
|Picture without people| | | 1.000|
|**Total** | | | |


## Best model info

To solve the problem we explore the applicability of transfer learning, by training a classifier on features extracted by a pre-trained deep convolutional neural network. *MobileNet* based features with NN classifier, is able to achieve best average testing and validation accuracy rate of 95%, for the shortest amount of time. MobileNet is commonly used for mobile and embedded vision applications because of its fewer number of parameters and high classification accuracy.

## Final architecture

To build the final model we implement a typical transfer-learning workflow using Keras :
 
Instantiate a MobileNet base model and load pre-trained weights into it by setting weights='imagenet' and input_shape=(224, 224, 3).
Freeze all layers in the base model by setting trainable = False and remove the fully connected layer with setting include_top=False.
Create a new model on top of the output from the base model which consists of one fully connected layer with 256 neurons and a soft-max layer with 5 neurons which corresponds to the number of categories.

## Benchmark

(slika so classification report, CM, ROC curves, Precision/Recall, Hardware)

|ROC curve| Classification report| Confusion Matrix| Precision/Recall| Hardware|
|---------|----------------------|-----------------|-----------------|---------|
|![ROC_curve]| | | | | |

---

## Conclusion

By using transfer learning which is a proven technique in the domain of image classification we have achieved to build and train a neural network with much less data, computational power and a low error.

Through the process we have created a CNN from scratch, tried other pre-trained CNNs combined with some data preparation techniques such as data augmentation. Also, techniques such as  fine-tuning, or cutting a neural network on intermediate layers were implemented.

Below is a table with the results:

|Model| Accuracy|
|-----|---------|
|Xception||
|InceptionV3||
|ResNet50||
|VGG16||
|VGG19||
|MobileNet||

The project can be further extended to more images, to tune more of the available parameters of the models, try different configurations for the top classifier etc.
