# Image recognition - beautif.ai
---
## Project description
This project is primarly created for a use in an application called *Beautif.ai* - Application that is designed to edit and enhance pictures on a mobile device. 

The steps for building this project are the following:

- Manually collecting dataset of 5.000 pictures
- Building a model that predicts 5 classes (detailes are explained below)
- Training the model with a training and validation set
- Succesfully testing the model with external additional 50 pictures

## Files
This project includes the following files:
- requirements.txt - all the frameworks used for creating this model
- Training_Beautif.ipynb - notebook that contains the script for preprocessing and training code
- Predictions_Beautif.ipynb - notebook that contains the script for prediction methods and the predictions of the used model
- README.md - summary of the results

## Dataset info

In order to create a good model, we need a good and balanced dataset. Therefore, we managed to collect 5.000 pictures (1.000 per each class), which were selected 
very carefully so we can achieve this.

For this project, we have splitted the pictures in five classes:

- Indoor selfie
- Outdoor selfie
- Indoor posing
- Outdoor posing
- Pictures with no humans included
- 

For the training part, the dataset was divided on **train 80%** and **validation 20%**.

| Class |Train set| Validation set|  Total No. of pictures |
|-------|---------|---------------|------------------|
|Indoor selfie| | |1.000|
|Outdoor selfie| | |1.000|
|Indoor posing| | |1.000|


## Best model info

In order to get the best model and a model that succesfully classifies one of five classes, we were trying several image classifiers.

Beside the CNN that we manually created (for which we got ...%), for the purpose of the project we decided to try six pre-trained CNNs and below is a table with the results we got:

|Model| Accuracy|
|-----|---------|
|Xception||
|InceptionV3||
|ResNet50||
|VGG16||
|VGG19||
|MobileNet||


From all of these models, we decided to go with **MobileNet** from Keras, because it gave great results and because it is a very small but fast and efficient model - exactly what was needed for the purpose of the webpage.

We instantiate a base model and load pre-trained weights from *imagenet* into it. Then we freezed all the layers and on top of the output of a layer we created a new model. 

We also tried models with cut on intermediate layers, but none of them gave better results then the ones above. We even faced with overfitting with one of them.
The new model was trained on the dataset.

We also tried models with cut on intermediate layers, but none of them gave better results then the ones above. We even faced with overfitting with one of them.

*In all of the models, augmentation method was used.*

---
## Conclusion
We think that one of the reasons we got these good results is because we managed to collect good and balanced dataset. Also, another very import reason is that many models were trained and that's how we saw the differences between the models, the different accuracies they gave and how they reacted on our dataset - Therefore, it was easy to choose MobileNet.

## What can be improved in future
Maybe if we can try to tune the parametars of the models more, we could get a slightly better results.



