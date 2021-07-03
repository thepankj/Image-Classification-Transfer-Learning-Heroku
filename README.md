# Multiclass Image Classification using Transfer Learning 

## Introduction
In machine learning, Transfer Learning is the process of using knowledge previously acquired by solving a problem on a new (current) problem, with or without changes to the knowledge.

This is possible because when a model is trained, the initial neurons of a CNN architecture extract only the high level information, which gets narrowed down to information about the specific classes with each neural layer. For example, if a model is trained to recognize flowers, it can be further trained to classify different flower species.

In this project, I have used 4 pre-trained models to predict the species of butterflies.

The full notebook is availavle [here](https://www.kaggle.com/pnkjgpt/multiclass-image-classification-transfer-learning)

## Dataset
The dataset for this project is taken from my Kaggle profile [here](https://www.kaggle.com/pnkjgpt/butterfly-classification-dataset)

The directory structure is in the following format:
```
Train   
│
└───adonis
│   │   000.jpg
│   │   001.jpg
│   │   ...
│
└───american anoot
│   │   000.jpg
│   │   001.jpg
│   │   ...
│
└───an 88
│   │   000.jpg
│   │   001.jpg
│   │   ...
...
...

...
│
└───zebra long wing
    │   000.jpg
    │   001.jpg
    │   ...
```

The dataset has around 4000 images of 50 different species of butterflies.
![NUmber of images of each species](https://github.com/thepankj/Image-Classification-Transfer-Learning-Heroku/blob/main/images/no%20of%20species.jpg)

## Methodology
To build the classifier, I used pretrained models from [Keras Applications](https://keras.io/api/applications/). I took just the base of the models ans added more layers to make it fit for this problem.

Then I saved the models and used StreamLit to create a webapp which can take in an image and predict to which class it may belong to.

For the deployment, I used Heroku. The final site is [this](https://butterfly-classification.herokuapp.com/)
![Website Deployed](https://github.com/thepankj/Image-Classification-Transfer-Learning-Heroku/blob/main/images/website.jpg)

## Result
The best accuracy is given by ResNet with 82% accuracy ans the worst is by VGG16 with 71.75% accuracy.

## Difficulties & Challenges Faced
* Did not know what ImageDataGenerator and flow_from_directory returns. So had difficulty to pass new images for prediction to the models. But it's basically a 4D Numpy array.
* The training was done on Colaboratory. To create and test the app, I had to download the trained model and run it locally, but due to low specs, it couldn't run on my laptop. So tried using ngrok on Colab which didn't work, but an alternative, remote.it worked perfectly.
* To deploy the app on Heroku, I used tensorflow 2.5.0 whid has a huge size so the slug was exceeding the limits (500MB). Then I used tensorflow-cpu 2.5 and it worked.
