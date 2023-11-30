# sign-language-detector-python

Sign language detector with Python, OpenCV and Mediapipe !

This code aims to gamify landmark classification with the use of Random Forest Classifier
on the hands of an individual for the purpose of self-teaching American Sign Language. 

More info can be found in the respective codes.

Repo Structure

- pictures
- music
- data collection [collect_images.py]
- creating datasets by collecting landmarks in data and labels [create_dataset.py]
- inference to do prediction of the labels [inference_classifier.py]
- training the Random Forest Classifier (RFC) [train_classifier_RFC.py]

- data.pickle is the data used for training the RFC
- model.p is the Random Forest Classifier Model
- train_csv is the training data for RFC
- train directory contains all the im
- resized_images is just the data used from Roboflow that is USED in the inference stage.

Note: sign-language-detector-python is the main directory.

- Danial C.
