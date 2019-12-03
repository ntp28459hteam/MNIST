# This file train and save the model to disk

from mnist import MNIST
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Loading dataset...")
mndata = MNIST("./data/")
images, labels = mndata.load_training()

# Load model from disk

filename = 'finalized_RF_model.sav'

clf_model = pickle.load(open(filename, 'rb'))

# Test on the next 1000 images:
# test_x = images[10000:11000]
# expected = labels[10000:11000].tolist()

import cv2
img = cv2.imread('digit.png', 0).flatten()
test_x = [img]
expected = [6]

import numpy as np
print (np.shape(test_x))

print("Compute predictions")
predicted = clf_model.predict(test_x)
print (predicted)

# print("Accuracy: ", accuracy_score(expected, predicted))
# result = clf_model.score(test_x, expected)
# print(result)