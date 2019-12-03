# This file train and save the model to disk

from mnist import MNIST
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Loading dataset...")
mndata = MNIST("./data/")
images, labels = mndata.load_training()

clf_model = RandomForestClassifier(n_estimators=100)

# Train on the first 10000 images:
train_x = images[:10000]
train_y = labels[:10000]

print("Train model")
clf_model.fit(train_x, train_y)

# Test on the next 1000 images:
test_x = images[10000:11000]
expected = labels[10000:11000].tolist()

print("Compute predictions")
predicted = clf_model.predict(test_x)

print("Accuracy: ", accuracy_score(expected, predicted))

# Save model to disk

filename = 'finalized_RF_model.sav'
pickle.dump(clf_model, open(filename, 'wb'))