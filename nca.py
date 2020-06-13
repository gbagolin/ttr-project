# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reshape import reshape
from upload_dataset import upload_dataset

NUM_CLASSES = 7
n_neighbors = 3
random_state = 0

dataset = [] 
labels = []

for i in range(7): 
    tmp_dataset, tmp_labels = upload_dataset('train/faces/{0}/'.format(i), i)
    dataset.append(tmp_dataset)
    labels.append(tmp_labels)


training_size = 800

x_train = []
y_train = []
x_test = []
y_test = []

for i in range(NUM_CLASSES): 
    x_train += (dataset[i])[:training_size]
    y_train += (labels[i])[:training_size]
    x_test += (dataset[i])[training_size:]
    y_test += (labels[i])[training_size:]


y_train = np.array(y_train)
y_test = np.array(y_test)

row,col = 200, 200

x_train_reshaped = reshape(x_train, row, col, len(x_train))
x_test_reshaped = reshape(x_test, row, col, len(x_test))

x_train_reshaped = x_train_reshaped.T
x_test_reshaped = x_test_reshaped.T

print(x_train_reshaped.shape,x_test_reshaped.shape)
print(len(y_train),len(y_test))

X_train = x_train_reshaped
X_test = x_test_reshaped


dim = len(x_train_reshaped[0])
print(dim)
n_classes = len(np.unique(labels))
print(n_classes)

# Reduce dimension to 2 with NeighborhoodComponentAnalysis
nca = make_pipeline(StandardScaler(),
                    NeighborhoodComponentsAnalysis(n_components=2,
                                                   random_state=random_state))

# Use a nearest neighbor classifier to evaluate the methods
knn = KNeighborsClassifier(n_neighbors=n_neighbors)


nca.fit(X_train, y_train)

knn.fit(nca.transform(X_train), y_train)

acc_knn = knn.score(nca.transform(X_test), y_test)
print("Test accuracy: {}".format(acc_knn))

