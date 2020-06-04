"""

This is the main application

"""

from os import listdir
from os.path import isfile, join
from feature_exctraction.resnet import FeaturesExtractor
from pca import getFeatures
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''

Dataset creation

'''

PATH_DATASET_YOUNG = 'train/face-a/'
PATH_DATASET_ELDER = 'train/face-b/'

filename_training_young = [f for f in listdir(PATH_DATASET_YOUNG) if isfile(join(PATH_DATASET_YOUNG, f))]
filename_training_elder = [f for f in listdir(PATH_DATASET_ELDER) if isfile(join(PATH_DATASET_ELDER, f))]


dataset_young = []
dataset_elder = []

dataset_complete = []

labels = []

for filename in filename_training_young: 
    path_image = PATH_DATASET_YOUNG + filename
    dataset_young.append((cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)))
    dataset_complete.append((cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)))
    #young = 1 
    labels.append(1)

for filename in filename_training_elder: 
    path_image = PATH_DATASET_ELDER + filename
    dataset_elder.append((cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)))
    dataset_complete.append((cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)))
    #elderly = 0
    labels.append(0)
''' 

Image reshape 

'''

# image = dataset_young[0]
# cv2.imshow('img', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

row,col = dataset_young[0].shape

len_dataset_young = len(dataset_young)
data_young = np.zeros((row * col, len_dataset_young))

for i in range(len_dataset_young):
    data_young[:,i] = np.reshape(dataset_young[i], row * col)

len_dataset_elder = len(dataset_elder)
data_elder = np.zeros((row * col, len_dataset_elder))

for i in range(len_dataset_elder):
  data_elder[:,i] = np.reshape(dataset_elder[i], row * col)


len_dataset_complete = len(dataset_complete)
data_complete = np.zeros((row * col, len_dataset_complete))

for i in range(len_dataset_complete):
  data_complete[:,i] = np.reshape(dataset_complete[i], row * col)


# print(data_complete.shape)

'''

Divide data in training and testing 

'''

training_set = data_complete[:,:400]
test_set = data_complete[:,400:]

training_labels = labels[:400]
test_labels = labels[400:]



# print(training_set.shape)
# print(test_set.shape)

'''

PCA 

'''

XT,Tr,media = getFeatures(training_set,0.9)

'''

BAYES 

'''

Xc_test = test_set - media[:,np.newaxis]

XT_test = np.dot(Xc_test.T, Tr)

# print(XT.shape)

classe1 = np.dot(data_young.T,Tr)
classe2 = np.dot(data_elder.T,Tr)

print(classe1.shape)
# print(classe2.shape)                          

m1 = np.mean(classe1, axis=0)
m2 = np.mean(classe2, axis=0)

C1 = np.cov(classe1,rowvar=False)
C2 = np.cov(classe2,rowvar=False)

from scipy.stats import multivariate_normal

lik1 = multivariate_normal.pdf(XT_test, m1, C1)
lik2 = multivariate_normal.pdf(XT_test, m2, C2)

loglik = np.log( np.vstack((lik1, lik2)))
prediction = np.argmax(loglik, axis=0)

accuracy = np.sum(prediction == test_labels)/len(test_labels)

print('Accuratezza del classificatore: ' + "{0:.2f}".format(accuracy*100) + "%")