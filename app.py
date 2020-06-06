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

PATH_DATASET = 'train/face-a/'

filename_images = [f for f in listdir(PATH_DATASET) if isfile(join(PATH_DATASET, f))]

dataset_complete = []

#from 0 to 30 label = 0, from 31 to 59 label = 1 
labels = []

for filename in filename_images: 
    path_image = PATH_DATASET + filename
    dataset_complete.append((cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)))
    index = filename.find('A') + 1
    age = 0 if int(filename[index:index+2]) < 30 else 1  
    labels.append(age)

labels = np.array(labels)

''' 

Image reshape 

'''

# image = dataset_complete[2]
# cv2.imshow('img', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


row,col = dataset_complete[0].shape

len_dataset_complete = len(dataset_complete)
dataset_reshaped_complete = np.zeros((len_dataset_complete, row * col))

# for i in range(len_dataset_complete):
#   dataset_reshaped_complete[i,:] = np.reshape(dataset_complete[i], (len_dataset_complete, row * col))

dataset_reshaped_complete = np.reshape(dataset_complete, (len_dataset_complete,row*col))

'''

Divide data in training and testing 

'''

print("dataset_reshaped_complete.shape " , dataset_reshaped_complete.shape )

training_set = dataset_reshaped_complete[:5000,:]
test_set = dataset_reshaped_complete[5000:,:]

training_labels = labels[:5000]
test_labels = labels[5000:]


print("Training_set.shape ", training_set.shape)

'''

PCA 

'''

tmp = np.array([training_set[0]])

print(tmp.shape)

XT,Tr,media = getFeatures(tmp,0.9)

'''

BAYES 

'''

# Xc_test = test_set - media[:,np.newaxis]

# XT_test = np.dot(Xc_test, Tr)

# print("XT.shape: ", XT.shape)

classe1 = dataset_reshaped_complete[labels == 0, :]
classe2 = dataset_reshaped_complete[labels == 1, :]



# classe1 = np.dot(classe1,Tr)
# classe2 = np.dot(classe2,Tr)

print("Classe1.shape: ", classe1.shape)
# print(classe2.shape)                          

# m1 = np.mean(classe1, axis=0)
# m2 = np.mean(classe2, axis=0)

# C1 = np.cov(classe1,rowvar=False)
# C2 = np.cov(classe2,rowvar=False)

# from scipy.stats import multivariate_normal

# lik1 = multivariate_normal.pdf(XT_test, m1, C1)
# lik2 = multivariate_normal.pdf(XT_test, m2, C2)

# loglik = np.log( np.vstack((lik1, lik2)))
# prediction = np.argmax(loglik, axis=0)

# accuracy = np.sum(prediction == test_labels)/len(test_labels)

# print('Accuratezza del classificatore: ' + "{0:.2f}".format(accuracy*100) + "%")