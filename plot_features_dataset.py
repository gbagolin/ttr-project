"""

This plots the features extracted by the feature extractor 

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
    img = cv2.imread(path_image)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    dataset_complete.append(img)
    index = filename.find('A') + 1
    age = 0 if int(filename[index:index+2]) < 30 else 1  
    labels.append(age)

labels = np.array(labels)

'''

Divide data in young and elderly 

'''
young = np.array(dataset_complete)[labels == 0]
elderly = np.array(dataset_complete)[labels == 1]

extractor = FeaturesExtractor()

features_young= extractor.getFeaturesOfList(young[:50])
features_elderly= extractor.getFeaturesOfList(elderly[:50])

for i in range(50): 
    plt.scatter(np.arange(len(features_young[i])), features_young[i], c = 'r')
    plt.scatter(np.arange(len(features_elderly[i])), features_elderly[i], c = 'b')

plt.show()