"""

This is the main application

"""

from os import listdir
from os.path import isfile, join
from feature_exctraction.resnet import FeaturesExtractor
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH_DATASET_YOUNG = 'train/face-a/'
PATH_DATASET_ELDER = 'train/face-b/'

filename_training_young = [f for f in listdir(PATH_DATASET_YOUNG) if isfile(join(PATH_DATASET_YOUNG, f))]
filename_training_elder = [f for f in listdir(PATH_DATASET_ELDER) if isfile(join(PATH_DATASET_ELDER, f))]

dataset_young = []
dataset_elder = []


for filename in filename_training_young: 
    path_image = PATH_DATASET_YOUNG + filename
    dataset_young.append((cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)))


for filename in filename_training_elder: 
    path_image = PATH_DATASET_ELDER + filename
    dataset_elder.append((cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)))

# cv2.imshow('funziona', dataset_elder[0]) 

# cv2.waitKey(0)  
  
# #closing all open windows  
# cv2.destroyAllWindows() 

training_young = dataset_young[:200]
training_elder = dataset_young[:200]

test_young = dataset_young[200:]
test_elder = dataset_elder[200:]

extractor = FeaturesExtractor()

img = training_young[0]
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

features = extractor.getFeatures(img)

features = np.sort(features)[::-1]

x_axis = np.arange(len(features))
plt.scatter(x_axis,features)
plt.show()

d = len(features)
y = np.cumsum(features)/np.sum(features)
plt.plot(np.arange(1,d+1),y)
plt.scatter(np.arange(1,d+1),y)
plt.show()
# print(features)



