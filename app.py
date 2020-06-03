from os import listdir
from os.path import isfile, join
import cv2
import numpy as np

PATH_DATASET_YOUNG = 'train/face-a/'
PATH_DATASET_ELDER = 'train/face-b/'

filename_training_young = [f for f in listdir(PATH_DATASET_YOUNG) if isfile(join(PATH_DATASET_YOUNG, f))]
filename_training_elder = [f for f in listdir(PATH_DATASET_ELDER) if isfile(join(PATH_DATASET_ELDER, f))]

dataset_young = []
dataset_elder = []


for filename in filename_training_young: 
    path_image = PATH_DATASET_YOUNG + filename
    # print(image)
    dataset_young.append(cv2.imread(path_image, cv2.IMREAD_GRAYSCALE))

# print(dataset_young)

for filename in filename_training_elder: 
    path_image = PATH_DATASET_ELDER + filename
    # print(image)
    dataset_elder.append(cv2.imread(path_image, cv2.IMREAD_GRAYSCALE))


training_young = dataset_young[:200]
training_elder = dataset_young[:200]

test_young = dataset_young[200:]
test_elder = dataset_elder[200:]




