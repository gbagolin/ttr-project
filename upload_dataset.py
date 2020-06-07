import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib import pyplot as plt


def upload_dataset(PATH, age, operator):

    filename_images = [f for f in listdir(PATH) if isfile(join(PATH, f))]

    dataset = []
    labels = []

    #from 0 to 30 label = 0, from 31 to 59 label = 1 

    for filename in filename_images: 
        path_image = PATH + filename
        index = filename.find('A') + 1
        dataset.append((cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)))

        if int(filename[index:index+2]) >= age:
                labels.append(0)
 
        elif int(filename[index:index+2]) <= age:
                dataset.append((cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)))
                labels.append(1)
    
    len_dataset = len(dataset) 
    row,col = dataset[0].shape

    print(row,col)

    dataset_reshaped = np.zeros((row * col,len_dataset))
    for i in range(len_dataset):
        dataset_reshaped[:,i] = np.reshape(dataset[i], row*col)


    return dataset_reshaped, np.array(labels)