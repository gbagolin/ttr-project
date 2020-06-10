import cv2
from os import listdir
from os.path import isfile, join
import numpy as np


def upload_dataset(PATH, age, N = None):

    filename_images = [f for f in listdir(PATH) if isfile(join(PATH, f))]

    dataset = []
    labels = []

    for filename in filename_images: 
        
        path_image = PATH + filename
        index = filename.find('A') + 1
        dataset.append(cv2.imread(path_image , 0))

        face_age = int(filename[index:index+2])

        if face_age >= age:
                labels.append(0)
 
        elif face_age <= age:
                labels.append(1)

        if N != None and len(dataset) == N: 
            break 

    return dataset, labels