''' 
This script is used to create a dataset 
Particularly, image of elderly and young people are taken from train/a and train/b directories 
Then, a face detection algorithm is used to detect faces in each photo. 
'''
from os import listdir
from os.path import isfile, join
from sifd import single_image_face_detection

PATH_A = 'train2/A'
PATH_B = 'train2/B'

PATH_FACE_A = 'train2/faces/A'
PATH_FACE_B = 'train2/faces/B'

IMAGES_DIRECTORY = 'train2/images'

dataset_a = [f for f in listdir(PATH_A) if isfile(join(PATH_A, f))]
dataset_b = [f for f in listdir(PATH_B) if isfile(join(PATH_B, f))]

for image in dataset_a: 
    # index = image.find('A') + 1
    # face_age = int(image[index:index+2])

    # index_folder = face_age // 10

    single_image_face_detection('train2/A/{0}'.format(image),PATH_FACE_A , IMAGES_DIRECTORY, image)

for image in dataset_b: 
    # index = image.find('A') + 1
    # face_age = int(image[index:index+2])

    # index_folder = face_age // 10

    single_image_face_detection('train2/B/{0}'.format(image),PATH_FACE_B , IMAGES_DIRECTORY, image)

# for image in dataset_b: 
#     single_image_face_detection('train/b/{0}'.format(image),PATH_FACE_B,IMAGES_DIRECTORY)



