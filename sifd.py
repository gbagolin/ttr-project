# Single image face detection

import cv2
import sys


faces_count = 0 #count the number of faces present in a image
image_count = 0 #count the number of images computed


def single_image_face_detection(image_path,save_directory):
    """Takes in input one image, detecs the face if there is any and
        create a new image with the cropped face

    Args:
        image_path (string): Path of the image
        save_directory (string): Path of the directory where image will be saved
    """


    global faces_count
    global image_count

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #declare and run the classifier
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    
    print("[INFO] Found {0} Faces!".format(len(faces)))

    #draw a rectangle for each face detected
    for (x, y, w, h) in faces:

        crop_img = image[y:y+h, x:x+w]
        status = cv2.imwrite("{0}/cropped_{1}.jpg".format(save_directory,index_image), crop_img)
        print(status)

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        faces_count += 1

    if len(faces) != 0: 
        status = cv2.imwrite('{0}/faces_detected_{1}.jpg'.format(save_directory,index2), image)
        print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

        image_count += 1

