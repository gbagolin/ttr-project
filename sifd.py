import cv2
import sys

index_image = 0
index2 = 0

def single_image_face_detection(image_path,save_directory):

    global index_image
    global index2

    imagePath = image_path

    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    
    print("[INFO] Found {0} Faces!".format(len(faces)))


    for (x, y, w, h) in faces:

        crop_img = image[y:y+h, x:x+w]
        status = cv2.imwrite("{0}/cropped_{1}.jpg".format(save_directory,index_image), crop_img)
        print(status)

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        index_image += 1

    if len(faces) != 0: 
        status = cv2.imwrite('{0}/faces_detected_{1}.jpg'.format(save_directory,index2), image)
        print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

        index2 += 1

