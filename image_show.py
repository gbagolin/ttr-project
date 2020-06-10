import cv2

def image_show(image): 
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
