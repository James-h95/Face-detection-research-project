import cv2
import numpy as np
from HOG import *

if __name__ == "__main__":
    test_img = np.zeros((3,3))
    test_img[0][0] = 13
    test_img[0][1] = 12
    test_img[0][2] = 4
    test_img[1][0] = 3
    test_img[1][1] = 5
    test_img[1][2] = 7
    test_img[2][0] = 2
    test_img[2][1]=  10
    test_img[2][2] = 8
    kernalx = np.array([[0,0,0],[-1,0,1],[0,0,0]])
    kernaly = np.array([[0,-1,0],[0,0,0],[0,1,0]])
    ts = cv2.filter2D(src = test_img, ddepth = -1, kernel = kernalx )
    ts2 = cv2.filter2D(src = test_img, ddepth = -1, kernel = kernaly )
    print(test_img)
    
    print(ts)
    print(ts2)

    hog = HOG()
    img = hog.load_image_as_gray_matrix("data/MIT/00001_male_back.jpg")
    features = hog.histogram(img)

    print("end")