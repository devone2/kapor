import glob

import numpy as np
import cv2 as cv
import utils as cp
import os

print(cp)

files = glob.glob("unlabeled/*.jpg")

print("Found " + str(len(files)) + " files to annotate")
cv.namedWindow('image1')
cv.namedWindow('image2')

for file in files:
    img = cv.imread(file, 0)
    cv.imshow('image1',img);
    cv.imshow('image2',cp.clearImage(img));
    cv.waitKey(1)
    name= input("Enter captcha: ")
    print("Moving " + file + " to labeled/" + name + ".jpg")
    os.rename(file, "labeled/" + name + ".jpg")
    
