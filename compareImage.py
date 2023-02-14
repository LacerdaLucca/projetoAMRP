import cv2 as cv
import numpy as np
import os

def mse(img1, img2):
   h, w = img1.shape
   diff = cv.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse, diff

input_dataset1 = 'erodedMImage'
input_dataset2 = 'freeOperations'

for i,filename in enumerate(os.listdir(input_dataset1)):
    img1 = cv.imread(input_dataset1+"/"+filename)
    img2 = cv.imread(input_dataset2+"/"+filename)

    rows,cols, c = img1.shape
    dim = (cols,rows)
    img2 = cv.resize(img2,dim,interpolation = cv.INTER_AREA)
    cv.imwrite("miniFO/"+os.path.splitext(filename)[0]+".png", img2)

    #error, diff = mse(img1, img2)
    #print("Image matching Error between "+ input_dataset1+"/"+filename + " and " + input_dataset2+"/"+filename + ":",error)

# 500 a 700 grid seeds
# 5 interaçoes
# c1 0.7
# c2 0.8
# Gradent variation