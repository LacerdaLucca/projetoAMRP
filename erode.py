from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import os
from scipy import ndimage
from tkinter import filedialog
from PIL import Image


srcImage = None
srcMarker = None
erosion_size = 0



## [main]
def main(input_dataset):
    
   global srcMarker, srcImage
   for i,filename in enumerate(os.listdir(input_dataset)):
       srcMarker = cv.imread(cv.samples.findFile(input_dataset+"/"+filename))
       srcImage = cv.imread(cv.samples.findFile("minicells"+"/"+filename))
       if srcMarker is None:
           print('Could not open or find the image: ', filename)
           exit(0)
       srcImage = cv.cvtColor(srcImage, cv.COLOR_BGR2GRAY)
       srcMarker = threshold()
       erosion(0,filename)
## [main]
def mimage(input_dataset):    
    global srcMarker, srcImage
    for i,filename in enumerate(os.listdir(input_dataset)):
        srcMarker = cv.imread(cv.samples.findFile("thresholdMImage/"+filename))
        srcImage = cv.imread(cv.samples.findFile("minicells"+"/"+filename))
        if srcMarker is None:
            print('Could not open or find the image: ', filename)
            exit(0)
        srcImage = cv.cvtColor(srcImage, cv.COLOR_BGR2GRAY)
        # srcMarker = threshold()
        erosion(0,filename)

def converter(input_dataset):    
    global srcMarker, srcImage
    for i,filename in enumerate(os.listdir(input_dataset)):
        srcMarker = cv.imread(cv.samples.findFile(input_dataset+"/"+filename))
        srcImage = cv.imread(cv.samples.findFile("minicells"+"/"+filename))
        if srcMarker is None:
            print('Could not open or find the image: ', filename)
            exit(0)
        srcImage = cv.cvtColor(srcImage, cv.COLOR_BGR2GRAY)
        saveScribbles(filename)



# optional mapping of values with morphological shapes
def morph_shape(val):
    if val == 0:
        return cv.MORPH_RECT
    elif val == 1:
        return cv.MORPH_CROSS
    elif val == 2:
        return cv.MORPH_ELLIPSE


## [erosion]
def erosion(val,filename):

    erosion_size = 3
    erosion_shape = morph_shape(val)

    ## [kernel]
    element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    ## [kernel]
    erosion_dst = cv.erode(srcMarker, element)
    #output_dataset = "freeOperations/"+filename
    output_dataset = "erodedMImage/"+filename
    if not cv.imwrite(output_dataset, erosion_dst):
        raise Exception("Could not write image")
## [erosion]

def threshold():
    ret,thresh1 = cv.threshold(srcImage,127,255,cv.THRESH_BINARY)
    #thresh1 = cv.bitwise_not(thresh1)
    return thresh1


def getBackground():
    global srcMarker

    for i in range(srcMarker.shape[0]):
        for j in range(srcMarker.shape[1]):
            if(i == 0 or i == (srcMarker.shape[0]-1) or j == 0 or j == (srcMarker.shape[1]-1)):
                if(srcMarker[i,j,0] != 255):
                    srcMarker[i,j,0] = 127
                else:
                    srcMarker[i,j,0] = 0
            else:
                srcMarker[i,j,0] = 0



def getMarkers():
    global srcMarker, srcImage
    markers = []
    objMarkers = 0
    markers_sizes = []
    getBackground()
    width, height, channels = srcMarker.shape
    
    image, number_of_objects = ndimage.label(srcMarker[:,:,2])
    blobs = ndimage.find_objects(image)
    # print(np.max(srcMarker[:,:,0]), 'Channel 0')
    # print(np.max(srcMarker[:,:,1]), 'Channel 1')
    # print(np.max(srcMarker[:,:,2]), 'Channel 2')
    for i,j in enumerate(blobs):
        marker = []
        for y in range(j[0].start,j[0].stop):
            for x in range(j[1].start,j[1].stop):
                if(image[y,x] != 0):
                    marker.append([x,y])
    
        markers_sizes.insert(0, len(marker))
        markers = marker + markers
        objMarkers += 1

    image, number_of_objects = ndimage.label(srcMarker[:,:,0])
    blobs = ndimage.find_objects(image)
    
    for i,j in enumerate(blobs):
      marker = []
      for y in range(j[0].start,j[0].stop):
        for x in range(j[1].start,j[1].stop):
          if(image[y,x] != 0):
            marker.append([x,y])
      
      markers_sizes.append(len(marker))
      markers = markers + marker    

    return markers, objMarkers, markers_sizes


def saveScribbles(filename):
    global srcMarker, srcImage
    markers, objMarkers, markers_sizes = getMarkers()
    # drawMarkers()
    print(len(markers_sizes),len(markers))
    if(len(markers) == 0): 
        return
    f = open("markersMI/"+os.path.splitext(filename)[0]+".txt", 'w')
    f.write("%d\n"%(len(markers_sizes)))
    f.write("%d\n"%(markers_sizes[0]))

    index_sizes=0
    acum=0

    for i in range(len(markers)-1):
        if(acum == markers_sizes[index_sizes]):
            index_sizes+=1
            acum=0
            f.write("%d\n"%(markers_sizes[index_sizes]))
        
        [x,y] = markers[i]
        f.write("%d;%d\n"%(x,y))
        acum+=1

    if(acum == markers_sizes[index_sizes]):
        index_sizes+=1
        acum=0
        f.write("%d\n"%(markers_sizes[index_sizes]))
    
    [x,y] = markers[-1]
    f.write("%d;%d\n"%(x,y))
    f.write("%d"%(objMarkers))
    f.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for Eroding and Dilating tutorial.')
    parser.add_argument('--input', help='Path to input image.', default='erodedMImage')
    args = parser.parse_args()
    #mimage(args.input)
    converter(args.input)