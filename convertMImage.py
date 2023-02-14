import pyift.pyift as ift
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import argparse
import sys


def normalize_by_band(feature):
    max_ = feature.max()
    min_ = feature.min()
    norm = 255 * ((feature - min_) / (max_ - min_))
    
    return norm

    
def decoder_fb(feature, kernel_importances = None, threshold=0.5):
    if(kernel_importances is None):
        kernel_importances = np.ones(feature.shape[3])
        
    img_size = [feature.shape[1], feature.shape[2]]
    positives = np.zeros(img_size)
    negatives = np.zeros(img_size)
    counter = 0
    n_counter = 0
    for b in range(feature.shape[3]):
        band_image = feature[0,:,:,b]
        band_mean = band_image.mean()
        norm_band_image = normalize_by_band(band_image)
        if(band_mean < threshold):
            positives += (norm_band_image*kernel_importances[b])
            counter+=kernel_importances[b]
        else:
            negatives += (norm_band_image*kernel_importances[b])
            n_counter+=kernel_importances[b]
    
    if(counter != 0):
        positives/=counter
        positives = normalize_by_band(positives)
    if(n_counter != 0):
        negatives/=n_counter
        negatives = normalize_by_band(negatives)
        
    decoded = positives - negatives
    
    decoded[decoded < 0] = 0
    
    # nb_components, components_image, stats, centroids = cv2.connectedComponentsWithStats((decoded>0).astype(np.uint8), connectivity=8)
    
    #---------This is a size filter. Uncomment if you want to use it----------
    #for c in range(1,nb_components):
    #    indices = np.argwhere(components_image==c)
    #    if(indices.shape[0] < 2000 or indices.shape[0] > 15000):
    #        for i in indices:
    #            decoded[i[0], i[1]] = 0
                
    if(decoded.max() == 0):
        return decoded
    decoded = normalize_by_band(decoded)
    
    return decoded

def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE


if __name__ == "__main__":
    try:
        ap = argparse.ArgumentParser()
    except:
        ap.print_help()
        sys.exit(0)
    #ap.add_argument("-i", "--input_dataset", required=True,	help="path to the folder with the feature images")
    #ap.add_argument("-o", "--output_dataset", required=True,	help="path to the folder for the decoded images")
    ap.add_argument("-t", "--threshold", required=False, default=0.5, help="Threshold to determine based on the mean saliency if the image is to be considered foreground")
    args = vars(ap.parse_args())
    input_dataset = "FLIM/activation1" 
    output_dataset = "test1/"
    #input_dataset = args["input_dataset"]
    #output_dataset = args["output_dataset"]
    threshold = float(args["threshold"])

    feature_ext = "mimg"
    out_ext = "png"
    
    if not os.path.exists(output_dataset):
        os.makedirs(output_dataset)
    
    dataset_size = len(os.listdir(input_dataset))
    toolbar_width = dataset_size
    print("Filtering images....")
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    for i,filename in enumerate(os.listdir(input_dataset)):
        #filename,ext = filename.split(".")
        featuref = os.path.join(input_dataset, filename)
        mfeature = ift.ReadMImage(featuref)
        feature = mfeature.AsNumPy()
        decoded = decoder_fb(feature)
        sys.stdout.write("-")
        sys.stdout.flush()
        print(os.path.splitext(filename))        
        cv2.imwrite(output_dataset+"/"+os.path.splitext(filename)[0]+"."+out_ext, decoded)
    sys.stdout.write("]\n") # this ends the progress bar
    print("Done!")
