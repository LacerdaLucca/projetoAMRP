import cv2 as cv
# Load two images
img1 = cv.imread('celulasPNG/getIT_23.png')
img2 = cv.imread('test1/getIT_23.mimg.png')
assert img1 is not None, "file could not be read, check with os.path.exists()"
assert img2 is not None, "file could not be read, check with os.path.exists()"
# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
dim = (cols,rows)
img1 = cv.resize(img1,dim,interpolation = cv.INTER_AREA)
print(img1.shape)
print(img2.shape)
# Now create a mask of logo and create its inverse mask also
# Now black-out the area of logo in ROI
dst = cv.bitwise_and(img1,img2,mask = None)
# Take only region of logo from logo image.
# Put logo in ROI and modify the main image
cv.imshow('res',dst)
cv.waitKey(0)
cv.destroyAllWindows()