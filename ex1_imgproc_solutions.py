# -*- coding: utf-8 -*-
"""
Event-based Robot Vision
Exercise 1
Apr 23 2020
@author: ggb
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys # use sys.exit() to prematurely end the execution at certain line

# %% Close all plots
plt.close('all')

# %% Download "Standard" test images
# http://www.imageprocessingplace.com/root_files_V3/image_databases.htm
# http://www.imageprocessingplace.com/downloads_V3/root_downloads/image_databases/standard_test_images.zip


# %% Auxiliary functions
"""
Auxiliary function to save an positive and negative image to disk by first
normalizing it to 8 bits (i.e., values in [0,255])
"""
def imwriteSymmetricImage(filename, src, maxabsval=None):

    if not maxabsval:
        maxabsval = np.amax(np.abs(src))

    # Conversion form the interval [-1,1]*maxabsval to [0,255]
    #   the value -maxabsval maps to 0
    #   the value  maxabsval maps to 255
    img_normalized = (src + maxabsval)*(255/(2*maxabsval))
    cv2.imwrite(filename, img_normalized)


# %% Load image from disk

#img = cv2.imread('/home/ggb/improc_py/images/TE100.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('/home/ggb/improc_py/images/tigerg.jpg', cv2.IMREAD_GRAYSCALE)
img = img.astype(float) # Convert to floating point numbers

# Get image size
height, width = img.shape

# Different ways of printing values on screen
print height, width
print("height=",height," width=",width)
print("height= {}, width={}".format(height, width))

# Check the type of the variables
print(type(img)) # numpy ndarray
print(img.dtype) # unit8 or float64

#sys.exit()

# %% Display an image

fig = plt.figure()
fig.suptitle('this is a title')
plt.figimage(img, cmap='gray')
plt.show()


# %% Select a Region of Interest (ROI) within an image

#img_sub = img[ 0:300, 0:400 ]

# Using variables to specify the ROI (border cropping, in this case)
border_r = 10 # row dimension
border_c = 30 # column dimension
img_sub = img[ border_r:-2*border_r, border_c:-2*border_c ]
fig = plt.figure()
fig.suptitle('subimage')
plt.figimage(img_sub, cmap='gray')
plt.show()


# %% Load a standard image or a grayscale ramp

# Grayscale ramp image
x = np.arange(0, 255, 0.4)
y = np.arange(0, 255, 0.666)
xx, yy = np.meshgrid(x, y)
print yy.shape
fig = plt.figure()
plt.figimage(xx.astype(int), cmap='gray')
plt.show()

#img_u8 = cv2.convertScaleAbs(xx.astype(int))
img_u8 = cv2.imread('/home/ggb/improc_py/images/lena_gray_512.tif', cv2.IMREAD_GRAYSCALE)

print img_u8.dtype
print(img_u8.dtype) # unit8


# %% Spatial resolution

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
for ax, k in zip(axs.flat, range(6)):
    scale_factor = 1./(2**k)
    img_small = cv2.resize(img_u8, (0,0), fx=scale_factor, fy=scale_factor)
    ax.imshow(img_small, interpolation='none', cmap='gray')
    ax.set_title("Scale: 1 / " + str(2**k))
plt.tight_layout()
plt.show()


# %% Grayscale (range) resolution.
# Number of intensity levels, number of bits (bit depth). Artefacts

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
#fig.patch.set_facecolor('white')
for ax, k in zip(axs.flat, range(6)):
    size_quant_interval = 2**k
    img_quantized = (img_u8 / size_quant_interval) * size_quant_interval
    ax.imshow(img_quantized, interpolation='none', cmap='gray')
    ax.set_title("Num gray levels: " + str(256/size_quant_interval))
plt.tight_layout()
plt.show()


# %% Convolution refresher
# Play movie


# %% Smoothing by a box filter
# kernel of 5x5 size
# kernel = np.ones((5,5),np.float32)/25
# img_box = cv2.filter2D(img,-1,kernel)
img_box = cv2.blur(img, (5,5))

# Display original and filtered image
plt.figure()
#plt.imshow(img, interpolation='none', cmap='gray'), plt.title('Original')
plt.figimage(img, cmap='gray')
plt.show()
plt.figure()
#plt.imshow(img_box, interpolation='none', cmap='gray'), plt.title('Averaging')
plt.figimage(img_box, cmap='gray')
plt.show()


# %% Write image to disk
# Example of reading and image, filtering it and saving the result
cv2.imwrite('img_box.png', img_box)


# %% Smoothing by Gaussian filtering
s = 2. # Parameter that controls the amount of smoothing
if s > 0:
    img_gauss = cv2.GaussianBlur(img, (0,0), sigmaX = s, sigmaY = s)

fig = plt.figure()
fig.suptitle('Gaussian smoothing')
#plt.imshow(img_gauss, interpolation='none', cmap='gray'), plt.title('Gaussian smoothing')
plt.figimage(img_gauss, cmap='gray')
plt.show()


# %% Spatial gradient. Sobel x and y
grad_x = cv2.Sobel(img, cv2.CV_64F, 1,0)
grad_y = cv2.Sobel(img, cv2.CV_64F, 0,1)

fig = plt.figure()
fig.suptitle('Image gradient x')
plt.figimage(grad_x, cmap='gray')
plt.show()

fig = plt.figure()
fig.suptitle('Image gradient y')
plt.figimage(grad_y, cmap='gray')
plt.show()


# %% Plot the filter mask or signal or kernel used
ksize = 5  # Size of the impulse image
impulse_img = np.zeros((ksize,ksize))
impulse_img[ksize/2,ksize/2] = 1
sobel_response_to_impulse = cv2.Sobel(impulse_img, cv2.CV_64F, 1,0)
print sobel_response_to_impulse

# Need to scale the gradient appropriately so that they have meaning of derivative
grad_x /= 8
grad_y /= 8

# %% Write derivative images to disk

# Plot histogram
fig = plt.figure()
plt.hist(grad_x.ravel(), bins=101)
plt.title('Histogram of grad_x')
plt.show()

# Convert from (-b,b) to (0,255)
#min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grad_x)
min_valx, max_valx, _, _ = cv2.minMaxLoc(grad_x)
min_valy, max_valy, _, _ = cv2.minMaxLoc(grad_y)
M_val = np.amax([np.abs(min_valx), max_valx, np.abs(min_valy), max_valy])
print M_val
imwriteSymmetricImage('grad_x.png', grad_x, M_val)
imwriteSymmetricImage('grad_y.png', grad_y, M_val)

fig = plt.figure()
fig.suptitle('Image gradient x')
plt.figimage(grad_x, cmap='gray')
plt.show()

fig = plt.figure()
fig.suptitle('Image gradient y')
plt.figimage(grad_y, cmap='gray')
plt.show()

# positive and negative values. How are they represented?
# Common way in image processing, and also in event-based vision.


# %% Convert gradient (x,y) to magnitude and phase
grad_mag, grad_dir = cv2.cartToPolar(grad_x, grad_y)

# Alternative commands
#grad_mag = np.hypot(grad_x, grad_y)
#grad_dir = np.arctan2(grad_y,grad_x)

fig = plt.figure()
fig.suptitle('Gradient magnitude')
plt.figimage(grad_mag, cmap='gray')
plt.show()

fig = plt.figure()
fig.suptitle('Gradient direction')
plt.figimage(grad_dir, cmap='gray')
plt.show()

cv2.imwrite('grad_mag.png', grad_mag * 255 / np.amax(grad_mag))
imwriteSymmetricImage('grad_dir.png', grad_dir)


# %% Temporal gradient

# DOWNLOAD slider_depth sequence (txt format) from DAVIS dataset
"""
import wget
print('Beginning file download with wget...')
url = 'http://rpg.ifi.uzh.ch/datasets/davis/slider_depth.zip'
wget.download(url, './slider_depth.zip')
"""

# Load images
id0 = 59
filename_prefix = '/home/ggb/improc_py/slider_depth/images/frame_'
filename_suffix = '.png'
filename1 = filename_prefix + ("%08d" % id0) + filename_suffix
filename2 = filename_prefix + ("%08d" % (id0+1)) + filename_suffix

# Which means:
#img1 = cv2.imread('/home/ggb/improc_py/slider_depth/images/frame_00000059.png', cv2.IMREAD_GRAYSCALE)
#img2 = cv2.imread('/home/ggb/improc_py/slider_depth/images/frame_00000060.png', cv2.IMREAD_GRAYSCALE)

# Read two consecutive images
img1 = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
img1 = img1.astype(float)
img2 = img2.astype(float)
height, width = img1.shape
print height, width

# Compute the (approximation to the) temporal derivative
img_difference = img2 - img1

# Display two images and the temporal derivative
fig = plt.figure()
fig.suptitle('Image 1')
plt.figimage(img1, cmap='gray')
plt.show()

fig = plt.figure()
fig.suptitle('Image 2')
plt.figimage(img2, cmap='gray')
plt.show()

fig = plt.figure()
fig.suptitle('Difference I2 - I1')
plt.figimage(img_difference, cmap='gray')
plt.show()

# Saving images to disk using the same symmetric rage for all 3 derivatives
grad_x = cv2.Sobel(img2, cv2.CV_64F, 1,0)
grad_y = cv2.Sobel(img2, cv2.CV_64F, 0,1)
min_valx, max_valx, _, _ = cv2.minMaxLoc(grad_x)
min_valy, max_valy, _, _ = cv2.minMaxLoc(grad_y)
min_valt, max_valt, _, _ = cv2.minMaxLoc(img_difference)
M_val = np.amax([np.abs(min_valx), max_valx, np.abs(min_valy), max_valy, np.abs(min_valt), max_valt])
print M_val
imwriteSymmetricImage('grad_x.png', grad_x, M_val)
imwriteSymmetricImage('grad_y.png', grad_y, M_val)
imwriteSymmetricImage('grad_t.png', img_difference)


# %% Other topics to practice:

# Median filter
# median = cv2.medianBlur(img,5)

# Bilateral filter
# blur = cv2.bilateralFilter(img,9,75,75)

# References:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#filtering
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html
