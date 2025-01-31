from glob import glob
import cv2, os
import numpy as np

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


for item in glob('images/*.jpg'):
    img = cv2.imread(item)
    #img = cv2.resize(img, (1640, 590))
    #img = image_resize(img, height = 590)
    
    img = image_resize(img, height = 1640)
    
    #zeros = np.zeros((590, 1640, 3)).astype(np.uint8)
    #h, w, c = img.shape
    #zeros[:h, :w] = img
    #img = zeros
    print(img.shape)
    cv2.imwrite('imageskeepap2/%s'%(os.path.basename(item)), img)
    break
