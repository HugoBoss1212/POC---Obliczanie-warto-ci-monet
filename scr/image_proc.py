import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage import exposure
from skimage.util import invert
from skimage.morphology import binary_closing, watershed, binary_erosion, binary_dilation, opening, erosion, square, dilation
from skimage.feature import peak_local_max, canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from scipy import ndimage as ndi
import cv2
import numpy as np


def calc(image, real_num):

    img = cv2.threshold(image, 0.50, 255, cv2.THRESH_BINARY)[1].astype(bool)
    img = ndi.binary_closing(img)
    img = sobel(img)
    img = binary_closing(img)
    img = ndi.binary_fill_holes(img)

    img = binary_dilation(img)
    img = binary_erosion(img)

    label_objects, nb_labels = ndi.label(img)
    sizes = np.bincount(label_objects.ravel())

    temp = 0
    while nb_labels != real_num:
        label_objects = binary_erosion(label_objects)
        label_objects, nb_labels = ndi.label(label_objects)
        temp += 1
    print(str(temp) + " number of erosions ")
    for i in range(0, temp):
        label_objects = binary_dilation(label_objects)

    label_objects, nb_labels = ndi.label(label_objects)
    sizes = np.bincount(label_objects.ravel())

    distance = ndi.distance_transform_edt(label_objects)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), min_distance=20,
                                labels=label_objects)
    markers = ndi.label(local_maxi)[0]
    img = watershed(- distance, markers, mask=label_objects)

    print(str(nb_labels) + " number of objects")
    # for i in sizes:
    #     print(str(i) + " wielkosci obiektow")

    plt.imshow(img, cmap="hot")
    plt.axis("off")
    plt.show()
