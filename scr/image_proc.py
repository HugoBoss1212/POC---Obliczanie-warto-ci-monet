import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage import exposure
from skimage.util import invert
from skimage.morphology import binary_closing, watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import cv2
import numpy as np


def calc(image):

    img = cv2.threshold(image, 0.50, 255, cv2.THRESH_BINARY)[1].astype(bool)
    img = ndi.binary_closing(img)
    img = sobel(img)
    img = binary_closing(img)
    img = ndi.binary_fill_holes(img)

    distance = ndi.distance_transform_edt(img)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                labels=img)
    markers = ndi.label(local_maxi)[0]
    img = watershed(- distance, markers, mask=img)

    plt.imshow(img, cmap="hot")
    plt.axis("off")
    plt.show()
