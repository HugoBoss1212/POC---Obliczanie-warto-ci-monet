import math
from scipy.spatial import distance
import matplotlib.pyplot as plt
from skimage.morphology import square
import cv2
import numpy as np


def cog2(points):
    mx = 0
    my = 0
    for (y, x) in points:
        mx = mx + x
        my = my + y
    mx = mx/len(points)
    my = my/len(points)

    return [my, mx]


def compute_bb(points):
    s = len(points)
    my, mx = cog2(points)
    r = 0
    for point in points:
        r = r + distance.euclidean(point, (my, mx))**2
    return s/(math.sqrt(2*math.pi*r))


def compute_f(points):
    px = [x for (y, x) in points]
    py = [y for (y, x) in points]

    fx = max(px) - min(px)
    fy = max(py) - min(py)

    return float(fy)/float(fx)


def compute_h(dis, length_):
    sum1 = dis
    sum2 = math.pow(dis, 2)
    return math.sqrt((math.pow(sum1, 2)) / (length_ * sum2 - 1))


def labeling(local_max):
    local_max = np.array(local_max, dtype=np.uint8)
    local_max = cv2.dilate(local_max, square(4))
    label = 1
    for i in range(local_max.shape[0]):
        for j in range(local_max.shape[1]):
            if local_max[i][j] == 1:
                label_object(local_max, i, j, label)
                label += 1

    local_max = cv2.erode(local_max, square(4))

    return local_max


def label_object(image, x, y, label):
    image[x][y] = label
    if image[x+1][y] == 1:
        label_object(image, x+1, y, label)
    if image[x][y+1] == 1:
        label_object(image, x, y+1, label)

