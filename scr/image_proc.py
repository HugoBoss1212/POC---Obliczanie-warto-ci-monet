import cv2
from skimage.exposure import adjust_gamma
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import math


def calc(image, real_num):

    img = cv2.pyrMeanShiftFiltering(image, 21, 53)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = adjust_gamma(gray, 10)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # kernel = np.ones((2, 2), np.uint8)
    # opening_ = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    #
    # sure_bg = cv2.dilate(opening_, kernel, iterations=10)
    #
    # dist_transform = cv2.distanceTransform(opening_, cv2.DIST_L2, 5)
    # ret, sure_fg = cv2.threshold(dist_transform, 0.35*dist_transform.max(), 255, 0)
    #
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg, sure_fg)
    #
    # ret, markers = cv2.connectedComponents(sure_fg)
    # markers = markers+1
    # markers[unknown == 255] = 0
    #
    # markers = cv2.watershed(img, markers)
    # img[markers == -1] = [255, 0, 0]

    d = ndimage.distance_transform_edt(thresh)
    localmax_ = peak_local_max(d, indices=False, min_distance=15,
                               labels=thresh)

    markers = ndimage.label(localmax_, structure=np.ones((3, 3)))[0]
    labels = watershed(-d, markers, mask=thresh)

    for label in np.unique(labels):
        if label == 0:
            continue

        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print("pole = " + str(int(math.pi*r*r)) + "\t nr: " + str(label))

    cv2.imshow("Output", image)
    cv2.waitKey(0)
