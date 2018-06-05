import cv2
from skimage.exposure import adjust_gamma
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import math
import reference


def calc(image, real_num):
    num = 0
    value = 0
    color_test = np.copy(image)

    img = cv2.pyrMeanShiftFiltering(image, 21, 53)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = adjust_gamma(gray, 10)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    d = ndimage.distance_transform_edt(thresh)
    local_max = peak_local_max(d, indices=False, min_distance=15, labels=thresh)
    markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
    labels = watershed(-d, markers, mask=thresh)
    (a, b, radius) = (0, 0, 0)
    for label in np.unique(labels):
        if label == 0:
            continue

        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
        ((x, y), r) = cv2.minEnclosingCircle(c)

        if math.sqrt(math.pow(int(x) - int(a), 2) + math.pow(int(y) - int(b), 2)) < int(r):
            continue
        (a, b, radius) = (x, y, r)

        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        pole = int(math.pi*r*r)
        num = label
        x_min = int(x - 15)
        x_max = int(x + 15)
        y_min = int(y - 15)
        y_max = int(y + 15)

        value += reference.check_value(pole, color_test[y_min:y_max, x_min:x_max])

    if num == real_num: print("Wykryto wszystkie elementy!")
    else: print("pominieto " + str(real_num - num) + " elementy!")
    print("Kwota na zdjęciu: " + str(value/100) + "zł")

    cv2.imshow("Output", image)
    cv2.waitKey(0)
