import math
import numpy as np
import cv2
from scipy.stats import itemfreq

coin_1 = [552, 530]
coin_2 = [770, 747] ## myli z 20
coin_5 = [938, 889]
coin_10 = [639, 634]
coin_20 = [799, 719] ## myli z 2
coin_50 = [1414, 1327]
coin_100 = [2136, 1965] ## myli z 500
coin_200 = [1655, 1645]
coin_500 = [2216, 2085] ## myli z 100

pola = []
pola.append((coin_1[0] + coin_1[1]) / 2)
pola.append((coin_2[0] + coin_2[1]) / 2)
pola.append((coin_5[0] + coin_5[1]) / 2)
pola.append((coin_10[0] + coin_10[1]) / 2)
pola.append((coin_20[0] + coin_20[1]) / 2)
pola.append((coin_50[0] + coin_50[1]) / 2)
pola.append((coin_100[0] + coin_100[1]) / 2)
pola.append((coin_200[0] + coin_200[1]) / 2)
pola.append((coin_500[0] + coin_500[1]) / 2)


def check_value(pole, image):
    test = []
    for i in pola:
        test.append(int(math.fabs(pole - i)))
    mn, idx = min((test[i], i) for i in range(len(test)))
    if idx == 1 or idx == 4:
        if dominant_color(image) < 100: return 2
        else: return 20
    if idx == 6 or idx == 8:
        if dominant_color(image) < 100: return 500
        else: return 100

    if idx == 0: return 1
    if idx == 2: return 5
    if idx == 3: return 10
    if idx == 5: return 50
    if idx == 7: return 200


def dominant_color(image):
    arr = np.float32(image)
    pixels = arr.reshape((-1, 3))
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    dominant_color_ = palette[np.argmax(itemfreq(labels)[:, -1])]
    return dominant_color_[0]
