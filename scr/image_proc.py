import cv2
from skimage.exposure import adjust_gamma
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import math
import reference
from scipy.spatial import distance


def calc(image, real_num, real_value):
    num = 0
    value = 0
    color_test = np.copy(image)
    monets_field = 0
    center_off_mass = []
    feret = []
    blair_bliss = []
    haralick = []

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
        if label == 0: continue

        # blair_bliss.append(compute_bb())
        # feret.append(compute_f())

        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
        ((x, y), r) = cv2.minEnclosingCircle(c)

        if math.sqrt(math.pow(int(x) - int(a), 2) + math.pow(int(y) - int(b), 2)) < int(r): continue
        center_off_mass.append((int(y), int(x)))
        haralick.append(compute_h(r, len(c)))
        (a, b, radius) = (x, y, r)

        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        pole = int(math.pi*r*r)
        monets_field += pole
        num = label
        x_min = int(x - 15)
        x_max = int(x + 15)
        y_min = int(y - 15)
        y_max = int(y + 15)

        value += reference.check_value(pole, color_test[y_min:y_max, x_min:x_max])

    print("Środki ciężkości: " + str(center_off_mass))
    print("Współczynniki Fereta: " + str(feret))
    print("Współczynniki Blaira-Blissa: " + str(blair_bliss))
    print("Współczynniki Haralicka: [ ", end="")
    for i in haralick: print("{:.2}, ".format(i), end="")
    print("]")
    print("Dokładność algorytmu: {:.2%}".format(check_alg(real_value, value/100, real_num, num)))
    print("Pole zajmowane przez obiekty: {:.2%}".format((monets_field*100)/(image.shape[0]*image.shape[1])/100))
    print()
    if num == real_num: print("Wykryto wszystkie elementy!")
    else: print("Pominieto " + str(real_num - num) + " elementy!")
    if value/100 == real_value: print("Pomyślnie odczytano wartość monet!")
    else: print("Wykryto złą wartość! pomyłka to: " + str(real_value - value/100))
    print("Kwota na zdjęciu: " + str(value/100) + "zł")
    print("### -------------------------------------------------- ###")
    print()

    # cv2.imshow("Output", image)
    # cv2.waitKey(0)


def check_alg(real_value, value, real_num, num):
    check = float((value / real_value) + (num / real_num))
    if check > 2: check = 2 - (math.fabs(2 - check))
    return (check * 100)/200


def compute_bb(points):
    s = len(points)
    my, mx = cog2(pts)
    r = 0
    for point in points:
        r = r + distance.euclidean(point, (my, mx))**2
    return s/(math.sqrt(2*math.pi*r))


def compute_f():
    pass


def compute_h(dis, length_):
    sum1 = dis
    sum2 = math.pow(dis, 2)
    return math.sqrt((math.pow(sum1, 2)) / (length_ * sum2 - 1))
