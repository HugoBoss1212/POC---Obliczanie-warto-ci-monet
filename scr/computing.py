import math
from scipy.spatial import distance


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
