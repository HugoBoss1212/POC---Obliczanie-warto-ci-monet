from skimage import io
from skimage.color import rgb2gray
import image_proc
import glob


def main_():
    for file in glob.glob("images/*.jpg"):
        count = int(file[len(file)-6:len(file)-4])
        image = io.imread(file)
        image = rgb2gray(image)
        image_proc.calc(image, count)


if __name__ == '__main__':
    main_()
