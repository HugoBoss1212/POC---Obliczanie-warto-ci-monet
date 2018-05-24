from skimage import io
from skimage.color import rgb2gray
import image_proc
import glob


def main_():
    for file in glob.glob("images/*.jpg"):
        image = io.imread(file)
        image = rgb2gray(image)
        image_proc.calc(image)


if __name__ == '__main__':
    main_()
