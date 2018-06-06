import image_proc
import glob
import cv2


def main_():
    for file in glob.glob("images/*.jpg"):
        count = int(file[len(file)-6:len(file)-4])
        value = float(file[7:12])
        print(file)
        image = cv2.imread(file)
        image_proc.calc(image, count, value)


if __name__ == '__main__':
    main_()
