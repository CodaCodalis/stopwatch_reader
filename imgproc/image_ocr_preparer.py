import cv2
import numpy as np
from PIL import Image


def prepare(image, number):
    img = np.array(Image.open(image))
    # img = cv2.imread(image)
    # convert to grey-scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # some gaussian blur
    img = cv2.GaussianBlur(img, (1, 1), 0)
    # threshold filter
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    # invert if more black than white pixels
    number_of_white_pix = np.sum(img == 255)
    number_of_black_pix = np.sum(img == 0)
    if number_of_black_pix > number_of_white_pix:
        img = cv2.bitwise_not(img)

    cv2.imwrite("resources/temp/temp" + str(number) + ".jpg", img)

    return img
