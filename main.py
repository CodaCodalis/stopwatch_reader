import glob
from PIL import Image
import pytesseract
import numpy as np
import cv2


def read_stopwatch(filename, choice):
    img = np.array(Image.open(filename))
    img = pre_process_image(img)

    custom_config = r'--psm 7 --oem 3 --tessdata-dir "tessdata" -c tessedit_char_whitelist=".:0123456789 "'

    if choice == "1":
        text = pytesseract.image_to_string(img, lang='lets', config=custom_config)
    elif choice == "2":
        text = pytesseract.image_to_string(img, lang='osd', config=custom_config)

    print("Result (%s): %s" % (filename, text))


def pre_process_image(img):
    # norm_img = np.zeros((img.shape[0], img.shape[1]))

    # normalize
    # img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)

    # grey-scale
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

    # opaque magical parameters
    # kernel = np.ones((2, 2), np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)
    # img = cv2.dilate(img, kernel, iterations=1)

    # cv2.imwrite("resources/temp/temp.jpg", img)
    # exit()

    return img


if __name__ == '__main__':
    directory = 'resources/digital_clock_fonts/*'
    choice = input("Choose font to recognize:\n [1] digital clock font\n [2] other font\n")

    for filename in sorted(glob.iglob(directory)):
        read_stopwatch(filename, choice)
