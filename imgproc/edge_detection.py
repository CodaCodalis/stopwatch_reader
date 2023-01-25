import cv2
import numpy as np
from PIL import Image
def process(image):
    img = np.array(Image.open(image))
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

    # for debugging
    # cv2.imwrite("resources/temp/temp.jpg", img)
    # exit()

    return img

def detect(image):
    # Read the original image
    # img = cv2.imread('test.jpg')
    # Display original image
    img = np.array(Image.open(image))
    cv2.imshow('Original', img)
    cv2.waitKey(0)
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    cv2.imshow('Sobel X', sobelx)
    cv2.waitKey(0)
    cv2.imshow('Sobel Y', sobely)
    cv2.waitKey(0)
    cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    cv2.waitKey(0)
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection
    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)
    # for debugging
    cv2.imwrite("resources/temp/temp.jpg", edges)
    exit(0)
    cv2.destroyAllWindows()