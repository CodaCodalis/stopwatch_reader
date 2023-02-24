import cv2 as cv
import numpy as np

def select():
    # Load Image
    img = cv.imread('resources/shapes/shapes.png')

    # Selecting ROI
    imgdraw = cv.selectROI(img)

    cropimg = img[int(imgdraw[1]):int(imgdraw[1] + imgdraw[3]),
              int(imgdraw[0]):int(imgdraw[0] + imgdraw[2])]  # displaying the cropped image as the output on the screen
    cv.imshow('Cropped_image', cropimg)

    blank = np.zeros(cropimg.shape[:2], dtype='uint8')  # creates a blank img, with the same size as our geometricShapes img
    gray = cv.cvtColor(cropimg, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)

    # Find edges using contours method
    ret, thresh = cv.threshold(blur, 125, 255, cv.THRESH_BINARY)
    # cv.imshow('Thresh', thresh)

    contours, hierachies = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(blank, contours, -1, (255, 255, 255), thickness=1)
    cv.imshow('Contours', blank)
    cv.waitKey(0)
