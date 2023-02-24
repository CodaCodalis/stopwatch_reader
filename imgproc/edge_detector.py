import cv2
import numpy as np
from PIL import Image


def detect(image, number):
    # Read the original image
    img = cv2.imread(image)
    # img = np.array(Image.open(image))

    # Display original image
    # if number <= 1:
    #    cv2.imshow('Original', img)
    #    cv2.waitKey(0)

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image for better edge detection
    # Documentation: https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html
    # img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # img_blur = cv2.blur(img, (5, 5))

    # Display blurred image
    # if number <= 1:
    #    cv2.imshow('Blurred', img_blur)
    #    cv2.waitKey(0)

    # Sobel Edge Detection
    # Documentation: https://docs.opencv.org/3.4/d5/d0f/tutorial_py_gradients.html
    # laplacian = cv2.Laplacian(img, cv2.CV_8U)
    # sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    # sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    # sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection

    # Display Sobel Edge Detection Images
    # cv2.imshow('Sobel X', sobelx)
    # cv2.waitKey(0)
    # cv2.imshow('Sobel Y', sobely)
    # cv2.waitKey(0)
    # cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    # cv2.waitKey(0)

    # Canny Edge Detection
    # Documentation: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
    # Lower Threshold
    t_lower = 90
    # Upper threshold
    t_upper = 120
    # Aperture size
    aperture_size = 3
    # L2Gradient Boolean
    l2_gradient = False
    edges = cv2.Canny(image=img, threshold1=t_lower, threshold2=t_upper, apertureSize=aperture_size, L2gradient=l2_gradient)  # Canny Edge Detection
    # edges = cv2.Canny(image=img_blur, threshold1=t_lower, threshold2=t_upper, apertureSize=aperture_size, L2gradient=l2_gradient)  # Canny Edge Detection
    # edges = cv2.Canny(image=laplacian, threshold1=t_lower, threshold2=t_upper, apertureSize=aperture_size, L2gradient=l2_gradient)  # Canny Edge Detection
    # edges = cv2.Canny(image=sobelxy, threshold1=t_lower, threshold2=t_upper, apertureSize=aperture_size, L2gradient=l2_gradient)  # Canny Edge Detection

    # Display Canny Edge Detection Image
    # cv2.imshow('Canny Edge Detection', edges)
    # cv2.waitKey(0)

    cv2.imwrite("resources/temp/temp" + str(number) + ".jpg", edges)
    # exit(0)
    # cv2.destroyAllWindows()
