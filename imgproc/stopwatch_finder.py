import cv2
import numpy as np
import os
import glob


def find_stopwatch(mask, img, number):
    # Convert the mask to grayscale
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the grayscale mask
    _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

    # Find the contour of the triangle in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stopwatch_contour = max(contours, key=cv2.contourArea)

    # Extract the bounding rectangle of the triangle
    x, y, w, h = cv2.boundingRect(stopwatch_contour)

    # Crop the binary mask to the dimensions of the bounding rectangle
    cropped_mask = binary_mask[y:y + h, x:x + w].astype(np.uint8)

    # Convert the cropped mask to grayscale
    cropped_mask = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR)

    # Perform template matching to locate the triangle in the original image
    result = cv2.matchTemplate(img, cropped_mask, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Extract the coordinates of the top-left corner of the template
    top_left = max_loc

    # Create a rectangle using the template dimensions
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

    # Crop the original image to the dimensions of the template
    cropped_img = img[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]

    # Save the cropped image
    cv2.imwrite('resources/cropped_mask_display_color/stopwatch_cropped_' + str(number) + '.jpg', cropped_img)

    # Display the original image, binary mask, and cropped image (for troubleshooting purposes)
    # cv2.imshow('Original image', img)
    # cv2.imshow('Binary mask', binary_mask)

    cv2.namedWindow('Cropped image', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Cropped image', cropped_img)
    cv2.resizeWindow('Cropped image', 400, 400)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_display(mask, img, number):
    # Convert the mask to grayscale
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the grayscale mask
    _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

    # Find the contour of the triangle in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stopwatch_contour = max(contours, key=cv2.contourArea)

    # Extract the bounding rectangle of the triangle
    x, y, w, h = cv2.boundingRect(stopwatch_contour)

    # Crop the binary mask to the dimensions of the bounding rectangle
    cropped_mask = binary_mask[y:y + h, x:x + w].astype(np.uint8)

    # Convert the cropped mask to grayscale
    cropped_mask = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR)

    # Perform template matching to locate the triangle in the original image
    result = cv2.matchTemplate(img, cropped_mask, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Extract the coordinates of the top-left corner of the template
    top_left = max_loc

    # Create a rectangle using the template dimensions
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

    # Crop the original image to the dimensions of the template
    cropped_img = img[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]

    # Save the cropped image
    cv2.imwrite('resources/cropped_display_near/display_cropped_' + str(number) + '.jpg', cropped_img)

    # Display the original image, binary mask, and cropped image (for troubleshooting purposes)
    # cv2.imshow('Original image', img)
    # cv2.imshow('Binary mask', binary_mask)

    # cv2.namedWindow('Cropped image', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('Cropped image', cropped_img)
    # cv2.resizeWindow('cropped', 400, 400)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def find2(mask, img, number):
    # Convert the mask to grayscale
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the grayscale mask
    _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

    # Find the contour of the stopwatch in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stopwatch_contour = max(contours, key=cv2.contourArea)

    # Extract the bounding rectangle of the stopwatch contour
    x, y, w, h = cv2.boundingRect(stopwatch_contour)

    # Crop the binary mask to the dimensions of the bounding rectangle
    cropped_mask = binary_mask[y:y + h, x:x + w].astype(np.uint8)

    # Convert the cropped mask to grayscale
    cropped_mask = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR)

    # Perform template matching to locate the stopwatch in the original image
    result = cv2.matchTemplate(img, cropped_mask, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Extract the coordinates of the top-left corner of the template
    top_left = max_loc

    # Create a rectangle using the template dimensions
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

    # Crop the original image to the dimensions of the template
    cropped_img = img[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]

    # Detect lines in the cropped image using the Hough Transform
    edges = cv2.Canny(cropped_img, 90, 120, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    # Draw the detected lines on the cropped image
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(cropped_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Calculate the angle of rotation based on the orientation of the detected lines
    angle = 0
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            if np.degrees(theta) > 45 and np.degrees(theta) < 135:
                angle = np.degrees(theta) - 90

    # Rotate the cropped image by the calculated angle
    rows, cols, _ = cropped_img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_img = cv2.warpAffine(cropped_img, M, (cols, rows))

    # Save the cropped and rotated image
    cv2.imwrite('resources/cropped/stopwatch_cropped_' + str(number) + '.jpg', rotated_img)

    # Display the cropped and rotated image (for troubleshooting purposes)
    cv2.namedWindow('Cropped image', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Cropped image', rotated_img)
    cv2.resizeWindow('Cropped image', 400, 400)

    cv2.namedWindow('Edges', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Edges', edges)
    cv2.resizeWindow('Edges', 400, 400)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
