import cv2
import numpy as np


def preprocess(image):
    # Load the image
    img = cv2.imread(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Perform morphological closing to fill in gaps in the contour
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # Find contours in the image
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest aspect ratio within a certain range
    aspect_ratio_min = 1.0
    aspect_ratio_max = 2.5
    largest_contour = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if aspect_ratio_min < aspect_ratio < aspect_ratio_max:
            if largest_contour is None or cv2.contourArea(contour) > cv2.contourArea(largest_contour):
                largest_contour = contour

    # Crop the image to the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = img[y:y + h, x:x + w]

    # return cropped

    # Display the cropped image
    cv2.namedWindow('cropped', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('cropped', cropped)
    cv2.resizeWindow('cropped', 400, 400)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find(image):
    # Load the image
    img = cv2.imread(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('gray', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('gray', gray)
    cv2.resizeWindow('gray', 400, 400)
    cv2.waitKey(0)

    # Apply a threshold to the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply Hough transform to find the lines in the image
    lines = cv2.HoughLines(thresh, 1, np.pi / 180, 100)

    # Create a list of the angles of the lines
    angles = []
    for line in lines:
        for rho, theta in line:
            angle = theta * 180 / np.pi
            angles.append(angle)

    # Compute the median angle of the lines
    median_angle = np.median(angles)

    # Rotate the image to align with the median angle
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if median_angle < -45 else cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Convert the rotated image to grayscale
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the rotated image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply Hough transform to find the lines in the rotated image
    lines = cv2.HoughLines(thresh, 1, np.pi / 180, 100)

    # Create a list of the angles of the lines
    angles = []
    for line in lines:
        for rho, theta in line:
            angle = theta * 180 / np.pi
            angles.append(angle)

    # Compute the median angle of the lines
    median_angle = np.median(angles)

    # Rotate the image again to align with the median angle
    rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE if median_angle < -45 else cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Convert the rotated image to grayscale
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the rotated image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Define the desired aspect ratio
    desired_aspect_ratio = 2.5

    # Find the contours in the rotated image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter the contours based on their aspect ratio
    closest_contour = None
    closest_aspect_ratio = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if closest_contour is None or abs(aspect_ratio - desired_aspect_ratio) < abs(
                closest_aspect_ratio - desired_aspect_ratio):
            closest_contour = contour
            closest_aspect_ratio = aspect_ratio

    # Find the largest contour in the rotated image, which should be the contour of the display
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the rotated image to the display region
    cropped = rotated[y:y + h, x:x + w]

    # Display the cropped image
    cv2.namedWindow('cropped', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('cropped', cropped)
    cv2.resizeWindow('cropped', 400, 400)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
