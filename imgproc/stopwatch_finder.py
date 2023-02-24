import cv2
import numpy as np
import os
import glob


def find():
    # Load binary image as mask
    mask = cv2.imread('resources/mask_stopwatch.png', cv2.IMREAD_GRAYSCALE)

    # Find contours of mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]

    # Find bounding rectangle of contour
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate center and scale
    center = (x + w / 2, y + h / 2)
    scale = max(w, h) / 2

    # Define rotation matrix
    angle = 45
    R = cv2.getRotationMatrix2D(center, angle, scale)

    # Define affine transformation matrix
    M = np.vstack([np.hstack([R, np.zeros((2, 1))]), np.array([0, 0, 1])])

    # Create directory for transformed images
    if not os.path.exists('resources/transformed'):
        os.mkdir('resources/transformed')

    # Loop through all images in directory
    for image in sorted(glob.iglob('resources/images/*')):
        # Load image
        img = cv2.imread(image)

        # Warp image using affine transformation
        warped = cv2.warpAffine(img, M[:2, :], (img.shape[1], img.shape[0]))

        # Crop image to size of mask
        cropped = cv2.getRectSubPix(warped, (w, h), center)

        # Get the filename from the full path
        filename = os.path.basename(image)

        # Save transformed image
        cv2.imwrite(os.path.join('resources/transformed', filename), cropped)