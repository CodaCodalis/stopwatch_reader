import cv2
import numpy as np
import os
import glob


def process():
    # Load the png image of the shape and the jpg images to be analyzed
    shape_img = cv2.imread('resources/mask2.png', cv2.IMREAD_UNCHANGED)
    input_dir = 'resources/images'
    jpg_imgs = glob.glob(os.path.join(input_dir, '*.jpg'))

    # Convert the png image to grayscale and create a binary mask of the shape
    shape_gray = cv2.cvtColor(shape_img, cv2.COLOR_BGR2GRAY)
    ret, shape_mask = cv2.threshold(shape_gray, 127, 255, cv2.THRESH_BINARY)

    # Find the contour of the shape in the binary mask and calculate its minimum bounding rectangle
    contours, hierarchy = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect = None
    if len(contours) > 0:
        shape_contour = contours[0]
        rect = cv2.minAreaRect(shape_contour)
        rect_points = cv2.boxPoints(rect).astype(int)
    else:
        # Handle the case when no contours are found (e.g., if the binary mask is empty)
        print("No contours found in the shape mask.")
        # Add any additional error handling code here, such as returning from the function or raising an exception.

    # Loop over each of the jpg images
    for jpg_path in jpg_imgs:
        # Load the jpg image and convert it to grayscale
        jpg_img = cv2.imread(jpg_path)

        jpg_gray = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2GRAY)

        # Create a binary mask of the jpg image
        ret, jpg_mask = cv2.threshold(jpg_gray, 127, 255, cv2.THRESH_BINARY)

        # Use OpenCV's matchTemplate function to find the location(s) of the shape in the image
        match = cv2.matchTemplate(jpg_gray, shape_gray, cv2.TM_CCOEFF_NORMED)
        locations = np.where(match >= 0.50)  # adjust threshold as needed

        # Loop over each location found
        for loc in zip(*locations[::-1]):
            # Calculate the rotation and tilt angles necessary to match the orientation of the shape in the png image
            # Note: you can use the "rect" variables from earlier to get the original orientation of the shape
            angle = rect[2]  # initial angle of the shape in the png image
            x, y = rect[0]  # center point of the shape in the png image
            tx, ty = loc  # location of the shape in the jpg image
            dx, dy = tx - x, ty - y  # vector from center of png shape to center of jpg shape
            r = np.sqrt(dx * dx + dy * dy)
            theta = np.arctan2(dy, dx) * 180 / np.pi
            rotation_angle = angle - theta
            tilt_angle = np.arctan2(r / 2, rect[1][1]) * 180 / np.pi

            # Apply these rotation and tilt angles to the image using OpenCV's warpAffine function
            M = cv2.getRotationMatrix2D((float(tx), float(ty)), rotation_angle, 1.0)
            rotated_img = cv2.warpAffine(jpg_img, M, (jpg_img.shape[1], jpg_img.shape[0]))

            M = cv2.getRotationMatrix2D((float(tx), float(ty)), tilt_angle, 1.0)
            tilted_img = cv2.warpAffine(rotated_img, M, (rotated_img.shape[1], rotated_img.shape[0]))

            # Crop the resulting image using the coordinates of the minimum bounding rectangle of the shape
            crop_rect = cv2.boundingRect(shape_contour)
            cropped_img = tilted_img[crop_rect[1]:crop_rect[1] + crop_rect[3], crop_rect[0]:crop_rect[0] + crop_rect[2]]

            # Save the resulting cropped and oriented image
            save_dir = 'resources/temp'  # directory to save the output images
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            filename = os.path.splitext(os.path.basename(jpg_path))[0] + '_cropped.jpg'
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, cropped_img)
