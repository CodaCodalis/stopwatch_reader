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


def recognize_rotated(mask, img, number):
    # Convert the mask and the input image to grayscale
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a SIFT object
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors in the mask and the input image
    kp_mask, des_mask = sift.detectAndCompute(gray_mask, None)
    kp_img, des_img = sift.detectAndCompute(gray_img, None)

    # Create a BFMatcher object
    bf = cv2.BFMatcher()

    # Match the descriptors in the mask and the input image
    matches = bf.knnMatch(des_mask, des_img, k=2)

    # Apply ratio test to filter out bad matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # If we have at least 4 good matches, extract the corresponding keypoints and compute the homography
    if len(good_matches) >= 4:
        src_pts = np.float32([kp_mask[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        M = M.astype(np.float32)

        # Use the homography to warp the mask to the input image
        h, w = mask.shape[:2]
        warped_mask = cv2.warpPerspective(mask, M, (w, h))

        # Extract the bounding rectangle of the warped mask
        _, binary_mask = cv2.threshold(warped_mask, 0.5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        triangle_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(triangle_contour)

        # Crop the input image to the dimensions of the bounding rectangle
        cropped_img = img[y:y + h, x:x + w]

        # Save the cropped image
        cv2.imwrite('resources/cropped/triangle_cropped_' + str(number) + '.jpg', cropped_img)

        # Display the cropped image
        cv2.imshow('Cropped image', cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print('Not enough matches to compute homography')
