import cv2
import json
import numpy as np


def get_outline():
    shape = cv2.imread('resources/mask2.png')
    edges = cv2.Canny(shape, 100, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = contours[0].tolist()
    with open('resources/outline.json', 'w') as f:
        json.dump(outline, f)
    return outline


def find_outline(outline):
    img = cv2.imread('resources/temp/temp3.jpg')

    # Perform image segmentation to identify potential object regions
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Perform template matching using the object's outline as the template
    template = np.array(outline)
    for cnt in contours:
        result = cv2.matchTemplate(cnt, template, cv2.TM_CCOEFF_NORMED)
        if result.max() > 0.8:
            match_coords = cnt.tolist()
            print("Match found at:", match_coords)
