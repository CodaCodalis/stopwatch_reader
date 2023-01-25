import pytesseract
import numpy as np

def analyze(image):


    # define custom config
    custom_config = r'--psm 7 --oem 3 --tessdata-dir "tessdata" -c tessedit_char_whitelist=".:0123456789 "'

    text = pytesseract.image_to_string(image, lang='7seg', config=custom_config)

    # use if other than 7seg font
    # text = pytesseract.image_to_string(img, lang='osd', config=custom_config)

    print("Erkannter Wert: %s" % text)