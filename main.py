import glob
from imgread import webcam
from imgrecog import ocr
from imgproc import (edge_detector,
                     image_ocr_preparer,
                     contours_finder,
                     roi_selector,
                     text_deskewer,
                     image_processor,
                     display_finder,
                     hough_lines_finder,
                     stopwatch_finder)


def print_main_menu():
    print("STOPPUHRLESER v0.1")
    print("[1] Echtzeit-Erkennung mit der Webcam")
    print("[2] Erkennung von Bildern aus Projektordner")
    print("[3] Formerkennung (Test)")
    print("[4] Beenden")


def menu(choice):
    number = 0
    if choice == "1":
        print("Das Display der Stoppuhr vor die Webcam halten!")
        input("Beliebige Taste drücken, um fortzufahren...")
        webcam.take_pictures()
        for image in sorted(glob.iglob('resources/realtime/*')):
            ocr.analyze(image_ocr_preparer.prepare(image))
    elif choice == "2":
        for image in sorted(glob.iglob('resources/images/*')):
            # detect edges
            number = number + 1
            edge_detector.detect(image, number)

            # form mask and crop everything but display section in original, translate and rotate image for a while
            # edge_detection.crop_and_so_on(image)

            # process original cropped image for ocr
            # ocr.analyze(image_ocr_processor.prepare(image))

    elif choice == "3":
        stopwatch_finder.find()
        # for image in sorted(glob.iglob('resources/images/*')):
        #    hough_lines_finder.preprocess(image)
        #    hough_lines_finder.find(image)

        # outline = display_finder.get_outline()
        # display_finder.find_outline(outline)

        # image_processor.process()
        # roi_selector.select()
        # contours_finder.find_contours()
        # number = 0
        # for image in sorted(glob.iglob('resources/images/*')):
        #    number = number + 1
        #    image_ocr_preparer.prepare(image, number)

        # for image in sorted(glob.iglob('resources/temp/*')):
        #    number = number + 1
        #    text_deskewer.deskew(image, number)

    elif choice == "4":
        exit(0)
    else:
        print("Unerwartete Eingabe...")


if __name__ == '__main__':
    while True:
        print_main_menu()
        menu(input("Ihre Auswahl: "))
