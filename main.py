import glob
from imgread import webcam
from imgrecog import ocr
from imgproc import edge_detection

def print_main_menu():
    print("STOPPUHRLESER v0.1")
    print("[1] Echtzeit-Erkennung mit der Webcam")
    print("[2] Erkennung von Bildern aus Projektordner")
    print("[3] Beenden")

def menu(choice):
    if choice == "1":
        print("Das Display der Stoppuhr vor die Webcam halten!")
        input("Beliebige Taste drücken, um fortzufahren...")
        webcam.take_pictures()
        for image in sorted(glob.iglob('resources/realtime/*')):
            ocr.analyze(edge_detection.process(image))
    elif choice == "2":
        for image in sorted(glob.iglob('resources/images/*')):
            # detect edges
            # edge_detection.detect(image)

            # form mask and crop everything but display section in original, translate and rotate image for a while
            # edge_detection.crop_and_so_on(image)

            # process original cropped image for ocr
            ocr.analyze(edge_detection.process(image))

    elif choice == "3":
        exit(0)
    else:
        print("Unerwartete Eingabe...")

if __name__ == '__main__':
    while True:
        print_main_menu()
        menu(input("Ihre Auswahl: "))
