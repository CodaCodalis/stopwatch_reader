import cv2

def take_pictures():
    cam = cv2.VideoCapture(0)
    s, img = cam.read()
    for i in range(10):
        cv2.imwrite('/mnt/storage/development/stopwatch_reader/resources/realtime/ocr' + str(i) + '.jpg', img)
    print("pictures taken")
    # return img