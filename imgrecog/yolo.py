import cv2
import numpy as np


def detect(img):
    # Load the YOLOv4 network
    net = cv2.dnn.readNet("resources/yolov4.weights", "resources/yolov4.cfg")

    # Get the names of the output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Convert image
    img = cv2.convertScaleAbs(img)

    # Resize the image to a fixed size
    # img = cv2.resize(img, (416, 416))
    # img = cv2.resize(img, None, fx=0.2, fy=0.2)

    # Normalize the image pixel values to the range [0, 1]
    # img = img / 255.0
    # cv2.imshow("Normalized", img)

    # Convert the image from BGR to RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow("BGR2RGB", img)

    # Run the YOLOv4 network on the image
    blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Get the dimensions of the input image
    height, width, channels = img.shape

    # Process the YOLOv4 output
    conf_threshold = 0.5
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                # Convert the relative coordinates to absolute coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression to remove overlapping bounding boxes
    nms_threshold = 0.4
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    classes = []
    with open("resources/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Draw the bounding boxes and class labels on the image
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(class_ids), 3))
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            conf = str(confidences[i])
            label = str(classes[class_ids[i]]) + " " + conf[0:4]
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 2, color, 2)

    # Show the resulting image
    cv2.imshow("YOLOv4 output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
