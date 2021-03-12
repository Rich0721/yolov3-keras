from PIL import Image
import cv2
import numpy as np

from yolo_predict import YOLO

yolo = YOLO("./yolov3/yolov3-148.h5")


predicts = []
with open("test.txt") as f:
    line = f.readline()
    while line:
        l = line.split()
        boxes = l[1].split(",")
        predicts.append([l[0], boxes[0], boxes[1], boxes[2], boxes[3]])
        line = f.readline()

#img = "./00010.jpg"
for index, pre in enumerate(predicts):

    img = pre[0]


    image = Image.open(img)
    image = yolo.detect_image(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #image = cv2.rectangle(image, (int(pre[1]), int(pre[2])), (int(pre[3]), int(pre[4])), (0, 0, 0), 3)
    #cv2.imshow("TEST", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
