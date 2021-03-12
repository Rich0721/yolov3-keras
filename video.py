from yolo_predict import YOLO
from PIL import Image
import cv2
import tensorflow as tf
import numpy as np
from time import time


det =  YOLO(weight_path="./weights/yolov3-148.h5")


def use_video():
    
    cap = cv2.VideoCapture("Video_07.mp4")
    _ = True
    while True:
        
        _, frame = cap.read()
        
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))
        frame = det.detect_image(frame)

        frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        
        cv2.imshow("video", frame)
        cv2.waitKey(1)
    

def use_image(image_path):

    image = cv2.imread(image_path)
    predict_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predict_image = Image.fromarray(np.uint8(predict_image))
    results = det.detected_bbox(predict_image)

    for r in results:
        center_x = (r[0] + r[2]) // 2
        center_y = (r[1] + r[3]) // 2
        image = cv2.circle(image, (center_x, center_y), 10, (255, 255, 0), -1)
    cv2.imwrite(image_path[:-4] + "_result.jpg", image)

if __name__ == "__main__":
    use_video()
    #use_image("./05685.jpg")