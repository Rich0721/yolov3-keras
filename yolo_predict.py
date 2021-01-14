import colorsys
import copy
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from PIL import Image, ImageDraw, ImageFont
from config import config

from networks.yolo import yolo_body, yolo_eval


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


class YOLO:

    def __init__(self, weight_path=None) -> None:
        self.input_shape = config.IMAGE_SIZE
        self.classes_name = config.CLASSES
        self.anchors = np.reshape(config.ANCHORS, (-1, 2))
        self.load_model(weight_path)
    
    def load_model(self, weight_path=None):

        self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors=len(self.anchors)//3, num_classes=len(self.classes_name))
        self.yolo_model.load_weights(weight_path)

        # Set every class' color
        hsv_tuples = [(x / len(self.classes_name), 1., 1.) for x in range(len(self.classes_name))]
        self.colors = list(map(lambda x:colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x:(int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
    
    @tf.function
    def get_pred(self, image):
        preds = self.yolo_model(image, training=False)
        return preds
    
    def detect_image(self, image):

        new_image = letterbox_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.array(new_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, axis=0)

        preds = self.get_pred(image_data)
        boxes, scores, classes = yolo_eval(preds, self.anchors, len(self.classes_name), image_shape=(image.size[1], image.size[0]),
                                score_threshold=config.SCORE_THRESHOLD, iou_threshold=config.IOU_THRESHOLD)
        
        font = ImageFont.truetype(font='simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0]

        for i, c in list(enumerate(classes)):

            predict_class = self.classes_name[c]
            box = boxes[i]
            score = scores[i]

            ymin, xmin, ymax, xmax = box
            ymin -= 5
            xmin -= 5
            xmax += 5
            ymax += 5

            xmin = max(0, np.floor(xmin + 0.5).astype('int32'))
            ymin = max(0, np.floor(ymin + 0.5).astype('int32'))
            xmax = min(image.size[0], np.floor(xmax + 0.5).astype('int32'))
            ymax = min(image.size[1], np.floor(ymax + 0.5).astype('int32'))

            label = "{}-{:.2f}".format(predict_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            if ymin - label_size[1] >= 0:
                text_origin = np.array((xmin, ymin - label_size[1]))
            else:
                text_origin = np.array((xmin, ymin + 1))
            
            for i in range(thickness):
                draw.rectangle(
                    [xmin + i, ymin+i, xmax-i, ymax-i],
                    outline=self.colors[int(c)]
                )
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)]
            )
            draw.text(text_origin, str(label, "UTF-8"), fill=(0, 0, 0), font=font)
            del draw
        return image