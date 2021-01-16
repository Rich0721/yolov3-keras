from __future__ import annotations
import colorsys
from utils.utils import compute_ap, compute_overlap
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from PIL import Image, ImageDraw, ImageFont
from config import config
from tqdm import tqdm
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

        self.solve_cudnn_error()
        self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors=len(self.anchors)//3, num_classes=len(self.classes_name))
        self.yolo_model.load_weights(weight_path)

        # Set every class' color
        hsv_tuples = [(x / len(self.classes_name), 1., 1.) for x in range(len(self.classes_name))]
        self.colors = list(map(lambda x:colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x:(int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
    
    def solve_cudnn_error(self):
        gpus = tf.config.experimental.list_physical_devices("GPU")

        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpu = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpu), "Logical GPUs")
            except RuntimeError as e:
                print(e)
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
        image = np.array(image)
        return image

    def evaluate(self, generator, num_test, iou_threshold=0.5, score_threshold=0.5):

        
        all_detections = [[None for i in range(len(self.classes_name))] for j in range(num_test)]
        all_annotations = [[None for i in range(len(self.classes_name))] for j in range(num_test)]

        for i in tqdm(range(num_test)):

            image, annotations = generator.load_image(i)
            new_image = letterbox_image(image, (self.input_shape[1], self.input_shape[0]))
            image_data = np.array(new_image, dtype='float32')
            image_data /= 255.
            image_data = np.expand_dims(image_data, axis=0)

            preds = self.get_pred(image_data)
            boxes, scores, classes = yolo_eval(preds, self.anchors, len(self.classes_name), image_shape=(image.size[1], image.size[0]),
                                    score_threshold=score_threshold, iou_threshold=iou_threshold)
            
            pred_boxes = []
            for index, c in list(enumerate(classes)):

                
                box = boxes[index]
                score = scores[index]

                ymin, xmin, ymax, xmax = box


                xmin = max(0, np.floor(xmin + 0.5).astype('int32'))
                ymin = max(0, np.floor(ymin + 0.5).astype('int32'))
                xmax = min(image.size[0], np.floor(xmax + 0.5).astype('int32'))
                ymax = min(image.size[1], np.floor(ymax + 0.5).astype('int32'))
                
                pred_boxes.append([xmin, ymin, xmax, ymax, score])
            
            if len(pred_boxes) > 0:
                pred_boxes = np.array(pred_boxes)
            else:
                pred_boxes = np.array([[]])

            classes = np.array(classes)
            scores = np.array(scores)
            #  sort the boxes and the labels according to scores
            score_sort = np.argsort(-scores)
            pred_label = classes[score_sort]
            pred_boxes = pred_boxes[score_sort]

            # copy detection to all_detections
            for label in range(len(self.classes_name)):
                all_detections[i][label] = pred_boxes[pred_label == label, :]
           
            # copy detections to all_annotaions
            for label in range(len(self.classes_name)):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
            
        
        average_precisions = {}
        
        for label in range(len(self.classes_name)):

            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0
            
            for i in range(num_test):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += annotations.shape[0]
                detected_annotations = []
                
                for d in detections:

                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue
                
                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
            
            if num_annotations == 0:
                average_precisions[self.classes_name[label]] = 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)
            
            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision = compute_ap(recall, precision)
            average_precisions[self.classes_name[label]] = average_precision

        return average_precisions
