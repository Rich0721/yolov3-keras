from random import shuffle
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import preprocess_input


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


class Generator:
    
    def __init__(self, batch_size, train_lines, val_lines, image_size, num_classes, anchors) -> None:
        
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.val_lines = val_lines
        self.image_size = image_size
        self.anchors = anchors
        self.num_classes = num_classes
    
    def data_argumation(self, annotation_line, input_shape, max_boxes=100, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):

        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])

        if not random:

            scale = min(w/iw, h/ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)/255

            # correct boxes
            box_data = np.zeros((max_boxes, 5))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data[:len(box)] = box
            return image_data, box_data

        # resize image
        new_ar = w/h * rand(1-jitter, 1+jitter) / rand(1-jitter, 1+jitter)
        scale = rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image)
        image = new_image

        flip = rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < 0.5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < 0.5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)

        # correct box
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:,[0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            if len(box)>max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box
        
        return image_data, box_data
    
    def preprocess_true_boxes(self, true_boxes):

        assert (true_boxes[..., 4]< self.num_classes).all(), 'class id must be less than num_classes'
        num_layers = len(self.anchors)//3
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(self.image_size[:2], dtype='int32')

        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]
       
        m = true_boxes.shape[0]
        grid_shapes = [input_shape // {0:32, 1:16, 2:8}[l] for l in range(num_layers)]

        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchors_mask[l]), 5 + self.num_classes),
                dtype='float32') for l in range(num_layers)]

        anchors = np.expand_dims(self.anchors, 0)
        anchor_maxs = anchors / 2.
        anchor_mins = -anchor_maxs

        valid_mask = boxes_wh[..., 0] > 0

        for b in range(m):
            
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0: continue

            # (n, 2) -> (n, 1, 2)
            wh = np.expand_dims(wh, -2)
            box_maxes = wh / 2.
            box_mins = -box_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxs)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]

            iou = intersect_area / (box_area + anchor_area - intersect_area)

            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                
                for l in range(num_layers):
                    if n in anchors_mask[l]:
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')

                        k = anchors_mask[l].index(n)
                        c = true_boxes[b, t, 4].astype('int32')

                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5+c] = 1
        return y_true

    def generator(self, train=True):

        while True:
            if train:
                shuffle(self.train_lines)
                lines = self.train_lines
            else:
                shuffle(self.val_lines)
                lines = self.val_lines
            
            inputs = []
            targets = []
            for annotation_line in lines:
                if train:
                    img, box = self.data_argumation(annotation_line, self.image_size[0:2], random=True)
                else:
                    img, box = self.data_argumation(annotation_line, self.image_size[0:2], random=False)
                
                inputs.append(img)
                targets.append(box)
                
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    y_true = self.preprocess_true_boxes(tmp_targets)
                    inputs = []
                    targets = []
                    yield [tmp_inp, *y_true], np.zeros(self.batch_size)
