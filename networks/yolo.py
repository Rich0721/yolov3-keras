import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model

from networks.model import darknet53, DarknetConv2D, DarknetConv2D_BN_Leaky, compose

def make_last_layers(inputs, filters, out_filters):

    x = DarknetConv2D_BN_Leaky(filters, (1, 1))(inputs)
    x = DarknetConv2D_BN_Leaky(filters*2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(filters*2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(filters, (1, 1))(x)
    
    y = DarknetConv2D_BN_Leaky(filters*2, (3, 3))(x)
    y = DarknetConv2D(out_filters, (1, 1))(y)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):

    feat1, feat2, feat3 = darknet53(inputs)

    # (13, 13, 1024) -> (13, 13, 512)
    x, y1 = make_last_layers(feat3, 512, num_anchors*(num_classes + 5))

    # (13, 13, 512) -> (26, 26, 256)
    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, feat2])

    # (26, 26, 256) -> (52, 52, 128)
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, feat1])

    # (52, 52, 256) -> (52, 52, 128)
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes + 5))
    return Model(inputs, [y1, y2, y3])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):

    num_anchors = len(anchors)

    anchors_tensor = K.reshape(K.constant(anchors), (1, 1, 1, num_anchors, 2))

    grid_shape = K.shape(feats)[1:3]
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), (-1, 1, 1, 1)), (1, grid_shape[1], 1, 1))
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), (1, -1, 1 ,1)), (grid_shape[0], 1, 1, 1))
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    # (b, 13, 13, num_anchors, num_classes+5)
    feats = K.reshape(feats, (-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5))

    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[..., ::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape - new_shape) /2. /input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxs = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],
        box_mins[..., 1:2],
        box_maxs[..., 0:1],
        box_maxs[..., 1:2]
    ])

    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):

    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, (-1, 4))
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, (-1, num_classes))
    return boxes, box_scores


# image prediction
def yolo_eval(yolo_outputs, anchors, num_classes, image_shape, max_boxes=20, score_threshold=0.5, iou_threshold=0.5):

    num_layers = len(yolo_outputs)
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []

    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchors_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):

        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        
        # nms
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_