import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from networks.yolo import yolo_head


def box_iou(b1, b2):
    
    b1 = K.expand_dims(b1, axis=-2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_min = b1_xy - (b1_wh / 2.)
    b1_max = b1_xy + (b1_wh / 2.)
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]

    b2 = K.expand_dims(b2, axis=0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_min = b2_xy - (b2_wh / 2.)
    b2_max = b2_xy + (b2_wh / 2.)
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]

    intersect_mins = K.maximum(b1_min, b2_min)
    intersect_maxs = K.minimum(b1_max, b2_max)
    intersect_wh = K.maximum(intersect_maxs - intersect_mins, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    iou = intersect_area / (b1_area + b2_area - intersect_area)
    return iou


def yolo_loss(args, anchors, num_classes, ignore_threshold=0.5, print_loss=False, normalize=True):

    num_layers = len(anchors) // 3

    y_true = args[num_layers:]
    yolo_outputs = args[:num_layers]

    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    num_pos = 0
    m = K.shape(yolo_outputs[0])[0] # Batch size
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):

        # feature maps location
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]
        
        # predict grid, class, xy, wh 
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l], anchors=anchors[anchors_mask[l]], 
                                        num_classes=num_classes, input_shape=input_shape, calc_loss=True)

        pred_box = K.concatenate([pred_xy, pred_wh])
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_threshold, K.dtype(true_box)))
            return b+1, ignore_mask
        
        _, ignore_mask = tf.while_loop(lambda b, *args: b<m, loop_body, [0, ignore_mask])

        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # Encode true bounding boxes
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][:] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchors_mask[l]] * input_shape[::-1])

        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))

        # True bounding box is to bigger, weights is less.
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]
        
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])

        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                        (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        xy_loss = K.sum(xy_loss)
        wh_loss = K.sum(wh_loss)
        confidence_loss = K.sum(confidence_loss)
        class_loss = K.sum(class_loss)

        # Compute positive sample
        num_pos += tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
        loss += xy_loss + wh_loss + confidence_loss + class_loss

        if print_loss:
            loss = tf.print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, tf.shape(ignore_mask)], summarize=100, message="loss: ")
        
    if normalize:
        loss = loss / num_pos
    else:
        loss = loss / mf
    return loss
