from config import config
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD

from networks.yolo import yolo_body
from loss import yolo_loss
from generator import data_argumation


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random=True):

    n = len(annotation_lines)
    i = 0

    while True:
        image_data = []
        boxes_data = []
        if i == 0:
            np.random.shuffle(annotation_lines)

        for b in range(batch_size):
            image, boxes = data_argumation(annotation_lines[i], input_shape, random=random)
            image_data.append(image)
            boxes_data.append(boxes)
            i = (i + 1) % n
        
        image_data = np.array(image_data)
        boxes_data = np.array(boxes_data)

        y_true = preprocess_true_boxes(boxes_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    
    num_layers = len(anchors) // 3
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    true_boxes = np.array(true_boxes, dtype=np.float32)
    input_shape = np.array(input_shape, dtype=np.int32)

    # compute centers, width and height.
    # Then normalization
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    # m: image numbers
    # grid shapes: grid's shape
    m = true_boxes.shape[0]
    grid_shapes = [input_shape //{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchors_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]
    
    # (9, 2) -> (1, 9, 2)
    anchors = np.expand_dims(anchors, axis=0)
    anchors_maxs = anchors / 2.
    anchors_mins = -anchors_maxs

    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue

        # (n, 2) -> (n, 1, 2)
        wh = np.expand_dims(wh, axis=1)
        boxes_maxs = wh/2.
        boxes_mins = -boxes_maxs

        # Compute true boxes' and anchors boxes' IoU
        intersect_mins = np.maximum(boxes_mins, anchors_mins)
        intersect_maxes = np.minimum(boxes_maxs, anchors_maxs)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        boxes_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (boxes_area + anchor_area - intersect_area)

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


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def learning(epoch):
    if epoch < 100:
    
        return 1e-4
    else:
        return 1e-5


if __name__ == "__main__":
    
    if not os.path.exists(config.TENSORBOARD_DIR):
        os.mkdir(config.TENSORBOARD_DIR)
    
    anchors = np.reshape(config.ANCHORS, (-1, 2))
    
    image_input = Input(shape=(None, None, 3))
    h, w = config.IMAGE_SIZE
    model_yolo = yolo_body(image_input, len(anchors)//3, len(config.CLASSES))

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], len(anchors)//3, len(config.CLASSES) +5)) for l in range(3)]
    
    loss_input = [*model_yolo.output, *y_true]
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
            arguments={'anchors':anchors, 'num_classes':len(config.CLASSES), 'ignore_threshold':config.IGNORE_THRESH, 'normalize':True})(loss_input)
    
    model = Model([model_yolo.input, *y_true], model_loss)
    model.summary()
    model.load_weights("./yolov3_sgd/yolov3-034.h5")
    # Callback set
    checkpoint = ModelCheckpoint(os.path.join(config.TENSORBOARD_DIR, config.WEIGHTS_FILE + "-{epoch:03d}.h5"),
                                monitor='val_loss', save_best_only=True, save_weights_only=True, mode='auto', verbose=1)
    learningRateScheduler = LearningRateScheduler(learning, verbose=1)
    #earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    if not os.path.exists(config.TENSORBOARD_DIR):
        os.mkdir(config.TENSORBOARD_DIR)

    val_split = 0.1
    with open(config.TRAIN_TEXT[0]) as f:
        lines = f.readlines()
    with open(config.TRAIN_TEXT[1]) as f:
        val_lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = len(val_lines)
    num_train = len(lines)
    
    # Train
    model.compile(optimizer=SGD(learning_rate=config.LEARNING_RATE), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    model.fit_generator(data_generator(lines, config.BATCH_SIZE, config.IMAGE_SIZE, anchors, len(config.CLASSES), random=True),
                    steps_per_epoch=num_train//config.BATCH_SIZE,
                    validation_data=data_generator(val_lines, config.BATCH_SIZE, config.IMAGE_SIZE, anchors, len(config.CLASSES), random=False),
                    validation_steps=num_val//config.BATCH_SIZE,
                    epochs=config.EPOCHS,
                    initial_epoch=33,
                    callbacks=[checkpoint, learningRateScheduler])
    model.save(config.TENSORBOARD_DIR, save_format='tf')
