from config import config
import tensorflow as tf
import numpy as np
import os
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#from networks.yolo_test import yolo_body
from networks.yolo import yolo_body
from loss import yolo_loss
from generator import Generator


def learning_schedulder(epoch):

    if epoch < 10:
        return 1e-4
    elif epoch < 120:
        return 1e-4
    else:
        return 1e-5
        
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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
    # Callback set
    checkpoint = ModelCheckpoint(os.path.join(config.TENSORBOARD_DIR, config.WEIGHTS_FILE + "-{epoch:03d}.h5"),
                                monitor='val_loss', save_best_only=True, save_weights_only=True, mode='auto', verbose=1)
    learningRateScheduler = LearningRateScheduler(learning_schedulder, verbose=1)
    
    val_split = 0.1
    with open(config.TRAIN_TEXT) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    gan = Generator(config.BATCH_SIZE, lines[:num_train], lines[num_train:], config.IMAGE_SIZE, len(config.CLASSES), anchors)
    # Train
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    model.fit_generator(gan.generator(True),
                    steps_per_epoch=num_train//config.BATCH_SIZE,
                    validation_data=gan.generator(False),
                    validation_steps=num_val//config.BATCH_SIZE,
                    epochs=config.EPOCHS,
                    callbacks=[checkpoint, learningRateScheduler])
    model.save(config.TENSORBOARD_DIR, save_format='tf')
