from generator import Generator
from yolo_predict import YOLO
import numpy as np
from config import config


yolo = YOLO("ep081.h5")


with open(config.VAL_TEXT) as f:
        lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines))

gen = Generator(config.BATCH_SIZE, train_lines=None, val_lines=lines, 
                image_size=config.IMAGE_SIZE, num_classes=len(config.CLASSES), anchors=config.ANCHORS)
ap = yolo.evaluate(gen, num_val)
print(ap)