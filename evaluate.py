from generator import Generator
from yolo_predict import YOLO
import numpy as np
from config import config



yolo = YOLO("./yolov3-148.h5")

with open("test.txt") as f:
        lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines))

gen = Generator(val_lines=lines)
ap = yolo.evaluate(gen, num_val)
x = 0
for key, values in ap.items():
        x+=values
        print("{}: {:.2f}".format(key, values))
print("Average :{:.2f}".format(x/31))
