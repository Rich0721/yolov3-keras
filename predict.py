from PIL import Image
from yolo_predict import YOLO

yolo = YOLO("./ep050.h5")

img = "./00010.jpg"
image = Image.open(img)
image = yolo.detect_image(image)
image.show()