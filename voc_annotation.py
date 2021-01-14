import xml.etree.ElementTree as ET
import os
from config import config
from glob import glob 

def convert_annotation(xml_file, text_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        cls = obj.find('name').text
        if cls not in config.CLASSES or int(difficult) == 1:
            continue
        cls_id = config.CLASSES.index(cls)
        bndbox = obj.find('bndbox')
        bbox = (int(bndbox.find("xmin").text), int(bndbox.find("ymin").text), int(bndbox.find("xmax").text), int(bndbox.find("ymax").text))
        text_file.write(" " + ",".join([str(b) for b in bbox]) + "," + str(cls_id))

if __name__ == "__main__":
    
    images = glob(os.path.join(config.DATASET, "JPEGImages", "*.jpg"))
    xmls = glob(os.path.join(config.DATASET, "Annotations", "*.xml"))
    assert len(images) == len(xmls), "Images number and XML files number need same."
    
    f = open(config.TRAIN_TEXT, "w")
    for image, xml in zip(images, xmls):
        print(image)
        f.write(image)
        convert_annotation(xml, f)
        f.write("\n")