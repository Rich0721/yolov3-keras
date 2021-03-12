import xml.etree.ElementTree as ET
from config import config
import numpy as np

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
    
    
    for index, file_name in enumerate(config.TRAIN_TEXT):
        
        with open(config.VOC_TEXT_FILE[index]) as f:
            lines = f.read().split()
        np.random.shuffle(lines)
        f = open(file_name, 'w')
        
        for line in lines:
            print(line)
            f.write(config.DATASET + "JPEGImages/" + line + ".jpg")
            convert_annotation(config.DATASET + "Annotations/" + line + ".xml", f)
            f.write("\n")
        f.close()