from pycocotools.coco import COCO
from config import config


def id2name(coco):
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes


def convert_annotation(image, coco, write_file, dir, class_dict):

    file_name = coco.loadImgs(image)[0]['file_name']
    print(file_name)
    write_file.write(dir + file_name)
    
    anns_ids = coco.getAnnIds(image)
    anns = coco.loadAnns(anns_ids)

    for ann in anns:
        class_name = class_dict[ann['category_id']]
        classes_index = config.CLASSES.index(class_name)
        if 'bbox' in ann:
            bbox = ann['bbox']
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[0] + bbox[2])
            ymax = int(bbox[1] + bbox[3])
            write_file.write(" {},{},{},{},{}".format(xmin, ymin, xmax, ymax, classes_index))

if __name__ == "__main__":

    for index, js in enumerate(config.COCO_JSON):
        coco = COCO(js)
        classes_dict = id2name(coco)
        write_file = open(config.COCO_TRAIN_TEXT[index], 'w')
        coco_imgs = coco.imgs
        for i, image in enumerate(coco_imgs):
            convert_annotation(image, coco, write_file, config.COCO_DATASER_FOLDER, classes_dict)
            write_file.write("\n")

        write_file.close()