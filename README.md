# yolov3-keras

This is use tensorflow and keras implements yolov3. The paper is [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767).  
You can use this code train self datasets, also can test voc dataset.  

## Requirments
```
tensorflow-gpu == 2.2.0
Pillow == 8.0.1
opencv-python == 4.0.1.23
```

## TODO
- [x] yolov3
- [x] Backbone:Darknet53
- [x] use voc dataset
- [ ] use coco dataset
- [x] predict image
- [ ] predict video
- [ ] yolov3-tiny
- [ ] more backbone

## Training
If you want to train self dataset, you need to modify `config.py`  
```
CLASSES : self classes.
TRAIN_TEXT: Train image information.(EX: path xmin, ymin, xmax, ymax, class_number xmin, ymin, xmax, ymax, class_number)
DATASET : Storage images' and annotations' folder.
```
And then you execute command line. `python train.py`.  


If you want to train `VOC dataset`, you need to use `voc_annotation.py` that will xml convert to text.  
If you want to train `COCO dataset`, you need to use `coco_annotation.py` that will xml convert to text.  


## Testing
Please execute `predict.py`.  


## Reference

[bubbliiiing: yolo3-keras](https://github.com/bubbliiiing/yolo3-keras/tree/bfe8f769b7ef02bf9caf75f22d86f5090303e4df)  
[qqwweee: keras-yolo3](https://github.com/qqwweee/keras-yolo3)  
