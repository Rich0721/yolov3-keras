
import tensorflow as tf
from tensorflow.keras import backend as K
from config import config
#from networks.yolo import yolo_body
#from networks.yolo_mobilenetv1 import yolo_body
#from networks.yolo_mobilenetv2 import yolo_body
#from networks.yolo_shuffle_mobilenet import yolo_body
#from networks.yolo_shufflenetv1 import yolo_body
from networks.yolo_shufflenetv2 import yolo_body
from tensorflow.keras.layers import Input


def get_flops(model_path):
    tf.compat.v1.disable_eager_execution()
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    anchors = config.ANCHORS

    with graph.as_default():
        with session.as_default():
            image_input = Input(shape=(416, 416, 3))
            h, w = config.IMAGE_SIZE
            model = yolo_body(image_input, len(anchors)//3, len(config.CLASSES))
            model.load_weights(model_path)
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)

            return flops.total_float_ops



if __name__ == '__main__':
    weights_path ="./yolov3-shufflenetv2/yolov3-shufflenetv2-150.h5"
    flops = get_flops(weights_path)
    print(flops)