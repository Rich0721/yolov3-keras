
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import DepthwiseConv2D, ReLU, UpSampling2D, Concatenate, Lambda, AveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from networks.yolo import make_last_layers
from networks.model import DarknetConv2D_BN_Leaky, compose
import numpy as np

def channel_split(inputs, name=''):

    channels = inputs.shape.as_list()[-1]
    split_channel = channels // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:split_channel], name="{}/split_{}slice".format(name, 0))(inputs)
    c = Lambda(lambda z: z[:, :, :, split_channel:], name="{}/split_{}slice".format(name, 1))(inputs)

    return c_hat, c


def channel_shuffle(inputs):
    height, width, channels = inputs.shape.as_list()[1:]
    channel_per_split = channels // 2
    x = K.reshape(inputs, [-1, height, width, 2, channel_per_split])
    x = K.permute_dimensions(x, (0, 1, 2, 3, 4))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio, strides=2, stage=1, block=1):

    if K.image_data_format() == 'channel_last':
        bn_axis = -1
    else:
        bn_axis = 1
    
    prefix = 'stage{}/block{}'.format(stage, block)

    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/split'.format(prefix))
        inputs = c
    
    bottlneck_channels = int(out_channels * bottleneck_ratio)

    x = Conv2D(bottlneck_channels, (1, 1), strides=1, padding='same', name="{}/conv1".format(prefix))(inputs)
    x = BatchNormalization(axis=bn_axis, momentum=0.999, epsilon=1e-3, name="{}/conv1_BN".format(prefix))(x)
    x = Activation("relu", name="{}/conv1_relu".format(prefix))(x)

    x = DepthwiseConv2D((3, 3), strides=strides, padding='same', name='{}/dwconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, momentum=0.999, epsilon=1e-3, name='{}/dwconv_BN'.format(prefix))(x)
    x = Activation("relu", name='{}/dwconv_relu'.format(prefix))(x)

    x = Conv2D(bottlneck_channels, (1, 1), strides=1, padding='same', name='{}/conv2'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, momentum=0.999, epsilon=1e-3, name="{}/conv2_BN".format(prefix))(x)
    x = Activation("relu", name="{}/conv2_relu".format(prefix))(x)

    if strides < 2:
        ret = Concatenate(name="{}/concat".format(prefix))([x, c_hat])
    else:
        shortcut = DepthwiseConv2D((3, 3), strides=strides, padding='same', name='{}/dwconv2'.format(prefix))(inputs)
        shortcut = BatchNormalization(axis=bn_axis, name='{}/dwconv2_BN'.format(prefix))(shortcut)

        shortcut = Conv2D(bottlneck_channels, (1, 1), strides=1, padding='same', name='{}/conv3'.format(prefix))(shortcut)
        shortcut = BatchNormalization(axis=bn_axis, momentum=0.999, epsilon=1e-3, name="{}/conv3_BN".format(prefix))(shortcut)
        shortcut = Activation("relu", name='{}/conv3_relu'.format(prefix))(shortcut)
        ret = Concatenate(name="{}/concat".format(prefix))([x, shortcut])
    
    
    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)
    return ret


def block(inputs, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(inputs, out_channels=channel_map[stage-1], bottleneck_ratio=bottleneck_ratio, strides=2, stage=stage, block=1)

    for i in range(1, repeat):
        x = shuffle_unit(x, out_channels=channel_map[stage-1], bottleneck_ratio=bottleneck_ratio, strides=1, stage=stage, block=i+1)
    return x

def shufflenetv2(inputs, alpha=1.0):
    
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    num_shuffle_units = [3, 7, 3]
    scale_factor=1.0
    bottleneck_ratio=1
    dims = {0.5:48, 1:116, 1.5:176, 2:244}
    
    exp = np.insert(np.arange(0, len(num_shuffle_units), dtype=np.float32), 0, 0)
    out_channels_in_stage = 2 ** exp
    out_channels_in_stage *= dims[bottleneck_ratio]
    out_channels_in_stage[0] = 24
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    
    stage1 = Conv2D(out_channels_in_stage[0], (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='relu', name='conv1')(inputs)
    stage1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool1')(stage1)

    stage2 = block(stage1, out_channels_in_stage, repeat=3, bottleneck_ratio=bottleneck_ratio, stage=2)
    feat1 = stage2
    stage3 = block(stage2, out_channels_in_stage, repeat=7, bottleneck_ratio=bottleneck_ratio, stage=3)
    feat2 = stage3
    stage4 = block(stage3, out_channels_in_stage, repeat=3, bottleneck_ratio=bottleneck_ratio, stage=4)
    feat3 = stage4
    
    return feat1, feat2, feat3


def yolo_body(inputs, num_anchors, num_classes):

    feat1, feat2, feat3 = shufflenetv2(inputs)

    # (13, 13, 1024) -> (13, 13, 512)
    x, y1 = make_last_layers(feat3, 512, num_anchors*(num_classes + 5))

    # (13, 13, 512) -> (26, 26, 256)
    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, feat2])

    # (26, 26, 256) -> (52, 52, 128)
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, feat1])

    # (52, 52, 256) -> (52, 52, 128)
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes + 5))
    return Model(inputs, [y1, y2, y3])
    