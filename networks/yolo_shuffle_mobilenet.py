
from tensorflow.keras.layers import Conv2D, BatchNormalization, ZeroPadding2D
from tensorflow.keras.layers import DepthwiseConv2D, ReLU, UpSampling2D, Concatenate, Lambda, AveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from networks.yolo import make_last_layers
from networks.model import DarknetConv2D_BN_Leaky, compose


def _make_divisible(v, divisor, min_value=None):

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor/2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def correct_pad(inputs, kernel_size):

    img_dim = 2 if K.image_data_format() == 'channels_first' else 1
    input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0]//2, kernel_size[1]//2)
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))
    

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


def res_block(inputs, expansion, strides, alpha, filters, block_id):

    channel_axis = 1 if K.image_data_format() =='chnnels_first' else -1

    in_channels = K.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    
    if strides == 1:
        c_hat, c = channel_split(inputs, "stage{}".format(block_id))
        x = c
    else:
        x = inputs
    
    prefix = 'block_{}_'.format(block_id)

    x = Conv2D(expansion * in_channels, 1, padding='same', use_bias=False, name=prefix + 'expand')(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
    x = ReLU(6, name=prefix + 'expand_relu')(x)

    x = DepthwiseConv2D(3, strides=strides, use_bias=False, padding='same', name=prefix + 'depthwise')(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    if strides == 1:
        x = Concatenate()([x, c_hat])
    else:
        shortcut = AveragePooling2D((3, 3), strides=(2, 2), padding='same', name=prefix+'/avg_pool')(inputs)
        x = Concatenate()([x, shortcut])
        
    # Project
    x = Conv2D(pointwise_filters, 1, padding='same', use_bias=False, name=prefix + 'project')(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.999, epsilon=1e-3, name=prefix + 'project_BN')(x)

    x = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(x)

    return x

def shuffle_mobilenet(inputs, alpha=1.0):
    
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    first_block_filters = _make_divisible(32 * alpha, 8)
    stage1 = ZeroPadding2D(padding=correct_pad(inputs, 3), name='conv1_pad')(inputs)
    stage1 = Conv2D(first_block_filters, 3, strides=(2, 2), padding='same', use_bias=False, name='conv1')(inputs)
    stage1 = BatchNormalization(axis=bn_axis,momentum=0.999, epsilon=1e-3,  name="conv1_bn")(stage1)
    stage1 = ReLU(6., name="conv1_relu")(stage1)
    stage1 = res_block(stage1, filters=16, alpha=alpha, strides=1, expansion=1, block_id=0)
    
    stage2 = res_block(stage1, filters=24, alpha=alpha, strides=2, expansion=6, block_id=1)
    stage2 = res_block(stage2, filters=24, alpha=alpha, strides=1, expansion=6, block_id=2)

    stage3 = res_block(stage2, filters=32, alpha=alpha, strides=2, expansion=6, block_id=3)
    stage3 = res_block(stage3, filters=32, alpha=alpha, strides=1, expansion=6, block_id=4)
    stage3 = res_block(stage3, filters=32, alpha=alpha, strides=1, expansion=6, block_id=5)
    feat1 = stage3

    stage4 = res_block(stage3, filters=64, alpha=alpha, strides=2, expansion=6, block_id=6)
    stage4 = res_block(stage4, filters=64, alpha=alpha, strides=1, expansion=6, block_id=7)
    stage4 = res_block(stage4, filters=64, alpha=alpha, strides=1, expansion=6, block_id=8)
    stage4 = res_block(stage4, filters=64, alpha=alpha, strides=1, expansion=6, block_id=9)
    stage4 = res_block(stage4, filters=96, alpha=alpha, strides=1, expansion=6, block_id=10)
    stage4 = res_block(stage4, filters=96, alpha=alpha, strides=1, expansion=6, block_id=11)
    stage4 = res_block(stage4, filters=96, alpha=alpha, strides=1, expansion=6, block_id=12)
    feat2 = stage4

    stage5 = res_block(stage4, filters=160, alpha=alpha, strides=2, expansion=6, block_id=13)
    stage5 = res_block(stage5, filters=160, alpha=alpha, strides=1, expansion=6, block_id=14)
    stage5 = res_block(stage5, filters=160, alpha=alpha, strides=1, expansion=6, block_id=15)
    stage5 = res_block(stage5, filters=320, alpha=alpha, strides=1, expansion=6, block_id=16)

    stage5 = Conv2D(1280, (1, 1), use_bias=False, name='conv_last')(stage5)
    stage5 = BatchNormalization(axis=bn_axis, epsilon=1e-3, momentum=0.999, name='conv_last_bn')(stage5)
    stage5 = ReLU(6., name='out_relu')(stage5)
    feat3 = stage5
    
    return feat1, feat2, feat3


def yolo_body(inputs, num_anchors, num_classes):

    feat1, feat2, feat3 = shuffle_mobilenet(inputs)

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
    