
from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, Activation, MaxPooling2D
from tensorflow.keras.layers import DepthwiseConv2D,  UpSampling2D, Concatenate, Lambda, AveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from networks.yolo import make_last_layers
from networks.model import DarknetConv2D_BN_Leaky, compose
import numpy as np

def _channel_shuffle(x, groups):

    """
    x: Input tensor of with 'channels_last' data fromat.
    group(int): number of groups per channel
    returns: channel shuffled output tensor
    """

    h, w, c = x.shape.as_list()[1:]
    channel_per_group = c // groups

    x = K.reshape(x, [-1, h, w, groups, channel_per_group])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3)) # transpose
    x = K.reshape(x, [-1, h, w, c])

    return x


def _group_conv(x, in_channels, out_channels, groups, kernel_size=1, strides=1, name=''):

    """
    x: Input tensor of with 'channels_last' data fromat.
    in_channels:  number of input channels
    out_channels: number of output channels
    group(int): number of groups per channel
    """
    
    if groups == 1:
        return Conv2D(out_channels, kernel_size, strides=strides, padding='same', use_bias=False, name=name)(x)
    
    # number of input channels per group
    input_group = in_channels // groups
    group_list = []
    
    assert out_channels % groups == 0
    
    for i in range(groups):
        
        offset = i * input_group
        group = Lambda(lambda z: z[:, :, :, offset:offset+input_group], name='{}/group_{}slice'.format(name, i))(x)
        group_list.append(Conv2D(int(0.5 + out_channels / groups), kernel_size=kernel_size, strides=strides, use_bias=False, padding='same',
                        name='{}/group{}'.format(name, i))(group))
    
    return Concatenate()(group_list)


def _shuffle_unit(inputs, in_channels, out_channels, groups, bottleneck_ratio, strides=2, stage=1, block=1):

    """
    inputs: Input tensor of with 'channels_last' data fromat.
    in_channels:  number of input channels
    out_channels: number of output channels
    group(int): number of groups per channel
    strides(int or list/tuple): specifying the strides of the convolution along the width and height.
    bottlneck_ratio(float): bottleneck ratio implies the ratio of bottleneck channels to output channels.
    """

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1
    
    prefix = 'stage{}/block{}'.format(stage, block)

    bottlneck_channels = int(out_channels * bottleneck_ratio)
    groups = (1 if stage==2 and block==1 else groups)

    x = _group_conv(inputs, in_channels, out_channels, groups, name=prefix + "/1x1_gconv_1")
    x = BatchNormalization(axis=bn_axis,momentum=0.999, epsilon=1e-3,  name=prefix+'/bn_gconv_1')(x)
    x = Activation('relu', name=prefix+'/relu_gconv_1')(x)

    x = Lambda(_channel_shuffle, arguments={'groups': groups}, name=prefix+'/channel_shuffle')(x)
    x = DepthwiseConv2D((3, 3), strides=strides, padding='same', use_bias=False, name=prefix+'/depthwise')(x)
    x = BatchNormalization(axis=bn_axis, momentum=0.999, epsilon=1e-3, name=prefix+'/depthwise_bn')(x)
    
    x = _group_conv(x, bottlneck_channels, out_channels=out_channels if strides == 1 else out_channels - in_channels, groups=groups, name=prefix + '/1x1_gconv_2')
    x = BatchNormalization(axis=bn_axis, momentum=0.999, epsilon=1e-3, name=prefix + '/bn_gconv_2_')(x)

    if strides < 2:
        ret = Add(name=prefix+'/add')([x, inputs])
    else:
        avg = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name=prefix+'/avg_pool')(inputs)
        ret =  Concatenate(name=prefix+'/concat')([x, avg])

    ret = Activation('relu', name=prefix+'/relu_out')(ret)

    return ret


def _block(x, channel_map, bottleneck_ratio, repeat=1, groups=1, stage=1):

    """
    creates a bottleneck block containing `repeat + 1` shuffle units
    x: Input tensor of with 'channels_last' data fromat
    channel_map(list): containing the number of output channels for a stage
    groups(int): number of groups per channel
    repeat(int): number of repetitions for a shuffle unit with stride 1
    bottlneck_ratio(float): bottleneck ratio implies the ratio of bottleneck channels to output channels.
    stage(int): stage number
    """
    
    x = _shuffle_unit(x, channel_map[stage-2], channel_map[stage-1], strides=2, groups=groups, bottleneck_ratio=bottleneck_ratio, stage=stage, block=1)

    for i in range(1, repeat+1):
        
        x = _shuffle_unit(x, channel_map[stage-1], channel_map[stage-1], strides=1, groups=groups, bottleneck_ratio=bottleneck_ratio, stage=stage, block=i+1)
    return x


def shufflenetv1(inputs, groups=3):
    
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    num_shuffle_units = [3, 7, 3]
    scale_factor=1.0
    bottleneck_ratio=1
    
    if groups == 1:
        dims = 144
    elif groups == 2:
        dims = 200
    elif groups == 3:
        dims = 240
    elif groups == 4:
        dims = 272
    elif groups == 8:
        dims = 384
    else:
        raise ValueError("Invalid number of groups. Please set groups in [1, 2, 3, 4, 8]")
    
    exp = np.insert(np.arange(0, len(num_shuffle_units), dtype=np.float32), 0, 0)
    out_channels_in_stage = 2 ** exp
    out_channels_in_stage *= dims
    out_channels_in_stage[0] = 24
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    # 416
    stage1 = Conv2D(out_channels_in_stage[0], (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='relu', name='conv1')(inputs)
    # 208
    stage1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool1')(stage1)
    # 104
    stage2 = _block(stage1, out_channels_in_stage, repeat=3, bottleneck_ratio=bottleneck_ratio, groups=groups, stage=2)
    feat1 = stage2

    stage3 = _block(stage2, out_channels_in_stage, repeat=7, bottleneck_ratio=bottleneck_ratio, groups=groups, stage=3)
    feat2 = stage3
    stage4 = _block(stage3, out_channels_in_stage, repeat=3, bottleneck_ratio=bottleneck_ratio, groups=groups, stage=4)
    feat3 = stage4
    
    return feat1, feat2, feat3


def yolo_body(inputs, num_anchors, num_classes):

    feat1, feat2, feat3 = shufflenetv1(inputs)

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
    