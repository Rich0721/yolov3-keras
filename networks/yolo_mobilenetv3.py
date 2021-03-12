
from tensorflow.keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, Dense, GlobalAveragePooling2D, Reshape
from tensorflow.keras.layers import DepthwiseConv2D, ReLU, UpSampling2D, Concatenate, add, Activation, Multiply
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from networks.yolo import make_last_layers
from networks.model import DarknetConv2D_BN_Leaky, compose


def relu6(x):
    return ReLU(6)(x)

def hard_swish(x):
    return x * K.relu(x + 3.0, max_value=6) / 6.0


def conv_block(inputs, filters, kernel, strides, nl):
    channels_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(momentum=0.999, epsilon=1e-3, axis=channels_axis)(x)

    if nl == 'RE':
        x = Activation(relu6)(x)
    elif nl == "HS":
        x = Activation(hard_swish)(x)
    
    return x


def _squeeze(inputs):

    input_channels = int(inputs.shape[-1])

    x = GlobalAveragePooling2D()(inputs)
    x = Dense(input_channels, activation='relu')(x)
    x = Dense(input_channels, activation='hard_sigmoid')(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])
    return x


def bottleneck(inputs, filters, kernel, e, strides, squeeze, nl, alpha=1):

    channels_axis = 1 if K.image_data_format() == 'channels_first' else -1
    input_shape = K.int_shape(inputs)

    t_channel = int(e)
    c_channel = int(alpha * filters)

    r = strides == (1, 1) and input_shape[3] == filters

    x = conv_block(inputs, t_channel, (1, 1), strides=(1, 1), nl=nl)

    x = DepthwiseConv2D(kernel, strides=strides, depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(momentum=0.999, epsilon=1e-3, axis=channels_axis)(x)
    
    if nl == 'RE':
        x = Activation(relu6)(x)
    elif nl == "HS":
        x = Activation(hard_swish)(x)
    
    if squeeze:
        x = _squeeze(x)
    
    x = Conv2D(c_channel, (1, 1), padding='same')(x)
    x = BatchNormalization(momentum=0.999, epsilon=1e-3, axis=channels_axis)(x)

    if r:
        x = add([x, inputs])
    return x


def mobilenetv3(inputs, alpha=1.0):
    
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    
    # 416
    conv1 = conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')
    
    # 208
    block2 = bottleneck(conv1, 16, (3, 3), e=16, strides=(1, 1), squeeze=False, nl='RE')
    block3 = bottleneck(block2, 24, (3, 3), e=64, strides=(2, 2), squeeze=False, nl='RE')
    
    # 104
    block4 = bottleneck(block3, 24, (3, 3), e=72, strides=(1, 1), squeeze=False, nl='RE')
    block5 = bottleneck(block4, 40, (5, 5), e=72, strides=(2, 2), squeeze=True, nl='RE')
    feat1 = block5
    # 52
    block6 = bottleneck(block5, 40, (5, 5), e=120, strides=(1, 1), squeeze=True, nl='RE')
    block7 = bottleneck(block6, 40, (5, 5), e=120, strides=(1, 1), squeeze=True, nl='RE')
    block8 = bottleneck(block7, 80, (3, 3), e=240, strides=(2, 2), squeeze=False, nl='HS')
    feat2 = block8
    # 26
    block9 = bottleneck(block8, 80, (3, 3), e=200, strides=(1, 1), squeeze=False, nl='HS')
    block10 = bottleneck(block9, 80, (3, 3), e=184, strides=(1, 1), squeeze=False, nl='HS')
    block11 = bottleneck(block10, 80, (3, 3), e=184, strides=(1, 1), squeeze=False, nl='HS')
    block12 = bottleneck(block11, 112, (3, 3), e=480, strides=(1, 1), squeeze=True, nl='HS')
    block13 = bottleneck(block12, 112, (3, 3), e=672, strides=(1, 1), squeeze=True, nl='HS')
    block14 = bottleneck(block13, 160, (5, 5), e=672, strides=(1, 1), squeeze=True, nl='HS')
    block15 = bottleneck(block14, 160, (5, 5), e=672, strides=(2, 2), squeeze=True, nl='HS')

    block16 = bottleneck(block15, 160, (5, 5), e=960, strides=(1, 1), squeeze=True, nl='HS')

    conv2 = conv_block(block16, 960, (1, 1), strides=(1, 1), nl='HS')
    conv3 = conv_block(conv2, 1280, (1, 1), strides=(1, 1), nl='HS')
    feat3 = conv3
    return feat1, feat2, feat3


def yolo_body(inputs, num_anchors, num_classes):

    feat1, feat2, feat3 = mobilenetv3(inputs)

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
    