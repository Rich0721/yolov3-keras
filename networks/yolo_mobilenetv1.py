
from tensorflow.keras.layers import Conv2D, BatchNormalization, ZeroPadding2D
from tensorflow.keras.layers import DepthwiseConv2D, ReLU, UpSampling2D, Concatenate

from tensorflow.keras.models import Model
from networks.yolo import make_last_layers
from networks.model import DarknetConv2D_BN_Leaky, compose


def depthwise_conv(inputs, pointwise_filter, alpha=1, depth_mutiplier=1, strides=(1, 1), name=None):

    pointwise_filter = int(pointwise_filter*alpha)

    if strides == (2, 2):
        x = ZeroPadding2D(((1, 1), (1, 1)), name="{}/padding".format(name))(inputs)
    else:
        x = inputs
    
    x = DepthwiseConv2D((3, 3),
                padding='same' if strides==(1, 1) else 'valid', strides=strides,
                depth_multiplier=depth_mutiplier, use_bias=False, name='{}/DW'.format(name))(x)
    x = BatchNormalization(name="{}/DW_BN".format(name))(x)
    x = ReLU(name="{}/DW_RELU".format(name))(x)

    x = Conv2D(pointwise_filter, (1, 1), strides=(1, 1), padding='same', use_bias=False, name='{}/PW'.format(name))(x)
    x = BatchNormalization(name="{}/PW_BN".format(name))(x)
    x = ReLU(name="{}/PW_RELU".format(name))(x)
    return x


def conv_block(inputs, filters, alpha=1, kernel=(3, 3), strides=(1, 1)):
    filters = int(filters * alpha)
    x = ZeroPadding2D(((1, 1), (1, 1)), name="conv1/padding")(inputs)
    x = Conv2D(filters, kernel, strides=strides, padding='valid', use_bias=False, name='conv1')(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001 ,name='conv1/BN')(x)
    x = ReLU(name='conv1/RELU')(x)
    return x


def mobilenetv1(inputs):
    
    # 416
    conv = conv_block(inputs, 32, alpha=1, strides=(2, 2))
    block1 = depthwise_conv(conv, 64, alpha=1, name='block1')
    
    # 208
    block2 = depthwise_conv(block1, 128, alpha=1, strides=(2, 2), name='block2')
    block3 = depthwise_conv(block2, 128, alpha=1, name='block3')
    
    # 104
    block4 = depthwise_conv(block3, 256, alpha=1, strides=(2, 2), name='block4')
    block5 = depthwise_conv(block4, 256, alpha=1, name='block5')
    feat1 = block5
    # 52
    block6 = depthwise_conv(block5, 512, alpha=1, strides=(2, 2), name='block6')
    block7 = depthwise_conv(block6, 512, alpha=1, name='block7')
    block8 = depthwise_conv(block7, 512, alpha=1, name='block8')
    block9 = depthwise_conv(block8, 512, alpha=1, name='block9')
    block10 = depthwise_conv(block9, 512, alpha=1, name='block10')
    block11 = depthwise_conv(block10, 512, alpha=1, name='block11')
    feat2 = block11
    # 26
    block12 = depthwise_conv(block11, 1024, alpha=1, strides=(2, 2), name='block12')
    block13 = depthwise_conv(block12, 1024, alpha=1, name='block13')
    feat3 = block13
    
    return feat1, feat2, feat3


def yolo_body(inputs, num_anchors, num_classes):

    feat1, feat2, feat3 = mobilenetv1(inputs)

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
    