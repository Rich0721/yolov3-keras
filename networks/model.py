from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Add, ZeroPadding2D
from tensorflow.keras.regularizers import l2
from tensorflow.python.framework.tensor_util import FastAppendBFloat16ArrayToTensorProto


def conv(inputs, filters, kernel_size, strides=(1, 1), reg=5e-4,normalization=True, name=None):

    if normalization:
        use_bias = False
    else:
        use_bias = True

    if strides == (2, 2):
        x = Conv2D(filters, kernel_size, strides=strides, padding='valid', use_bias=use_bias, kernel_regularizer=l2(reg), name=name)(inputs)
    else:
        x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias, kernel_regularizer=l2(reg), name=name)(inputs)

    if normalization:
        x = BatchNormalization(name=name + "/bn")(x)
        x = LeakyReLU(alpha=0.1, name=name + 'leaky')(x)
    return x


def resdual_block(inputs, filters, stage=1, num_blocks=1):

    x = ZeroPadding2D(((1, 0), (0, 1)))(inputs)
    x = conv(x, filters, (3, 3), strides=(2, 2), name="conv{}_{}".format(stage, 1))
    block = 2
    for i in range(num_blocks):
        y = conv(x, filters//2, (1, 1), strides=(1, 1), name="conv{}_{}".format(stage, block))
        y = conv(y, filters, (3, 3), strides=(1, 1), name="conv{}_{}".format(stage, block+1))
        block += 2
        x = Add()([x, y])
    return x


def darknet53(inputs):

    x = conv(inputs, 32, (3, 3), name='conv1')
    x = resdual_block(x, 64, stage=2, num_blocks=1)
    x = resdual_block(x, 128, stage=3, num_blocks=2)
    x = resdual_block(x, 256, stage=4, num_blocks=8)
    feat1 = x
    x = resdual_block(x, 512, stage=5, num_blocks=8)
    feat2 = x
    x = resdual_block(x, 1024, stage=6, num_blocks=4)
    feat3 = x
    return feat1, feat2, feat3
