from functools import wraps, reduce
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Add, ZeroPadding2D
from tensorflow.keras.regularizers import l2
from tensorflow.python.framework.tensor_util import FastAppendBFloat16ArrayToTensorProto


def compose(*funcs):

    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_kwargs = {'kernel_regularizer':l2(5e-4)}
    darknet_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):

    no_bias_kwargs = {'use_bias':False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1)
    )



def resdual_block(inputs, filters, num_blocks=1):

    x = ZeroPadding2D(((1,0),(0,1)))(inputs)
    x = DarknetConv2D_BN_Leaky(filters, (3, 3), strides=(2, 2))(x)

    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(filters//2, (1, 1))(x)
        y = DarknetConv2D_BN_Leaky(filters, (3, 3))(y)
        x = Add()([x, y])
    return x


def darknet53(inputs):

    x = DarknetConv2D_BN_Leaky(32, (3, 3))(inputs)
    x = resdual_block(x, 64,  num_blocks=1)
    x = resdual_block(x, 128, num_blocks=2)
    x = resdual_block(x, 256, num_blocks=8)
    feat1 = x
    x = resdual_block(x, 512, num_blocks=8)
    feat2 = x
    x = resdual_block(x, 1024, num_blocks=4)
    feat3 = x
    return feat1, feat2, feat3
