import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, PReLU, Input
from tensorflow.keras.models import Model

def fsrcnn_model(input_shape=(256, 256, 3), scale=3, d=56, s=12, m=4):
    """
    FSRCNN model architecture for super-resolution.
    Args:
        input_shape: Shape of the input LR image.
        scale: Upscaling factor.
        d, s, m: Hyperparameters of FSRCNN as in the original paper.
    Returns:
        Keras Model.
    """
    x_in = Input(shape=input_shape)

    # Feature extraction
    x = Conv2D(d, (5, 5), padding='same')(x_in)
    x = PReLU(shared_axes=[1, 2])(x)

    # Shrinking
    x = Conv2D(s, (1, 1), padding='same')(x)
    x = PReLU(shared_axes=[1, 2])(x)

    # Mapping
    for _ in range(m):
        x = Conv2D(s, (3, 3), padding='same')(x)
        x = PReLU(shared_axes=[1, 2])(x)

    # Expanding
    x = Conv2D(d, (1, 1), padding='same')(x)
    x = PReLU(shared_axes=[1, 2])(x)

    # Deconvolution (Upsampling)
    x = Conv2DTranspose(3, (9, 9), strides=(scale, scale), padding='same')(x)

    return Model(inputs=x_in, outputs=x, name='FSRCNN_RGB_x{}'.format(scale))
