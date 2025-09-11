from functools import partial
from typing import Tuple, Optional

import keras # type: ignore


def get_small_cnn(
    input_shape: Tuple[int, ...] = (64, 64, 1),
    batch_size: Optional[int] = None,
    num_classes: int = 10,
    base_filters: int = 32,
):
    """
    A small, fully convolutional neural network (around 1M parameters).
    """
    act = keras.layers.LeakyReLU(negative_slope=0.1)
    conv = partial(
        keras.layers.Conv2D,
        kernel_size=(3, 3),
        padding="same",
        activation=None,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
    )
    norm = partial(keras.layers.BatchNormalization, dtype="float32")
    inputs = keras.Input(shape=input_shape, batch_size=batch_size)
    x = inputs
    x = conv(base_filters)(x)
    x = norm()(x)
    x = act(x)
    x = conv(base_filters)(x)
    x = norm()(x)
    x = act(x)
    x = conv(base_filters * 2)(x)
    x = norm()(x)
    x = act(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = conv(base_filters * 2)(x)
    x = norm()(x)
    x = act(x)
    x = conv(base_filters * 2)(x)
    x = norm()(x)
    x = act(x)
    x = conv(base_filters * 4)(x)
    x = norm()(x)
    x = act(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = conv(base_filters * 4)(x)
    x = norm()(x)
    x = act(x)
    x = conv(base_filters * 4)(x)
    x = norm()(x)
    x = act(x)
    x = conv(base_filters * 8)(x)
    x = norm()(x)
    x = act(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = conv(num_classes, dtype="float32")(x)
    outputs = keras.layers.GlobalAveragePooling2D()(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
