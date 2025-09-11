import keras # type: ignore
import typing_extensions as tx
import typing, warnings
from typing import Tuple

from . import layers, utils

ConfigDict = tx.TypedDict(
    "ConfigDict",
    {
        "dropout": float,
        "mlp_dim": int,
        "num_heads": int,
        "num_layers": int,
        "hidden_size": int,
    },
)

CONFIG_B: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 3072,
    "num_heads": 12,
    "num_layers": 12,
    "hidden_size": 768,
}

CONFIG_L: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 4096,
    "num_heads": 16,
    "num_layers": 24,
    "hidden_size": 1024,
}

BASE_URL = "https://github.com/faustomorales/vit-keras/releases/download/dl"
WEIGHTS = {"imagenet21k": 21_843, "imagenet21k+imagenet2012": 1_000}
SIZES = {"B_16", "B_32", "L_16", "L_32"}

def build_model(
    input_shape: Tuple[int, ...] ,
    patch_size: int,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    name: str,
    mlp_dim: int,
    classes: int,
    dropout=0.1,
    activation="linear",
    include_top=True,
    representation_size=None,
):
    """Build a ViT model.

    Args:
        input_shape: The shape of the input images.
        patch_size: The size of each patch (must fit evenly in image_size)
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        num_layers: The number of transformer layers to use.
        hidden_size: The number of filters to use
        num_heads: The number of transformer heads
        mlp_dim: The number of dimensions for the MLP output in the transformers.
        dropout_rate: fraction of the units to drop for dense layers.
        activation: The activation to use for the final layer.
        include_top: Whether to include the final classification layer. If not,
            the output will have dimensions (batch_size, hidden_size).
        representation_size: The size of the representation prior to the
            classification layer. If None, no Dense layer is inserted.
    """
    assert input_shape[0] % patch_size == 0, (
        f"input_shape must be a multiple of patch_size {input_shape[0]} % {patch_size} != 0"
    )
    assert input_shape[-1] == 3, "Input shape must have 3 channels (RGB)."
    x = keras.layers.Input(shape=input_shape)
    y = keras.layers.Conv2D(
        filters=hidden_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="embedding",
    )(x)
    y = keras.layers.Reshape((y.shape[1] * y.shape[2], hidden_size))(y)
    y = layers.ClassToken(name="class_token")(y)
    y = layers.AddPositionEmbs(name="Transformer:posembed_input")(y)
    for n in range(num_layers):
        y, _ = layers.TransformerBlock(
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"Transformer:encoderblock_{n}",
        )(y)
    y = keras.layers.LayerNormalization(
        epsilon=1e-6, name="Transformer:encoder_norm"
    )(y)
    y = keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(y)
    if representation_size is not None:
        y = keras.layers.Dense(
            representation_size, name="pre_logits", activation="tanh"
        )(y)
    if include_top:
        y = keras.layers.Dense(classes, name="head", activation=activation)(y)
    return keras.models.Model(inputs=x, outputs=y, name=name)


def validate_pretrained_top(
    include_top: bool, pretrained: bool, classes: int, weights: str
):
    """Validate that the pretrained weight configuration makes sense."""
    assert weights in WEIGHTS, f"Unexpected weights: {weights}."
    expected_classes = WEIGHTS[weights]
    if classes != expected_classes:
        warnings.warn(
            f"Can only use pretrained_top with {weights} if classes = {expected_classes}. Setting manually.",
            UserWarning,
        )
    assert include_top, "Can only use pretrained_top with include_top."
    assert pretrained, "Can only use pretrained_top with pretrained."
    return expected_classes


def load_pretrained(
    size: str,
    weights: str,
    pretrained_top: bool,
    model: keras.models.Model,
    input_shape: Tuple[int, ...],
    patch_size: int,
):
    """Load model weights for a known configuration."""
    fname = f"ViT-{size}_{weights}.npz"
    origin = f"{BASE_URL}/{fname}"
    local_filepath = keras.utils.get_file(fname, origin, cache_subdir="weights")
    success = False
    try:
        utils.load_weights_numpy(
            model=model,
            params_path=local_filepath,
            pretrained_top=pretrained_top,
            num_x_patches=input_shape[1] // patch_size,
            num_y_patches=input_shape[0] // patch_size,
        )
        success = True
    except Exception as e:
        success = False
        warnings.warn(f"Error loading weights: {e}", UserWarning)
    if success:
        print(f"... succesfully loaded weights for ViT-{size} from {origin}")

def vit_b8(
    input_shape: Tuple[int, ...] = (224, 224, 3),
    classes=1000,
    activation="linear",
    include_top=True,
    pretrained=True,
    pretrained_top=True,
    weights="imagenet21k+imagenet2012",
):
    """Build ViT-B16. All arguments passed to build_model."""
    if pretrained_top:
        classes = validate_pretrained_top(
            include_top=include_top,
            pretrained=pretrained,
            classes=classes,
            weights=weights,
        )
    model = build_model(
        **CONFIG_B,
        name="vit-b16",
        patch_size=8,
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )

    if pretrained:
        load_pretrained(
            size="B_16",
            weights=weights,
            model=model,
            pretrained_top=pretrained_top,
            input_shape=input_shape,
            patch_size=8,
        )
    return model


def vit_b16(
    input_shape: Tuple[int, ...] = (224, 224, 3),
    classes=1000,
    activation="linear",
    include_top=True,
    pretrained=True,
    pretrained_top=True,
    weights="imagenet21k+imagenet2012",
):
    """Build ViT-B16. All arguments passed to build_model."""
    if pretrained_top:
        classes = validate_pretrained_top(
            include_top=include_top,
            pretrained=pretrained,
            classes=classes,
            weights=weights,
        )
    model = build_model(
        **CONFIG_B,
        name="vit-b16",
        patch_size=16,
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )

    if pretrained:
        load_pretrained(
            size="B_16",
            weights=weights,
            model=model,
            pretrained_top=pretrained_top,
            input_shape=input_shape,
            patch_size=16,
        )
    return model


def vit_b32(
    input_shape: Tuple[int, ...] = (224, 224, 3),
    classes=1000,
    activation="linear",
    include_top=True,
    pretrained=True,
    pretrained_top=True,
    weights="imagenet21k+imagenet2012",
):
    """Build ViT-B32. All arguments passed to build_model."""
    if pretrained_top:
        classes = validate_pretrained_top(
            include_top=include_top,
            pretrained=pretrained,
            classes=classes,
            weights=weights,
        )
    model = build_model(
        **CONFIG_B,
        name="vit-b32",
        patch_size=32,
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )
    if pretrained:
        load_pretrained(
            size="B_32",
            weights=weights,
            model=model,
            pretrained_top=pretrained_top,
            patch_size=32,
            input_shape=input_shape,
        )
    return model


def vit_l16(
    input_shape: Tuple[int, ...] = (384, 384, 3),
    classes=1000,
    activation="linear",
    include_top=True,
    pretrained=True,
    pretrained_top=True,
    weights="imagenet21k+imagenet2012",
):
    """Build ViT-L16. All arguments passed to build_model."""
    if pretrained_top:
        classes = validate_pretrained_top(
            include_top=include_top,
            pretrained=pretrained,
            classes=classes,
            weights=weights,
        )
    model = build_model(
        **CONFIG_L,
        patch_size=16,
        name="vit-l16",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=1024 if weights == "imagenet21k" else None,
    )
    if pretrained:
        load_pretrained(
            size="L_16",
            weights=weights,
            model=model,
            pretrained_top=pretrained_top,
            patch_size=16,
            input_shape=input_shape,
        )
    return model


def vit_l32(
    input_shape: Tuple[int, ...] = (384, 384, 3),
    classes=1000,
    activation="linear",
    include_top=True,
    pretrained=True,
    pretrained_top=True,
    weights="imagenet21k+imagenet2012",
):
    """Build ViT-L32. All arguments passed to build_model."""
    if pretrained_top:
        classes = validate_pretrained_top(
            include_top=include_top,
            pretrained=pretrained,
            classes=classes,
            weights=weights,
        )
    model = build_model(
        **CONFIG_L,
        patch_size=32,
        name="vit-l32",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=1024 if weights == "imagenet21k" else None,
    )
    if pretrained:
        load_pretrained(
            size="L_32",
            weights=weights,
            model=model,
            pretrained_top=pretrained_top,
            patch_size=32,
            input_shape=input_shape,
        )
    return model