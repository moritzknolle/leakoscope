import math
from enum import Enum
from functools import partial
from typing import Callable, List, Optional

import keras # type: ignore
import numpy as np
import tensorflow as tf # type: ignore
from keras import ops # type: ignore
from keras.optimizers.schedules import LearningRateSchedule # type: ignore

import wandb

class LabelType(Enum):
    """Enum for label types."""

    BINARY = 1
    MULTICLASS = 2
    MULTILABEL = 3
    CHEXPERT = 4
    SIMCLR = 5


def get_label_mode(label_mode: str) -> LabelType:
    """
    Get the label mode as a LabelType enum.

    Args:
        label_mode (str): The label mode as a string.

    Returns:
        LabelType: The corresponding LabelType enum.

    Raises:
        ValueError: If the label mode is invalid.
    """
    print(label_mode.lower())
    if label_mode.lower() == "binary":
        return LabelType.BINARY
    elif label_mode.lower() == "multiclass":
        return LabelType.MULTICLASS
    elif label_mode.lower() == "multilabel":
        return LabelType.MULTILABEL
    elif label_mode.lower() == "chexpert":
        return LabelType.CHEXPERT
    elif label_mode.lower() == "simclr":
        return LabelType.SIMCLR
    else:
        raise ValueError("Invalid label mode.")


def stateless_random_rotate(img: tf.Tensor, seed: tf.Tensor, max_degrees=10):
    """
    Apply a random rotation to an image tensor.

    Args:
        img (tf.Tensor): The input image tensor.
        seed (tf.Tensor): The seed tensor for randomness.
        max_degrees (int, optional): The maximum degrees for rotation. Defaults to 10.

    Returns:
        tf.Tensor: The rotated image tensor.
    """
    angle_degrees = tf.random.stateless_uniform(
        [1], seed=seed, minval=-float(max_degrees), maxval=float(max_degrees)
    )
    angle_radians = angle_degrees * (math.pi / 180.0)
    angle = angle_radians[0]

    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    transform = tf.stack([cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0, 0.0, 0.0])
    transform = tf.expand_dims(transform, 0)
    expanded_image = tf.expand_dims(img, 0)
    output_shape = tf.shape(img)[:2]
    rotated_image = tf.raw_ops.ImageProjectiveTransformV2(
        images=expanded_image,
        transforms=transform,
        output_shape=output_shape,
        fill_mode="REFLECT",
        interpolation="BILINEAR",
    )
    return tf.squeeze(rotated_image, axis=0)


def random_pixel_shifts(img: tf.Tensor, seed: tf.Tensor, shift: float = 0.125):
    """
    Apply random pixel shifts to an image tensor.

    Args:
        img (tf.Tensor): The input image tensor.
        seed (tf.Tensor): The seed tensor for randomness.
        shift (float, optional): The shift factor. Defaults to 0.125.

    Returns:
        tf.Tensor: The image tensor with random pixel shifts.
    """
    x = img
    original_shape = img.shape
    shift = 0 if shift == 0.0 else int(shift * original_shape[1])
    x = tf.pad(x, [[shift] * 2, [shift] * 2, [0] * 2], mode="REFLECT")
    x = tf.image.stateless_random_crop(x, original_shape, seed=seed)
    return x


def create_random_aug_fn(aug_fn_list: List[Callable], rng: tf.random.Generator):
    """
    Create a random augmentation function from a list of augmentation functions.

    Args:
        aug_fn_list (List[Callable]): List of augmentation functions.
        rng (tf.random.Generator): Random number generator.

    Returns:
        Callable: The random augmentation function.
    """
    print(
        f"... creating augmentation function with {len(aug_fn_list)} augmentation(s): \n{aug_fn_list}"
    )

    def random_aug(x):
        if len(aug_fn_list) > 0:
            seeds = rng.make_seeds(len(aug_fn_list))
            for i, aug_fn in enumerate(aug_fn_list):
                x = aug_fn(x, seed=seeds[:, i])
        return x

    return random_aug


def get_aug_fn(aug_strength: str):
    """
    Get the augmentation function based on the augmentation strength.

    Args:
        aug_strength (str): The augmentation strength as a string.

    Returns:
        Callable: The augmentation function.

    Raises:
        ValueError: If the augmentation strength is invalid.
    """
    if aug_strength == "trivial":
        augs = [tf.image.stateless_random_flip_left_right]
    elif aug_strength == "weak":
        augs = [tf.image.stateless_random_flip_left_right, random_pixel_shifts]
    elif aug_strength == "medium":
        augs = [
            tf.image.stateless_random_flip_left_right,
            random_pixel_shifts,
            stateless_random_rotate,
        ]
    elif aug_strength == "strong":
        augs = [
            tf.image.stateless_random_flip_left_right,
            random_pixel_shifts,
            stateless_random_rotate,
            partial(tf.image.stateless_random_contrast, lower=0.0, upper=1.0),
            partial(tf.image.stateless_random_brightness, max_delta=0.25),
        ]
    elif aug_strength == "diffusion":
        augs = [
        partial(stateless_random_rotate, max_degrees=5),
        partial(tf.image.stateless_random_contrast, lower=0.0, upper=0.5),
        partial(tf.image.stateless_random_brightness, max_delta=0.1),
        ]
    elif aug_strength == "rotate_constrast_brightness":
        augs = [
            stateless_random_rotate,
            partial(tf.image.stateless_random_contrast, lower=0.0, upper=0.5),
            partial(tf.image.stateless_random_brightness, max_delta=0.1),
        ]
    elif aug_strength == "rotate":
        augs = [stateless_random_rotate]
    elif aug_strength == "none":
        augs = []
    else:
        raise ValueError("Invalid augmentation strength.")
    random_aug_fn = create_random_aug_fn(
        aug_fn_list=augs, rng=tf.random.Generator.from_seed(420, alg="philox")
    )
    return random_aug_fn


# Custom cosine learning rate decay (Carlini et al., 2021)
class MyCosineDecay(LearningRateSchedule):
    """
    Custom cosine learning rate decay schedule.

    Args:
        base_lr (float): The base learning rate.
        steps (int): The total number of steps.
        name (str, optional): The name of the schedule. Defaults to "CosineDecay".
        relative_lr_warmup_steps (float, optional): The relative learning rate warmup steps. Defaults to 0.025.
    """

    def __init__(
        self,
        base_lr: float,
        steps: int,
        name: str = "CosineDecay",
        relative_lr_warmup_steps: float = 0.025,
    ):
        super().__init__()

        self.base_lr = base_lr
        self.steps = steps
        self.name = name
        self.warmup_factor = ops.convert_to_tensor(1 / relative_lr_warmup_steps)
        print(
            f"... custom cosine lr schedule: warmup steps={int(relative_lr_warmup_steps*steps)}, total decay steps={steps}"
        )
        if self.steps <= 0:
            raise ValueError(
                "Argument `steps` must be > 0. " f"Received: steps={self.steps}"
            )

    def _decay_function(
        self, step: int, steps: int, decay_from_lr: float, dtype=tf.float32
    ):
        """
        Compute the decayed learning rate.

        Args:
            step (int): The current step.
            steps (int): The total number of steps.
            decay_from_lr (float): The initial learning rate.
            dtype: The data type.

        Returns:
            tf.Tensor: The decayed learning rate.
        """
        completed_fraction = step / steps
        completed_fraction = ops.clip(completed_fraction, 0, 1)
        pi = ops.cast(np.pi, dtype=dtype)
        cosine_decayed = decay_from_lr * ops.cos(
            completed_fraction * (7 * pi) / (2 * 8)
        )
        cosine_decayed = cosine_decayed * ops.clip(
            completed_fraction * self.warmup_factor, 0, 1
        )  # modified so that linear lr warmup is slighlty longer
        return cosine_decayed

    def __call__(self, step: int):
        """
        Call the learning rate schedule.

        Args:
            step (int): The current step.

        Returns:
            tf.Tensor: The learning rate for the current step.
        """
        initial_learning_rate = ops.convert_to_tensor(self.base_lr)
        dtype = initial_learning_rate.dtype
        steps = ops.cast(self.steps, dtype)
        global_step_recomp = ops.cast(step, dtype)
        decayed_lr = self._decay_function(
            global_step_recomp, steps, initial_learning_rate, dtype
        )
        return decayed_lr

    def get_config(self):
        """
        Get the configuration of the learning rate schedule.

        Returns:
            dict: The configuration dictionary.
        """
        return {
            "base_lr": self.base_lr,
            "steps": self.steps,
            "name": self.name,
            "relative_lr_warmup_steps": self.warmup_factor,
        }


class WandbLogger(keras.callbacks.Callback):
    """
    Callback for efficiently logging performance metrics and learning rate to wandb during training.

    Args:
        model (keras.Model): The Keras model.
    """

    def __init__(self, model: keras.Model):
        super().__init__()
        assert isinstance(model, keras.Model)
        self.optimizer = model.optimizer

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """
        Log metrics and learning rate to wandb at the end of an epoch.

        Args:
            epoch (int): The current epoch.
            logs (Optional[dict], optional): The logs dictionary. Defaults to None.
        """
        lr = self.optimizer.learning_rate
        to_log = {"lr": lr} if logs is None else {**logs, "lr": lr._value}
        try:
            wandb.log(to_log, commit=True)
        except Exception as e:
            print(
                f"WARNING: ecountered Exception while trying to log wandb results: \n{to_log} \n {e}"
            )


def grayscale_to_rgb(img: np.ndarray):
    """
    Preprocess a grayscale image for fine-tuning a pre-trained ImageNet model.

    Args:
        img (np.ndarray): Grayscale image in [-1, 1] range.

    Returns:
        np.ndarray: Three channel image preprocessed with ImageNet normalization.
    """
    IMAGENET_MEAN = 2 * np.array([0.485, 0.456, 0.406]) - np.ones(
        3
    )  # since the normal IMAGENET_MEAN is for RGB images in [0, 1] range
    IMAGENET_STD = 2 * np.array(
        [0.229, 0.224, 0.225]
    )  # scaling factor is constant so standard deviation remains the same
    img = keras.ops.repeat(img, 3, axis=-1)  # repeat color channel
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img
