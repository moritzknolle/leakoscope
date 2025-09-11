import os
from pathlib import Path

# set keras backend to jax and enable compilation caching
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
import keras  # type: ignore
import numpy as np
from absl import app, flags  # type: ignore

from src.data_utils.dataset_factory import get_dataset
from src.train_utils.models.model_factory import get_model
from src.train_utils.training import train_and_eval, train_random_subset
from src.train_utils.utils import MyCosineDecay, get_aug_fn, grayscale_to_rgb

FLAGS = flags.FLAGS
flags.DEFINE_integer("epochs", 30, "Number of training steps.")
flags.DEFINE_float("lr", 1e-1, "Learning rate.")
flags.DEFINE_float("weight_decay", 1e-8, "L2 weight decay.")
flags.DEFINE_integer("batch_size", 512, "Batch size.")
flags.DEFINE_float("momentum", 0.9, "Momentum parameter.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("log_wandb", True, "Whether to log metrics to weights & biases.")
flags.DEFINE_boolean(
    "ema", True, "Whether to use exponential moving average for parameters."
)
flags.DEFINE_string("model", "wrn_28_2", "Name of the model to use.")
flags.DEFINE_enum("lr_schedule", "cosine", ["constant", "cosine"], "LR schedule.")
flags.DEFINE_float(
    "lr_warmup", 0.1, "Relative steps to perform linear learning rate warmup."
)
flags.DEFINE_float(
    "decay_steps",
    1.0,
    "Relative fraction of total steps until learning rate is decayed to 1/10 times the original value. A value smaller than one means faster decay and likewise a bigger value leads to slower decay.",
)
flags.DEFINE_float("ema_decay", 0.995, "EMA decay.")
flags.DEFINE_integer("grad_accum_steps", 2, "Number of gradient accumulation steps.")
flags.DEFINE_float("dropout", 0.0, "Dropout rate.")
flags.DEFINE_string(
    "augment",
    "rotate_constrast_brightness",
    "What type of data augmentations strength to apply.",
)
flags.DEFINE_list("img_size", [64, 48], "Image size.")
flags.DEFINE_boolean(
    "mixed_precision",
    True,
    "Whether to perform mixed precision training to reduce training time.",
)
flags.DEFINE_bool("full_dataset", True, "Whether to train and evaluate on full dataset (True) or train on random subset for privacy auditing (False).")
flags.DEFINE_integer(
    "n_runs", 200, "Number of leave-many-out re-training runs to perform."
)
flags.DEFINE_float(
    "subset_ratio", 0.5, "Ratio of the training data to use for each re-training run."
)
flags.DEFINE_integer(
    "eval_views", 16, "Number of augmentations to query when saving train/test logits."
)
flags.DEFINE_string(
    "ckpt_file_path",
    "./tmp/ckpts/",
    "Path to root folder where the model checkpoint files are stored.",
)
flags.DEFINE_string(
    "save_root",
    "/home/moritz/data_fast/npy",
    "Path to root folder where the memmap files are stored.",
)
flags.DEFINE_string(
    "logdir",
    "./logs/embed/",
    "Path to logdir.",
)


def get_compiled_model(train_steps: int, num_classes: int = 4):
    # create model, lr schedule and optimizer
    model = get_model(
        model_name=FLAGS.model,
        input_shape=(int(FLAGS.img_size[0]), int(FLAGS.img_size[1]), 1),
        num_classes=num_classes,
        dropout=FLAGS.dropout,
    )
    schedule = MyCosineDecay(
        base_lr=FLAGS.lr,
        steps=int(FLAGS.decay_steps * train_steps),
        relative_lr_warmup_steps=FLAGS.lr_warmup,
    )
    opt = keras.optimizers.SGD(
        learning_rate=(
            schedule if FLAGS.lr_schedule == "cosine" else FLAGS.lr
        ),
        momentum=FLAGS.momentum,
        weight_decay=FLAGS.weight_decay,
        use_ema=FLAGS.ema,
        ema_momentum=FLAGS.ema_decay,
        gradient_accumulation_steps=FLAGS.grad_accum_steps,
    )
    # compile model
    model.compile(
        optimizer=opt,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.AUC(from_logits=True),
        ],
    )
    return model


def get_callbacks(is_ema: bool):
    callbacks = []
    if is_ema:
        callbacks += [keras.callbacks.SwapEMAWeights(swap_on_epoch=True)]
    return callbacks


def main(argv):
    if FLAGS.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    IMG_SIZE = [int(FLAGS.img_size[0]), int(FLAGS.img_size[1])]

    if FLAGS.full_dataset:
        train_dataset, test_dataset = get_dataset(
            dataset_name="embed",
            img_size=IMG_SIZE,
            csv_root=Path("./data/csv"),
            data_root=Path("/home/moritz/data_massive/embed_small/png/1024x768"),
            save_root=Path(FLAGS.save_root),
            get_numpy=True,
            load_from_disk=True,
            overwrite_existing=False,
        )
        STEPS = len(train_dataset) // FLAGS.batch_size * FLAGS.epochs
        model = get_compiled_model(train_steps=STEPS)
        _ = train_and_eval(
            compiled_model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=FLAGS.batch_size,
            aug_fn=get_aug_fn(FLAGS.augment),
            augment=True if FLAGS.augment != "None" else False,
            epochs=FLAGS.epochs,
            target_metric="val_auc",
            callbacks=get_callbacks(FLAGS.ema),
            ckpt_file_path=Path(FLAGS.ckpt_file_path),
            seed=FLAGS.seed,
            log_wandb=FLAGS.log_wandb,
            wandb_project_name="embed",
        )
        raise StopIteration("Evaluation only, stopping execution.")
    else:
        train_dataset, test_dataset = get_dataset(
            dataset_name="embed",
            img_size=IMG_SIZE,
            csv_root=Path("./data/csv"),
            data_root=Path(
                "/home/moritz/data_massive/embed_small/png/1024x768"
            ),
            save_root=Path(FLAGS.save_root),
            get_numpy=True,
            load_from_disk=True,
            overwrite_existing=False,
        )
        STEPS = (
            len(train_dataset)
            * FLAGS.subset_ratio
            // FLAGS.batch_size
            * FLAGS.epochs
        )
        model = get_compiled_model(train_steps=STEPS)
        train_random_subset(
            compiled_model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            patient_id_col="empi_anon",
            batch_size=FLAGS.batch_size,
            aug_fn=get_aug_fn(FLAGS.augment),
            augment=True if FLAGS.augment != "None" else False,
            epochs=FLAGS.epochs,
            seed=FLAGS.seed,
            target_metric="val_auc",
            logdir=Path(FLAGS.logdir),
            n_total_runs=FLAGS.n_runs,
            subset_ratio=FLAGS.subset_ratio,
            n_eval_views=FLAGS.eval_views if FLAGS.augment != "none" else 1,
            callbacks=get_callbacks(FLAGS.ema),
            ckpt_file_path=Path(FLAGS.ckpt_file_path),
            log_wandb=FLAGS.log_wandb,
            wandb_project_name="embed",
        )

if __name__ == "__main__":
    app.run(main)
