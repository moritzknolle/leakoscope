from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from sklearn.metrics import roc_auc_score, roc_curve

from convenience_utils import get_color
from src.privacy_utils.common import load_score
from src.privacy_utils.lira import compute_scores
from src.privacy_utils.rmia import perform_rmia, rmia_transform
from src.train_utils.utils import get_label_mode

FONT_SIZE = 6
plt.style.use("default")
plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "figure.figsize": (2.2, 2.0),
        "font.family": "sans serif",
        "font.sans-serif": "Inter",
        "axes.grid": False,
        "grid.alpha": 0.1,
        "axes.axisbelow": True,
        # "figure.constrained_layout.use": True,
        # "pdf.fonttype":42,
    }
)
color_a = "#3b3f3f"
color_b = "#982649"


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "reference_model_logdir",
    "./logs/torchxrayvision/reference",
    "Path to the attack model log directory.",
)
flags.DEFINE_string(
    "chex_model_logdir",
    "./logs/torchxrayvision/chexpert",
    "Path to the target model log directory.",
)
flags.DEFINE_string(
    "mimic_model_logdir",
    "./logs/torchxrayvision/mimic",
    "Path to the target model log directory.",
)
flags.DEFINE_string(
    "out_dir",
    "./figs/torchxrayvision",
    "Path to the target model log directory.",
)
flags.DEFINE_boolean("multiprocessing", True, "Use multiprocessing.")
flags.DEFINE_integer("threads", 8, "Number of threads.")


def main(argv):
    rmia_transform_func = partial(rmia_transform, is_mimic_or_chexpert=True)
    rmia_load_func = partial(
        load_score, logit_transform_func=rmia_transform_func, verbose=True
    )
    assert Path(
        FLAGS.reference_model_logdir
    ).is_dir(), f"{FLAGS.reference_model_logdir} is not a directory."
    assert Path(
        FLAGS.chex_model_logdir
    ).is_dir(), f"{FLAGS.chex_model_logdir} is not a directory."
    assert Path(
        FLAGS.mimic_model_logdir
    ).is_dir(), f"{FLAGS.mimic_model_logdir} is not a directory."
    reference_scores, reference_mask, _ = compute_scores(
        log_dirs=[Path(FLAGS.reference_model_logdir)],
        load_score_func=rmia_load_func,
        multi_processing=FLAGS.multiprocessing,
        threads=FLAGS.threads,
    )
    chexpert_model_scores, chexpert_mask, _ = compute_scores(
        log_dirs=[Path(FLAGS.chex_model_logdir)],
        load_score_func=rmia_load_func,
        multi_processing=FLAGS.multiprocessing,
        threads=FLAGS.threads,
    )
    mimic_model_scores, mimic_mask, _ = compute_scores(
        log_dirs=[Path(FLAGS.mimic_model_logdir)],
        load_score_func=rmia_load_func,
        multi_processing=FLAGS.multiprocessing,
        threads=FLAGS.threads,
    )
    # add augmentation axis
    reference_scores = np.expand_dims(reference_scores, axis=-1)
    chexpert_model_scores = np.expand_dims(chexpert_model_scores, axis=-1)
    mimic_model_scores = np.expand_dims(mimic_model_scores, axis=-1)
    print(
        "reference scores:",
        reference_scores.shape,
        "reference mask:",
        reference_mask.shape,
    )
    print(
        "chexpert scores:",
        chexpert_model_scores.shape,
        "target mask:",
        chexpert_mask.shape,
    )
    print("mimic scores:", mimic_model_scores.shape, "target mask:", mimic_mask.shape)
    label_mode = get_label_mode("multiclass")
    rmia_preds_chex, rmia_targets_chex = perform_rmia(
        train_scores=reference_scores,
        train_masks=reference_mask,
        test_scores=chexpert_model_scores,
        test_masks=chexpert_mask,
        find_optimal_a=True,
        offline=True,
        n_pop=100,
        a=0.8,
        label_type=label_mode,
        labels=None,
        multi_processing=FLAGS.multiprocessing,
        threads=FLAGS.threads,
    )
    rmia_preds_mimic, rmia_targets_mimic = perform_rmia(
        train_scores=reference_scores,
        train_masks=reference_mask,
        test_scores=mimic_model_scores,
        test_masks=~mimic_mask,
        find_optimal_a=True,
        offline=True,
        n_pop=100,
        a=0.8,
        label_type=label_mode,
        labels=None,
        multi_processing=FLAGS.multiprocessing,
        threads=FLAGS.threads,
    )
    rmia_preds_chex = rmia_preds_chex.squeeze()
    rmia_targets_chex = rmia_targets_chex.squeeze()
    rmia_preds_mimic = rmia_preds_mimic.squeeze()
    rmia_targets_mimic = rmia_targets_mimic.squeeze()
    fig, ax = plt.subplots(layout="tight")
    fpr_chex, tpr_chex, _ = roc_curve(rmia_targets_chex, rmia_preds_chex)
    auc_chex = roc_auc_score(rmia_targets_chex, rmia_preds_chex)
    fpr_mimic, tpr_mimic, _ = roc_curve(rmia_targets_mimic, rmia_preds_mimic)
    auc_mimic = roc_auc_score(rmia_targets_mimic, rmia_preds_mimic)
    ax.plot(
        fpr_chex,
        tpr_chex,
        label=f"CheXpert ({auc_chex:.2f})",
        color=get_color("chexpert"),
    )
    ax.plot(
        fpr_mimic,
        tpr_mimic,
        label=f"MIMIC-CXR ({auc_mimic:.2f})",
        color=get_color("mimic"),
    )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
    ax.spines[["right", "top"]].set_visible(False)
    lin_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xticks(lin_ticks)
    ax.set_yticks(lin_ticks)
    #ax.legend(loc="lower right")  # bbox_to_anchor=(0.3, 1.22)
    outdir = Path(FLAGS.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "roc_curve.pdf", bbox_inches="tight")
    log_ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([1e-5, 1])
    ax.set_ylim([1e-5, 1])
    ax.set_xticks(log_ticks)
    ax.set_yticks(log_ticks)
    plt.savefig(outdir / "roc_curve_log.pdf", bbox_inches="tight")


if __name__ == "__main__":
    app.run(main)
