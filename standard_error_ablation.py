from functools import partial
from pathlib import Path

import joblib #type: ignore
import matplotlib.pyplot as plt #type: ignore
import numpy as np
from absl import app, flags #type: ignore
import pandas as pd #type: ignore
from matplotlib.patches import Patch #type: ignore

from src.data_utils.dataset_factory import get_dataset
from src.data_utils.utils import get_dataset_str
from src.privacy_utils.common import load_score
from src.privacy_utils.lira import (
    compute_scores,
    loss_logit_transform_multiclass,
    loss_logit_transform_binary,
    perform_lira,
    preprocess_scores,
)
from src.privacy_utils.common import roc_analysis_from_gaussian_samplestats
from src.privacy_utils.plot_utils import (
    disease_group_plots,
    mia_auc_esf_plot,
    plot_images,
    subgroup_plots,
)
from src.privacy_utils.rmia import perform_rmia, rmia_transform
from src.privacy_utils.utils import (
    confidence_roc_plot,
    convert_patientmask_to_recordmask,
)
from src.train_utils.utils import get_label_mode
from convenience_utils import fig_dir_exists, get_patient_col, get_data_root

# force jax to use CPU
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

FONT_SIZE = 7
plt.style.use("default")
plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "figure.figsize": (2, 2),
        "font.family": "sans serif",
        "font.sans-serif": "Inter",
        "axes.grid": False,
        "grid.alpha": 0.1,
        "axes.axisbelow": True,
        "figure.constrained_layout.use": True,
        # "pdf.fonttype":42,
    }
)
color_a = "#7c8483"
color_b = "#982649"

LOGDIRS = {
    "chexpert": "./logs/chexpert/wrn_28_2",
    "mimic": "./logs/mimic/wrn_28_2",
    "mimic-iv-ed": "./logs/mimic-iv/resnet_300_6",
    "embed": "./logs/embed/wrn_28_2",
    "fairvision": "./logs/fairvision/wrn_28_2",
    "fitzpatrick": "./logs/fitzpatrick/wrn_28_2",
    "ptb-xl": "./logs/ptb-xl/resnet1d_128",
}

flags.DEFINE_list(
    "n_models_each",
    [10, 25, 50, 100],
    "Number of target models used to computed patient-level sucess estimates.",
)
flags.DEFINE_string(
    "csv_root", "./data/csv/", "The directory where the csv files can be found."
)
flags.DEFINE_string(
    "save_root",
    "/home/moritz/data_fast/npy",
    "The directory where the pre-computed .npy dataset files can be found.",
)
flags.DEFINE_boolean(
    "multiprocessing", True, "Whether to use multiprocessing for parallel computation."
)
flags.DEFINE_integer("threads", 16, "Number of threads to use for multiprocessing.")
flags.DEFINE_integer("seed", 21, "Random seed.")
FLAGS = flags.FLAGS


def main(argv):
    np.random.seed(FLAGS.seed)
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.75), sharey=True)
    # load empirical sampling distributions IN/OUT for all datasets and log directories
    for i, (dataset_name, logdir) in enumerate(LOGDIRS.items()):
        train_dataset, _ = get_dataset(
            dataset_name=dataset_name,
            img_size=[64, 64],  # not of relevance here
            csv_root=Path(FLAGS.csv_root),
            save_root=Path(FLAGS.save_root),
            data_root=get_data_root(dataset_name),
            load_from_disk=dataset_name != "ptb-xl",
        )
        is_mimic_or_chex = dataset_name in ["mimic", "chexpert"]
        is_binary = dataset_name == "mimic-iv-ed"
        logit_transform_func = partial(
            (
                loss_logit_transform_binary
                if is_binary
                else loss_logit_transform_multiclass
            ),
            is_mimic_or_chexpert=is_mimic_or_chex,
        )
        logit_load_func = partial(load_score, logit_transform_func=logit_transform_func)
        logdir = Path(logdir)
        assert logdir.exists(), f"Log directory {logdir} does not exist."
        assert logdir.is_dir(), f"Log directory {logdir} is not a directory."
        ldirs = [d for d in logdir.iterdir() if d.is_dir()]
        scores, masks, _ = compute_scores(
            log_dirs=ldirs,
            load_score_func=logit_load_func,
            multi_processing=FLAGS.multiprocessing,
            threads=FLAGS.threads,
        )
        masks = convert_patientmask_to_recordmask(
            patient_masks=masks,
            patient_ids=train_dataset.dataframe[get_patient_col(dataset_name)].unique(),
            dataset=train_dataset,
            patient_id_col=get_patient_col(dataset_name),
        )
        assert scores.shape[1] == len(
            train_dataset
        ), f"Shape mismatch: {scores.shape} vs {len(train_dataset)}"
        assert masks.shape[1] == len(
            train_dataset
        ), f"Shape mismatch: {masks.shape} vs {len(train_dataset)}"
        print(
            f"scores: shape={scores.shape} mean={scores.mean():.3f} max={scores.max():.3f} min={scores.min():.3f}, masks: shape={masks.shape}"
        )
        print(
            "---------------------------------------------------------------------------"
        )
        in_scores, out_scores = preprocess_scores(
            scores=scores,
            masks=masks,
        )
        plot_data = []
        plot_data_nine_nine = []
        for n in FLAGS.n_models_each:
            # select random subset of n target models
            in_idcs = np.random.choice(
                range(in_scores.shape[0]),
                size=min(n, in_scores.shape[0]),
                replace=False,
            )
            in_score_subset = in_scores[in_idcs, :]
            out_idcs = np.random.choice(
                range(out_scores.shape[0]),
                size=min(n, out_scores.shape[0]),
                replace=False,
            )
            out_score_subset = out_scores[out_idcs, :]
            _, _, aucs, se_aucs = roc_analysis_from_gaussian_samplestats(
                mean_in=np.mean(in_score_subset, axis=0),
                mean_out=np.mean(out_score_subset, axis=0),
                std_in=np.std(in_score_subset, axis=0),
                std_out=np.std(out_score_subset, axis=0),
                N_in=in_score_subset.shape[0],
                N_out=out_score_subset.shape[0],
                verbose=False,
                eps=1e-10,
            )
            # standard error for all estimated AUCs
            plot_data.append(se_aucs)
            # get indices of records in the 99th record-leve MIA AUC percentile
            auc_cutoff = np.percentile(aucs, 99)
            # standard error for records in the 99th percentile
            plot_data_nine_nine.append(se_aucs[aucs >= auc_cutoff])
        positions=np.arange(len(FLAGS.n_models_each))
        axs.flat[i].axhline(y=0.05, color="black", linestyle="--", linewidth=1, alpha=0.5, zorder=0)
        axs.flat[i].boxplot(
            plot_data,
            tick_labels=None,
            positions=positions-0.15,
            medianprops={'color': 'black'},
            showfliers=False,
            zorder=1,
        )
        bplot = axs.flat[i].boxplot(
            plot_data_nine_nine,
            tick_labels=None,
            patch_artist=True,
            positions=positions+0.15,
            medianprops={'color': 'black'},
            showfliers=False,
            zorder=1,
        )
        # compute IQRs
        iqr = [(np.percentile(data, 75)-np.percentile(data, 25)) for data in plot_data]
        iqr_nine_nine = [(np.percentile(data, 75)-np.percentile(data, 25)) for data in plot_data_nine_nine]
        # plot small colored dot above boxes where the box color is otherwise not visible because the box (IQR) is too small
        for j, val in enumerate(iqr):
            if val < 0.004:  # if the box is too small, plot a dot above the upper whisker
                height = np.median(plot_data[j]) + 0.01
                axs.flat[i].plot(
                    positions[j] - 0.15,
                    height,
                    marker="s",
                    color="white",
                    markeredgecolor="black",
                    markersize=3,
                )
        for j, val in enumerate(iqr_nine_nine):
            if val < 0.004:  # if the box is too small, plot a dot above the upper whisker
                height = np.median(plot_data_nine_nine[j]) + 0.01
                axs.flat[i].plot(
                    positions[j] + 0.15,
                    height,
                    marker="s",
                    color="#fb887f",
                    markeredgecolor="black",
                    markersize=3,
                )
        for patch in bplot['boxes']:
            patch.set_facecolor("#fb887f")
        axs.flat[i].set_title(
            get_dataset_str(dataset_name),
            fontsize=FONT_SIZE,
        )
        fig.supylabel("Standard Error", fontsize=FONT_SIZE)
        fig.supxlabel("Number of Target Models", fontsize=FONT_SIZE)
        axs.flat[i].set_xticks(positions, labels=[n*2 for n in FLAGS.n_models_each])
        axs.flat[i].spines[["right", "top"]].set_visible(False)
        #axs.flat[i].set_ylim(-0.02, 0.15)
    axs.flat[-1].axis("off")
    axs.flat[-1].legend(
        handles=[
            Patch(facecolor="#fb887f",edgecolor="black", label="99th Percentile"),
            Patch(facecolor="white", edgecolor="black", label="All Records"),
        ],
        loc="upper right",
        fontsize=FONT_SIZE,
        frameon=False,
    )
    plt.savefig("./figs/standard_error_ablation.pdf", bbox_inches="tight")


if __name__ == "__main__":
    app.run(main)
