import json
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt #type: ignore
import numpy as np
import scipy
from absl import app, flags

from mem_inf_stats import get_patient_col
from src.data_utils.dataset_factory import get_dataset
from src.privacy_utils.common import aggregate_by_patient
from src.privacy_utils.common import load_score
from src.privacy_utils.utils import convert_patientmask_to_recordmask
from src.privacy_utils.lira import (
    compute_scores,
    loss_logit_transform_multiclass,
)
from src.privacy_utils.common import record_MIA_ROC_analysis

FONT_SIZE = 6
plt.style.use("default")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force jax to use CPU
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
        # "pdf.fonttype": 42, # embed fonts in pdf
    }
)
FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_name", "fitzpatrick", "Name of the dataset to plot data for ('chexpert' or 'fitzpatrick').")
flags.DEFINE_integer("r_seed", 21, "Random seed.")
flags.DEFINE_float(
    "ylim_upper", 0.935, "upper y-limit for test performance metric plot"
)
flags.DEFINE_float("ylim_lower", 0.8, "lower y-limit for test performance metric plot")

FITZ_LOG_DIRS = {
    "WRN-28-2": "./logs/fitzpatrick/wrn_28_2",
    "WRN-40-4": "./logs/fitzpatrick/wrn_40_4",
    "ViT-B/16-64": "./logs/fitzpatrick/vit_b_16",
    "ViT-B/16-128": "./logs/fitzpatrick/vit_b_16_128x128",
    "ViT-L/16-64": "./logs/fitzpatrick/vit_l_16",
}
COLORS = {
    "WRN-28-2": "#F3BAE6",
    "WRN-40-4": "#4C0C54",
    "ViT-B/16-64": "#E5DA70",
    "ViT-B/16-128": "#E47E34",
    "ViT-L/16-64": "#522D31",
}
CHEX_LOGDIRS = {
    "WRN-28-2": "./logs/chexpert/wrn_28_2",
    "WRN-40-4": "./logs/chexpert/wrn_40_4",
    "ViT-B/16-64": "./logs/chexpert/vit_b_16",
    "ViT-B/16-128": "./logs/chexpert/vit_b_16_128x128",
    "ViT-L/16-64": "./logs/chexpert/vit_l_16",
}

MODEL_SIZE = {
    "WRN-28-2": "1.5",
    "WRN-40-4": "9",
    "ViT-B/16-64": "86",
    "ViT-B/16-128": "86",
    "ViT-L/16-64": "303",
}


def get_perf_from_json(log_dir: Path, metric_name: str = "accuracy"):
    metric_vals = []
    valid_dirs = [l for l in log_dir.iterdir() if l.is_dir()]
    for subdir in valid_dirs:
        try:
            with open(subdir / "info.json", "r") as f:
                metrics = json.load(f)
                test_metrics = metrics["test_metrics"]
                for m_name, _ in test_metrics.items():
                    contains_numb = any(char.isdigit() for char in metric_name)
                    if contains_numb:
                        m_name_fixed = "_".join(
                            m_name.split("_")[-1]
                        )  # get rid off annoying number suffix that keras sometimes adds to metric names
                        metrics["test_metrics"][m_name_fixed] = metrics[
                            "test_metrics"
                        ].pop(m_name)
                metric_vals.append(metrics["test_metrics"][metric_name])
        except Exception as e:
            print(f"Encountered error loading {metric_name} from info.json: {e}")
            continue
    assert len(metric_vals) == len(
        valid_dirs
    ), f"Found {len(metric_vals)} values for {metric_name} but expected {len(valid_dirs)} \n found keys: {test_metrics.keys()}"
    return metric_vals


def main(argv):
    np.random.seed(FLAGS.r_seed)
    out_dir = Path(f"./figs/{FLAGS.dataset_name}")
    train_dataset, _ = get_dataset(
        dataset_name=FLAGS.dataset_name,
        img_size=[64, 64],
        csv_root=Path(FLAGS.csv_root),
        save_root=Path(FLAGS.save_root),
        data_root=Path(""),
    )
    LOG_DIRS = CHEX_LOGDIRS if FLAGS.dataset_name == "chexpert" else FITZ_LOG_DIRS
    test_metric_name = (
        "_val_macro_auroc(cxp)" if FLAGS.dataset_name == "chexpert" else "_val_auroc"
    )
    is_mimic_or_chexpert = FLAGS.dataset_name == "chexpert"
    logit_transform_func = partial(
        loss_logit_transform_multiclass, is_mimic_or_chexpert=is_mimic_or_chexpert
    )
    load_func = partial(load_score, logit_transform_func=logit_transform_func)
    fig, axes = plt.subplots(
        1, 2, figsize=(3.4, 1.8), layout="tight", width_ratios=[1.45, 1.1]
    )
    test_metric_means, test_metric_stds = [], []
    for model_name, base_dir in LOG_DIRS.items():
        base_dir = Path(base_dir)
        logdirs = [d for d in base_dir.iterdir()]
        test_metric_vals = get_perf_from_json(base_dir, metric_name=test_metric_name)
        test_metric_means.append(np.mean(test_metric_vals))
        test_metric_stds.append(np.std(test_metric_vals))
        print(
            f"... average test {test_metric_name} for {model_name}: {np.mean(test_metric_vals)}"
        )
        scores, masks, _ = compute_scores(
            log_dirs=logdirs,
            load_score_func=load_func,
            multi_processing=True,
            threads=16,
        )
        masks = convert_patientmask_to_recordmask(
            patient_masks=masks,
            patient_ids=train_dataset.dataframe[
                get_patient_col(FLAGS.dataset_name)
            ].unique(),
            dataset=train_dataset,
            patient_id_col=get_patient_col(FLAGS.dataset_name),
        )
        _, _, aucs, _ = record_MIA_ROC_analysis(scores=scores, masks=masks)
        if FLAGS.dataset_name == "chexpert":
            patient_col = get_patient_col(FLAGS.dataset_name)
            patient_ids = train_dataset.dataframe[patient_col]
            patient_auc_df = aggregate_by_patient(aucs=aucs, patient_ids=patient_ids)
            aucs = patient_auc_df["max"]
        res = scipy.stats.ecdf(aucs)
        conf = res.sf.confidence_interval(confidence_level=0.95)
        res.sf.plot(
            ax=axes[0],
            color=COLORS[model_name],
            label=f"{model_name}({np.mean(aucs):.2f})",
            alpha=0.8,
        )
        conf.low.plot(
            ax=axes[0], color=COLORS[model_name], linestyle="--", lw=1, alpha=0.8
        )
        conf.high.plot(
            ax=axes[0], color=COLORS[model_name], linestyle="--", lw=1, alpha=0.8
        )
    axes[1].bar(
        range(len(LOG_DIRS)),
        test_metric_means,
        color=[COLORS[model_name] for model_name in LOG_DIRS.keys()],
        yerr=test_metric_stds,
        capsize=2,
    )
    axes[1].set_xticks(range(len(LOG_DIRS)))
    # axes[1].set_xticklabels(LOG_DIRS.keys(), rotation=45)
    axes[1].set_xticklabels([MODEL_SIZE[model_name] for model_name in LOG_DIRS.keys()])
    #axes[1].ticklabel_format(useMathText=True)
    axes[1].set_xlabel("Model Size\n(Million Parameters)")
    axes[1].set_ylabel("Diagnostic Performance")
    axes[1].grid(False)
    axes[1].set_ylim((FLAGS.ylim_lower, FLAGS.ylim_upper))
    #axes[0].legend(loc="upper right", framealpha=0.5)
    axes[0].set_xlim((0.475, 1.01))
    ylim = 1 / len(aucs)
    axes[0].set_ylim((ylim, 1))
    axes[0].spines[["right", "top"]].set_visible(False)
    axes[1].spines[["right", "top"]].set_visible(False)
    xlabel = "MIA AUC (Patient-level)"
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("1 - Cumulative Probability")
    axes[0].set_yscale("log")
    plt.savefig(
        out_dir / f"{FLAGS.dataset_name}_model_scaling_eSF.pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    app.run(main)
