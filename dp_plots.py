import json
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt #type: ignore
import numpy as np
import scipy #type: ignore
from absl import app, flags #type: ignore
from scipy.stats import norm #type: ignore
from dp_accounting.pld import privacy_loss_distribution #type: ignore

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
from convenience_utils import get_data_root

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
flags.DEFINE_string("dataset_name", "embed", "Name of the dataset to plot data for ('ptb-xl' or 'embed').")
flags.DEFINE_integer("r_seed", 21, "Random seed.")
flags.DEFINE_float(
    "ylim_upper", 0.95, "upper y-limit for test performance metric plot"
)
flags.DEFINE_float("ylim_lower", 0.75, "lower y-limit for test performance metric plot")
flags.DEFINE_boolean("patient_level", True, "Whether to plot patient-level MIA AUCs or record-level AUCs.")

PTB_LOG_DIRS = {
    "eps1": "./logs/ptb-xl/dp/eps1",
    "eps10": "./logs/ptb-xl/dp/eps10",
    "eps100": "./logs/ptb-xl/dp/eps100",
    "eps1000": "./logs/ptb-xl/dp/eps1000",
    "epsinf": "./logs/ptb-xl/dp/epsinf",
}
EMBED_LOG_DIRS = {
    "eps1": "./logs/embed/dp/eps1",
    "eps10": "./logs/embed/dp/eps10",
    "eps100": "./logs/embed/dp/eps100",
    "eps1000": "./logs/embed/dp/eps1000",
    "epsinf": "./logs/embed/dp/epsinf_clip",
}
COLORS = {
    "eps1": "#FFE19C",
    "eps10": "#B28B84",
    "eps100": "#A3CEF1",
    "eps1000": "#274C77",
    "epsinf": "#74011E",
}
def compute_mu_uniform(
    steps: int, noise_multiplier: float, sample_rate: float
) -> float:
    c = sample_rate * np.sqrt(steps)
    return (
        np.sqrt(2)
        * c
        * np.sqrt(
            np.exp(noise_multiplier ** (-2)) * norm.cdf(1.5 / noise_multiplier)
            + 3 * norm.cdf(-0.5 / noise_multiplier)
            - 2
        )
    )

def calculate_auc_from_mu(mu:float) -> float:
    auc =norm.cdf(
        (mu) / np.sqrt(1 + 1)
    )  
    return auc

def get_privacy_profile(eps:np.ndarray, sigma:float, p:float, steps:int, group_size:int=1)-> np.ndarray:
    """
    See page 12 https://doi.org/10.29012/jpc.726
    """
    mech = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=sigma,
        sampling_prob=p,
        sensitivity=group_size,
    ).self_compose(steps)
    return mech.get_delta_for_epsilon(eps)

def profile2tradeoff(alpha, eps, delta):
    "Equation (5) in https://arxiv.org/pdf/1905.02383"
    term1 = 1.0 - delta - np.exp(eps)*alpha
    term2 = (1.0 - delta - alpha)*np.exp(-eps)
    return np.maximum(0.0, np.maximum(term1, term2))

def ADP2fDP(alphas, epsilons, deltas):
    "Privacy Profile to Trade-off Function"
    betas = []
    for alpha in alphas:
        B = profile2tradeoff(alpha, np.asarray(epsilons), np.asarray(deltas))
        betas.append(np.max(B))
    return np.array(betas)

def compute_auc(alphas, betas, plot:bool=True, name:str=""):
    powers = 1 - betas  # TPR@alpha
    auc = np.trapezoid(powers, alphas)
    if plot:
        plt.figure()
        plt.plot(alphas, powers)
        plt.xlabel("alpha")
        plt.ylabel("TPR@alpha")
        plt.plot([0, 1],[0, 1],color="black",linestyle="--",alpha=0.5)
        plt.title(f"AUC={auc:.3f}")
        plt.savefig(f"./figs/tradeoff_auc_{name}.pdf", bbox_inches="tight")
    return auc

def calculate_dp_bound(logdir: Path, group_size=1):
    valid_dirs = [l for l in logdir.iterdir() if l.is_dir()]
    with open(valid_dirs[0] / "info.json", "r") as f:
        config = json.load(f)["wandb_config"]
        if config["dp"] == False:
            return 1.0
        epsilon = config["epsilon"]
        train_size = config["train_size"]
        batch_size = config["batch_size"]
        noise_multiplier = config["noise_multiplier"]
        steps = config["train_steps"]
    sample_rate = batch_size / train_size
    # for small epsilon we use the privacy profile to compute the AUC bound
    if epsilon <= 20:
        alphas = np.linspace(0, 1, 10_000)
        epsila = np.linspace(-10, 10, 5000)
        pprofile = get_privacy_profile(eps=epsila, sigma=noise_multiplier, p=sample_rate, steps=steps, group_size=group_size)
        betas = ADP2fDP(alphas, epsila, pprofile)
        auc_bound = compute_auc(alphas, betas, name=f"eps={epsilon}", plot=False)
        print(f"\n******** Computed AUC bound using privacy profile: AUC={auc_bound:.3f} for (eps,delta)-DP with eps={epsilon:.3f}, delta={1/train_size:.1e}\n")
    # otherwise we compute a cheaper bound using mu-GDP and a CLT assumption which is usually tight enough for large epsilon
    else:
        mu = compute_mu_uniform(
            steps=steps, noise_multiplier=noise_multiplier, sample_rate=sample_rate
        )
        auc_bound = calculate_auc_from_mu(mu)
        print(f"\n******** Computed AUC bound using mu-GDP: AUC={auc_bound:.3f} for (eps,delta)-DP with eps={epsilon:.3f}, delta={1/train_size:.1e}\n")
    return auc_bound


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

def get_label(model_name: str) -> str:
    if model_name == "eps1":
        return r"$\varepsilon=10^0$"
    elif model_name == "eps5":
        return r"$\varepsilon=5$"
    elif model_name == "eps10":
        return r"$\varepsilon=10^1$"
    elif model_name == "eps100":
        return r"$\varepsilon=10^2$"
    elif model_name == "eps1000":
        return r"$\varepsilon=10^3$"
    elif model_name == "epsinf":
        return r"$\varepsilon=\infty $"
    else:
        raise ValueError(f"Unknown model name {model_name}")

def get_eps_val(model_name: str) -> str:
    if model_name == "eps1":
        return r"$10^0$"
    elif model_name == "eps5":
        return r"$5$"
    elif model_name == "eps10":
        return r"$10^1$"
    elif model_name == "eps100":
        return r"$10^2$"
    elif model_name == "eps1000":
        return r"$10^3$"
    elif model_name == "epsinf":
        return r"$\infty $"
    else:
        raise ValueError(f"Unknown model name {model_name}")

def main(argv):
    np.random.seed(FLAGS.r_seed)
    out_dir = Path(f"./figs/{FLAGS.dataset_name}")
    train_dataset, _ = get_dataset(
        dataset_name=FLAGS.dataset_name,
        img_size=[64, 64],
        csv_root=Path(FLAGS.csv_root),
        save_root=Path(FLAGS.save_root),
        data_root=get_data_root(FLAGS.dataset_name),
    )
    LOG_DIRS = PTB_LOG_DIRS if FLAGS.dataset_name == "ptb-xl" else EMBED_LOG_DIRS
    test_metric_name = (
        "_val_macro_auroc" if FLAGS.dataset_name == "ptb-xl" else "_val_auc"
    )
    logit_transform_func = partial(
        loss_logit_transform_multiclass, is_mimic_or_chexpert=False
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
            f"... test {test_metric_name} for {model_name}: {np.mean(test_metric_vals):.3f} ± {np.std(test_metric_vals):.3f}"
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
        if FLAGS.patient_level:
            patient_col = get_patient_col(FLAGS.dataset_name)
            patient_ids = train_dataset.dataframe[patient_col]
            patient_auc_df = aggregate_by_patient(aucs=aucs, patient_ids=patient_ids)
            aucs = patient_auc_df["max"]
        if model_name == "epsinf":
            auc_bound = 1.0
        else:
            #group_size = train_dataset.dataframe.groupby(get_patient_col(FLAGS.dataset_name)).size().max()
            #auc_bound_group = calculate_dp_bound(base_dir, group_size=group_size)
            auc_bound_record = calculate_dp_bound(base_dir, group_size=1)
        res = scipy.stats.ecdf(aucs)
        conf = res.sf.confidence_interval(confidence_level=0.95)
        res.sf.plot(
            ax=axes[0],
            color=COLORS[model_name],
            label=f"{get_label(model_name)} ({np.mean(aucs):.2f})[{auc_bound_record:.2f}]",
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
    ticklabels = [get_eps_val(model_name) for model_name in LOG_DIRS.keys()]
    axes[1].set_xticklabels(ticklabels, fontsize=6)
    axes[1].set_xlabel(r"Privacy Budget ($\varepsilon$)")
    axes[1].set_ylabel("Diagnostic Performance")
    axes[1].grid(False)
    axes[1].set_ylim((FLAGS.ylim_lower, FLAGS.ylim_upper))
    #axes[0].legend(loc="upper right", framealpha=0.5)
    axes[0].set_xlim((0.475, 1.01))
    ylim = 1 / len(aucs)
    axes[0].set_ylim((ylim, 1))
    axes[0].spines[["right", "top"]].set_visible(False)
    axes[1].spines[["right", "top"]].set_visible(False)
    xlabel = "MIA AUC (Patient-level)" if FLAGS.patient_level else "MIA AUC (Record-level)"
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("1 - Cumulative Probability")
    axes[0].set_yscale("log")
    suffix = "patient_level" if FLAGS.patient_level else "record_level"
    plt.savefig(
        out_dir / f"{FLAGS.dataset_name}_dp_eSF_{suffix}.pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    app.run(main)
