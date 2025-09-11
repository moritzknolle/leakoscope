from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from skimage import exposure
from mpl_toolkits.axes_grid1 import Divider, Size

from ..colors import CXR_COLORS, FITZPATRICK_COLORS
from ..data_utils.constants import (
    CXR_SHORT_LABELS,
    CXR_LABELS,
    EMBED_LABELS,
    EMBED_LABELS_SHORT,
    FAIRVISION_LABELS,
    FAIRVISION_LABELS_SHORT,
    FITZPATRICK_LABELS_COARSE,
    FITZPATRICK_LABELS_FINE,
    FITZPATRICK_LABELS_FINE_SHORT,
    FITZPATRICK_LABELS_COARSE_SHORT,
    EMBED_EXAM_LABELS,
    EMBED_EXAM_LABELS_BIRADS,
    ECG_LABELS,
)
from ..data_utils.datasets import BaseDataset
from .common import aggregate_by_patient

my_blue = "#03012d"


def mia_auc_esf_plot(
    aucs: np.ndarray,
    patient_ids: pd.Series = None,
    conf_level: float = 0.95,
    aggregation_mode: str = "max",
    save_path: Optional[str] = None,
    color: str = my_blue,
    alpha: float = 0.9,
    log_scale: bool = True,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Plot the empirical survival function of MIA AUCs.
    If patient_ids is provided, aggregate record-level MIA AUCs by patient ID.
        Args:
            aucs: np.ndarray, MIA AUCs
            patient_ids: pd.Series, patient IDs
            aggregation_mode: str, how to obtain patient-level scores from record-level ones. One of either max or mean.
            conf_level: float, confidence level
            save_path: str, path to save the plot
            alpha: float, alpha value for curves
            color: str, color of the plot
            label: str, optional legend label
        Returns:
            None
    """
    # maybe aggregate record-level MIA AUCs by patient ID
    if patient_ids is not None:
        if not aggregation_mode in ["max", "mean"]:
            raise ValueError(
                "Please provide a valid aggregation model ('max' or 'mean')."
            )
        patient_auc_df = aggregate_by_patient(aucs, patient_ids)
        aucs = patient_auc_df[aggregation_mode]
        print(
            f"... plotting MIA AUCS aggregated by patient ID: n_patients={len(aucs)}, n_records={len(patient_ids)}"
        )
    ylim = 1 / len(aucs)
    set_plot_style = False
    if ax is None:
        plt.figure()
        ax = plt.gca()
        set_plot_style = True
    res = scipy.stats.ecdf(aucs)
    conf = res.sf.confidence_interval(confidence_level=conf_level)
    res.sf.plot(ax=ax, color=color, label=label, alpha=alpha)
    conf.low.plot(ax=ax, color=color, linestyle="--", lw=1, alpha=0.8)
    conf.high.plot(ax=ax, color=color, linestyle="--", lw=1, alpha=0.8)
    if set_plot_style:
        plt.xlim((0.475, 1))
        plt.ylim((ylim, 1))
        metric_str = "Record-level" if patient_ids is None else "Patient-level"
        plt.xlabel(f"MIA AUC ({metric_str})")
        plt.ylabel("1 - Cumulative Probability")
        if log_scale:
            plt.semilogy()
    ax.spines[["right", "top"]].set_visible(False)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


def group_mia_auc_esf_plot(
    df: pd.DataFrame,
    group_col: str,
    mia_result_col: str,
    patient_id_col: str,
    color_dict: Dict,
    patient_level: bool = True,
    log_scale: bool = True,
    inset_xlim: Optional[Tuple[float, float]] = None,
    inset_ylim: Optional[Tuple[float, float]] = None,
    inset_params: Optional[List[float]] = None,
    figsize=(2.25, 2),
    save_path: Optional[Path] = None,
    ylim_min: Optional[float] = None,
    xlim_max: Optional[float] = None,
):
    """
    Plot the empirical survival function of MIA AUCs for different groups.

        Args:
            df: pd.DataFrame, data frame
            group_col: str, column name of the group
            mia_result_col: str, column name of the MIA result
            patient_id_col: str, column name of the patient ID
            color_dict: Dict, color dictionary
            save_path: str, path to save the plot
        Returns:
            None
    """
    assert (
        df[mia_result_col].isna().sum() == 0
    ), f"Missing values in {mia_result_col}, {df[mia_result_col].isna().sum()}"
    if figsize:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()
    axins = None
    smallest_group_n = 1e10
    for group, group_df in df.groupby(group_col):
        if group in color_dict:
            group_n = group_df[patient_id_col].nunique()
            if group_n < smallest_group_n:
                smallest_group_n = group_n
            group_color = color_dict[group]
            group_ids = group_df[patient_id_col]
            mia_auc_esf_plot(
                aucs=group_df[mia_result_col].values,
                patient_ids=group_ids,
                label=group,
                ax=ax,
                color=group_color,
            )
            # inset plot
            if inset_xlim and inset_ylim:
                if not axins:
                    axins = ax.inset_axes(inset_params, transform=ax.transAxes)
                mia_auc_esf_plot(
                    aucs=group_df[mia_result_col].values,
                    patient_ids=group_ids,
                    label=group,
                    ax=axins,
                    color=group_color,
                )
    if xlim_max is not None:
        ax.set_xlim((0.49, xlim_max))
    else:
        ax.set_xlim((0.49, ax.get_xlim()[1]))
    print(
        f"... found smallest group size: {smallest_group_n} setting ylim to 1/{smallest_group_n}"
    )
    ylim = 1 / smallest_group_n if ylim_min is None else ylim_min
    ax.set_ylim((ylim, 1))
    metric_str = "Patient-level" if patient_level else "Record-level"
    ax.set_xlabel(f"MIA AUC ({metric_str})")
    ax.set_ylabel("1 - Cumulative Probability")
    if log_scale:
        ax.set_yscale("log")

    if inset_xlim and inset_ylim:
        legend_loc = "lower left"
        axins.set_xlim(inset_xlim)
        axins.set_ylim(inset_ylim)
        axins.grid(visible=False)
        ax.indicate_inset_zoom(axins)
        axins.tick_params(axis="both", labelsize=5)
    else:
        legend_loc = "upper right"
    ax.legend(loc=legend_loc)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()


def get_sig_str(p_val: float, bonferroni_correction_factor:Optional[float]=None) -> str:
    thresholds = [0.05, 0.01, 0.001]
    if bonferroni_correction_factor is not None:
        thresholds = [t/bonferroni_correction_factor for t in thresholds]
    if p_val <= thresholds[0]:
        sig_str = "*"
        if p_val <= thresholds[1]:
            sig_str = "**"
            if p_val <= thresholds[2]:
                sig_str = "***"
    else:
        sig_str = "n.s"
    return sig_str


def composition_comparison_plot_overlapping_groups(
    df: pd.DataFrame,
    group_cols: List[str],
    xlabel: str,
    top_k_percent: int = 1,
    save_path: Optional[Path] = None,
    bar_width: float = 0.15,
    inner_padding: float = 0.75,
    save_csv: bool = True,
):
    """
    Create a bar plot of the relative difference in the subgroup composition between the overall population and the top N most vulnerable patients.
    This version is for overlapping groups, e.g. multiple disease labels. Assumes labels in group_cols are in binary format.

    Args:
        df: pd.DataFrame, data frame containing the AUCs and patient information
        group_cols: List, column names of the groups
        top_k_percent: int, top k% of AUCs to consider
        save_path: Path, path to save the plot
        xlabel: str, x-axis label
        figsize: Tuple, figure size
    Returns:
        None
    """
    assert isinstance(df, pd.DataFrame), f"Expected pd.DataFrame, got {type(df)}"
    top_n = int(top_k_percent / 100 * len(df))
    attr_vals = group_cols
    for group_col in group_cols:
        assert (
            group_col in df.columns
        ), f"Group column {group_col} not found in data frame. Columns: {df.columns}"
        assert (
            df[group_col].isna().sum() == 0
        ), f"Missing values in {group_col}, {df[group_col].isna().sum()}"
    inner_width = bar_width * len(attr_vals)
    inner_height = 1.0
    # Define fixed sizes for the axes (in inches)
    left_margin = 0.25
    right_margin = 0.15
    bottom_margin = 0.25
    top_margin = 0.25
    # Total width and height of the figure
    total_width = inner_width + left_margin + right_margin
    total_height = inner_height + bottom_margin + top_margin
    # Create the figure
    fig = plt.figure(figsize=(total_width, total_height))
    # Create a divider to allocate space for the axes
    h = [Size.Fixed(left_margin), Size.Fixed(inner_width), Size.Fixed(right_margin)]
    v = [Size.Fixed(bottom_margin), Size.Fixed(inner_height), Size.Fixed(top_margin)]
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax = fig.add_axes(
        divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1)
    )
    ax.hlines(y=0, xmin=-100, xmax=100, colors="gray", lw=1, alpha=0.25, zorder=-2)
    # filter out the top N most vulnerable patients
    top_n_df = df[df["MIA_AUC"] > df["MIA_AUC"].quantile(1 - top_n / len(df))]
    assert len(top_n_df) == top_n, f"Expected {top_n} patients, got {len(top_n_df)}"
    expected_counts = df[group_cols].mean().values * top_n
    observed_counts = top_n_df[group_cols].sum().values
    residual = (observed_counts - expected_counts) / np.sqrt(expected_counts)
    rel_diff = (observed_counts - expected_counts) / expected_counts
    print("... expected composition (overall dataset):\n", df[group_cols].mean(), "\n... observed composition (99th percentile):\n", top_n_df[group_cols].mean())
    print(f"... relative differences (%): {rel_diff}")
    print(f"... Pearson residuals: {residual}")
    # color by group size in overall population
    cm = plt.get_cmap("Blues")
    overall_counts = df[group_cols].mean().values * 100
    max_norm_val = 50
    norm = plt.Normalize(0, max_norm_val)
    colors = [cm(norm(c)) for c in overall_counts]
    bars = ax.bar(
        attr_vals,
        residual,
        color=colors,
        linewidth=1,
        edgecolor="black",
        width=4 * bar_width,
        align="center",
    )
    bar_positions = [bar.get_x() + bar.get_width() / 2 for bar in bars]
    ax.set_ylabel("Pearson Residual")
    ax.grid(visible=False, axis="both")
    ax.set_xlim(
        (
            bar_positions[0] - (bar_width / 2) - inner_padding,
            bar_positions[-1] + (bar_width / 2) + inner_padding,
        )
    )
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)  # ScalarMappable for the colorbar
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        shrink=0.66,
        pad=0.025,
        fraction=1.0,
        anchor=(1.0, 0.3),
    )
    cbar.ax.set_title("%", pad=5, loc="center")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlabel("")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        results_dict = {
            "Groups": attr_vals,
            "Pearson Residual": residual,
            "Relative Delta": rel_diff,
            "Relative Frequency": overall_counts,
        }
        if save_csv:
            csv_dir = Path("/".join(save_path.parts[:-1]) + "/scatter_data")
            csv_dir.mkdir(exist_ok=True)
            csv_path = csv_dir / f"{xlabel}_ns.csv"
            results_df = pd.DataFrame.from_dict(results_dict)
            results_df.to_csv(csv_path)


def composition_comparison_plot(
    df: pd.DataFrame,
    group_col: str,
    col_vals: List[str],
    xlabel: str,
    top_k_percent: int = 1,
    bonferroni_correction_factor=None,
    color_dict: Optional[Dict] = None,
    save_path: Optional[Path] = None,
    draw_xticks: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    bar_width: float = 0.15,
    inner_padding: float = 0.75,
    save_csv: bool = True,
):
    """
    Create a bar plot of the relative difference in subgroup composition between the overall population and the top k % most vulnerable patients.
    Also performs a Chi-squared test to determine statistical significance.

    Args:
        df: pd.DataFrame, data frame containing the AUCs and patient information
        group_col: str, column name of the group
        top_k_percent: int, top k% of AUCs to consider
        top_n: int, number of most vulnerable patients
        save_path: Path, path to save the plot
        xlabel: str, x-axis label
        figsize: Tuple, figure size
    Returns:
        None
    """
    print(f"... Subgroup composition comparison: {group_col}")
    assert isinstance(df, pd.DataFrame), f"Expected pd.DataFrame, got {type(df)}"
    assert (
        group_col in df.columns
    ), f"Group column {group_col} not found in data frame. Columns: {df.columns}"
    if color_dict is not None:
        # drop patients with group_col value not in color_dict (other/unknown value)
        df = df[df[group_col].isin(color_dict.keys())]
    top_n = int(top_k_percent / 100 * len(df))
    attr_vals = df[group_col].unique()
    inner_width = bar_width * len(attr_vals)
    inner_height = 1.0
    # Define fixed sizes for the axes (in inches)
    left_margin = 0.25
    right_margin = 0.15
    bottom_margin = 0.25
    top_margin = 0.25
    # Total width and height of the figure
    total_width = inner_width + left_margin + right_margin
    total_height = inner_height + bottom_margin + top_margin
    # Create the figure
    fig = plt.figure(figsize=(total_width, total_height))
    # Create a divider to allocate space for the axes
    h = [Size.Fixed(left_margin), Size.Fixed(inner_width), Size.Fixed(right_margin)]
    v = [Size.Fixed(bottom_margin), Size.Fixed(inner_height), Size.Fixed(top_margin)]
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax = fig.add_axes(
        divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1)
    )
    ax.hlines(y=0, xmin=-100, xmax=100, colors="gray", lw=1, alpha=0.25, zorder=-2)
    assert (
        df[group_col].isna().sum() == 0
    ), f"Missing values in {group_col}, {df[group_col].isna().sum()}"
    # filter out the top N most vulnerable patients
    top_n_df = df[df["MIA_AUC"] > df["MIA_AUC"].quantile(1 - top_n / len(df))]
    assert len(top_n_df) == top_n, f"Expected {top_n} patients, got {len(top_n_df)}"
    # chi-squared test for significance
    expected_counts = (
        df[group_col].value_counts(normalize=True, sort=False).values * top_n
    )
    observed_counts = top_n_df[group_col].value_counts(sort=False)
    observed_counts = observed_counts.reindex(attr_vals, fill_value=0).values
    assert len(observed_counts) == len(
        expected_counts
    ), f"Expected {len(expected_counts)} values, got {len(observed_counts)}"
    print(
        "... expected composition (overall dataset):\n",
        df[group_col].value_counts(normalize=True, sort=False),
        "\n... observed composition (99th percentile):\n",
        top_n_df[group_col].value_counts(normalize=True, sort=False).reindex(attr_vals, fill_value=0),
    )
    chisq_statistic, p_value = scipy.stats.chisquare(
        f_obs=observed_counts, f_exp=expected_counts
    )
    print(f"... chi-squared statistic: {chisq_statistic}, p-value: {p_value}")
    residual = (observed_counts - expected_counts) / (np.sqrt(expected_counts))
    rel_diff = (observed_counts - expected_counts) / expected_counts
    print(f"... relative differences: {rel_diff}")
    print(f"... Pearson residuals: {residual}")
    print("---------------------------------------------------------------------------")
    # color by group size in overall population
    cm = plt.get_cmap("Blues")
    overall_counts = df[group_col].value_counts(normalize=True, sort=False).values * 100
    max_norm_val = 50
    norm = plt.Normalize(0, max_norm_val)
    colors = [cm(norm(c)) for c in overall_counts]
    if not isinstance(attr_vals[0], str):
        attr_vals = [str(a) for a in attr_vals]
    assert set(attr_vals) == set(
        col_vals
    ), f"Found unexpected column values, expected {set(col_vals)} and got {set(attr_vals)}"
    # make sure the bars are in the correct order
    ordered_data = [
        (cat, val, color)
        for cat, val, color in zip(attr_vals, residual, colors)
        if cat in attr_vals
    ]
    ordered_data.sort(key=lambda x: col_vals.index(x[0]))
    cats_sorted, data_sorted, colors_sorted = zip(*ordered_data)
    bars = ax.bar(
        cats_sorted,
        data_sorted,
        color=colors_sorted,
        linewidth=1,
        edgecolor="black",
        width=4 * bar_width,
        align="center",
    )
    bar_positions = [bar.get_x() + bar.get_width() / 2 for bar in bars]
    ax.set_ylabel("Pearson Residual")
    ax.grid(visible=False, axis="both")
    ax.set_xlim(
        (
            bar_positions[0] - (bar_width / 2) - inner_padding,
            bar_positions[-1] + (bar_width / 2) + inner_padding,
        )
    )
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)  # ScalarMappable for the colorbar
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        shrink=0.66,
        pad=0.025,
        fraction=1.0,
        anchor=(1.0, 0.3),
    )
    cbar.ax.set_title("%", pad=5, loc="center")
    ax.grid(visible=False, axis="x")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.spines[["right", "top"]].set_visible(False)
    textstr = r"{}, $\chi^2$={}".format(get_sig_str(p_value, bonferroni_correction_factor=bonferroni_correction_factor), round(chisq_statistic, 1))
    props = dict(facecolor="#FAFAFA", boxstyle="round", alpha=1.0, lw=0.75)
    ax.text(
        0.95,
        1.15,
        textstr,
        transform=ax.transAxes,
        fontsize=5,
        verticalalignment="top",
        bbox=props,
    )
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        results_dict = {
            "Groups": attr_vals,
            "Pearson Residual": residual,
            "Relative Delta": rel_diff,
            "Relative Frequency": overall_counts,
        }
        if save_csv:
            csv_dir = Path("/".join(save_path.parts[:-1]) + "/scatter_data")
            csv_dir.mkdir(exist_ok=True)
            csv_path = (
                csv_dir / f"{group_col}.csv"
                if p_value <= 0.05
                else csv_dir / f"{group_col}_ns.csv"
            )
            results_df = pd.DataFrame.from_dict(results_dict)
            results_df.to_csv(csv_path)


def disease_group_plots(dataset_name: str, results_df: pd.DataFrame, out_path: Path):
    out_path = out_path / "subgroup_plots"
    out_path.mkdir(parents=True, exist_ok=True)
    if dataset_name == "mimic" or dataset_name == "chexpert":
        results_df = results_df.rename(
            columns={a: b for a, b in zip(CXR_LABELS, CXR_SHORT_LABELS)}
        )
        results_df[CXR_SHORT_LABELS] = results_df[CXR_SHORT_LABELS].replace(
            {-1: 1}
        )  # map uncertain labels to positive
        color_dict = {k: plt.cm.tab20(i) for i, k in enumerate(CXR_SHORT_LABELS)}
        composition_comparison_plot_overlapping_groups(
            df=results_df,
            group_cols=CXR_SHORT_LABELS,
            save_path=out_path / f"{dataset_name}_composition_plot_disease.pdf",
            xlabel="Disease",
        )
    elif dataset_name == "fairvision":
        color_dict = {k: plt.cm.tab20(i) for i, k in enumerate(FAIRVISION_LABELS_SHORT)}
        # map integer labels to string labels
        results_df["disease_label"] = results_df["disease_label"].replace(
            {
                0: "healthy",
                1: "non-vision threatening dr",
                2: "vision threatening dr",
                3: "glaucoma",
                4: "early amd",
                5: "intermediate amd",
                6: "late amd",
            }
        )
        results_df["disease_label"] = results_df["disease_label"].replace(
            {a: b for a, b in zip(FAIRVISION_LABELS, FAIRVISION_LABELS_SHORT)}
        )
        composition_comparison_plot(
            df=results_df,
            group_col="disease_label",
            save_path=out_path / f"{dataset_name}_composition_plot_disease.pdf",
            xlabel="Disease",
            draw_xticks=False,
            col_vals=FAIRVISION_LABELS_SHORT,
            bonferroni_correction_factor=4,
        )
    elif dataset_name == "fitzpatrick":
        color_dict = {
            k: plt.cm.tab20(i) for i, k in enumerate(FITZPATRICK_LABELS_COARSE)
        }
        results_df["three_partition_label"] = results_df[
            "three_partition_label"
        ].replace(
            {
                a: b
                for a, b in zip(
                    FITZPATRICK_LABELS_COARSE, FITZPATRICK_LABELS_COARSE_SHORT
                )
            }
        )
        composition_comparison_plot(
            df=results_df,
            group_col="three_partition_label",
            save_path=out_path / f"{dataset_name}_composition_plot_disease_coarse.pdf",
            xlabel="Disease",
            draw_xticks=False,
            col_vals=FITZPATRICK_LABELS_COARSE_SHORT,
            bonferroni_correction_factor=3,
        )
        color_dict = {
            k: plt.cm.tab20(i) for i, k in enumerate(FITZPATRICK_LABELS_FINE_SHORT)
        }
        results_df["nine_partition_label"] = results_df["nine_partition_label"].replace(
            {
                a: b
                for a, b in zip(FITZPATRICK_LABELS_FINE, FITZPATRICK_LABELS_FINE_SHORT)
            }
        )
        composition_comparison_plot(
            df=results_df,
            group_col="nine_partition_label",
            save_path=out_path / f"{dataset_name}_composition_plot_disease.pdf",
            xlabel="Disease",
            draw_xticks=False,
            col_vals=FITZPATRICK_LABELS_FINE_SHORT,
            bonferroni_correction_factor=3,
        )
    elif dataset_name == "embed":
        color_dict = {k: plt.cm.tab20(i) for i, k in enumerate(EMBED_LABELS_SHORT)}
        # drop all cases with exam label X (unknown)
        results_df = results_df.loc[results_df["asses"] != "X"]
        birads_labels = ["BIRADS A", "BIRADS B", "BIRADS C", "BIRADS D"]
        results_df["tissueden"] = results_df["tissueden"].replace(
            {a: b for a, b in zip(birads_labels, EMBED_LABELS_SHORT)}
        )
        composition_comparison_plot(
            df=results_df,
            group_col="tissueden",
            save_path=out_path / f"{dataset_name}_composition_plot_density.pdf",
            xlabel="BIRADS Density",
            col_vals=EMBED_LABELS_SHORT,
            bonferroni_correction_factor=5,
        )
        results_df["asses_birads"] = results_df["asses"].replace(
            {a: b for a, b in zip(EMBED_EXAM_LABELS, EMBED_EXAM_LABELS_BIRADS)}
        )
        composition_comparison_plot(
            df=results_df,
            group_col="asses_birads",
            save_path=out_path / f"{dataset_name}_composition_plot_disease.pdf",
            xlabel="BIRADS Assesment",
            col_vals=[str(x) for x in EMBED_EXAM_LABELS_BIRADS],
            bonferroni_correction_factor=5,
        )
    elif dataset_name == "mimic-iv-ed":
        results_df['outcome_hospitalization'] = results_df['outcome_hospitalization'].astype("int8")
        composition_comparison_plot(
            df=results_df,
            group_col="outcome_hospitalization",
            save_path=out_path / f"{dataset_name}_composition_plot_disease.pdf",
            xlabel="Hospitalization Outcome",
            col_vals=["0", "1"],
            bonferroni_correction_factor=6,
        )
    elif dataset_name == "ptb-xl":
        composition_comparison_plot_overlapping_groups(
            df=results_df,
            group_cols=ECG_LABELS,
            save_path=out_path / f"{dataset_name}_composition_plot_disease.pdf",
            xlabel="Disease",
        )
    else:
        raise ValueError("Dataset not supported")


def subgroup_plots(dataset_name: str, results_df: pd.DataFrame, out_path: Path):
    out_path = out_path / "subgroup_plots"
    out_path.mkdir(parents=True, exist_ok=True)
    print("... subgroup plots")
    if dataset_name == "mimic":
        composition_comparison_plot(
            df=results_df,
            group_col="race",
            color_dict=CXR_COLORS,
            save_path=out_path / f"{dataset_name}_composition_plot_race.pdf",
            xlabel="Race",
            col_vals=["Black", "White", "Asian"],
            bonferroni_correction_factor=3,
        )
        composition_comparison_plot(
            df=results_df,
            group_col="Sex",
            color_dict=CXR_COLORS,
            save_path=out_path / f"{dataset_name}_composition_plot_sex.pdf",
            figsize=(1.25, 1.75),
            xlabel="Sex",
            col_vals=["Female", "Male"],
            bonferroni_correction_factor=3,
        )
        views = ["AP", "PA", "LATERAL", "LL"]
        mpl_colors = plt.cm.get_cmap("tab20")
        view_colors_dict = {v: mpl_colors(i) for i, v in enumerate(views)}
        composition_comparison_plot(
            df=results_df,
            group_col="ViewPosition",
            color_dict=view_colors_dict,
            save_path=out_path / f"{dataset_name}_composition_plot_view.pdf",
            xlabel="Imaging View",
            col_vals=["AP", "PA", "LATERAL", "LL"],
            bonferroni_correction_factor=3,
        )
    elif dataset_name == "chexpert":
        composition_comparison_plot(
            df=results_df,
            group_col="race",
            color_dict=CXR_COLORS,
            save_path=out_path / f"{dataset_name}_composition_plot_race.pdf",
            xlabel="Race",
            col_vals=["Black", "White", "Asian"],
            bonferroni_correction_factor=3,
        )
        composition_comparison_plot(
            df=results_df,
            group_col="Sex",
            color_dict=CXR_COLORS,
            save_path=out_path / f"{dataset_name}_composition_plot_sex.pdf",
            xlabel="Sex",
            col_vals=["Female", "Male"],
            bonferroni_correction_factor=3,
        )
        results_df["view"] = results_df["AP/PA"]
        views = ["AP", "PA"]
        mpl_colors = plt.cm.get_cmap("tab20")
        view_colors_dict = {v: mpl_colors(i) for i, v in enumerate(views)}
        composition_comparison_plot(
            df=results_df,
            group_col="view",
            color_dict=view_colors_dict,
            save_path=out_path / f"{dataset_name}_composition_plot_view.pdf",
            xlabel="Imaging View",
            col_vals=views,
            bonferroni_correction_factor=3,
        )
    elif dataset_name == "fairvision":
        results_df["race"] = results_df["race"].str.capitalize()
        results_df["gender"] = results_df["gender"].str.capitalize()
        composition_comparison_plot(
            df=results_df,
            group_col="race",
            color_dict=CXR_COLORS,
            save_path=out_path / f"{dataset_name}_composition_plot_race.pdf",
            xlabel="Race",
            col_vals=["Black", "White", "Asian"],
            save_csv=False,
            bonferroni_correction_factor=4,
        )
        # control for confounding by disease label
        filtered_df = results_df[
            results_df["disease_label"].isin([0, 1])
        ]  # only healthy patients or those with non-vision-threatening DR
        composition_comparison_plot(
            df=filtered_df,
            group_col="race",
            color_dict=CXR_COLORS,
            save_path=out_path / f"{dataset_name}_composition_plot_race@H_NVTDR.pdf",
            xlabel="Race",
            col_vals=["Black", "White", "Asian"],
            bonferroni_correction_factor=4,
        )
        composition_comparison_plot(
            df=results_df,
            group_col="gender",
            color_dict=CXR_COLORS,
            save_path=out_path / f"{dataset_name}_composition_plot_sex.pdf",
            xlabel="Sex",
            col_vals=["Female", "Male"],
            bonferroni_correction_factor=4,
        )
    elif dataset_name == "fitzpatrick":
        results_df["Fitzpatrick Skin Type"] = (
            results_df["Fitzpatrick Skin Type"].str.split(" ").str[1]
        )
        fitz_colors = {k.split(" ")[1]: v for k, v in FITZPATRICK_COLORS.items()}
        composition_comparison_plot(
            df=results_df,
            group_col="Fitzpatrick Skin Type",
            color_dict=fitz_colors,
            save_path=out_path / f"{dataset_name}_composition_plot_fitzpatrick.pdf",
            xlabel="Fitzpatrick Skin Type\n",
            col_vals=["I/II", "III/IV", "V/VI"],
            bonferroni_correction_factor=3,
        )
    elif dataset_name == "embed":
        composition_comparison_plot(
            df=results_df,
            group_col="RACE_DESC",
            color_dict=CXR_COLORS,
            save_path=out_path / f"{dataset_name}_composition_plot_race.pdf",
            xlabel="Race",
            col_vals=["Black", "White", "Asian"],
            save_csv=True,
            bonferroni_correction_factor=5,
        )
        composition_comparison_plot(
            df=results_df,
            group_col="FinalImageType",
            color_dict={"2D": None, "cview": None},
            save_path=out_path / f"{dataset_name}_composition_plot_imagetype.pdf",
            xlabel="Image Type",
            col_vals=["2D", "cview"],
            save_csv=True,
            bonferroni_correction_factor=5,
        )
        views = ["MLO", "CC"]
        mpl_colors = plt.cm.get_cmap("tab20")
        view_colors_dict = {v: mpl_colors(i) for i, v in enumerate(views)}
        composition_comparison_plot(
            df=results_df,
            group_col="ViewPosition",
            color_dict=view_colors_dict,
            save_path=out_path / f"{dataset_name}_composition_plot_view.pdf",
            xlabel="Imaging View",
            col_vals=["MLO", "CC"],
            bonferroni_correction_factor=5,
        )
    elif dataset_name == "mimic-iv-ed":
        # correct field 'race' to only include 'Black', 'White', 'Asian' and 'Other/Unknown'
        black = 'Black'
        white = 'White'
        asian = 'Asian'
        other = "Other/Unknown"

        black_mask = (results_df.race.str.contains("BLACK", na=False))
        white_mask = (results_df.race.str.contains("WHITE", na=False))
        asian_mask = (results_df.race.str.contains("ASIAN", na=False))
        other_mask = np.logical_not(black_mask | white_mask | asian_mask)

        results_df.loc[black_mask, "race"] = black
        results_df.loc[white_mask, "race"] = white
        results_df.loc[asian_mask, "race"] = asian
        results_df.loc[other_mask, "race"] = other
        print(results_df['eci_Obesity'].value_counts(), results_df['insurance'].value_counts())
        print(list(results_df.columns))
        print(results_df['gender'].value_counts())
        results_df['gender'].replace("F", "Female", inplace=True)
        results_df['gender'].replace("M", "Male", inplace=True)
        composition_comparison_plot(
            df=results_df,
            group_col="race",
            color_dict=CXR_COLORS,
            save_path=out_path / f"{dataset_name}_composition_plot_race.pdf",
            xlabel="Race",
            col_vals=["Black", "White", "Asian"],
            bonferroni_correction_factor=6,
        )
        composition_comparison_plot(
            df=results_df,
            group_col="gender",
            color_dict=CXR_COLORS,
            save_path=out_path / f"{dataset_name}_composition_plot_sex.pdf",
            xlabel="Sex",
            col_vals=["Female", "Male"],
            bonferroni_correction_factor=6,
        )
        composition_comparison_plot(
            df=results_df,
            group_col="cci_Cancer1",
            color_dict={0: None, 1:None},
            save_path=out_path / f"{dataset_name}_composition_plot_cancer.pdf",
            xlabel="Cancer Status",
            col_vals=["0", "1"],
            bonferroni_correction_factor=6,
        )
        composition_comparison_plot(
            df=results_df,
            group_col="insurance",
            color_dict={"Medicare": None, "Medicaid":None},
            save_path=out_path / f"{dataset_name}_composition_plot_insurance.pdf",
            xlabel="Insurance Type",
            col_vals=["Medicare", "Medicaid"],
            bonferroni_correction_factor=6,
        )
        composition_comparison_plot(
            df=results_df,
            group_col="eci_Obesity",
            color_dict={0: None, 1:None},
            save_path=out_path / f"{dataset_name}_composition_plot_obesity.pdf",
            xlabel="Obesity Status",
            col_vals=["0", "1"],
            bonferroni_correction_factor=6,
        )
    elif dataset_name == "ptb-xl":
        print(results_df)
        results_df["extra_beat_binary"] = 0
        results_df.loc[~results_df["extra_beats"].isna(), "extra_beat_binary"] = 1
        print(results_df["extra_beat_binary"].value_counts())
        composition_comparison_plot(
            df=results_df,
            group_col="sex",
            color_dict={0: None, 1:None},
            save_path=out_path / f"{dataset_name}_composition_plot_sex.pdf",
            xlabel="Sex",
            col_vals=["0", "1"],
            bonferroni_correction_factor=2,
        )
        composition_comparison_plot(
            df=results_df,
            group_col="extra_beat_binary",
            color_dict={0: None, 1:None},
            save_path=out_path / f"{dataset_name}_composition_plot_extra_beat.pdf",
            xlabel="Extra Beats",
            col_vals=["0", "1"],
            bonferroni_correction_factor=2,
        )
    else:
        raise ValueError("Dataset not supported")


def plot_images(
    risk_scores: np.ndarray,
    dataset: BaseDataset,
    dataset_name: str,
    highest: bool = True,
    out_dir: Optional[Path] = None,
):
    fig, axes = plt.subplots(4, 5, figsize=(7.25, 7.25))
    sorted_idcs = (
        np.argsort(-risk_scores) if highest else np.random.permutation(len(risk_scores))
    )
    for i, ax in enumerate(axes.flat):
        idx = int(sorted_idcs[i])
        img, target = dataset.__getitem__(idx)
        img = np.array(img)
        assert (
            len(img.shape) == 3
        ), f"Expected color or gray-scale image, found: {img.shape}"
        if img.shape[-1] not in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        if img.min() < 0:
            # re-scale image to [0, 1]
            img = (img + 1) / 2
        target_idc = np.argmax(target)
        disease_idcs = np.nonzero(np.array(target, np.uint8))[0]
        if dataset_name in ["mimic", "chexpert"]:
            labels = [CXR_SHORT_LABELS[i] for i in disease_idcs]
            ax.imshow(img, cmap="gray")
            label_str = ", ".join(labels)
        elif dataset_name == "fitzpatrick":
            ax.imshow(img)
            label_str = FITZPATRICK_LABELS_COARSE[target_idc]
        elif dataset_name == "embed":
            label_str = EMBED_LABELS[target_idc]
            ax.imshow(img, cmap="gray")
        elif dataset_name == "fairvision":
            label_str = FAIRVISION_LABELS[target_idc]
            img = exposure.equalize_hist(img)
            ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(f"{label_str} \n {risk_scores[idx]:.3f}")
    save_str = "highest" if highest else "random"
    if out_dir is not None:
        plt.savefig(
            out_dir / f"{dataset_name}_imgs_auc_{save_str}.pdf",
            bbox_inches="tight",
            dpi=500,
        )
    plt.show()
