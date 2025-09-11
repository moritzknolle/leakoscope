import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch
import scipy
import scipy.stats

from plots import get_color, get_dataset_name


def get_marker_style(attr_val: str):
    # sex style
    if (
        attr_val.lower() == "sex"
        or attr_val.split("_")[0] == "gender"
        or attr_val.lower().split("_")[0] == "sex"
    ):
        tick_style = "o"
    # disease label style
    elif (
        "label" in attr_val.split("_")
        or attr_val == "asses_birads"
        or attr_val == "tissueden"
        or attr_val.split("_")[0] == "Disease"
        or "_".join(attr_val.split("_")[:-1]) == "extra_beat_binary"
        or attr_val == "outcome_hospitalization"
        or attr_val == "eci_Obesity"
        or attr_val == "cci_Cancer1"
        or attr_val == "insurance"
    ):
        tick_style = "s"
    # race style
    elif (
        attr_val.lower() == "race"
        or attr_val == "RACE_DESC"
        or attr_val == "Fitzpatrick Skin Type_ns"
        or attr_val.split("_")[0] == "race"
    ):
        tick_style = "d"
    # imaging protocol style
    elif (
        attr_val == "view"
        or attr_val == "ViewPosition"
        or attr_val.split("_")[0] == "FinalImageType"
        or attr_val.split("_")[0] == "view"
    ):
        tick_style = "v"
    else:
        raise ValueError(f"Unknown attribute value: {attr_val}")
    return tick_style


def create_scatter_plot():
    FONT_SIZE = 6
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
    
    fig_dir = Path("figs")
    
    alpha = 0.5
    fig, axs = plt.subplots(2, 4, figsize=(7, 3.5))
    datasets = []
    xs, ys = [], []
    i = 0
    for sub_dir in fig_dir.iterdir():
        if sub_dir.is_dir():
            if Path(sub_dir / "subgroup_plots/scatter_data").exists():
                xs = []
                ys = []
                dataset = sub_dir.parts[1]
                datasets.append(dataset)
                print("... dataset:", dataset)
                for file in (sub_dir / "subgroup_plots/scatter_data").iterdir():
                    if file.is_file():
                        df = pd.read_csv(file)
                        deltas = df["Pearson Residual"]
                        rel_freq = df["Relative Frequency"]
                        xs.extend(rel_freq)
                        ys.extend(deltas)
                        print(f"strat variable: {file.stem}, marker: {get_marker_style(file.stem)}")
                        axs.flat[i].scatter(x=rel_freq, y=deltas, label=file.stem, marker=get_marker_style(file.stem), s=6, alpha=alpha, c=get_color(dataset))
                        axs.flat[i].axhline(y=0, linestyle="--", alpha=0.15, color="black")
                        axs.flat[i].spines[['right', 'top',]].set_visible(False)
                rho, _ = scipy.stats.spearmanr(xs, ys)
                r, _ = scipy.stats.pearsonr(xs, ys)
                # text box with correlation values
                props = dict(facecolor='white', edgecolor='grey', boxstyle='round', alpha=0.5)
                axs.flat[i].text(0.75, 0.75, r"$\rho$={}".format(f"{rho:.2f}") + "\n" + r"$r$={}".format(f"{r:.2f}"), transform=axs.flat[i].transAxes, fontsize=6, bbox=props)
                axs.flat[i].set_title(get_dataset_name(dataset))
                i += 1

    dataset_colors = [get_color(dataset) for dataset in datasets]
    datasets = [get_dataset_name(dataset) for dataset in datasets]
    dataset_handles = [Patch(color=color, label=dataset, alpha=alpha) 
                       for dataset, color in zip(datasets, dataset_colors)]
    markers = ["o", "s", "d", "v"]
    marker_names = ["Sex", "Disease", "Race", "Imaging Protocol"]
    marker_handles = [plt.Line2D([0], [0], marker=marker, color="w", markerfacecolor="black", markersize=6, label=marker_name) 
                      for marker, marker_name in zip(markers, marker_names)]
    fig.supxlabel("Group Size (%)")
    fig.supylabel("Pearson Residual")
    # two column legend with column titles, one column for datasets, one column for markers
    axs.flat[-1].legend(handles=marker_handles, ncol=1, loc="upper right", fontsize=FONT_SIZE, framealpha=0.3, bbox_to_anchor=(1.05, 1.0))
    axs.flat[-1].axis("off")
    #ax.set_xlim(-2, 90)
    #plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    plt.savefig("./figs/group_size_vs_residual.pdf", bbox_inches="tight")


if __name__ == "__main__":
    create_scatter_plot()