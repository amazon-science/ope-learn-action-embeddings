from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


plt.style.use(['science', 'no-latex'])

registered_colors = {
    "MIPS": "tab:gray",
    "MIPS (true)": "tab:orange",
    "MIPS (w/ SLOPE)": "tab:orange",
    "IPS": "tab:red",
    "DR": "tab:blue",
    "DM": "tab:purple",
    "SNIPS": "lightblue",
    "SwitchDR": "tab:pink",
    "Learned MIPS OneHot": "tab:olive",
    "Learned MIPS FineTune": "green",
    "Learned MIPS Combined": "tab:brown",
}

markers_dict = {
    "MIPS": "X",
    "MIPS (true)": "X",
    "MIPS (w/ SLOPE)": "X",
    "IPS": "v",
    "DR": "v",
    "DM": "v",
    "SNIPS": "v",
    "SwitchDR": "v",
    "Learned MIPS OneHot": "v",
    "Learned MIPS FineTune": "X",
    "Learned MIPS Combined": "X",
}


def export_legend(ax, filename="legend.pdf"):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False, loc='lower center', ncol=10, handlelength=1.5)
    for legobj in legend.legendHandles:
        legobj.set_linewidth(2.5)
        legobj._markeredgecolor = 'white'
        legobj._markeredgewidth = 0.5
        legobj._markersize = 8
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def remove_estimators(df: pd.DataFrame, estimators: list):
    return df[~df.est.isin(estimators)]


def plot_line(
    result_df: pd.DataFrame,
    fig_path: Path,
    x: str,
    xlabel: str,
    xticklabels: list,
    estimators: list = None,
    exclude=[],
    markers=True
):
    result_df = remove_estimators(result_df, exclude)

    def plot_part(column, ylabel, fig_name, legend=True, log_scale=True):
        plt.close()
        fig, ax = plt.subplots(figsize=(11, 7), tight_layout=True)
        sns.lineplot(
            linewidth=5,
            legend=legend,
            markers=markers_dict if markers else False,
            markersize=15,
            markeredgecolor='white',
            dashes=False,
            x=x,
            y=column,
            hue="est",
            style="est",
            ax=ax,
            palette=palette,
            data=result_df.query(query),
        )

        if legend:
            l = ax.legend(
                loc="upper left",
                fontsize=25,
            )
            for legobj in l.legendHandles:
                legobj.set_linewidth(4)
                legobj.set_markersize(12)

        # yaxis
        if log_scale:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=25)
        ax.tick_params(axis="y", labelsize=18)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        # ax.set_ylim(top=1.05e-1, bottom=4e-3)

        # xaxis
        if x in ["n_actions", "n_val_data"]:
            ax.set_xscale("log")

        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_xticks(xticklabels)
        ax.set_xticklabels(xticklabels, fontsize=18)
        ax.xaxis.set_label_coords(0.5, -0.1)
        plt.savefig(fig_path / fig_name, bbox_inches="tight")
        return ax

    if estimators is None:
        estimators = [est for est in result_df.est.unique() if est in registered_colors]
    query = "(" + " or ".join([f"est == '{est}'" for est in estimators]) + ")"
    palette = [registered_colors[est] for est in estimators]

    fig_path.mkdir(exist_ok=True, parents=True)
    print(estimators)

    plot_part("se", "MSE", "mse.png")
    ax = plot_part("se", "MSE", "mse.pdf")
    export_legend(ax, fig_path / "legend.pdf")
    plot_part("se", "MSE", "mse_no_legend.pdf", legend=False)
    plot_part("se", "MSE", "mse_no_log.pdf", log_scale=False)
    plot_part("se", "MSE", "mse_no_log_no_legend.pdf", legend=False, log_scale=False)
    plot_part("variance", "Variance", "variance.pdf")
    plot_part("variance", "Variance", "variance_no_legend.pdf", legend=False)
    plot_part("variance", "Variance", "variance_no_log.pdf", log_scale=False)
    plot_part("variance", "Variance", "variance_no_log_no_legend.pdf", legend=False, log_scale=False)
    plot_part("bias", "Squared bias", "bias_no_log.pdf", log_scale=False)
    plot_part("bias", "Squared bias", "bias_no_log_no_legend.pdf", legend=False, log_scale=False)


def plot_cdf(
    result_df: pd.DataFrame,
    fig_path: str,
    relative_to: str = 'MIPS (w/ SLOPE)',
    remove_legend=False,
    exclude=[]
):
    result_df = remove_estimators(result_df, exclude)
    estimators = [est for est in result_df.est.unique() if est in registered_colors]
    query = "(" + " or ".join([f"est == '{est}'" for est in estimators]) + ")"
    result_df = result_df.query(query).reset_index()
    relative_index = result_df[result_df.est == relative_to].index[0]
    rel_result_df = result_df.groupby(result_df.index // len(estimators)) \
        .apply(lambda df: pd.DataFrame({'est': df['est'], 'se': df['se'] / df.iloc[relative_index]['se']}))
    palette = [registered_colors[est] for est in estimators[::-1]]

    fig, ax = plt.subplots(figsize=(10, 6.5), tight_layout=True)
    sns.ecdfplot(
        linewidth=3.5,
        palette=palette,
        data=rel_result_df,
        x="se",
        hue="est",
        hue_order=estimators[::-1],
        ax=ax
    )

    ax.legend(estimators, loc="upper left", fontsize=22)
    if remove_legend:
        ax.legend(estimators, loc="upper left", fontsize=22).remove()

    for i in range(len(ax.lines)):
        line = ax.lines[i]
        if line._x.max() == 1:
            continue
        idx = np.abs(line._x - 1).argmin()
        y = line._y[idx]
        ax.axhline(y, 0, 0.52, color=line.get_color(), linewidth=1.5, linestyle=(0, (5, 3)))

    ax.set_ylabel("Cumulative distribution", fontsize=22)
    ax.tick_params(axis="y", labelsize=18)
    ax.yaxis.set_label_coords(-0.08, 0.5)

    ax.set_xscale("log")
    ax.set_xlim(0.3, 3)
    ax.set_xlabel(f"Relative MSE w.r.t. {relative_to}", fontsize=22)
    ax.tick_params(axis="x", labelsize=18)
    ax.xaxis.set_label_coords(0.5, -0.1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
