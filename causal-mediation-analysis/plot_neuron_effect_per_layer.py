"""Script to generate a plot with effect per-layer.

Requirement:
You have to have run `compute_and_save_neuron_agg_effect.py` for each of the models you want
to investigate. That script will save several intermediate result csvs. To reduce computational
overhead, this file expects those intermediate result files.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_context("talk")
sns.set_style("whitegrid")


def get_top_perc_per_layer(df, ntype, ccat = None, n=10):
    """Get avg indirect effect of top n% neurons"""

    # Get only entries for the negation type of interest
    ntype_df = df.loc[df["negation_type_"] == ntype]
    # If splitting by correctness category, get entries for the category of interest
    if not ccat == None:
        ntype_df = ntype_df.loc[ntype_df["correctness_cat_"] == ccat]
    num_neurons = int(ntype_df.groupby("layer_direct_mean").size()[0] * n / 100.0)
    return (
        ntype_df.groupby("layer_direct_mean")
        .apply(lambda x: x.nlargest(num_neurons, ["odds_ratio_indirect_mean"]))
        .reset_index(drop=True)
        .groupby("layer_direct_mean")
        .agg("mean")[["odds_ratio_indirect_mean", "odds_ratio_indirect_std"]]
        .reset_index()
    )

def get_bottom_perc_per_layer(df, ntype, ccat = None, n=10):
    """Get avg indirect effect of bottom n% neurons"""

    ntype_df = df.loc[df["negation_type_"] == ntype]
    if not ccat == None:
        ntype_df = ntype_df.loc[ntype_df["correctness_cat_"] == ccat]
    num_neurons = int(ntype_df.groupby("layer_direct_mean").size()[0] * n / 100.0)
    return (
        ntype_df.groupby("layer_direct_mean")
        .apply(lambda x: x.nsmallest(num_neurons, ["odds_ratio_indirect_mean"]))
        .reset_index(drop=True)
        .groupby("layer_direct_mean")
        .agg("mean")[["odds_ratio_indirect_mean", "odds_ratio_indirect_std"]]
        .reset_index()
    )


def main(folder_name, save_dir):
    # For plotting purposes.

    cmap = ["#9b59b6", "#3498db", "#2ecc71", "#e74c3c", "#34495e", "#95a5a6"]

    # Load results for all the models.
    neuron_effect_fnames = [f for f in os.listdir(folder_name) if "neuron_effects" in f]
    modelname_to_effects = {}
    for f in neuron_effect_fnames:
        modelname = f.split("_")[0]
        effects = f.split("_")[1]
        path = os.path.join(folder_name, f)
        try:
            modelname_to_effects[modelname][effects] = pd.read_csv(path)
        except KeyError:
            modelname_to_effects[modelname] = {}
            modelname_to_effects[modelname][effects] = pd.read_csv(path)

    sanitise_ntype = {"v-action": "Verbal (content)", "v-existential": "Verbal (existential)", "np-nonexistential": "NP (nonexistential)", "np-existential": "NP (existential)", "np-num2none": "NP (number-to-none)", "sw": "Sentence-wide"}


    for ntype in ["v-action", "v-existential", "np-nonexistential", "np-existential", "np-num2none", "sw"]:
        plt.figure(figsize=(10, 5))
        color_index = 0
        # Plot 6 figures for each negation type, showing the NIEs of the two model sizes
        for k, v in modelname_to_effects.items():
            # Get top 5% neurons
            vals = get_top_perc_per_layer(v["ntype"], ntype, n = 5)

            # Plot a line for mean per layer.
            plt.plot(
                vals["layer_direct_mean"].values,
                vals["odds_ratio_indirect_mean"].values,
                label=k,
                color=cmap[color_index],
            )
            # Fill in between standard deviation.
            plt.fill_between(
                vals["layer_direct_mean"].values,
                vals["odds_ratio_indirect_mean"].values
                - vals["odds_ratio_indirect_std"].values,
                vals["odds_ratio_indirect_mean"].values
                + vals["odds_ratio_indirect_std"].values,
                alpha=0.08,
                color=cmap[color_index],
            )
            color_index += 1
        plt.title(sanitise_ntype[ntype] + " negation",  fontsize=24)
        plt.xlabel("Layer index", fontsize=24)
        plt.ylabel("Indirect effect", fontsize=24)
        plt.yticks([1 + i / 100 for i in range(0, 6)], [str(i / 100) for i in range(0, 6)])
        plt.legend(loc='upper left')

        plt.savefig(
            os.path.join(save_dir, ntype + "_neuron_layer_effect.pdf"),
            format="pdf",
            bbox_inches="tight",
        )


        # Now do all available models individually showing the different correctness categories
        for k, v in modelname_to_effects.items():
            color_index = 0

            plt.figure(figsize=(10, 5))

            tops = []
            bottoms = []
            for ccat in ["c_c", "c_i", "i_c", "i_i"]:
                vals_top = get_top_perc_per_layer(v["ntype-ccat"], ntype, ccat, n = 5)
                vals_bottom = get_bottom_perc_per_layer(v["ntype-ccat"], ntype, ccat, n = 5)
                vals = vals_top.join(vals_bottom, lsuffix = "_top", rsuffix = "_bottom")

                # Plot top 5% of neurons
                tops.append(plt.plot(
                    vals["layer_direct_mean_top"].values,
                    vals["odds_ratio_indirect_mean_top"].values,
                    label=ccat.replace("_", "-") + " top",
                    color=cmap[color_index],
                )[0])

                plt.fill_between(
                    vals["layer_direct_mean_top"].values,
                    vals["odds_ratio_indirect_mean_top"].values
                    - vals["odds_ratio_indirect_std_top"].values,
                    vals["odds_ratio_indirect_mean_top"].values
                    + vals["odds_ratio_indirect_std_top"].values,
                    alpha=0.08,
                    color=cmap[color_index],
                )

                # Plot bottom 5% of neurons
                bottoms.append(plt.plot(
                    vals["layer_direct_mean_bottom"].values,
                    vals["odds_ratio_indirect_mean_bottom"].values,
                    label=ccat.replace("_", "-") + " bottom",
                    color=cmap[color_index],
                    linestyle='dashed'
                )[0])

                plt.fill_between(
                    vals["layer_direct_mean_top"].values,
                    vals["odds_ratio_indirect_mean_bottom"].values
                    - vals["odds_ratio_indirect_std_bottom"].values,
                    vals["odds_ratio_indirect_mean_bottom"].values
                    + vals["odds_ratio_indirect_std_bottom"].values,
                    alpha=0.08,
                    color=cmap[color_index],
                )
                color_index += 1
            plt.title(k + " " + sanitise_ntype[ntype] + " negation", fontsize=24)
            plt.xlabel("Layer index", fontsize=24)
            plt.ylabel("Indirect effect", fontsize=24)
            plt.yticks([1 + i / 100 for i in range(-5, 6)], [str(i / 100) for i in range(-5, 6)])

            # Add separate legends for the top and bottom lines
            l1 = plt.legend(tops, ["c-c", "c-i", "i-c", "i-i"], loc='upper left')
            plt.gca().add_artist(l1)
            plt.legend(bottoms, ["c-c", "c-i", "i-c", "i-i"], loc='lower left')

            plt.savefig(
                os.path.join(save_dir, ntype + "_neuron_layer_effect_" + k + ".pdf"),
                format="pdf",
                bbox_inches="tight",
            )

    # Plot 2 figures showing all 6 negation types for each model
    for k, v in modelname_to_effects.items():
        plt.figure(figsize=(10, 5))
        color_index = 0

        for ntype in ["v-action", "v-existential", "np-nonexistential", "np-existential", "np-num2none", "sw"]:
            # Get top 5% neurons
            vals = get_top_perc_per_layer(v["ntype"], ntype, n = 5)

            # Plot a line for mean per layer.
            plt.plot(
                vals["layer_direct_mean"].values,
                vals["odds_ratio_indirect_mean"].values,
                label=sanitise_ntype[ntype],
                color=cmap[color_index],
            )
            # Fill in between standard deviation.
            plt.fill_between(
                vals["layer_direct_mean"].values,
                vals["odds_ratio_indirect_mean"].values
                - vals["odds_ratio_indirect_std"].values,
                vals["odds_ratio_indirect_mean"].values
                + vals["odds_ratio_indirect_std"].values,
                alpha=0.08,
                color=cmap[color_index],
            )
            color_index += 1
        plt.title(k,  fontsize=24)
        plt.xlabel("Layer index", fontsize=24)
        plt.ylabel("Indirect effect of top neurons", fontsize=24)
        plt.yticks([1 + i / 100 for i in range(0, 6)], [str(i / 100) for i in range(0, 6)])
        plt.legend(loc='upper left')

        plt.savefig(
            os.path.join(save_dir, k + "_neuron_layer_effect.pdf"),
            format="pdf",
            bbox_inches="tight",
        )

    print("Success, all figures were written.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: python ", sys.argv[0], "<neuron effect files directory> <output directory for figures>")
    # e.g., results/20191114...
    folder_name = sys.argv[1]
    save_dir = sys.argv[2]
    main(folder_name, save_dir)
