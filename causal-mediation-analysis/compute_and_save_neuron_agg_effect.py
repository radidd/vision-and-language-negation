"""
This script save the direct/indirect effects for each neuron averaging across different groups depending
on negation type and correctness category.

Usage:
    python compute_and_save_neuron_agg_effect.py $result_file_path $model_name $negation_test_set_file
"""
import os
import sys
import json

import pandas as pd


def get_correctness_category(results_df):

    orig_label = results_df['orig_label'].all()
    neg_label = results_df['neg_label'].all()

    orig_assigned = False if float(results_df['candidate1_orig_prob'].unique()) > 0.5 else True
    neg_assigned = False if float(results_df['candidate1_neg_prob'].unique()) > 0.5 else True

    orig_correctness = "c" if orig_label == orig_assigned else "i"
    neg_correctness = "c" if neg_label == neg_assigned else "i"

    return "_".join([orig_correctness, neg_correctness])

def analyze_effect_results(results_df, effect):
    # Calculate response variable under the null condition and with the neuron intervention
    if results_df["orig_label"].all() == True:
        odds_base = (
            results_df["candidate1_orig_prob"] / results_df["candidate2_orig_prob"]
        )
        odds_intervention = (
            results_df["candidate1_prob"] / results_df["candidate2_prob"]
        )
    else:
        odds_base = (
            results_df["candidate2_orig_prob"] / results_df["candidate1_orig_prob"]
        )
        odds_intervention = (
            results_df["candidate2_prob"] / results_df["candidate1_prob"]
        )
    odds_ratio = odds_intervention / odds_base
    results_df["odds_ratio"] = odds_ratio

    # Add correctness category to dataframe
    results_df["correctness_cat"] = get_correctness_category(results_df=results_df)

    # Get the odds ratio for each neuron in each layer
    results_df = results_df.pivot("neuron", "layer", "odds_ratio")


def get_all_effects(fname):
    """
    Give fname from a direct effect file
    """
    # Step 1: Load results for current file

    indirect_result_df = pd.read_csv(fname)
    analyze_effect_results(
        results_df=indirect_result_df, effect="indirect"
    )
    fname = fname.replace("_indirect_", "_direct_")
    direct_result_df = pd.read_csv(fname)
    analyze_effect_results(
        results_df=direct_result_df, effect="direct"
    )

    # Step 2: Join the two DF's
    total_df = direct_result_df.join(
        indirect_result_df, lsuffix="_direct", rsuffix="_indirect"
    )[
        [
            "layer_direct",
            "neuron_direct",
            "odds_ratio_indirect",
            "odds_ratio_direct"
        ]
    ]
    total_df["total_causal_effect"] = (
        total_df["odds_ratio_indirect"] + total_df["odds_ratio_direct"] - 1
    )
    total_df["correctness_cat"] = direct_result_df["correctness_cat"]

    return total_df


def main(folder_name, model_name, neg_test_file = 'neg_test.json'):

    # Load the negation type for each negated sample
    neg_types = {}
    with open(neg_test_file, 'r') as test_set:
        for line in test_set:
            js_line = json.loads(line)
            try:
                neg_types[str(js_line["identifier"])] = str(js_line["negation_type"])
            except KeyError:
                neg_types[str(js_line["identifier"])] = None

    # Get all direct and indirect effect files
    fnames = [
        f
        for f in os.listdir(folder_name)
        if "_" + model_name + ".csv" in f and f.endswith("csv")
    ]
    paths = [os.path.join(folder_name, f) for f in fnames]

    files = [
        f
        for f in paths
        if "indirect" in f
        if os.path.exists(f.replace("indirect", "direct"))
    ]

    # Prepare dataframes for each of the 6 negation types
    va_dfs = []
    ve_dfs = []
    npne_dfs = []
    npe_dfs = []
    n2n_dfs = []
    sw_dfs = []
    ntypes = {"v-action" : va_dfs, "v-existential" : ve_dfs, "np-nonexistential" : npne_dfs, "np-existential" : npe_dfs, "np-num2none" : n2n_dfs, "sw" : sw_dfs}


    for path in files:
        # Get negation type of current example
        id_neg = path.split("/")[-1].split("_")[1]
        ntype = neg_types[id_neg]

        # Get indirect and direct effects
        ntypes[ntype].append(get_all_effects(path))

    va_df = pd.concat(va_dfs)
    ve_df = pd.concat(ve_dfs)
    npne_df = pd.concat(npne_dfs)
    npe_df = pd.concat(npe_dfs)
    n2n_df = pd.concat(n2n_dfs)
    sw_df = pd.concat(sw_dfs)

    va_df["negation_type"] = "v-action"
    ve_df["negation_type"] = "v-existential"
    npne_df["negation_type"] = "np-nonexistential"
    npe_df["negation_type"] = "np-existential"
    n2n_df["negation_type"] = "np-num2none"
    sw_df["negation_type"] = "sw"

    overall_df = pd.concat([va_df, ve_df, npne_df, npe_df, n2n_df, sw_df])

    del va_df
    del ve_df
    del npne_df
    del npe_df
    del n2n_df
    del sw_df

    overall_df["neuron"] = (
        overall_df["layer_direct"].map(str) + "-" + overall_df["neuron_direct"].map(str)
    )

    # Calculate the effects by negation type and correctness category for each neuron
    neuron_effect_by_ntype_ccat_df = (
        overall_df.groupby(["negation_type", "correctness_cat", "neuron"])
        .agg(
            {
                "layer_direct": ["mean"],
                "neuron_direct": ["mean"],
                "odds_ratio_indirect": ["mean", "std"],
                "odds_ratio_direct": ["mean", "std"],
                "total_causal_effect": ["mean", "std"],
            }
        )
        .reset_index()
    )

    neuron_effect_by_ntype_ccat_df.columns = [
        "_".join(col).strip() for col in neuron_effect_by_ntype_ccat_df.columns.values
    ]

    # Calculate the effects by negation type for each neuron
    neuron_effect_by_ntype_df = (
        overall_df.groupby(["negation_type","neuron"])
        .agg(
            {
                "layer_direct": ["mean"],
                "neuron_direct": ["mean"],
                "odds_ratio_indirect": ["mean", "std"],
                "odds_ratio_direct": ["mean", "std"],
                "total_causal_effect": ["mean", "std"],
            }
        )
        .reset_index()
    )

    neuron_effect_by_ntype_df.columns = [
        "_".join(col).strip() for col in neuron_effect_by_ntype_df.columns.values
    ]

    # Calculate the effects regardless of negation type for each neuron
    neuron_effect_overall_df = (
        overall_df.groupby("neuron")
        .agg(
            {
                "layer_direct": ["mean"],
                "neuron_direct": ["mean"],
                "odds_ratio_indirect": ["mean", "std"],
                "odds_ratio_direct": ["mean", "std"],
                "total_causal_effect": ["mean", "std"],
            }
        )
        .reset_index()
    )

    neuron_effect_overall_df.columns = [
        "_".join(col).strip() for col in neuron_effect_overall_df.columns.values
    ]

    # Write to files
    ntype_ccat_path_name = os.path.join(folder_name, model_name + "_ntype-ccat_neuron_effects.csv")
    ntype_path_name = os.path.join(folder_name, model_name + "_ntype_neuron_effects.csv")
    overall_path_name = os.path.join(folder_name, model_name + "_overall_neuron_effects.csv")
    neuron_effect_by_ntype_ccat_df.to_csv(ntype_ccat_path_name)
    neuron_effect_by_ntype_df.to_csv(ntype_path_name)
    neuron_effect_overall_df.to_csv(overall_path_name)
    print("Effect by negation type and correctness category saved to {}".format(ntype_ccat_path_name))
    print("Effect by negation type saved to {}".format(ntype_path_name))
    print("Overall effect saved to {}".format(overall_path_name))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("USAGE: python ", sys.argv[0], "<interventions folder> <model type> <negation_test_file>")
    # e.g., results/20191114...
    folder_name = sys.argv[1]
    # UNITER-base, UNITER-large
    model_name = sys.argv[2]
    # negation test set files
    neg_test_file = sys.argv[3]

    main(folder_name, model_name, neg_test_file)
