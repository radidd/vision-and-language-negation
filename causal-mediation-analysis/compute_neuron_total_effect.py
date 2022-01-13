import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, median_abs_deviation
import json


def get_orig_correctness(results_df):
    ''' Returns the correctness of the original example. '''
    orig_label = results_df['orig_label'].all()
    orig_assigned = False if float(results_df['candidate1_orig_prob'].unique()) > 0.5 else True
    orig_correctness = "orig_correct" if orig_label == orig_assigned else "orig_incorrect"

    return orig_correctness

def main(folder_name, model_name, neg_test_file = 'neg_test.json'):

    # Load the negation type for each example
    neg_types = {}
    with open(neg_test_file, 'r') as test_set:
        for line in test_set:
            js_line = json.loads(line)
            try:
                neg_types[str(js_line["identifier"])] = str(js_line["negation_type"])
            except KeyError:
                neg_types[str(js_line["identifier"])] = None

    # Get the indirect effect files
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

    # Create dictionary for means per negation type and original correctness
    means_dict = {}
    for ntype in ["v-action", "v-existential", "np-nonexistential", "np-existential", "np-num2none", "sw", "overall"]:
        means_dict[ntype] = {}
        for oc in ["orig_correct", "orig_incorrect"]:
            means_dict[ntype][oc] = []


    for path in files:
        temp_df = pd.read_csv(path)
        #print(path)

        # Get negation type
        id_neg = path.split("/")[-1].split("_")[1]
        ntype = neg_types[id_neg]

        # Get correctness of original example
        orig_correctness = get_orig_correctness(temp_df)

        # Response variable under the negate and null operations
        if temp_df["orig_label"].all() == True:
            temp_df["orig_effect"] = (
                temp_df["candidate1_orig_prob"] / temp_df["candidate2_orig_prob"]
            )
            temp_df["neg_effect"] = (
                temp_df["candidate1_neg_prob"] / temp_df["candidate2_neg_prob"]
            )
        else:
            temp_df["orig_effect"] = (
                temp_df["candidate2_orig_prob"] / temp_df["candidate1_orig_prob"]
            )
            temp_df["neg_effect"] = (
                temp_df["candidate2_neg_prob"] / temp_df["candidate1_neg_prob"]
            )

        temp_df["total_effect"] = temp_df["neg_effect"] / temp_df["orig_effect"]
        mean_total = temp_df["total_effect"].mean()

        # Record this based on the appropriate negation type and for the overall total effect
        means_dict[ntype][orig_correctness].append(mean_total - 1)
        means_dict["overall"][orig_correctness].append(mean_total - 1)

    total_effects_dict = {}
    for ntype in means_dict.keys():
        for oc in means_dict[ntype].keys():
            try:
                total_effects_dict[ntype][oc] = {}
                total_effects_dict[ntype][oc]["mean"] = np.mean(means_dict[ntype][oc])
                total_effects_dict[ntype][oc]["std"] = np.std(means_dict[ntype][oc])
                total_effects_dict[ntype][oc]["max"] = np.amax(means_dict[ntype][oc])
            except KeyError:
                total_effects_dict[ntype] = {}
                total_effects_dict[ntype][oc] = {}
                total_effects_dict[ntype][oc]["mean"] = np.mean(means_dict[ntype][oc])
                total_effects_dict[ntype][oc]["std"] = np.std(means_dict[ntype][oc])
                total_effects_dict[ntype][oc]["max"] = np.amax(means_dict[ntype][oc])
            print(
            "The total effect of this model for {} {} is {:.3f} (std: {:.3f}, max: {:.3f})".format(ntype, oc, total_effects_dict[ntype][oc]["mean"], total_effects_dict[ntype][oc]["std"], total_effects_dict[ntype][oc]["max"])
    )

    with open(os.path.join(folder_name, model_name + "_total_effects.json"), "w+") as fout:
        json.dump(total_effects_dict, fout)

    print("Results saved in {}".format(str(os.path.join(folder_name, model_name + "_total_effects.json"))))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("USAGE: python ", sys.argv[0], "<folder_name> <model_name> <negation_test_file>")
    # e.g., results/20191114...
    folder_name = sys.argv[1]
    # UNITER-base, UNITER-large
    model_name = sys.argv[2]
    # negation test set file
    neg_test_file = sys.argv[3]
    main(folder_name, model_name, neg_test_file)
