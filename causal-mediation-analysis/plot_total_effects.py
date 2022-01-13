import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from math import log

def save_barplot(total_effects_dict, save_loc, correctness):
    """
    """
    sanitise_ntype = {"v-action": "Verbal (content)", "v-existential": "Verbal (existential)", "np-nonexistential": "NP (nonexistential)", "np-existential": "NP (existential)", "np-num2none": "NP (number-to-none)", "sw": "Sentence-wide", "overall" : "Overall"}
    sns.set_context("talk")
    sns.set_style("whitegrid")

    x = np.arange(len(total_effects_dict["base"].keys()))  # the label locations
    width = 0.25  # the width of the bars

    x_labels = ["v-action", "v-existential", "np-nonexistential", "np-existential", "np-num2none", "sw", "overall"]
    fig, ax = plt.subplots(figsize=(10, 8))

    if correctness == "orig_correct":
        rects1 = ax.bar(x - width/2, [round(total_effects_dict["base"][l]["mean"], 2) for l in x_labels], width, label='UNITER-base', color = "#2ecc71")
        rects2 = ax.bar(x + width/2, [round(total_effects_dict["large"][l]["mean"], 2) for l in x_labels], width, label='UNITER-large', color = "#34495e")
        ax.set_ylabel('Total effect')
        ax.set_title('Originally correct examples')
    else:
        rects1 = ax.bar(x - width/2, [round(total_effects_dict["base"][l]["mean"],2) for l in x_labels], width, label='UNITER-base', color = "#2ecc71")
        rects2 = ax.bar(x + width/2, [round(total_effects_dict["large"][l]["mean"],2) for l in x_labels], width, label='UNITER-large', color = "#34495e")
        ax.set_ylabel('Total effect')
        ax.set_title('Originally incorrect examples')

    ax.set_ylim(bottom = -1, top = 1000000)
    ax.set_xticks([-0.6, 0.3, 1.3, 2.45, 3.2, 4.4, 5.75])
    ax.set_xticklabels([sanitise_ntype[n] for n in x_labels], rotation = 45, fontsize = 18)
    plt.yscale('symlog')
    ax.legend()

    ax.bar_label(rects1, padding=2, fontsize = 14)
    ax.bar_label(rects2, padding=2, fontsize = 14)

    plt.savefig(
        save_loc,
        format="pdf",
        bbox_inches="tight",
    )

def main(base_effects, large_effects, save_folder):
    # Load the effects for both models
    orig_correct_effects = {"base":{}, "large": {}}
    orig_incorrect_effects = {"base": {}, "large": {}}
    with open(base_effects, "r") as b, open(large_effects, "r") as l:
        base_dict = json.load(b)
        large_dict = json.load(l)

    for k,v in base_dict.items():
        orig_correct_effects["base"][k] = v["orig_correct"]
        orig_incorrect_effects["base"][k] = v["orig_incorrect"]
    for k,v in large_dict.items():
        orig_correct_effects["large"][k] = v["orig_correct"]
        orig_incorrect_effects["large"][k] = v["orig_incorrect"]

    orig_correct_save_loc = os.path.join(save_folder, "total_effects_orig_correct.pdf")
    orig_incorrect_save_loc = os.path.join(save_folder, "total_effects_orig_incorrect.pdf")
    save_barplot(orig_correct_effects, orig_correct_save_loc, correctness = "orig_correct")
    save_barplot(orig_incorrect_effects, orig_incorrect_save_loc, correctness = "orig_incorrect")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("USAGE: python ", sys.argv[0], "<base_total_effects> <large_total_effects> <save_folder>")

    base_effects = sys.argv[1]
    large_effects = sys.argv[2]
    save_folder = sys.argv[3]

    main(base_effects, large_effects, save_folder)
