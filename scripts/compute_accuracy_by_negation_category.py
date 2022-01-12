import json
import numpy as np
import argparse

# Usage: python compute_accuracy_by_negation_category.py --predictions_file <predictions_file> --labels_file <labels_file> --output_file <output_file>

def compute_consistency(n_assign, n_label, p_assign, p_label):
    ''' Determines the consistency between a negated and non-negated sample.
        Returns:
        - c_c if both are correctly predicted
        - i_c if the non-negated sample is incorrectly predicted, but the negated sample is correctly predicted
        - c_i if the negated sample is incorrectly predicted, but the non-negated sample correctly predicted
        - i_i if both are incorrectly predicted
    '''
    if n_assign == n_label and p_assign == p_label:
        return "c_c"
    elif n_assign == n_label:
        return "i_c"
    elif p_assign == p_label:
        return "c_i"
    else:
        return "i_i"

def get_accuracy_by_original_correctness(consistency_dict):
    acc_by_orig_correctness = {}
    for k,v in consistency_dict.items():
        count_orig_correct = consistency_dict[k]["c_c"] + consistency_dict[k]["c_i"]
        count_orig_incorrect = consistency_dict[k]["i_c"] + consistency_dict[k]["i_i"]
        acc_orig_correct = 100 * float(consistency_dict[k]["c_c"]) / count_orig_correct
        acc_orig_incorrect = 100 * float(consistency_dict[k]["i_c"]) / count_orig_incorrect

        acc_by_orig_correctness[k] = {}
        acc_by_orig_correctness[k]["orig_correct"] = (acc_orig_correct, count_orig_correct)
        acc_by_orig_correctness[k]["orig_incorrect"] = (acc_orig_incorrect, count_orig_incorrect)
    return acc_by_orig_correctness

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", help = "File that contains the model predictions (csv).")
    parser.add_argument("--labels_file", help = "File that contains the negation test set (jsonl).")
    parser.add_argument("--output_file", help = "File to write results.")
    args = parser.parse_args()

    # Load predictions from file
    preds = dict()
    with open(args.predictions_file) as f:
        for line in f:
            identifier, assignment = line.strip().split(',')
            preds[identifier] = assignment

    # Load the non-negated examples
    pos_targets = dict()
    with open(args.labels_file) as f:
        for line in f:
            example = json.loads(line)
            if "negation_type" in example.keys():
                continue
            else:
                identifier = example["identifier"]
                pos_targets[identifier] = {}
                label = example["label"].lower()
                pos_targets[identifier]["label"] = label

    ### Create structures to keep track of accuracy
    consistency_dict = dict()
    neg_corrects_dict = dict()
    pos_corrects_dict = dict()
    pos_sets = dict()
    for ntype in ["v-action", "v-existential", "np-nonexistential", "np-existential", "np-num2none", "sw", "overall"]:
        neg_corrects_dict[ntype] = list()
        pos_corrects_dict[ntype] = list()
        pos_sets[ntype] = set()
        consistency_dict[ntype] = {}
        for cons_type in ["c_c", "c_i", "i_c", "i_i"]:
            consistency_dict[ntype][cons_type] = {}
            consistency_dict[ntype][cons_type] = 0




    # Loop through the negated samples and compare predicted and golden labels, also for the corresponding non-negated samples
    with open(args.labels_file) as f:
        for line in f:
            example = json.loads(line)
            if "negation_type" not in example.keys():
                # Skip non-negated examples
                continue
            else:
                # Grab the true label and the predicted label
                identifier = example["identifier"]
                gold_label = example["label"].lower()
                predicted_label = preds[identifier].lower()

                # Grab the prediction and true label of the coresponding non-negated sample
                pos_identifier = example["original_identifier"]
                pos_gold_label = pos_targets[pos_identifier]["label"]
                pos_predicted_label = preds[pos_identifier].lower()

                # Get consistency type and keep track of consistency across negation types
                cons_type = compute_consistency(predicted_label, gold_label, pos_predicted_label, pos_gold_label)
                consistency_dict[example["negation_type"]][cons_type] += 1
                consistency_dict["overall"][cons_type] += 1

                if predicted_label in {'true','false'}:
                    neg_corrects_dict[example["negation_type"]].append(int(predicted_label == gold_label))
                    neg_corrects_dict["overall"].append(int(predicted_label == gold_label))

                    if pos_identifier not in pos_sets[example["negation_type"]]:
                        # When computing the accuracy of the corresponding non-negated samples for each negation category, make sure to not double count the same samples
                        pos_corrects_dict[example["negation_type"]].append(int(pos_predicted_label == pos_gold_label))
                        pos_corrects_dict["overall"].append(int(pos_predicted_label == pos_gold_label))
                        pos_sets[example["negation_type"]].add(pos_identifier)



    # Calculate accuracy by original correctness
    acc_by_orig_correctness = get_accuracy_by_original_correctness(consistency_dict)

    ### Write to file
    tidy_ntype = {"v-action": "Verbal (content)", "v-existential": "Verbal (existential)", "np-nonexistential": "NP (nonexistential)", "np-existential": "NP (existential)", "np-num2none": "NP (number-to-none)", "sw": "Sentence-wide", "overall": "Overall"}
    with open(args.output_file, "w+") as fout:
        fout.write("Accuracy of negated and non-negated examples:\n-------------------------------------------------------------------------------------------------\n")
        for ntype in ["v-action", "v-existential", "np-nonexistential", "np-existential", "np-num2none", "sw", "overall"]:
            fout.write(tidy_ntype[ntype] + " negated: " +  str(100. * np.mean(np.array(neg_corrects_dict[ntype]))) + " (total {})".format(str(len(neg_corrects_dict[ntype])))+"\n")
            fout.write(tidy_ntype[ntype] + " corresponding non-negated: " +  str(100. * np.mean(np.array(pos_corrects_dict[ntype]))) + " (total {})".format(str(len(pos_corrects_dict[ntype])))+"\n")
            fout.write("-----------------------\n")

        fout.write("-------------------------------------------------------------------------------------------------\n")
        fout.write("Accuracy by original correctness: % out of originally correct; % out of originally incorrect\n-------------------------------------------------------------------------------------------------\n")
        for ntype in ["v-action", "v-existential", "np-nonexistential", "np-existential", "np-num2none", "sw", "overall"]:
            fout.write(tidy_ntype[ntype] + ": {}% out of {} orig correct; {}% out of {} orig incorrect; ".format(round(acc_by_orig_correctness[ntype]["orig_correct"][0], 2), acc_by_orig_correctness[ntype]["orig_correct"][1], round(acc_by_orig_correctness[ntype]["orig_incorrect"][0], 2), acc_by_orig_correctness[ntype]["orig_incorrect"][1])+"\n")
