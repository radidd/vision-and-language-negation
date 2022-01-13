from copy import deepcopy

import pandas as pd

def batch(iterable, bsize=1):
    total_len = len(iterable)
    for ndx in range(0, total_len, bsize):
        yield list(iterable[ndx : min(ndx + bsize, total_len)])

def convert_results_to_pd(
    example_pair, intervention_results, layer_fixed=None, neuron_fixed=None
):
    """
    Convert intervention results to data frame

    Args:
        example_pair: a tuple with the (non-negated, negated) versions
        intervention_results: tuple with the intervention results for the current example pair
    """

    results = []

    # print(example_pair[0]['targets'])
    # print(example_pair[1]['targets'])

    intervention = example_pair
    # Candidate 1 is False, Candidate 2 is True
    (
        candidate1_orig_prob,
        candidate2_orig_prob,
        candidate1_neg_prob,
        candidate2_neg_prob,
        candidate1_probs,
        candidate2_probs,
    ) = intervention_results
    # we have results for all layers and all neurons

    results_base = {
        "orig_id": intervention[0]['qids'],
        "neg_id": intervention[1]['qids'],
        "orig_label": bool(intervention[0]['targets']),
        "neg_label": bool(intervention[1]['targets']),
        "candidate1": "False",
        "candidate2": "True",
        # base probs
        "candidate1_orig_prob": float(candidate1_orig_prob),
        "candidate2_orig_prob": float(candidate2_orig_prob),
        "candidate1_neg_prob": float(candidate1_neg_prob),
        "candidate2_neg_prob": float(candidate2_neg_prob),
    }
    if layer_fixed is None:
        for layer in range(candidate1_probs.size(0)):
            for neuron in range(candidate1_probs.size(1)):
                c1_prob, c2_prob = (
                    candidate1_probs[layer][neuron],
                    candidate2_probs[layer][neuron],
                )
                results_single = deepcopy(results_base)
                results_single.update(
                    {  # strings
                        # intervention probs
                        "candidate1_prob": float(c1_prob),
                        "candidate2_prob": float(c2_prob),
                        "layer": layer,
                        "neuron": neuron,
                    }
                )
                results.append(results_single)
    # we have results for all neurons in one layer
    elif neuron_fixed is None:
        for neuron in range(candidate1_probs.size(1)):
            c1_prob, c2_prob = (
                candidate1_probs[0][neuron],
                candidate2_probs[0][neuron],
            )
            results_single = deepcopy(results_base)
            results_single.update(
                {  # strings
                    # intervention probs
                    "candidate1_prob": float(c1_prob),
                    "candidate2_prob": float(c2_prob),
                    "layer": layer_fixed,
                    "neuron": neuron,
                }
            )
            results.append(results_single)
    # we have result for a specific neuron and layer
    else:
        c1_prob, c2_prob = candidate1_probs, candidate2_probs
        results_single = deepcopy(results_base)
        results_single.update(
            {  # strings
                # intervention probs
                "candidate1_prob": float(c1_prob),
                "candidate2_prob": float(c2_prob),
                "layer": layer_fixed,
                "neuron": neuron_fixed,
            }
        )
        results.append(results_single)
    return pd.DataFrame(results)
