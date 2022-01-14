# Investigating Negation in Pre-trained Vision-and-language Models

Code and data for the experiments in the paper [Investigating Negation in Pre-trained Vision-and-language Models](https://aclanthology.org/2021.blackboxnlp-1.27/).

## Under construction
This repository is still under construction. The negation test set is available (see details below). Experiment code is coming soon. Thank you for your patience!

## Requirements
The experiments were conducted using the LXMERT and UNITER models.

Requirements for LXMERT:
- found in lxmert/requirements.txt
+ numpy

Requirements for UNITER:
> We provide Docker image for easier reproduction. Please install the following:
> - [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+),
> - [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) (19.03+),
> - [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).
>
> Our scripts require the user to have the docker group membership so that docker commands can be run without sudo. We only support Linux with NVIDIA GPUs. We test on Ubuntu 18.04 and V100 cards. We use mixed-precision training hence GPUs with Tensor Cores are recommended.

## Data
The negation test set: (``` negation-test-set/negation_test_set.jsonl ```) contains both the negated instances and their corresponding original instances from NLVR2. Negated instances contain a unique identifier, the negated sentence, the negation type, the label and the original identifier, e.g.:

``` {"identifier": "test1-621-0-0-n-1", "sentence": "The left image does not contain a light blue wooden cabinet.", "negation_type": "v-existential", "label": "True", "original_identifier": "test1-621-0-0"}```

The negated instance identifier corresponds to the original identifier with the addition of “-n” signifying that it is negated and “-\[number\]” to uniquely identify all negated instances of the same original sample.

The negation categories are:
- np-existential
- np-nonexistential
- np-num2none
- v-existential
- v-action
- sw

We provide pre-processed data for use with LXMERT (``` negation-test-set/lxmert_negation_test.jsonl ```) and UNITER (``` negation-test-set/uniter_negation_test/nlvr2_negation_test_set.db/ ```). To pre-process the data yourself run:
- LXMERT: ``` python scripts/preprocess_negtest_lxmert.py ```
- UNITER: ``` bash scripts/preprocess_negtest_uniter.sh ```

<!-- ## Models
Finetune models for NLVR2 following the instructions in the relevant submodules.

## Evaluation on negation test set

Run inference on negation test set:

#### LXMERT
./scripts/run_inference_lxmert.sh <model location> <output dir> <device>

#### UNITER
Launch docker container specifying location of model to be used:
``` bash scripts/launch_model_container.sh <path to model> <gpu ids> ```

Inside container:
``` bash scripts/run_inference_uniter.sh <model location> <model checkpoint> <output dir>```

Model location is “nlvr-base” if using the model provided by the authors of UNITER and checkpoint is “6500”. If using your own trained model, specify “nlvr2/default” or “nlvr2/large” and the relevant checkpoint.

#### Results
To get accuracy on the negation test set, run the following script:
python scripts/compute_accuracy_by_negation_category.py --predictions_file <predictions_file> --labels_file <labels_file> --output_file <output_file>

## Causal mediation analysis

We perform causal mediation analysis on the triplet version of UNITER.
Provide triplet training instructions.



To run neuron interventions:
./launch_container.sh <TXT_DB> <IMG_DB> <MODEL_LOCATION>

Inside the docker, run the following command, supplying the checkpoint to use and whether the model is base or large:
python run_negation_neuron_interventions.py --txt_db /txt_db/nlvr2_negation_test_set.db --img_db /img_db/nlvr2_test --train_dir /model --ckpt <checkpoint> --model <UNITER-base, UNITER-large>

By default results are saved under ./results/, use --out_dir to specify output directory.


To compute total and indirect effects, first create a virtual environment
virtualenv venv -p python3
source venv/bin/activate
pip install -r requirements.txt


To compute total effects:
python compute_neuron_total_effect.py <interventions results directory>  <model type (UNITER-base/UNITER-large)> <negation test set>
Run this for both the base and large versions of UNITER

To plot the total effects of UNITER base and UNITER large:
python plot_total_effects.py <base total effects file> <large total effects file> <save directory>
This results in two figures for the total effects of originally correct and originally incorrect examples, correspinding to Figure 3 in the paper.


To compute neuron indirect effects (for one model) run:
python compute_and_save_neuron_agg_effect.py <interventions folder> <model type> <negation test set file>
This computes indirect effects for each neuron and averages them across negation and correctness categories. The effects are saved in three files in the interventions directory:
- Effect by negation type and correctness category: UNITER-base_ntype-ccat_neuron_effects.csv
- Effect by negation type: UNITER-base_ntype_neuron_effects.csv
- Overall effect: UNITER-base_overall_neuron_effects.csv

To plot neuron effects per layer, run:
python plot_neuron_effect_per_layer.py <neuron effect files directory> <output directory for figures>

-->
