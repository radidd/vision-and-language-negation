# Investigating Negation in Pre-trained Vision-and-language Models

Code and data for the experiments in the paper [Investigating Negation in Pre-trained Vision-and-language Models](https://aclanthology.org/2021.blackboxnlp-1.27/).

## Under construction
This repository is still under construction. The negation test set is available (see details below). Experiment code is coming soon. Thank you for your patience!

## Requirements
The experiments were conducted using the [LXMERT](https://github.com/airsplay/lxmert) and [UNITER](https://github.com/ChenRocks/UNITER) models.

All experiments that include UNITER require Docker. From the UNITER instructions:
> We provide Docker image for easier reproduction. Please install the following:
> - [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+),
> - [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) (19.03+),
> - [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).
>
> Our scripts require the user to have the docker group membership so that docker commands can be run without sudo. We only support Linux with NVIDIA GPUs. We test on Ubuntu 18.04 and V100 cards. We use mixed-precision training hence GPUs with Tensor Cores are recommended.

Additionally, please install the packages in requirements.txt:
```
virtualenv venv -p python3
source venv/bin/activate
pip install -r requirements.txt
```

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

## Models
Finetune models for NLVR2 following the instructions in the relevant submodules. To reproduce all experiments from the paper the following models are needed:
- LXMERT (standard settings; model also provided in original repository)
- UNITER paired-attention base (standard settings; model also provided in original repository)
- UNITER triplet base (training configuration file: config/train-nlvr2-base-triplet.json)
- UNITER triplet large (training configuration file: config/train-nlvr2-large-triplet.json)

## Evaluation on negation test set

To evaluate the models on the negation test set, first run inference.

#### LXMERT
```bash scripts/run_inference_lxmert.sh <model location> <output dir> <device>```

#### UNITER
Launch docker container specifying location of model to be used and GPU IDs:

``` bash scripts/launch_model_container.sh <path to model> <gpu ids> ```

Inside container:

``` bash scripts/run_inference_uniter.sh <model location> <model checkpoint> <output dir>```

Model location is “nlvr-base” if using the model provided by the authors of UNITER and checkpoint is “6500”. If using your own trained model, specify “nlvr2/default” or “nlvr2/large” and the relevant checkpoint.

#### Results
To get accuracy on the negation test set, run the following script:

```python scripts/compute_accuracy_by_negation_category.py --predictions_file <predictions file> --labels_file <labels file> --output_file <output file>```

## Causal mediation analysis

Causal mediation analysis is conducted on the triplet version of UNITER. See above for training instructions.

First, we need to run neuron interventions. Launch the docker container:
```
cd causal-mediation-analysis
bash launch_container.sh <location of text database> <location of image database> <model location>
```

Inside the docker, run the following command, supplying the checkpoint to use and whether the model is base or large:
```
python run_negation_neuron_interventions.py --txt_db /txt_db/nlvr2_negation_test_set.db --img_db /img_db/nlvr2_test --train_dir /model --ckpt <checkpoint> --model <UNITER-base/UNITER-large>
```

By default results are saved under ./results/, use --out_dir to specify output directory.

#### Total effects
To compute the total effect, run (outside docker):

```python compute_neuron_total_effect.py <interventions results directory>  <model type (UNITER-base/UNITER-large)> <negation test set>```

Output is saved in the interventions results directory.

To plot the total effects of UNITER-base and UNITER-large (requires outputs of the total effects script for both base and large):

```python plot_total_effects.py <UNITER-base total effects file> <UNITER-large total effects file> <save directory>```

This results in two figures for the total effects of originally correct and originally incorrect examples, correspinding to Figure 3 in the paper.

#### Natural indirect effect
To compute neuron indirect effects (for one model) run:
```
python compute_and_save_neuron_agg_effect.py <interventions results directory> <model type (UNITER-base/UNITER-large)> <negation test set>
```

This computes indirect effects for each neuron and averages them across negation and correctness categories. The effects are saved in three files in the interventions directory:
- Effect by negation type and correctness category: UNITER-base_ntype-ccat_neuron_effects.csv
- Effect by negation type: UNITER-base_ntype_neuron_effects.csv
- Overall effect: UNITER-base_overall_neuron_effects.csv

To plot neuron indirect effects per layer, run:
```
python plot_neuron_effect_per_layer.py <interventions results directory> <output directory for figures>
```
