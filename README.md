# Investigating Negation in Pre-trained Vision-and-language Models

Code and data for the experiments in the paper [Investigating Negation in Pre-trained Vision-and-language Models](https://aclanthology.org/2021.blackboxnlp-1.27/).

### Under construction
This repository is still under construction, thank you for your patience.

## Requirements
The experiments were conducted using the LXMERT and UNITER models.

Requirements for LXMERT:


Requirements for UNITER:
> We provide Docker image for easier reproduction. Please install the following:
> - nvidia driver (418+),
> - Docker (19.03+),
> - nvidia-container-toolkit.
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

## Models
Finetune models for NLVR2 following the instructions in the relevant submodules.

## Evaluation on negation test set

Run inference on negation test set:

### LXMERT

### UNITER
Launch docker container specifying location of model to be used:
``` bash scripts/launch_model_container.sh <path to model> <gpu ids> ```

Inside container:
``` bash scripts/run_inference_uniter.sh <model location> <model checkpoint> <output dir>``` 

Model location is “nlvr-base” if using the model provided by the authors of UNITER and checkpoint is “6500”. If using your own trained model, specify “nlvr2/default” or “nlvr2/large” and the relevant checkpoint. 

### Results

## Causal mediation analysis

We perform causal mediation analysis on the triplet version of UNITER. 
<!-- Provide triplet training instructions. -->

