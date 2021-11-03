#!/bin/bash

PATH_TO_STORAGE=$1
CUDA_VISIBLE_DEVICES=$2

source scripts/launch_container.sh $(pwd)/negation-test-set/uniter_negation_test $PATH_TO_STORAGE/img_db \
           $PATH_TO_STORAGE/finetune $PATH_TO_STORAGE/pretrained
