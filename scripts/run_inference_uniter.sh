#!/bin/bash

MODEL_LOCATION=$1
CKPT=$2
OUTPUT_DIR=$3

python UNITER/inf_nlvr2.py \
       --txt_db /txt/nlvr2_negation_test_set.db/ \
       --img_db /img/nlvr2_test/ \
       --train_dir /storage/$MODEL_LOCATION/ \
       --ckpt $CKPT \
       --output_dir $OUTPUT_DIR \
       --fp16
