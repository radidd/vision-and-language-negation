# Original script from https://github.com/airsplay/lxmert

MODEL_LOCATION=$1
OUTPUT_DIR=$2
DEVICES=$3
RUN_DIR=$(pwd)

cp negation-test-set/lxmert_negation_test.jsonl lxmert/data/nlvr2/negation_test.json
cd lxmert

CUDA_VISIBLE_DEVICES=$DEVICES PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/nlvr2.py \
    --tiny --llayers 9 --xlayers 5 --rlayers 5 \
    --tqdm --output $RUN_DIR/$OUTPUT_DIR \
    --load $RUN_DIR/$MODEL_LOCATION --test negation_test --batchSize 1024
