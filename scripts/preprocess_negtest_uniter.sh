# Original script from https://github.com/ChenRocks/UNITER
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

OUT_DIR="negation-test-set/uniter_negation_test/"
ANN_DIR="negation-test-set"

set -e

URL='https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data'
if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi
if [ ! -d $ANN_DIR ]; then
    mkdir -p $ANN_DIR
fi

SPLIT='negation_test_set'

echo "preprocessing ${SPLIT} annotations..."

docker run --ipc=host --rm -it \
        --mount src=$(pwd),dst=/src,type=bind \
        --mount src=$(pwd)/$OUT_DIR,dst=/txt_db,type=bind \
	--mount src=$(pwd)/$ANN_DIR,dst=/ann,type=bind,readonly \
	-w /src chenrocks/uniter   \
	python scripts/preprocess_negtest_uniter.py --annotation /ann/$SPLIT.jsonl \
                         --output /txt_db/nlvr2_${SPLIT}.db


echo "done"
