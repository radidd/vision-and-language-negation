# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

TXT_DB=$1
IMG_DB=$2
MODEL_DIR=$3

UNITER_DIR=$(pwd)/../UNITER

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi


docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount source=$UNITER_DIR,destination=/UNITER,type=bind,readonly \
    --mount source=$TXT_DB,destination=/txt_db,type=bind,readonly \
    --mount source=$IMG_DB,destination=/img_db,type=bind,readonly \
    --mount source=$MODEL_DIR,destination=/model,type=bind,readonly \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /src chenrocks/uniter
