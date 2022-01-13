"""Run all the extraction for a model across many templates.
"""
# add UNITER to directories to search for python packages
import sys
sys.path.append('/UNITER')

import argparse
import os
from datetime import datetime
import json

import torch
from torch.utils.data import DataLoader

from experiment import Model
from utils_cma import convert_results_to_pd

from horovod import torch as hvd

from data import (DetectFeatLmdb, TxtTokLmdb,
                  PrefetchLoader, TokenBucketSampler,
                  Nlvr2TripletEvalDataset, nlvr2_triplet_eval_collate)

from model.model import UniterConfig

from utils.misc import Struct


def load_examples(opts, train_opts):
    ''' Loads examples in pairs; returns a list of tuples, where each entry is the input for (non-negated version, negated version)
    '''

    eval_collate_fn = nlvr2_triplet_eval_collate
    img_db = DetectFeatLmdb(opts.img_db,
                            train_opts.conf_th, train_opts.max_bb,
                            train_opts.min_bb, train_opts.num_bb,
                            opts.compressed_db)
    txt_db = TxtTokLmdb(opts.txt_db, -1)
    dset = Nlvr2TripletEvalDataset(txt_db, img_db, train_opts.use_img_type)

    data_dict = {}

    eval_dataloader = DataLoader(dset, batch_sampler=None,
                                 num_workers=opts.n_workers,
                                 pin_memory=opts.pin_mem,
                                 collate_fn=eval_collate_fn)
    eval_dataloader = PrefetchLoader(eval_dataloader)

    for i, item in enumerate(eval_dataloader):
        data_dict[item['qids'][0]] = item

    corresponding_examples = []

    # Grab the IDs of all negated examples
    for id in dset.ids:
        if "-n-" in id:
            # Grab the corresponding non-negated IDs
            corresponding_id = "-".join(id.split("-", 4)[0:4])
            # Add the (non-negated, negated) pair to the list of examples
            corresponding_tuple = (data_dict[corresponding_id], data_dict[id])
            corresponding_examples.append(corresponding_tuple)

    data_dict.clear()

    return corresponding_examples

def run_all(
    opts,
    model_type="UNITER-base",
    device="cuda",
    out_dir=".",
    random_weights=False,
    batch_size = 500
):

    # Load trained model configuration
    train_opts = Struct(json.load(open(f'{opts.train_dir}/log/hps.json')))

    # Load the negation test set as a list of (non-negated, negated) tuples
    loaded_examples = load_examples(opts, train_opts)
    intervention_types = ["negate_direct", "negate_indirect"]

    # Load finetuned model
    ckpt_file = f'{opts.train_dir}/ckpt/model_step_{opts.ckpt}.pt'
    model_config = UniterConfig.from_json_file(f'{opts.train_dir}/log/model.json')
    model = Model(ckpt_file, model_config, opts, device=device, random_weights=random_weights)

    # Set up folder if it does not exist.
    dt_string = datetime.now().strftime("%Y%m%d")
    folder_name = dt_string + "_neuron_intervention"
    base_path = os.path.join(out_dir, "results", folder_name)
    if random_weights:
        base_path = os.path.join(base_path, "random")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Iterate over each pair of (non-negated, negated) examples
    for example_pair in loaded_examples:

        # id1 - non-negated, id2 - negated
        id1 = example_pair[0]['qids'][0]
        id2 = example_pair[1]['qids'][0]
        print(f"Running with ids: {id1}, {id2}", flush=True)

        for itype in intervention_types:

            print("\t Running with intervention: {}".format(itype), flush=True)

            intervention_results = model.neuron_intervention_single_experiment(
                example_pair, itype, alpha=1.0, bsize = batch_size
            )

            df = convert_results_to_pd(example_pair, intervention_results)
            # Generate file name.
            #temp_string = "_".join(temp.replace("{}", "X").split())
            model_type_string = model_type
            fname = "_".join([id1, id2, itype, model_type_string])
            # Finally, save each exp separately.
            df.to_csv(os.path.join(base_path, fname + ".csv"))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hvd.init()

    parser = argparse.ArgumentParser(description="Run negation interventions.")

    parser.add_argument("--txt_db",
                        type=str, required=True,
                        help="The input test corpus.")
    parser.add_argument("--img_db",
                        type=str, required=True,
                        help="The input test images.")

    parser.add_argument("--train_dir", type=str, required=True,
                        help="The directory storing NLVR2 finetuning output")
    parser.add_argument("--ckpt", type=int, required=True,
                        help="Model checkpoint.")

    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    parser.add_argument('--fp16', action='store_true',
                        help="fp16 inference")
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')

    parser.add_argument("--batch_size", type=int, default = 500,
                        help="neuron batch size")

    parser.add_argument("--model", type=str, default="UNITER-base",
                        help="UNITER-base or UNITER-large")

    parser.add_argument("--out_dir", default=".", type=str,
                        help="Path of the result folder.")


    opts = parser.parse_args()


    run_all(
        opts,
        opts.model,
        device,
        opts.out_dir,
        batch_size = opts.batch_size
    )
