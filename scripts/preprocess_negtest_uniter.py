"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess NLVR annotations into LMDB
"""
import sys
sys.path.append('./')
sys.path.append('UNITER')

import argparse
import json
import os
from os.path import exists

from cytoolz import curry
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer

from UNITER.data.data import open_lmdb


@curry
def bert_tokenize(tokenizer, text):
    neg_list = ["not", "isn't", "aren't", "doesn't", "don't", "can't", "cannot", "shouldn't", "won't", "wouldn't", "no", "none", "nobody", "nothing", "nowhere", "neither", "nor", "never", "without"]
    ids = []
    neg_pos_ids = []
    words = []
    for i, word in enumerate(text.strip().split()):
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        if word.lower()in neg_list:
            neg_pos_ids.extend([len(ids) + idx for idx in range(len(ws))])
        words.extend(ws)
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids, neg_pos_ids, words


def process_nlvr2(jsonl, db, tokenizer, missing=None):
    id2len = {}
    txt2img = {}  # not sure if useful
    id2negidx = {}
    id2tok_sent = {}
    for line in tqdm(jsonl, desc='processing NLVR2'):
        example = json.loads(line)
        id_ = example['identifier']
        img_id = '-'.join(id_.split('-')[:3])
        img_fname = (f'nlvr2_{img_id}-img0.npz', f'nlvr2_{img_id}-img1.npz')
        if missing and (img_fname[0] in missing or img_fname[1] in missing):
            continue
        input_ids, neg_pos_ids, tok_words = tokenizer(example['sentence'])
        if 'label' in example:
            target = 1 if example['label'] == 'True' else 0
        else:
            target = None
        txt2img[id_] = img_fname
        id2len[id_] = len(input_ids)
        id2negidx[id_] = neg_pos_ids
        id2tok_sent[id_] = tok_words
        example['input_ids'] = input_ids
        example['img_fname'] = img_fname
        example['target'] = target
        db[id_] = example
    return id2len, txt2img, id2negidx, id2tok_sent


def main(opts):
    if not exists(opts.output):
        os.makedirs(opts.output)
    else:
        raise ValueError('Found existing DB. Please explicitly remove '
                         'for re-processing')
    meta = vars(opts)
    meta['tokenizer'] = opts.toker
    toker = BertTokenizer.from_pretrained(
        opts.toker, do_lower_case='uncased' in opts.toker)
    tokenizer = bert_tokenize(toker)
    meta['UNK'] = toker.convert_tokens_to_ids(['[UNK]'])[0]
    meta['CLS'] = toker.convert_tokens_to_ids(['[CLS]'])[0]
    meta['SEP'] = toker.convert_tokens_to_ids(['[SEP]'])[0]
    meta['MASK'] = toker.convert_tokens_to_ids(['[MASK]'])[0]
    meta['v_range'] = (toker.convert_tokens_to_ids('!')[0],
                       len(toker.vocab))
    with open(f'{opts.output}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)

    open_db = curry(open_lmdb, opts.output, readonly=False)
    with open_db() as db:
        with open(opts.annotation) as ann:
            if opts.missing_imgs is not None:
                missing_imgs = set(json.load(open(opts.missing_imgs)))
            else:
                missing_imgs = None
            id2lens, txt2img, id2negidx, id2tok_sent = process_nlvr2(ann, db, tokenizer, missing_imgs)

    with open(f'{opts.output}/id2len.json', 'w') as f:
        json.dump(id2lens, f)
    with open(f'{opts.output}/txt2img.json', 'w') as f:
        json.dump(txt2img, f)
    with open(f'{opts.output}/id2negidx.json', 'w') as f:
        json.dump(id2negidx, f)
    with open(f'{opts.output}/id2tok_sent.json', 'w') as f:
        json.dump(id2tok_sent, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', required=True,
                        help='annotation JSON')
    parser.add_argument('--missing_imgs',
                        help='some training image features are corrupted')
    parser.add_argument('--output', required=True,
                        help='output dir of DB')
    parser.add_argument('--toker', default='bert-base-cased',
                        help='which BERT tokenizer to used')
    args = parser.parse_args()
    main(args)
