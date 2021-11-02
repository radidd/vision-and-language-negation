import json
import os

DATA_ROOT = './negation-test-set'

split2fname = {
    'negation_test': 'negation_test_set'
}

for split, fname in split2fname.items():
    with open(os.path.join(DATA_ROOT, fname + '.jsonl')) as f:
        new_data = []
        for i, line in enumerate(f):
            datum = json.loads(line)
            id_stem = '-'.join(datum['identifier'].split('-')[:3])
            new_datum = {
                'identifier': datum['identifier'],
                'img0': '%s-img0' % id_stem,
                'img1': '%s-img1' % id_stem,
                'label': 1 if datum['label'] == 'True' else 0,
                'sent': datum['sentence'],
                'uid': 'nlvr2_%s_%d' % (split, i),
            }
            new_data.append(new_datum)
    
    with open(os.path.join(DATA_ROOT, 'lxmert_'+ split + '.jsonl'), 'w') as g:
        json.dump(new_data, g, sort_keys=True, indent=4)
