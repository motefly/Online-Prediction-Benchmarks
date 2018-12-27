#!/usr/bin/env python3

import argparse, csv, sys

from common import *
from tqdm import tqdm

if len(sys.argv) == 1:
    sys.argv.append('-h')

from common import *

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+6))
parser.add_argument('-t', '--threshold', type=int, default=int(10))
parser.add_argument('-i', '--numI', type=int, default=int(13))
parser.add_argument('-c', '--numC', type=int, default=int(26))
parser.add_argument('csv_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())

def gen_hashed_fm_feats(feats, nr_bins):
    feats = ['{0}:{1}:1'.format(field-1, hashstr(feat, nr_bins)) for (field, feat) in feats]
    return feats

frequent_feats = read_freqent_feats(args['threshold'])

with open(args['out_path'], 'w') as f:
    for row in tqdm(csv.DictReader(open(args['csv_path']))):
        feats = []
        for feat in gen_cate_feats(row, args['numI'], args['numC']):
            field = feat.split('$')[0]
            Type, field = field[0], int(field[1:])
            if Type == 'C' and feat not in frequent_feats:
                feat = feat.split('$')[0]+'less'
            if Type == 'C':
                field += args['numI']
            feats.append((field, feat))
        feats = gen_hashed_fm_feats(feats, args['nr_bins'])
        f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
