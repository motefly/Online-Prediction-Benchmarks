#!/usr/bin/env python3

import argparse, csv, sys, json

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
parser.add_argument('-p', '--phase', type=str, default='train')
parser.add_argument('-d', '--data', type=str, default='criteo_offline0')
parser.add_argument('-u', '--update', type=str, default=None)
parser.add_argument('csv_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())

def gen_hashed_fm_feats(feats, nr_bins):
    feats = ['{0}:{1}:1'.format(field-1, hashstr(feat, nr_bins)) for (field, feat) in feats]
    return feats

# Need to test the update function
def train_preprocess():
    update = False
    meta_info = {}
    if args['update'] != None:
        update = True
    frequent_feats, cate_emb_arr, num_mean_arr, _, _ = read_freqent_feats(args['threshold'], data=args['data'], Type='idv')
    with open(args['out_path'], 'w') as f:
        for row in tqdm(csv.DictReader(open(args['csv_path']))):
            feats = []
            for idx,feat in enumerate(gen_num_feats(row, args['numI'], args['numC'])):
                field = feat.split('$')[0]
                Type = field[0]
                if update:
                    if feat not in cate_emb_arr[field]:
                        feat = feat.split('$')[0]+'less'
                if Type == 'C' and feat not in frequent_feats:
                    feat = feat.split('$')[0]+'less'
                if Type == 'C':
                    if field not in meta_info:
                        meta_info[field] = {}
                    meta_info[field][feat] = str(cate_emb_arr[field][feat]['Idx'])
                    feats.append(meta_info[field][feat])
                elif Type == 'I':
                    val = feat.split('$')[1]
                    Mean = round(num_mean_arr[field]['Sum']/num_mean_arr[field]['Cnt'],5)
                    if val == 'mean':
                        feats.append(str(Mean))
                    else:
                        feats.append(val)
                    meta_info[field] = str(Mean)
            f.write(row['Label'] + ',' + ','.join(feats) + '\n')
        for idx in range(args['numC']):
            less = 'C' + str(idx+1) + 'less'
            field = 'C' + str(idx+1)
            if less not in meta_info[field]:
                meta_info[field][less] = str(len(cate_emb_arr[field]))
        with open('meta_data/' + args['data'] + '_idv_meta_info.json', 'w') as jf:
            jstr = json.dumps(meta_info)
            jf.write(jstr)

def test_preprocess():
    meta_info = json.loads(open('meta_data/' + args['data'] + '_idv_meta_info.json', 'r').readline())
    frequent_feats = read_freqent_feats(args['threshold'], data=args['data'], Type='idv')[0]
    with open(args['out_path'], 'w') as f:
        for row in tqdm(csv.DictReader(open(args['csv_path']))):
            feats = []
            for idx,feat in enumerate(gen_num_feats(row, args['numI'], args['numC'])):
                field = feat.split('$')[0]
                Type = field[0]
                if Type == 'C':
                    if feat not in frequent_feats:
                        feat = feat.split('$')[0]+'less'
                    feats.append(meta_info[field][feat])
                else:
                    val = feat.split('$')[1]
                    if val == 'mean':
                        feats.append(meta_info[field])
                    else:
                        feats.append(val)
            f.write(row['Label'] + ',' + ','.join(feats) + '\n')

if args['phase'] == 'train':
    train_preprocess()
else:
    test_preprocess()
