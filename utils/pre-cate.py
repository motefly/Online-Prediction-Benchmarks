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
    if args['update'] != None:
        update = True
        jstr = open('meta_data/' + args['update'] + '_cate_meta_info.json', 'r').readline()
        meta_info = json.loads(jstr)
    frequent_feats = read_freqent_feats(args['threshold'], data=args['data'], Type='cate')
    with open(args['out_path'], 'w') as f:
        cate_emb_arr = [{} for i in range(args['numI'] + args['numC'])]
        if update:
            for i in range(args['numI'] + args['numC']):
                for j in meta_info[i]:
                    cate_emb_arr[i][j] = meta_info[i+args['numI']][j][1]
        for row in tqdm(csv.DictReader(open(args['csv_path']))):
            feats = []
            for idx,feat in enumerate(gen_cate_feats(row, args['numI'], args['numC'])):
                field = feat.split('$')[0]
                Type, field = field[0], int(field[1:])
                if update:
                    if feat not in cate_emb_arr[idx]:
                        feat = feat.split('$')[0]+'less'
                if Type == 'C' and feat not in frequent_feats:
                    feat = feat.split('$')[0]+'less'
                if Type == 'C':
                    field += args['numI']
                if feat not in cate_emb_arr[idx]:
                    Idx = len(cate_emb_arr[idx])
                    cate_emb_arr[idx][feat] = str(Idx)
                feats.append(cate_emb_arr[idx][feat])
            f.write(row['Label'] + ',' + ','.join(feats) + '\n')
        for idx in range(args['numI'] + args['numC']):
            if idx < args['numI']:
                less = 'I' + str(idx+1) + 'less'
            else:
                less = 'C' + str(idx-args['numI']+1) + 'less'
            if less not in cate_emb_arr[idx]:
                cate_emb_arr[idx][less] = str(len(cate_emb_arr[idx]))
        with open('meta_data/' + args['data'] + '_cate_meta_info.json', 'w') as jf:
            jstr = json.dumps(cate_emb_arr)
            jf.write(jstr)

def test_preprocess():
    meta_info = json.loads(open('meta_data/' + args['data'] + '_cate_meta_info.json', 'r').readline())
    frequent_feats = read_freqent_feats(args['threshold'], data=args['data'], Type='cate')
    with open(args['out_path'], 'w') as f:
        for row in tqdm(csv.DictReader(open(args['csv_path']))):
            feats = []
            for idx,feat in enumerate(gen_cate_feats(row, args['numI'], args['numC'])):
                field = feat.split('$')[0]
                Type, field = field[0], int(field[1:])
                if Type == 'C':
                    if feat not in frequent_feats:
                        feat = feat.split('$')[0]+'less'
                if feat not in meta_info[idx]:
                    feat = Type + str(idx+1) + 'less'
                # if Type == 'C':
                #     field += args['numI']
                feats.append(meta_info[idx][feat])
            f.write(row['Label'] + ',' + ','.join(feats) + '\n')

if args['phase'] == 'train':
    train_preprocess()
else:
    test_preprocess()
