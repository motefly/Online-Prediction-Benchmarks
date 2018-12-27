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
parser.add_argument('csv_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())

def train_preprocess():
    frequent_feats = read_freqent_feats(args['threshold'])
    with open(args['out_path'], 'w') as f:
        featsLst = []
        cate_emb_arr = [{} for i in range(args['numC'])]
        num_mean_arr = [{'cnt':0, 'val':0} for i in range(args['numI'])]
        MaxIdx = [0 for i in range(args['numC'])]
        for row in tqdm(csv.DictReader(open(args['csv_path']))):
            feats = []
            for feat in gen_num_feats(row, args['numI'], args['numC']):
                field = feat.split('$')[0]
                Type, field = field[0], int(field[1:])
                if Type == 'C' and feat not in frequent_feats:
                    feat = feat.split('$')[0]+'less'
                if Type == 'C':
                    fieldIdx = field-1
                    if feat not in cate_emb_arr[fieldIdx]:
                        item = {'Idx':len(cate_emb_arr[fieldIdx]), 'Cnt':0, 'Label':0}
                        cate_emb_arr[fieldIdx][feat] = item
                        MaxIdx[fieldIdx] = max(MaxIdx[fieldIdx], len(cate_emb_arr[fieldIdx]))
                    cate_emb_arr[fieldIdx][feat]['Cnt'] += 1
                    cate_emb_arr[fieldIdx][feat]['Label'] += float(row['Label'])
                elif Type == 'I':
                    val = feat.split('$')[1]
                    if val != 'mean':
                        num_mean_arr[field-1]['val'] += eval(val)
                        num_mean_arr[field-1]['cnt'] += 1
                feats.append((Type, field, feat))
            featsLst.append(feats)
        meta_info = [{} for i in range(args['numI'] + args['numC'])]
        for feats in tqdm(featsLst):
            feat_row = []
            for Type, field, feat in feats:
                if Type == 'C':
                    begin = len(feat_row)
                    MaxBit = len(bin(MaxIdx[field-1])) - 2
                    bin_code = bin(cate_emb_arr[field-1][feat]['Idx'])[2:][::-1]
                    for idx in range(MaxBit):
                        if idx < len(bin_code):
                            feat_row.append(bin_code[idx])
                        else:
                            feat_row.append(str(0))
                    feat_row.append(str(cate_emb_arr[field-1][feat]['Cnt'] / len(featsLst)))
                    feat_row.append(str(cate_emb_arr[field-1][feat]['Label'] / cate_emb_arr[field-1][feat]['Cnt']))
                    meta_info[field-1+args['numI']][feat] = feat_row[begin:]
                elif Type == 'I':
                    begin = len(feat_row)
                    val = feat.split('$')[1]
                    if val == 'mean':
                        feat_row.append(str(num_mean_arr[field-1]['val']/num_mean_arr[field-1]['cnt']))
                    else:
                        feat_row.append(val)
                    meta_info[field-1]['mean'] = str(num_mean_arr[field-1]['val']/num_mean_arr[field-1]['cnt'])
            f.write(row['Label'] + ',' + ','.join(feat_row) + '\n')
        for idx in range(args['numC']):
            less = 'C' + str(idx+1) + 'less'
            if less not in meta_info[idx+args['numI']]:
                MaxBit = len(bin(MaxIdx[idx])) - 2
                bin_code = bin(MaxIdx[idx])[2:][::-1]
                less_encode = []
                for jdx in range(MaxBit):
                    if jdx < len(bin_code):
                        less_encode.append(bin_code[jdx])
                    else:
                        less_encode.append(str(0))
                less_encode.extend(['0','0'])
                meta_info[idx+args['numI']][less] = less_encode
        with open('num_meta_info.json', 'w') as jf:
            jstr = json.dumps(meta_info)
            jf.write(jstr)

def test_preprocess():
    jstr = open('num_meta_info.json', 'r').readline()
    meta_info = json.loads(jstr)
    frequent_feats = read_freqent_feats(args['threshold'])
    with open(args['out_path'], 'w') as f:
        for row in tqdm(csv.DictReader(open(args['csv_path']))):
            feat_row = []
            for feat in gen_num_feats(row, args['numI'], args['numC']):
                field = feat.split('$')[0]
                Type, field = field[0], int(field[1:])
                if Type == 'C':
                    if feat not in frequent_feats:
                        feat = feat.split('$')[0]+'less'
                    # if fea not in meta_info[field-1+args['numI']]):
                    #    feat = feat.split('$')[0]+'less'
                if Type == 'C':
                    feat_row.extend(meta_info[field-1+args['numI']][feat])
                elif Type == 'I':
                    val = feat.split('$')[1]
                    if val == 'mean':
                        feat_row.append(meta_info[field-1]['mean'])
                    else:
                        feat_row.append(val)
            f.write(row['Label'] + ',' + ','.join(feat_row) + '\n')
            
if args['phase'] == 'train':
    train_preprocess()
else:
    test_preprocess()
