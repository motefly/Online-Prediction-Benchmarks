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

def train_preprocess():
    update = False
    frequent_feats, cate_emb_arr, num_mean_arr, MaxIdx, _ = read_freqent_feats(args['threshold'], data=args['data'])
    Samples = 0
    meta_info = {'Samples':Samples}
    if args['update'] != None:
        update = True
        jstr = open('meta_data/' + args['update'] + '_num_meta_info.json', 'r').readline()
        meta_info = json.loads(jstr)
        Samples = meta_info['Samples']
    with open(args['out_path'], 'w') as f:
        cate_count = {}
        # to do
        if update:
            for field in meta_info:
                cate_count[field] = {}
                Type = field[0]
                if Type == 'C':
                    for feat in meta_info['field']:
                        cate_count[field][feat] = meta_info[field][feat][1]
        for row in tqdm(csv.DictReader(open(args['csv_path']))):
            Samples += 1
            feats = []
            for feat in gen_num_feats(row, args['numI'], args['numC']):
                begin = len(feats)
                field = feat.split('$')[0]
                Type = field[0]
                if Type == 'C':
                    if feat not in frequent_feats:
                        feat = feat.split('$')[0]+'less'
                    if update and feat not in cate_count[field]:
                        feat = feat.split('$')[0]+'less'
                if Type == 'C':
                    if field not in cate_count:
                        cate_count[field] = {}
                    if feat not in cate_count[field]:
                        item = {'Cnt':0, 'Label':0}
                        cate_count[field][feat] = item
                    cate_count[field][feat]['Cnt'] += 1
                    cate_count[field][feat]['Label'] += float(row['Label'])
                    feats.append(str(round(cate_count[field][feat]['Label']/cate_count[field][feat]['Cnt'],6)))
                    feats.append(str(round(cate_count[field][feat]['Cnt']/Samples*1000,6)))
                    MaxBit = len(bin(MaxIdx[field])) - 2
                    bin_code = bin(cate_emb_arr[field][feat]['Idx'])[2:][::-1]
                    for idx in range(MaxBit):
                        if idx < len(bin_code):
                            feats.append(bin_code[idx])
                        else:
                            feats.append(str(0))
                    if field not in meta_info:
                        meta_info[field] = {}
                    meta_info[field][feat] = [feats[begin:], cate_count[field][feat]]
                elif Type == 'I':
                    val = feat.split('$')[1]
                    Mean = round(num_mean_arr[field]['Sum']/num_mean_arr[field]['Cnt'],5)
                    if val == 'mean':
                        feats.append(str(Mean))
                    else:
                        feats.append(val)
                    meta_info[field] = str(Mean)
            f.write(row['Label'] + ',' + ','.join(feats) + '\n')
        meta_info['Samples'] = Samples
        for idx in range(args['numC']):
            less = 'C' + str(idx+1) + 'less'
            field = 'C'+str(idx+1)
            if less not in meta_info[field]:
                MaxBit = len(bin(MaxIdx[field])) - 2
                bin_code = bin(MaxIdx[field])[2:][::-1]
                less_encode = ['0','0']
                for jdx in range(MaxBit):
                    if jdx < len(bin_code):
                        less_encode.append(bin_code[jdx])
                    else:
                        less_encode.append(str(0))
                meta_info[field][less] = [less_encode, {'Cnt':0, 'Label':0}]
        with open('meta_data/' + args['data'] + '_num_meta_info.json', 'w') as jf:
            jstr = json.dumps(meta_info)
            jf.write(jstr)

def test_preprocess():
    jstr = open('meta_data/' + args['data'] + '_num_meta_info.json', 'r').readline()
    meta_info = json.loads(jstr)
    frequent_feats = read_freqent_feats(args['threshold'], data=args['data'])
    with open(args['out_path'], 'w') as f:
        for row in tqdm(csv.DictReader(open(args['csv_path']))):
            feats = []
            for feat in gen_num_feats(row, args['numI'], args['numC']):
                field = feat.split('$')[0]
                Type = field[0]
                if Type == 'C':
                    if feat not in frequent_feats:
                        feat = feat.split('$')[0]+'less'
                    # if fea not in meta_info[field-1+args['numI']]):
                    #    feat = feat.split('$')[0]+'less'
                if Type == 'C':
                    feats.extend(meta_info[field][feat][0])
                elif Type == 'I':
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
