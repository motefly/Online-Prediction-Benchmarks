#!/usr/bin/env python3

import argparse, csv, sys, collections
from tqdm import tqdm
from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
parser.add_argument('data', type=str)
parser.add_argument('--update', type=str, default=None)
parser.add_argument('--numC', type=int, default=26)
parser.add_argument('--numI', type=int, default=13)
parser.add_argument('--thres', type=int, default=10)
args = vars(parser.parse_args())

cate_counts = collections.defaultdict(lambda : [0, 0])
num_counts = collections.defaultdict(lambda : [0, 0])
update = False
if args['update'] != None:
    update = True
    for row in csv.DictReader(open('meta_data/' + args['update'] + '_cate_counts.csv')):
        cate_counts[row['Field']+','+row['Value']][0] = int(row['Label'])
        cate_counts[row['Field']+','+row['Value']][1] = int(row['Total'])

    for row in csv.DictReader(open('meta_data/' + args['update'] + '_num_counts.csv')):
        num_counts[row['Field']][0] = eval(row['Sum'])
        num_counts[row['Field']][1] = int(row['Cnt'])

for i, row in tqdm(enumerate(csv.DictReader(open(args['csv_path'])), start=1)):
    label = row['Label']
    for j in range(1, args['numC']+1):
        field = 'C{0}'.format(j)
        value = row[field]
        if update and field+','+value not in cate_counts:
            continue
        cate_counts[field+','+value][0] += eval(label)
        cate_counts[field+','+value][1] += 1
    
    for j in range(1, args['numI']+1):
        field = 'I{0}'.format(j)
        value = row[field]
        if value != '':
            num_counts[field][0] += eval(value)
            num_counts[field][1] += 1

with open('meta_data/' +args['data']+'_cate_counts.csv', 'w') as f:
    f.write('Field,Value,Label,Total\n')
    for key, (label, total) in tqdm(sorted(cate_counts.items(), key=lambda x: x[1][1], reverse=True)):
        if total < args['thres']:
            continue
        # ratio = round(float(pos)/total, 5)
        f.write(key+','+str(label)+','+str(total)+'\n')

with open('meta_data/' +args['data']+'_num_counts.csv', 'w') as f:
    f.write('Field,Sum,Cnt\n')
    for key, (Sum, cnt) in tqdm(sorted(num_counts.items(), key=lambda x: x[0][1], reverse=True)):
        f.write(key+','+str(Sum)+','+str(cnt)+'\n')
