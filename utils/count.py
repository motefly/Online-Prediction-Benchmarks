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
args = vars(parser.parse_args())

counts = collections.defaultdict(lambda : 0)
if args['update'] != None:
    for row in csv.DictReader(open('meta_data/' + args['update'] + '_fc.trva.t10.txt')):
        #counts[row['Field']+','+row['Value']][0] = int(row['Neg'])
        #counts[row['Field']+','+row['Value']][1] = int(row['Pos'])
        counts[row['Field']+','+row['Value']] = int(row['Total'])

for i, row in tqdm(enumerate(csv.DictReader(open(args['csv_path'])), start=1)):
    label = row['Label']
    for j in range(1, args['numC']+1):
        field = 'C{0}'.format(j)
        value = row[field]
        # if label == '0':
            # counts[field+','+value][0] += 1
        # else:
            # counts[field+','+value][1] += 1
        counts[field+','+value] += 1
with open('meta_data/' +args['data']+'_fc.trva.t10.txt', 'w') as f:
    f.write('Field,Value,Total\n')
    for key, total in tqdm(sorted(counts.items(), key=lambda x: x[1])):
        # if total < 10:
            #continue
        # ratio = round(float(pos)/total, 5)
        f.write(key+','+str(total)+'\n')  #+str(neg)+','+str(pos)+','+str(total)+','+str(ratio)+'\n')
