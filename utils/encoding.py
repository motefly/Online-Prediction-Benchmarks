import pandas as pd
import numpy as np
import category_encoders as ce
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+6))
parser.add_argument('-t', '--threshold', type=int, default=int(10))
parser.add_argument('-r', '--thresrate', type=float, default=0.99)
parser.add_argument('-i', '--numI', type=int, default=int(13))
parser.add_argument('-c', '--numC', type=int, default=int(26))
parser.add_argument('-p', '--phase', type=str, default='train')
parser.add_argument('-d', '--data', type=str, default='criteo_offline0')
parser.add_argument('-u', '--update', type=str, default=None)
parser.add_argument('csv_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())

class Num_encoder(object):
    def __init__(self, cate_col, nume_col, threshold, thresrate):
        self.label_name = 'Label'
        # cate_col = list(df.select_dtypes(include=['object']))
        self.cate_col = cate_col 
        # nume_col = list(set(list(df)) - set(cate_col))
        self.nume_col = nume_col
        self.encoder = ce.BinaryEncoder(cols=cate_col,verbose=1)
        self.threshold = threshold
        self.thresrate = thresrate
        # for online update, to du
        self.save_avgs = None

    def transform_fit(self, inPath, outPath):
        df = pd.read_csv(inPath)
        print('Filtering and fillna features')
        for item in tqdm(self.cate_col):
            value_counts = df[item].value_counts()
            num = value_counts.shape[0]
            value_filter_t = set(value_counts[value_counts<self.threshold].index)
            value_filter_l = set(value_counts[int(num*self.thresrate):].index)
            value_filter = list(value_filter_t | value_filter_l)
            df[item] = df[item].replace(value_filter,'<LESS>')
            df[item] = df[item].fillna('<UNK>')

        for item in tqdm(self.nume_col):
            df[item] = df[item].fillna(df[item].mean())

        print('Binary encoding cate features')
        # binary_encoding
        encode_df = self.encoder.fit_transform(df)
        
        print('Target encoding cate features')
        # dynamic_targeting_encoding
        data_len = df.shape[0]
        for item in tqdm(self.cate_col):
            feats = df[item].values
            feat_encoding = {'mean':[], 'count':[]}
            for idx in range(data_len):
                temp = df[:idx]
                cur_feat = feats[idx]
                avgs = temp.groupby(by=item)[self.label_name].agg(["mean", "count"])
                # smoothing optional
                if cur_feat in avgs.index:
                    feat_encoding['mean'].append(avgs.loc[cur_feat]['mean'])
                    feat_encoding['count'].append(avgs.loc[cur_feat]['count'])
                else:
                    feat_encoding['mean'].append(0)
                    feat_encoding['count'].append(0)
            encode_df[item+'_t_mean'] = feat_encoding['mean']
            encode_df[item+'_t_count'] = feat_encoding['count']

        self.save_avgs = df.groupby(by=cate_col)[self.label_name].agg(["mean", "count"])
        encode_df.to_csv(outPath, index=False)

    # for test dataset
    def fit(self, inPath, outPath):
        # to do
        pass
    
    # for update online dataset
    def update(self, inPath, outPath):
        # to do
        pass

if __name__ == '__main__':
    cate_col = ['C'+str(i) for i in range(1, args['numC']+1)]
    nume_col = ['I'+str(i) for i in range(1, args['numI']+1)]
    ec = Num_encoder(cate_col, nume_col, args['threshold'], args['thresrate'])
    ec.transform_fit(args['csv_path'], args['out_path'])
