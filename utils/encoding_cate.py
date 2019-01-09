import pandas as pd
import numpy as np
import category_encoders as ce
from tqdm import tqdm
import argparse, collections, os
parser = argparse.ArgumentParser()

parser.add_argument('-b', '--num_bins', type=int, default=int(32))
parser.add_argument('-t', '--threshold', type=int, default=int(10))
parser.add_argument('-r', '--thresrate', type=float, default=0.99)
parser.add_argument('-i', '--numI', type=int, default=int(13))
parser.add_argument('-c', '--numC', type=int, default=int(26))

parser.add_argument('--train_csv_path', type=str)
parser.add_argument('--test_csv_path', type=str)
parser.add_argument('out_dir', type=str)

parser.add_argument('--online', action='store_true')
parser.add_argument('--data', type=str)
parser.add_argument('--num_onlines', type=int)

args = vars(parser.parse_args())

class CateEncoder(object):
    def __init__(self, cate_col, nume_col, threshold, thresrate, bins):
        self.label_name = 'Label'
        # cate_col = list(df.select_dtypes(include=['object']))
        self.cate_col = cate_col 
        # nume_col = list(set(list(df)) - set(cate_col))
        self.dtype_dict = {}
        for item in cate_col:
            self.dtype_dict[item] = 'str'
        for item in nume_col:
            self.dtype_dict[item] = 'float'
        self.nume_col = nume_col
        self.encoder = ce.ordinal.OrdinalEncoder(cols=cate_col+nume_col)
        self.threshold = threshold
        self.thresrate = thresrate
        self.bins = bins
        # for online update, to do
        self.save_value_filter = {}
        self.save_num_bins = {}
        self.samples = 0

    def save2npy(self, df, out_dir):
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        result = {'label':[], 'index':[],'feature_sizes':[]}
        result['label'] = df[self.label_name].values
        result['index'] = df[self.cate_col+self.nume_col].values
        for item in self.cate_col+self.nume_col:
            result['feature_sizes'].append(df[item].max()+1)
        for item in result:
            result[item] = np.array(result[item])
            np.save(out_dir + '_' + item +'.npy', result[item])

    def fit_transform(self, inPath, outPath):
        print('----------------------------------------------------------------------')
        print('Fitting and Transforming %s .'%inPath)
        print('----------------------------------------------------------------------')
        df = pd.read_csv(inPath, dtype=self.dtype_dict)
        print('Filtering and fillna features')
        for item in tqdm(self.cate_col):
            value_counts = df[item].value_counts()
            num = value_counts.shape[0]
            self.save_value_filter[item] = list(value_counts[:int(num*self.thresrate)][value_counts>self.threshold].index)
            rm_values = set(value_counts.index)-set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)
            df[item] = df[item].fillna('<UNK>')

        print('Fillna and Bucketize numeric features')
        for item in tqdm(self.nume_col):
            q_res = pd.qcut(df[item], self.bins, labels=False, retbins=True, duplicates='drop')
            df[item] = q_res[0].fillna(-1).astype('int')
            self.save_num_bins[item] = q_res[1]

        print('Ordinal encoding cate features')
        # ordinal_encoding
        df = self.encoder.fit_transform(df)
        self.save2npy(df, outPath)
        # df.to_csv(outPath, index=False)

    # for test dataset
    def transform(self, inPath, outPath):
        print('----------------------------------------------------------------------')
        print('Transforming %s .'%inPath)
        print('----------------------------------------------------------------------')
        df = pd.read_csv(inPath, dtype=self.dtype_dict)
        print('Filtering and fillna features')
        for item in tqdm(self.cate_col):
            value_counts = df[item].value_counts()
            rm_values = set(value_counts.index)-set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)
            df[item] = df[item].fillna('<UNK>')

        for item in tqdm(self.nume_col):
            df[item] = pd.cut(df[item], self.save_num_bins[item], labels=False, include_lowest=True).fillna(-1).astype('int')

        print('Ordinal encoding cate features')
        # ordinal_encoding
        df = self.encoder.transform(df)
        self.save2npy(df, outPath)
        # df.to_csv(outPath, index=False)

        
if __name__ == '__main__':
    cate_col = ['C'+str(i) for i in range(1, args['numC']+1)]
    nume_col = ['I'+str(i) for i in range(1, args['numI']+1)]
    if not os.path.isdir(args['out_dir']):
        os.mkdir(args['out_dir'])
    ec = CateEncoder(cate_col, nume_col, args['threshold'], args['thresrate'], args['num_bins'])
    def online_encoding():
        in_map = lambda x : args['data']+'_online%d.csv'%x
        out_map = lambda x : args['out_dir']+'/online%d/'%x
        
        ec.fit_transform(in_map(0), out_map(0))
        # ec.transform(in_map(1), out_map(1)+'_test')
        
        for idx in range(1, args['num_onlines']):
            # ec.refit_transform(in_map(idx), out_map(idx)+'_train')
            ec.transform(in_map(idx), out_map(idx))

    if args['online']:
        online_encoding()
    else:
        ec.fit_transform(args['train_csv_path'], args['out_dir']+'/train/')
        ec.transform(args['test_csv_path'], args['out_dir']+'/test/')
