import pandas as pd
import numpy as np
import category_encoders as ce
from tqdm import tqdm
import argparse, collections, os
import gc
import pdb

parser = argparse.ArgumentParser()

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

def unpackbits(x,num_bits):
    xshape = list(x.shape)
    x = x.reshape([-1,1])
    to_and = 2**np.arange(num_bits).reshape([1,num_bits])
    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])

class NumEncoder(object):
    def __init__(self, cate_col, nume_col, threshold, thresrate):
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
        self.tgt_nume_col = []
        self.encoder = ce.ordinal.OrdinalEncoder(cols=cate_col)
        self.threshold = threshold
        self.thresrate = thresrate
        # for online update, to do
        self.save_cate_avgs = {}
        self.save_value_filter = {}
        self.save_num_embs = {}
        self.Max_len = {}
        self.samples = 0

    def fit_transform(self, inPath, outPath):
        print('----------------------------------------------------------------------')
        print('Fitting and Transforming %s .'%inPath)
        print('----------------------------------------------------------------------')
        df = pd.read_csv(inPath, dtype=self.dtype_dict)
        self.samples = df.shape[0]
        print('Filtering and fillna features')
        for item in tqdm(self.cate_col):
            value_counts = df[item].value_counts()
            num = value_counts.shape[0]
            self.save_value_filter[item] = list(value_counts[:int(num*self.thresrate)][value_counts>self.threshold].index)
            rm_values = set(value_counts.index)-set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)
            df[item] = df[item].fillna('<UNK>')
            del value_counts
            gc.collect()

        for item in tqdm(self.nume_col):
            df[item] = df[item].fillna(df[item].mean())
            self.save_num_embs[item] = {'sum':df[item].sum(), 'cnt':df[item].shape[0]}

        print('Ordinal encoding cate features')
        # ordinal_encoding
        df = self.encoder.fit_transform(df)

        print('Target encoding cate features')
        # dynamic_targeting_encoding
        for item in tqdm(self.cate_col):
            feats = df[item].values
            labels = df[self.label_name].values
            feat_encoding = {'mean':[], 'count':[]}
            self.save_cate_avgs[item] = collections.defaultdict(lambda : [0, 0])
            for idx in range(self.samples):
                cur_feat = feats[idx]
                # smoothing optional
                if cur_feat in self.save_cate_avgs[item]:
                    feat_encoding['mean'].append(round(self.save_cate_avgs[item][cur_feat][0]/self.save_cate_avgs[item][cur_feat][1],6))
                    feat_encoding['count'].append(round(self.save_cate_avgs[item][cur_feat][1]/idx,6))
                else:
                    feat_encoding['mean'].append(0)
                    feat_encoding['count'].append(0)
                self.save_cate_avgs[item][cur_feat][0] += labels[idx]
                self.save_cate_avgs[item][cur_feat][1] += 1
            df[item+'_t_mean'] = feat_encoding['mean']
            df[item+'_t_count'] = feat_encoding['count']
            self.tgt_nume_col.append(item+'_t_mean')
            self.tgt_nume_col.append(item+'_t_count')
        
        print('Start manual binary encode')
        rows = None
        for item in tqdm(self.nume_col+self.tgt_nume_col):
            feats = df[item].values
            if rows is None:
                rows = feats.reshape((-1,1))
            else:
                rows = np.concatenate([rows,feats.reshape((-1,1))],axis=1)
            del feats
            gc.collect()
        for item in tqdm(self.cate_col):
            feats = df[item].values
            Max = df[item].max()
            bit_len = len(bin(Max)) - 2
            samples = self.samples
            self.Max_len[item] = bit_len
            res = unpackbits(feats, bit_len).reshape((samples,-1))
            rows = np.concatenate([rows,res],axis=1)
            del feats
            gc.collect()
        trn_y = np.array(df[self.label_name].values).reshape((-1,1))
        del df
        gc.collect()
        trn_x = np.array(rows)
        np.save(outPath+'_features.npy', trn_x)
        np.save(outPath+'_labels.npy', trn_y)

    # for test dataset
    def transform(self, inPath, outPath):
        print('----------------------------------------------------------------------')
        print('Transforming %s .'%inPath)
        print('----------------------------------------------------------------------')
        df = pd.read_csv(inPath, dtype=self.dtype_dict)
        samples = df.shape[0]
        print('Filtering and fillna features')
        for item in tqdm(self.cate_col):
            value_counts = df[item].value_counts()
            rm_values = set(value_counts.index)-set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)
            df[item] = df[item].fillna('<UNK>')

        for item in tqdm(self.nume_col):
            mean = self.save_num_embs[item]['sum'] / self.save_num_embs[item]['cnt']
            df[item] = df[item].fillna(mean)

        print('Ordinal encoding cate features')
        # ordinal_encoding
        df = self.encoder.transform(df)

        print('Target encoding cate features')
        # dynamic_targeting_encoding
        for item in tqdm(self.cate_col):
            avgs = self.save_cate_avgs[item]
            df[item+'_t_mean'] = df[item].map(lambda x: round(avgs[x][0]/avgs[x][1],6) if x in avgs else 0)
            df[item+'_t_count'] = df[item].map(lambda x: round(avgs[x][1]/self.samples) if x in avgs else 0)
        
        print('Start manual binary encode')
        rows = None
        for item in tqdm(self.nume_col+self.tgt_nume_col):
            feats = df[item].values
            if rows is None:
                rows = feats.reshape((-1,1))
            else:
                rows = np.concatenate([rows,feats.reshape((-1,1))],axis=1)
            del feats
            gc.collect()
        for item in tqdm(self.cate_col):
            feats = df[item].values
            bit_len = self.Max_len[item]
            res = unpackbits(feats, bit_len).reshape((samples,-1))
            rows = np.concatenate([rows,res],axis=1)
            del feats
            gc.collect()
        vld_y = np.array(df[self.label_name].values).reshape((-1,1))
        del df
        gc.collect()
        vld_x = np.array(rows)
        np.save(outPath+'_features.npy', vld_x)
        np.save(outPath+'_labels.npy', vld_y)
    
    # for update online dataset
    def refit_transform(self, inPath, outPath):
        print('----------------------------------------------------------------------')
        print('Refitting and Transforming %s .'%inPath)
        print('----------------------------------------------------------------------')
        df = pd.read_csv(inPath, dtype=self.dtype_dict)
        samples = df.shape[0]
        print('Filtering and fillna features')
        for item in tqdm(self.cate_col):
            value_counts = df[item].value_counts()
            rm_values = set(value_counts.index)-set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)
            df[item] = df[item].fillna('<UNK>')

        for item in tqdm(self.nume_col):
            self.save_num_embs[item]['sum'] += df[item].sum()
            self.save_num_embs[item]['cnt'] += df[item].shape[0]
            mean = self.save_num_embs[item]['sum'] / self.save_num_embs[item]['cnt']
            df[item] = df[item].fillna(mean)

        print('Ordinal encoding cate features')
        # ordinal_encoding
        df = self.encoder.transform(df)

        print('Target encoding cate features')
        # dynamic_targeting_encoding
        for item in tqdm(self.cate_col):
            feats = df[item].values
            labels = df[self.label_name].values
            feat_encoding = {'mean':[], 'count':[]}
            for idx in range(samples):
                cur_feat = feats[idx]
                
                feat_encoding['mean'].append(round(self.save_cate_avgs[item][cur_feat][0]/self.save_cate_avgs[item][cur_feat][1],6))
                feat_encoding['count'].append(round(self.save_cate_avgs[item][cur_feat][1]/(self.samples+idx),6))
                self.save_cate_avgs[item][cur_feat][0] += labels[idx]
                self.save_cate_avgs[item][cur_feat][1] += 1
            df[item+'_t_mean'] = feat_encoding['mean']
            df[item+'_t_count'] = feat_encoding['count']

        self.samples += samples
            
        print('Start manual binary encode')
        rows = None
        for item in tqdm(self.nume_col+self.tgt_nume_col):
            feats = df[item].values
            if rows is None:
                rows = feats.reshape((-1,1))
            else:
                rows = np.concatenate([rows,feats.reshape((-1,1))],axis=1)
            del feats
            gc.collect()
        for item in tqdm(self.cate_col):
            feats = df[item].values
            bit_len = self.Max_len[item]
            res = unpackbits(feats, bit_len).reshape((samples,-1))
            rows = np.concatenate([rows,res],axis=1)
            del feats
            gc.collect()
        vld_y = np.array(df[self.label_name].values).reshape((-1,1))
        del df
        gc.collect()
        vld_x = np.array(rows)
        np.save(outPath+'_features.npy', vld_x)
        np.save(outPath+'_labels.npy', vld_y)
        # to do
        pass

if __name__ == '__main__':
    cate_col = ['C'+str(i) for i in range(1, args['numC']+1)]
    nume_col = ['I'+str(i) for i in range(1, args['numI']+1)]
    if not os.path.isdir(args['out_dir']):
        os.mkdir(args['out_dir'])
    ec = NumEncoder(cate_col, nume_col, args['threshold'], args['thresrate'])
    def online_encoding():
        in_map = lambda x : args['data']+'_online%d.csv'%x
        out_map = lambda x : args['out_dir']+'/online%d'%x
        
        ec.fit_transform(in_map(0), out_map(0)+'_train')
        ec.transform(in_map(1), out_map(1)+'_test')
        
        for idx in range(1, args['num_onlines']-1):
            ec.refit_transform(in_map(idx), out_map(idx)+'_train')
            ec.transform(in_map(idx+1), out_map(idx+1)+'_test')

    if args['online']:
        online_encoding()
    else:
        ec.fit_transform(args['train_csv_path'], args['out_dir']+'/train')
        ec.transform(args['test_csv_path'], args['out_dir']+'/test')
