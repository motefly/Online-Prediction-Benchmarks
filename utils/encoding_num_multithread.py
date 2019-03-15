import pandas as pd
import numpy as np
import category_encoders as ce
from tqdm import tqdm
import argparse, collections
import gc
import pdb
from threading import Thread, Event

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--threshold', type=int, default=int(10))
parser.add_argument('-r', '--thresrate', type=float, default=0.99)
parser.add_argument('-i', '--numI', type=int, default=int(13))
parser.add_argument('-c', '--numC', type=int, default=int(26))

parser.add_argument('train_csv_path', type=str)
parser.add_argument('train_out_path', type=str)
parser.add_argument('test_csv_path', type=str)
parser.add_argument('test_out_path', type=str)
args = vars(parser.parse_args())

class Num_encoder(object):
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
        self.encoder = ce.ordinal.OrdinalEncoder(cols=cate_col)
        self.threshold = threshold
        self.thresrate = thresrate
        # for online update, to do
        self.save_cate_avgs = {}
        self.save_value_filter = {}
        self.save_num_embs = {}
        self.samples = 0

    def fit_transform(self, inPath, outPath):
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
        rows = []
        gc.collect()
        
        print('Start manual binary encode')
        with open(outPath, 'w') as f:
            def cate_binary_encode(df_item, samples, rows, item):
                print("Running thread for %s"%item)
                feats = df_item.values
                Max = df_item.max()
                del df_item
                gc.collect()
                bit_len = len(bin(Max)) - 2
                row = [[] for idx in range(bit_len)]
                for idx in range(samples):
                    bin_code = bin(feats[idx])[2:][::-1]
                    for jdx in range(bit_len):
                        row[jdx].append(eval(bin_code[jdx]) if jdx < len(bin_code) else 0)
                    if idx % 200000 == 0:
                        print("#Thread %s# iteration is "%item + str(idx))
                for idx in range(bit_len):
                    rows.append(row[idx])
                del row
                gc.collect()
                print("#Thread %s# iteration done"%item + str(idx))

            for item in self.nume_col:
                print(item+': ')
                feats = df[item].values
                row = []
                for idx in tqdm(range(self.samples)):
                    row.append(feats[idx])
                rows.append(row)
                del feats
                gc.collect()
            threads = []
            for item in self.cate_col:
                t = Thread(target = cate_binary_encode,
                           args = (df[item], self.samples, rows, item,))
                t.start()
                threads.append(t)
                # print(item+': ')
                # feats = df[item].values
                # Max = df[item].max()
                # bit_len = len(bin(Max)) - 2
                # row = [[] for idx in range(bit_len)]
                # for idx in tqdm(range(self.samples)):
                #     bin_code = bin(feats[idx])[2:][::-1]
                #     for jdx in range(bit_len):
                #         row[jdx].append(eval(bin_code[jdx]) if jdx < len(bin_code) else 0)
                # for idx in range(bit_len):
                #     rows.append(row[idx])
                # del feats
                # gc.collect()
            for idx,t in enumerate(threads):
                if t.is_alive():
                    print('#Thread %d# is running'%idx)
                else:
                    print('Something wrong in #Thread %d#'%idx)
                
            for idx,t in enumerate(threads):
                t.join()
    
        import pdb
        pdb.set_trace()
        trn_y = np.asarray(df[self.label_name].values)
        del df
        gc.collect()
        trn_x = np.asarray(rows).transpose((1,0))
        self.mean = np.mean(trn_x, axis=0)
        self.std = np.std(trn_x, axis=0)
        trn_x = (trn_x - self.mean) / (self.std + 1e-5)
        np.save(outPath+'_features.npy', trn_x)
        np.save(outPath+'_labels.npy', trn_y)
        
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
        self.nume_col.append(item+'_t_mean')
        self.nume_col.append(item+'_t_count')
            
        print('Binary encoding cate features')
        # binary_encoding
        df[cate_col] = self.encoder.fit_transform(df[cate_col])
        df.to_csv(outPath, index=False)

    # for test dataset
    def transform(self, inPath, outPath):
        df = pd.read_csv(inPath, dtype=self.dtype_dict)
        print('Filtering and fillna features')
        for item in tqdm(self.cate_col):
            value_counts = df[item].value_counts()
            rm_values = set(value_counts.index)-set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)
            df[item] = df[item].fillna('<UNK>')

        for item in tqdm(self.nume_col):
            mean = self.save_num_embs[item]['sum'] / self.save_num_embs[item]['cnt']
            df[item] = df[item].fillna(mean)

        print('Target encoding cate features')
        # dynamic_targeting_encoding
        for item in tqdm(self.cate_col):
            value_counts = set(df[item].value_counts().index)
            # for feat in value_counts:
            avgs = self.save_cate_avgs[item]
            df[item+'_t_mean'] = df[item].map(lambda x: round(avgs[x][0]/avgs[x][1],6) if x in avgs else 0)
            df[item+'_t_count'] = df[item].map(lambda x: round(avgs[x][1]/self.samples) if x in avgs else 0)

        print('Binary encoding cate features')
        # binary_encoding
        df = self.encoder.transform(df)
        df.to_csv(outPath, index=False)

        # vld_x = (vld_x - self.mean) / (self.std + 1e-5)
    
    # for update online dataset
    def update_transform(self, inPath, outPath):
        # to do
        pass

if __name__ == '__main__':
    cate_col = ['C'+str(i) for i in range(1, args['numC']+1)]
    nume_col = ['I'+str(i) for i in range(1, args['numI']+1)]
    ec = Num_encoder(cate_col, nume_col, args['threshold'], args['thresrate'])
    ec.fit_transform(args['train_csv_path'], args['train_out_path'])
    print('Start transform test dataset')
    ec.transform(args['test_csv_path'], args['test_out_path'])
