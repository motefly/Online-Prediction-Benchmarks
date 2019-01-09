# -*- coding:utf-8 -*-

from utils import data_preprocess
from model import DeepFM, PNN
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--numI', type=int, default=int(13))
parser.add_argument('-c', '--numC', type=int, default=int(26))
parser.add_argument('-d', '--data', type=str, default='criteo_offline0')
parser.add_argument('-m', '--model', type=str, default='deepFM')
# parser.add_argument('train_csv_path', type=str)
# parser.add_argument('test_csv_path', type=str)

args = vars(parser.parse_args())

result_dict = data_preprocess.read_new_criteo_data('data/%s/train/'%args['data'])#,'meta_data/' + args['data'] + '_cate_meta_info.json', args['numI'], args['numC'])
test_dict = data_preprocess.read_new_criteo_data('data/%s/test/'%args['data'])#, 'meta_data/' + args['data'] + '_cate_meta_info.json', args['numI'], args['numC'])

field_size = args['numI'] + args['numC']
feature_sizes = result_dict['feature_sizes']

if args['model'] == 'deepFM':
    deepfm = DeepFM.DeepFM(field_size,feature_sizes,verbose=True,use_cuda=True, weight_decay=0.001,use_fm=True,use_ffm=False,use_deep=True,batch_size=1024).cuda()
    deepfm.fit(result_dict['index'], result_dict['value'], result_dict['label'],
               test_dict['index'], test_dict['value'], test_dict['label'],
               ealry_stopping=True,refit=True)

elif args['model'] == 'PNN':
    pnn = PNN.PNN(field_size, feature_sizes, batch_size=1024, verbose=True, use_cuda=True,weight_decay=0.00001, use_inner_product=True, use_outer_product=True).cuda()
    pnn.fit(result_dict['index'], result_dict['value'], result_dict['label'],
            test_dict['index'], test_dict['value'], test_dict['label'],
            ealry_stopping=True,refit=True)
    pass
