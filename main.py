# -*- coding:utf-8 -*-

from utils import data_preprocess
from model import DeepFM
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--numI', type=int, default=int(13))
parser.add_argument('-c', '--numC', type=int, default=int(26))
parser.add_argument('-d', '--data', type=str, default='criteo_offline0')
parser.add_argument('train_csv_path', type=str)
parser.add_argument('test_csv_path', type=str)

args = vars(parser.parse_args())

result_dict = data_preprocess.read_criteo_data('data/%s_train'%args['data'], args['train_csv_path'],'meta_data/' + args['data'] + '_cate_meta_info.json', args['numI'], args['numC'])
test_dict = data_preprocess.read_criteo_data('data/%s_test'%args['data'], args['test_csv_path'], 'meta_data/' + args['data'] + '_cate_meta_info.json', args['numI'], args['numC'])

deepfm = DeepFM.DeepFM(args['numI'] + args['numC'],result_dict['feature_sizes'],verbose=True,use_cuda=True, weight_decay=0.0001,use_fm=True,use_ffm=False,use_deep=True).cuda()
deepfm.fit(result_dict['index'], result_dict['value'], result_dict['label'],
           test_dict['index'], test_dict['value'], test_dict['label'],ealry_stopping=True,refit=True)
