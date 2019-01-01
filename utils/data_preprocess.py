# -*- coding:utf-8 -*-

"""
Created on Dec 10, 2017
@author: jachin,Nie

This script is used to preprocess the raw data file

"""

import sys
import math
import argparse
import hashlib, csv, math, os, pickle, subprocess
import json
import numpy as np
from tqdm import tqdm

def gen_criteo_category_index(file_path):
    cate_dict = []
    for i in range(26):
        cate_dict.append({})
    for line in open(file_path, 'r'):
        datas = line.replace('\n','').split('\t')
        for i, item in enumerate(datas[14:]):
            if not cate_dict[i].has_key(item):
                cate_dict[i][item] = len(cate_dict[i])
    return cate_dict

def write_criteo_category_index(file_path, cate_dict_arr):
    f = open(file_path,'w')
    for i, cate_dict in enumerate(cate_dict_arr):
        for key in cate_dict:
            f.write(str(i)+','+key+','+str(cate_dict[key])+'\n')

def load_criteo_category_index(file_path):
    f = open(file_path,'r')
    cate_dict = []
    for i in range(39):
        cate_dict.append({})
    for line in f:
        datas = line.strip().split(',')
        cate_dict[int(datas[0])][datas[1]] = int(datas[2])
    return cate_dict

def read_raw_criteo_data(file_path, embedding_path, type):
    """
    :param file_path: string
    :param type: string (train or test)
    :return: result: dict
            result['continuous_feat']:two-dim array
            result['category_feat']:dict
            result['category_feat']['index']:two-dim array
            result['category_feat']['value']:two-dim array
            result['label']: one-dim array
    """
    begin_index = 1
    if type != 'train' and type != 'test':
        print("type error")
        return {}
    elif type == 'test':
        begin_index = 0
    cate_embedding = load_criteo_category_index(embedding_path)
    result = {'continuous_feat':[], 'category_feat':{'index':[],'value':[]}, 'label':[], 'feature_sizes':[]}
    for i, item in enumerate(cate_embedding):
        result['feature_sizes'].append(len(item))
    f = open(file_path)
    for line in f:
        datas = line.replace('\n', '').split('\t')

        indexs = []
        values = []
        flag = True
        for i, item in enumerate(datas[begin_index + 13:]):
            if not cate_embedding[i].has_key(item):
                flag = False
                break
            indexs.append(cate_embedding[i][item])
            values.append(1)
        if not flag:
            continue
        result['category_feat']['index'].append(indexs)
        result['category_feat']['value'].append(values)

        if type == 'train':
            result['label'].append(int(datas[0]))
        else:
            result['label'].append(0)

        continuous_array = []
        for item in datas[begin_index:begin_index+13]:
            if item == '':
                continuous_array.append(-10.0)
            elif float(item) < 2.0:
                continuous_array.append(float(item))
            else:
                continuous_array.append(math.log(float(item)))
        result['continuous_feat'].append(continuous_array)

    return result

def read_criteo_data(dir_path,file_path,emb_file,numI,numC):
    result = {'label':[], 'index':[],'value':[],'feature_sizes':[]}
    failed = False
    if os.path.isdir(dir_path):
        for item in result:
            try:
                result[item] = np.load(dir_path + '_' + item +'.npy')
            except:
                failed = True
                break
        if not failed:
            print("loaded from %s."%dir_path)
            return result
    else:
        os.mkdir(dir_path)    
    jstr = open(emb_file, 'r').readline()
    meta_info = json.loads(jstr)
    
    # cate_dict = load_criteo_category_index(emb_file)
    # for item in cate_dict:
    #     result['feature_sizes'].append(len(item))
    for item in meta_info:
        result['feature_sizes'].append(len(item))
        # Type = item[0]
        # field = eval(item.replace('less','').split('$')[0][1:])
        # if Type == 'C':
        #     result['feature_sizes'][eval(item[1:])+numI-1] += 1
        # elif Type =='I':
        #     result['feature_sizes'][eval(item[1:])-1] += 1
    f = open(file_path,'r')
    for line in tqdm(f):
        datas = line.strip().split(',')
        result['label'].append(int(datas[0]))
        indexs = [eval(item) for item in datas[1:]]
        values = [1 for i in range(numI+numC)]
        result['index'].append(indexs)
        result['value'].append(values)
    
    for item in result:
        result[item] = np.array(item)
        np.save(dir_path + '_' + item +'.npy', result[item])
    print("loaded from raw and saved to %s"%dir_path)
    return result

def gen_criteo_category_emb_from_libffmfile(filepath, dir_path):
    fr = open(filepath)
    cate_emb_arr = [{} for i in range(39)]
    for line in fr:
        datas = line.strip().split(' ')
        for item in datas[1:]:
            [field, index, value] = item.split(':')
            field = int(field)
            if field > 12:
                index = int(index)
                if index not in cate_emb_arr[field]:
                    cate_emb_arr[field][index] = len(cate_emb_arr[field])

    with open(dir_path, 'w') as f:
        for i,item in enumerate(cate_emb_arr):
            for key in item:
                f.write(str(i)+','+str(key)+','+str(item[key])+'\n')

def gen_emb_input_file(filepath, emb_file, dir_path):
    cate_dict = load_criteo_category_index(emb_file)
    fr = open(filepath,'r')
    fw = open(dir_path,'w')
    for line in fr:
        row = []
        datas = line.strip().split(' ')
        row.append(datas[0])
        for item in datas[1:]:
            [field, index, value] = item.split(':')
            field = int(field)
            if field > 12:
                row.append(str(cate_dict[field][index]))
            else:
                row.append(str(index))
        fw.write(','.join(row)+'\n')



# result_dict = read_criteo_data('../data/tiny_test.txt', '../data/category_index.csv', 'test')
#
# for item in result_dict['continuous_feat']:
#     print item
# cate_dict = gen_criteo_category_index('../data/train.txt')
# write_criteo_category_index('../data/category_index.csv',cate_dict)

# gen_criteo_category_emb_from_libffmfile('../data/tiny_criteo.ffm','../data/tiny_criteo_emb.csv')
# import pdb
# pdb.set_trace()
# gen_emb_input_file('../data/tiny_criteo.ffm','../data/tiny_criteo_emb.csv','../data/tiny_train_input.csv')

# result = read_criteo_data('../data/tiny_train_input.csv', '../data/tiny_criteo_emb.csv')

# print(len(result['label']), len(result['index']), len(result['value']))
# print(result['feature_sizes'])
# for item in result['index']:
#     print(len(item))
