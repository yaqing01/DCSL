import lmdb
import os
os.environ['LMDB_FORCE_CFFI'] = '1'

root = '../'
caffe_root = root + 'caffe/'  # this file is expected to be in {caffe_root}/examples
import cv2
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

import glob

# import cv2
import numpy as np
from matplotlib import pyplot as plt

import random

#import png
import itertools

from skimage.io import imread,imsave
import shutil
import getopt
from multiprocessing import Pool 
import time

# CUHK01
import cuhk01_util
# CUHK03
import cuhk03_util
# VIPER
import viper_util
# i-LIDS
import ilids_util
# MARKET
import market_util
# utils
import util

set_no = 1
save_p = 'train_lmdb'
shuffle = 1
M=160
N=80
transformer = caffe.io.Transformer({'data': (1,3,M,N)})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# for HNM
HNM = 0
GPU_ID = 0
MODEL_FILE = '../experiments/reid_earlyfusion/set01/deploy_google_bigmap.prototxt'
PRETRAINED = '../experiments/reid_earlyfusion_google_bigmap/set01/Snapshots/set01_cuhk01_iter_60000.caffemodel'

dataset_list = ['CUHK01','CUHK03','ViPeR','iLIDS','MARKET']  #,'3DPeS','PRID2011']
dataset_path = ['cuhk01/cuhk01/','cuhk03/cuhk03_release/', 'Viper/VIPeR/','i-LIDS/data/','Market-1501/'] #] #,'Viper/VIPeR/','i-LIDS/']
dataset_root = ['cuhk01/','cuhk03/cuhk03_release/','Viper/','i-LIDS/','Market-1501/']
dataset_util = [cuhk01_util,cuhk03_util,viper_util,ilids_util,market_util]
dataset_test_id_number = [100,100,316,59,752]
dataset_setno = [set_no,set_no,set_no,set_no,0]
dataset_usage = [0,1,0,0,0]
dataset_hnm = [HNM,HNM,HNM,HNM,HNM]
dataset_numpos = [25,20,40,40,20]

dataset_info_dict = {}
count = 0
for setname in dataset_list:
    if not dataset_info_dict.has_key(setname):
        dataset_info_dict[setname] = {}
    dataset_info_dict[setname]['DatasetPath'] = dataset_path[count]
    dataset_info_dict[setname]['Util'] = dataset_util[count]
    dataset_info_dict[setname]['Used'] = dataset_usage[count]
    dataset_info_dict[setname]['Set_no'] =  dataset_setno[count]
    dataset_info_dict[setname]['Num_pos'] = dataset_numpos[count]
    dataset_info_dict[setname]['HNM'] = dataset_hnm[count]
    if setname=='CUHK03':
        dataset_info_dict[setname]['PartitionTrain']  = dataset_root[count] + 'exp_set/set%02d_train_noval.txt'%((dataset_setno[count]))
        dataset_info_dict[setname]['PartitionTest'] = dataset_root[count] + 'exp_set/set%02d_test_noval.txt'%((dataset_setno[count]))
    else:
        dataset_info_dict[setname]['PartitionTrain']  = dataset_root[count] + 'exp_set/testid%03d_set%02d_train.txt'%(dataset_test_id_number[count],(dataset_setno[count]))
        dataset_info_dict[setname]['PartitionTest'] = dataset_root[count] + 'exp_set/testid%03d_set%02d_test.txt'%(dataset_test_id_number[count],(dataset_setno[count]))
        
    count+=1
for key in dataset_info_dict.keys():
    print key + ':'
    print dataset_info_dict[key]

# collecting training pairs
PHASE = 'train'

training_pairs = []
for setname in dataset_list:
    if dataset_info_dict[setname]['Used'] == 1:
        filename_part = dataset_info_dict[setname]['PartitionTrain']
        if not setname=='MARKET':
            main_dict=dataset_info_dict[setname]['Util'].collect_data(dataset_info_dict[setname]['DatasetPath'])
            this_list=dataset_info_dict[setname]['Util'].partition_file_to_list(filename_part)
            sub_dict=util.get_sub_dict(main_dict,this_list)
        else:
            main_dict=dataset_info_dict[setname]['Util'].collect_data(dataset_info_dict[setname]['DatasetPath']+'bounding_box_%s/'%PHASE)
            sub_dict=main_dict
            this_list=sub_dict.keys()
        if dataset_info_dict[setname]['HNM']==0:
            this_pairs = util.generate_training_pairs(this_list, sub_dict, 4, dataset_info_dict[setname]['Num_pos'])
        else:
            this_pairs = util.generate_training_pairs_hnm(this_list[:], sub_dict, 2, dataset_info_dict[setname]['Num_pos'], 400, MODEL_FILE, PRETRAINED, GPU_ID)
        training_pairs.extend(this_pairs)
        print setname + ' training pairs: %d'%len(this_pairs)
print 'All training pairs: %d'%len(training_pairs)    
if util.check_list(training_pairs)==1:
    print 'pass list check'
else:
    print 'list check failed'
    
if shuffle==1:
    shuffled_pairs=util.random_shuffle(training_pairs)
    
save_root = save_p + '/set%02d_%s'%(set_no,PHASE)
if HNM==1:
    save_root = save_root + '_hnm'
print 'saved to: ' + save_root
if os.path.exists(save_root):
    shutil.rmtree(save_root)
os.makedirs(save_root) 
X = np.zeros((2000, 6, 160, 80), dtype=np.float32)
map_size = X.nbytes * 1000
env = lmdb.open(save_root, map_size=map_size)

util.process(shuffled_pairs,M,N,transformer,env)   

PHASE = 'test'
training_pairs = []
for setname in dataset_list:
    if dataset_info_dict[setname]['Used'] == 1:
        filename_part = dataset_info_dict[setname]['PartitionTest']
        if not setname=='MARKET':
            main_dict=dataset_info_dict[setname]['Util'].collect_data(dataset_info_dict[setname]['DatasetPath'])
            this_list=dataset_info_dict[setname]['Util'].partition_file_to_list(filename_part)
            sub_dict=util.get_sub_dict(main_dict,this_list)
        else:
            main_dict=dataset_info_dict[setname]['Util'].collect_data(dataset_info_dict[setname]['DatasetPath']+'bounding_box_%s/'%PHASE)
            sub_dict=main_dict
            this_list=sub_dict.keys()
        this_pairs = util.generate_training_pairs(this_list, sub_dict, 4, 2)
        training_pairs.extend(this_pairs)
        print setname + ' training pairs: %d'%len(this_pairs)
print 'All training pairs: %d'%len(training_pairs)    
if util.check_list(training_pairs)==1:
    print 'pass list check'
else:
    print 'list check failed'
    
if shuffle==1:
    shuffled_pairs=util.random_shuffle(training_pairs)
    
save_root = save_p + '/set%02d_%s'%(set_no,PHASE)
if HNM==1:
    save_root = save_root + '_hnm'
    
print 'saved to: ' + save_root
if os.path.exists(save_root):
    shutil.rmtree(save_root)
os.makedirs(save_root) 
X = np.zeros((2000, 6, 160, 80), dtype=np.float32)
map_size = X.nbytes * 1000
env = lmdb.open(save_root, map_size=map_size)

util.process(shuffled_pairs,M,N,transformer,env)   
