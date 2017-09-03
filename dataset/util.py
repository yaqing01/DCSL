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

def get_sub_dict(main_dict,this_list):
    sub_dict = {}
    for name in this_list:
        sub_dict[name] = main_dict[name]
    return sub_dict
def read_image(galleryName):
    transformer_testing = caffe.io.Transformer({'data_test': (1,3,160,80)})
    transformer_testing.set_transpose('data_test', (2,0,1))
    transformer_testing.set_mean('data_test', np.array([ 104,  117,  123])) # mean pixel
    transformer_testing.set_raw_scale('data_test', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer_testing.set_channel_swap('data_test', (2,1,0))  # the reference model has channels 
    return transformer_testing.preprocess('data_test', caffe.io.load_image(galleryName))
    
def generate_training_pairs(id_list, sub_dict, num_neg_base, num_pos):
    num_neg = num_neg_base*num_pos
    training_pairs = []
    for id in id_list:
        if not sub_dict.has_key(id):
            print 'error!'
        else:
            for ViewIdx in range(len(sub_dict[id])):
                Viewsamples = sub_dict[id][ViewIdx]
                for samplepath in Viewsamples:
                    for posIdx in range(num_pos):
                        randView = random.randint(0,len(sub_dict[id])-2)
                        if randView>=ViewIdx:
                            randView+=1
                        PairSamples = sub_dict[id][randView]
                        #print len(PairSamples)
                        while len(PairSamples)==0:
                            randView = random.randint(0,len(sub_dict[id])-1)
                            PairSamples = sub_dict[id][randView]
                        pairpath = random.choice(PairSamples)
                        if os.path.isfile(pairpath):
                            pair_name = samplepath + ' ' + pairpath + ' 1'
                            training_pairs.append(pair_name)
                    for posIdx in range(int(num_pos/3)):
                        pairpath = random.choice(Viewsamples)
                        if os.path.isfile(pairpath):
                            pair_name = samplepath + ' ' + pairpath + ' 1'
                            training_pairs.append(pair_name)
                    for negIdx in range(num_neg):
                        # choose subdict:
                        randID = random.choice(id_list)
                        if randID==id:
                            continue
                        else:
                            randView = random.randint(0,len(sub_dict[randID])-1)
                            PairSamples = sub_dict[randID][randView]
                            while len(PairSamples)==0:
                                randView = random.randint(0,len(sub_dict[randID])-1)
                                PairSamples = sub_dict[randID][randView]
                            pairpath = random.choice(PairSamples)
                            if os.path.isfile(pairpath):
                                pair_name = samplepath + ' ' + pairpath + ' 0'
                                training_pairs.append(pair_name)       
    return training_pairs

def generate_training_pairs_hnm(id_list, sub_dict, num_neg_base, num_pos, num_test, MODEL_FILE, PRETRAINED, DEVICE_ID):
    print "loading parameters..."
    caffe.set_device(DEVICE_ID)
    caffe.set_mode_gpu()
    net = caffe.Classifier(MODEL_FILE, PRETRAINED,caffe.TEST)

    
    print "mining hard samples..."
    num_neg = num_neg_base*num_pos
    training_pairs = []
    id_idx=0
    pool = Pool(processes=4)
    for id in id_list:
        id_idx+=1
        start=time.time()

        
        if not sub_dict.has_key(id):
            print 'error!'
        else:
            for ViewIdx in range(len(sub_dict[id])):
                Viewsamples = sub_dict[id][ViewIdx]
                
                for samplepath in Viewsamples:
                    
                    for posIdx in range(num_pos):
                        randView = random.randint(0,len(sub_dict[id])-2)
                        if randView>=ViewIdx:
                            randView+=1
                        PairSamples = sub_dict[id][randView]
                        #print len(PairSamples)
                        while len(PairSamples)==0:
                            randView = random.randint(0,len(sub_dict[id])-1)
                            PairSamples = sub_dict[id][randView]
                        pairpath = random.choice(PairSamples)
                        if os.path.isfile(pairpath):
                            pair_name = samplepath + ' ' + pairpath + ' 1'
                            training_pairs.append(pair_name)
                    for posIdx in range(int(num_pos/3)):
                        pairpath = random.choice(Viewsamples)
                        if os.path.isfile(pairpath):
                            pair_name = samplepath + ' ' + pairpath + ' 1'
                            training_pairs.append(pair_name)
                    
                    testing_pairs = []
                    testIdx=0
                    while True:
                        if testIdx==num_test:
                            break
                        randID = random.choice(id_list)

                        if randID==id:
                            continue
                        else:
                            randView = random.randint(0,len(sub_dict[randID])-1)
                            PairSamples = sub_dict[randID][randView]
                            if(len(PairSamples))==0:
                                continue
                            else:
                                pairpath = random.choice(PairSamples)
                                if os.path.isfile(pairpath):
                                    testing_pairs.append(pairpath)    
                                    testIdx+=1
                    
                    galleryIdx=0
                    probeScoreLists=[]
                    batchSize=num_test
                    batchNum=int(num_test/batchSize)
                    probeImage=read_image(samplepath)
                    C,H,W=probeImage.shape
                    probeData=np.zeros((batchSize,C,H,W))
                    probeData[:,:,:,:]=probeImage 
                    
                    
                    while galleryIdx<len(testing_pairs):

                        galleryDataList=[]
                        imageNameList=[]
                        galleryNames = testing_pairs[galleryIdx:galleryIdx+batchSize]
                        for batchIdx in range(batchSize): 
                            if galleryIdx>=len(testing_pairs):
                                break
                            else:
                                galleryName=testing_pairs[galleryIdx]
                                imageNameList.append(galleryName)
                                galleryIdx+=1
                        galleryDataList.extend(pool.map(read_image,galleryNames))

                        galleryData=np.asarray(galleryDataList)
                        N,C,H,W=galleryData.shape
                        net.blobs['data'].reshape(N,C,H,W)
                        net.blobs['data_p'].reshape(N,C,H,W)
                        net.blobs['data'].data[:] = probeData[0:N,:]
                        net.blobs['data_p'].data[:] = galleryData
                        net.forward()
                        outScore=net.blobs['softmax_score'].data[:,(0,1)] 
                        similarScore=outScore[:,1]
                        probeScoreLists.extend(similarScore.tolist())
                    RankList=np.argsort(probeScoreLists)[::-1]
                    
                    
                    
                    
                    for i in range(num_neg):
                        imagename_pair = imageNameList[RankList[i]]
                        if os.path.isfile(imagename_pair):
                            pair_name = samplepath + ' ' + imagename_pair + ' 0'
                            training_pairs.append(pair_name) 
 
                    for negIdx in range(int(num_neg/4)):
                        # choose subdict:
                        randID = random.choice(id_list)
                        if randID==id:
                            continue
                        else:
                            randView = random.randint(0,len(sub_dict[randID])-1)
                            PairSamples = sub_dict[randID][randView]
                            while len(PairSamples)==0:
                                randView = random.randint(0,len(sub_dict[randID])-1)
                                PairSamples = sub_dict[randID][randView]
                            pairpath = random.choice(PairSamples)
                            if os.path.isfile(pairpath):
                                pair_name = samplepath + ' ' + pairpath + ' 0'
                                training_pairs.append(pair_name)    
        finish=time.time()
        sys.stdout.write('\r  Processing %d/%s ids. %fs '%(id_idx,len(id_list),finish-start))
        sys.stdout.flush()  
    pool.close()
    pool.join()
    return training_pairs

def transform(I,rows,cols,pts1,pts2):
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(I,M,(cols,rows))
    return dst

def process_one(filename):
    M=160
    N=80
    transformer = caffe.io.Transformer({'data': (1,3,M,N)})
    transformer.set_transpose('data', (2,0,1))
    #transformer.set_mean('data', np.array([ 104.00698793,  116.66876762,  122.67891434])) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    
    filename_first = filename.split(' ')[0]
    filename_second = filename.split(' ')[1]
    label = int(filename.split(' ')[2])
    map_all_parts = np.zeros([6,M,N]);
    
    
    ratio = 0.05
    ratio_s = 0.05
    
    # the first image
    #image
    I_cv = caffe.io.load_image(filename_first)
    M_I,N_I,c = I_cv.shape
    pts1 = np.float32([[0,0],[N_I,0],[0,M_I]])
    dx = random.uniform(N_I*(-ratio), N_I*(ratio))
    dy = random.uniform(M_I*(-ratio), M_I*(ratio))
    ds = random.uniform(-ratio_s,ratio_s)
    ds_x = (N_I-(1+ds)*N_I)/2
    ds_y = (M_I-(1+ds)*M_I)/2
    if random.uniform(0,1) > 0.3:
        pts2 = np.float32([[dx+ds_x,dy+ds_y],[N_I+dx-ds_x,dy+ds_y],[dx+ds_x,M_I+dy-ds_y]])
    else:
        pts2 = np.float32([[N_I+dx-ds_x,dy+ds_y],[dx+ds_x,dy+ds_y],[N_I+dx-ds_x,M_I+dy-ds_y]])
    rows,cols,c = I_cv.shape
    dst = transform(I_cv,rows,cols,pts1,pts2)
    image=transformer.preprocess('data', dst)
    map_all_parts[0:3,:] = image

    #the second image
    #image
    I_cv = caffe.io.load_image(filename_second)
    M_I,N_I,c = I_cv.shape
    pts1 = np.float32([[0,0],[N_I,0],[0,M_I]])
    dx = random.uniform(N_I*(-ratio), N_I*(ratio))
    dy = random.uniform(M_I*(-ratio), M_I*(ratio))
    ds = random.uniform(-ratio_s,ratio_s)
    ds_x = (N_I-(1+ds)*N_I)/2
    ds_y = (M_I-(1+ds)*M_I)/2
    if random.uniform(0,1) > 0.3:
        pts2 = np.float32([[dx+ds_x,dy+ds_y],[N_I+dx-ds_x,dy+ds_y],[dx+ds_x,M_I+dy-ds_y]])
    else:
        pts2 = np.float32([[N_I+dx-ds_x,dy+ds_y],[dx+ds_x,dy+ds_y],[N_I+dx-ds_x,M_I+dy-ds_y]])
    rows,cols,c = I_cv.shape
    dst = transform(I_cv,rows,cols,pts1,pts2)
    image_p=transformer.preprocess('data', dst)
    map_all_parts[3:6,:] = image_p
    return map_all_parts

def process(imageList,M,N,transformer,env):
    batchSize= 1000
    imageData=np.zeros([batchSize,6,M,N]);
    #imageData_p=np.zeros([batchSize,3,M,N]);
    labelData = np.zeros([batchSize,1])
    pool = Pool(processes=4)
    #inputMap=np.zeros([batchSize,1,M,N])
    random.shuffle(imageList)
    for batchIdx in range(len(imageList)/batchSize):
        start=time.time()

        batchList=imageList[batchIdx*batchSize:(batchIdx+1)*batchSize]
        map_all_parts = pool.map(process_one,batchList)
        for imgIdx in range(len(batchList)):
            imagename= batchList[imgIdx]
            labelData[imgIdx,]=int(imagename.split(' ')[2])
        finish1=time.time()

        imageData[:]=map_all_parts
        with env.begin(write=True) as txn:
    # txn is a Transaction object
            for i in range(batchSize):
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = imageData.shape[1]
                datum.height = imageData.shape[2]
                datum.width = imageData.shape[3]
                datum.data = np.uint8(imageData[i]).tobytes()  # or .tostring() if numpy < 1.9
                datum.label = int(labelData[i])
                str_id = '{:08}'.format(batchIdx*batchSize+i)

        # The encode is only essential in Python 3
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
        finish=time.time()
        #print('\r  Processed %d pairs. All: %f, Process: %f '%((batchIdx+1)*batchSize,finish-start,finish1-start))
        sys.stdout.write('\r  Processed %d pairs. All: %f, Process: %f '%((batchIdx+1)*batchSize,finish-start,finish1-start))
        sys.stdout.flush()    
    # the last batch
    start=time.time()
    batchList=imageList[len(imageList)/batchSize*batchSize:len(imageList)]
    map_all_parts = pool.map(process_one,batchList)
    labelData = np.zeros([len(batchList),1])
    imageData=np.zeros([len(batchList),6,M,N]);
    for imgIdx in range(len(batchList)):
        imagename= batchList[imgIdx]
        labelData[imgIdx,]=int(imagename.split(' ')[2])
    finish1=time.time()
    
    imageData[:]=map_all_parts
    with env.begin(write=True) as txn:
        for i in range(len(batchList)):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = imageData.shape[1]
            datum.height = imageData.shape[2]
            datum.width = imageData.shape[3]
            datum.data = np.uint8(imageData[i]).tobytes()  # or .tostring() if numpy < 1.9
            datum.label = int(labelData[i])
            str_id = '{:08}'.format(len(imageList)/batchSize*batchSize+i)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
    finish=time.time()
    #print('\r  Processed %d pairs. All: %f, Process: %f '%(len(imageList),finish-start,finish1-start))
    sys.stdout.write('\r  Processed %d pairs. All: %f, Process: %f '%(len(imageList),finish-start,finish1-start))
    sys.stdout.flush()    
    pool.close()
    pool.join() 
    return 1


def check_list(pair_list):
    for name in pair_list:
        name1 = name.split(' ')[0]
        name2 = name.split(' ')[1]
        if not os.path.isfile(name1):
            print 'no file:' + name1
            return 0
        if not os.path.isfile(name2):
            print 'no file:' + name2
            return 0
    return 1

def random_shuffle(pair_list):
    pair_list_shuffle = pair_list[:]
    random.shuffle(pair_list_shuffle)
    new_list = []
    for name in pair_list_shuffle:
        name1 = name.split(' ')[0]
        name2 = name.split(' ')[1]
        label = name.split(' ')[2]
        if random.randint(0,1)==0:
            new_list.append(name2 + ' ' + name1 + ' ' + label)
        else:
            new_list.append(name1 + ' ' + name2 + ' ' + label)
    return new_list