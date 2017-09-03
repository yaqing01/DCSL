import numpy as np
root = '../../'
caffe_root = root + 'caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

def readList(list_name): 
    import random
    import os
    file_object = open(list_name)
    try:
        all_the_text = file_object.read()
    finally:
        file_object.close()

    lines = all_the_text.split('\n')
    #print all_the_text
    DATA_DIR='../../dataset/cuhk03/cuhk03_release/'
    probes=[]
    gallerys=[]
    for filename in lines:
        if filename!='' :
            campair_no = int(filename.split(',')[0])
            person_id = int(filename.split(',')[1])
            #print int(campair_no)
            while True:
                probe_no=random.randint(1,5)
                probe_filename = DATA_DIR +  'data/campair_%d/'%campair_no + '%02d_%04d_%02d.jpg'%(campair_no,person_id,probe_no)
                if os.path.isfile(probe_filename):
                    probes.append(probe_filename)
                    break
            while True:
                gallery_no=random.randint(6,10)  
                gallery_filename = DATA_DIR +  'data/campair_%d/'%campair_no + '%02d_%04d_%02d.jpg'%(campair_no,person_id,gallery_no)
                if os.path.isfile(gallery_filename):
                    gallerys.append(gallery_filename)
                    break
    if len(probes)!=len(gallerys):
        print('something wrong! list length does not match!/n')
        return 0
    else:
        return probes,gallerys

def generateScoreList(net,probes,gallerys):
    transformer = caffe.io.Transformer({'data': (1,3,160,80)})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.array([ 104.00698793,  116.66876762,  122.67891434])) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    
    scoreList=[]
    from time import clock
    start=clock()
    for idx in range(len(probes)):
        probeName=probes[idx]
        probeImage=transformer.preprocess('data', caffe.io.load_image(probeName))
        batchSize= 100
        C,H,W=probeImage.shape
        probeData=np.zeros((batchSize,C*5,H,W))
        probeData[:,0:3,:,:]=probeImage
        probeData[:,3:6,:,:]=probeImage
        probeData[:,6:9,:,:]=probeImage
        probeData[:,9:12,:,:]=probeImage
        probeData[:,12:15,:,:]=probeImage
        galleryIdx=0
        probeScoreLists=[]
        while galleryIdx<len(gallerys):
            galleryDataList=[]
            for batchIdx in range(batchSize):
                if galleryIdx>=len(gallerys):
                    break
                else:
                    galleryName=gallerys[galleryIdx]
                    galleryImage=transformer.preprocess('data', caffe.io.load_image(galleryName))
                    galleryImageSet=np.zeros((C*5,H,W))
                    galleryImageSet[0:3,:,:]=galleryImage
                    galleryImageSet[3:6,:,:]=galleryImage
                    galleryImageSet[6:9,:,:]=galleryImage
                    galleryImageSet[9:12,:,:]=galleryImage
                    galleryImageSet[12:15,:,:]=galleryImage
                    galleryDataList.append(galleryImageSet)
                    galleryIdx+=1
            galleryData=np.asarray(galleryDataList)
            #init net data with images
            N,C,H,W=galleryData.shape
            net.blobs['data'].reshape(N,C,H,W)
            net.blobs['data_p'].reshape(N,C,H,W)
            net.blobs['data'].data[:] = probeData[0:N,:]
            net.blobs['data_p'].data[:] = galleryData
            #net forwad
            net.forward()
            #get output score
            outScore=net.blobs['softmax_score'].data[:,(0,1)]    #softmax_score[0] and softmax_score[1]
            #score_sum=np.exp(outScore[:,0]*1.0)+np.exp(outScore[:,1]*1.0)
            similarScore=outScore[:,1]#np.exp(outScore[:,1]*1.0)/score_sum
            probeScoreLists.extend(similarScore.tolist())
        scoreList.append(probeScoreLists)
        sys.stdout.write('\r%3d/%d, '%(idx+1,len(probes))+probeName)
        sys.stdout.flush()
    predictLists=[]
    for score in scoreList:
        probeRankList=np.argsort(score)[::-1]
        predictLists.append(probeRankList)
    finish=clock()
    print('\r  Processing %dx%d pairs cost %f second time'%(len(probes),len(gallerys),(finish-start)))
    return scoreList,predictLists

def calCMC(net,set_no,rand_times=10):
    from cmc import evaluateCMC
    DATA_DIR= '../../dataset/cuhk03/cuhk03_release/'
    list_name=DATA_DIR+'exp_set/set%02d_test_noval.txt'%(set_no)
    print list_name+'\n'
    #rand 10 times for stable result
    cmc_list=[]
    for i in range(rand_times):
        print 'Round %d with rand list:'%i
        probes,gallerys=readList(list_name)
        scoreList,predictLists=generateScoreList(net,probes,gallerys)
        gtLabels=range(len(probes))
        cmc=evaluateCMC(gtLabels,predictLists)
        cmc_list.append(cmc)
    return np.average(cmc_list,axis=0)

def getCVPRcmc():
    #return the cmc values, 100 dim vetor
    import numpy as np
    cmcIndex=[0,4,8,12,16,21,25,29,33,37,41,45,49,53]
    cmcOfCVPRImproved=[0.5474,0.8753,0.9293,0.9712,0.9764,0.9811,0.9899,0.9901,0.9912,0.9922,0.9937,0.9945,0.9951,1]
    pOfCVPRImproved = np.poly1d(np.polyfit(cmcIndex,cmcOfCVPRImproved,10))
    x_line=range(50)
    cmc=pOfCVPRImproved(x_line)
    return cmc

def plotCMC(cmcDict):
    import matplotlib.pyplot as plt
    get_ipython().magic(u'matplotlib inline')
    from matplotlib.legend_handler import HandlerLine2D
    import numpy as np

    #plot the cmc curve, record CVPR from the pdf paper.cmc[0,4,8,12,16,21,25,29,33,37,41,45,49]
    cmcIndex=[0,4,8,12,16,21,25,29,33,37,41,45,49,53]
    colorList=['g','b','r','y','c','m','k']
    #start to plot
    plt.figure()
    sortedCmcDict = sorted(cmcDict.items(), key=lambda (k, v): v[1])[::-1]
    for idx in range(len(sortedCmcDict)):
        cmc_dictList=sortedCmcDict[idx]
        cmc_name=cmc_dictList[0]
        cmc_list=cmc_dictList[1]
        #print cmc_name,": ",cmc_list
        #x for plot
        x_point=[item+1 for item in cmcIndex[0:-1]]
        x_line=range(50)
        x_plot=[temp+1 for temp in x_line]
        #start plot
        plt.plot(x_plot, cmc_list[x_line],colorList[idx],label="%02.02f%% %s"%(100*cmc_list[0],cmc_name))
        plt.plot(x_point,cmc_list[cmcIndex[0:-1]],colorList[idx]+'*')
        #plt.legend(loc=4,handler_map={line: HandlerLine2D(numpoints=1)})
        #idx of color +1
        idx+=1
    #something to render
    plt.xlabel('Rank')
    plt.ylabel('Identification Rate')
    plt.xticks(np.arange(0,51,5))
    plt.yticks(np.arange(0,1.01,0.1))
    plt.grid()
    plt.legend(loc=4)
    plt.show()
    #end of show
    
def main():
    test_list=range(3,4) #use set 1-10 for test (total 20)
    cmc_list=[]
    for set_no in test_list:
        #init net
        MODEL_FILE = '../../experiments/reid_simplemodel/set%02d/'%(set_no)+'deploy.prototxt'
        PRETRAINED = '../../experiments/reid_simplemodel/set%02d/'%(set_no)+'Snapshots/set%02d_iter_120000.caffemodel'%(set_no)
        caffe.set_device(0)
        caffe.set_mode_gpu()
        net = caffe.Classifier(MODEL_FILE, PRETRAINED,caffe.TEST)
        #caculate CMC
        cmc=calCMC(net,set_no,rand_times=10)
        cmc_list.append(cmc)
    cmc_all=np.average(cmc_list,axis=0)
    print('\nCMC from rank 1 to rank %d:'%(len(cmc_all)))
    print(cmc_all)
    plotCMC(cmc)
    
if __name__ == '__main__':
    main()

