#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#Description: Tracking metrics, IOU and precision
#Date: 2021/04/11
#Author: Steven Huang, Auckland, NZ
import os 
import torch
import torchvision.ops.boxes as bops
import numpy as np
import matplotlib.pyplot as plt
from predict import getFileList,parseFileName
from commonPath import pathsFiles,getFileName,deleteFile
from commonPath import getFileNameNo,createPath
from plotCommon import plotSub
import pandas as pd
import matplotlib 

matplotlib.rcParams['figure.dpi'] = 200 #high resolution when save plot
"""
 ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', \
'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', \
    'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
"""
#matplotlib.use('template')

SMALL_SIZE = 6
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

#plt.rc('figure.dpi', 300)
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#plt.rc('font', family='Times New Roman')
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : array
        Keys: [x1, y1, w, h]
        The (x1, y1) position is at the top left corner,
        the (w, h) position is at the bottom right corner
    bb2 : array
        Keys: [x2, y2, w, h]
        The (x2, y2) position is at the top left corner,
        the (w, h) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[0]+bb1[2], bb2[0]+bb2[2])
    y_bottom = min(bb1[1]+bb1[3], bb2[1]+bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def testIOU():
    #bb1 = {'x1':281.0,'x2':300.0, 'y1':87.0, 'y2':275.0}
    #bb2 = {'x1':282.0,'x2':300.0, 'y1':87.0, 'y2':275.0}
    #iou = get_iou2(bb1,bb2)
    
    bb1 = [281.0, 300.0, 87.0, 275.0] 
    bb2 = [281.0, 300.0, 87.5, 275.0]
    iou = get_iou(bb1,bb2)
    
    print('iou=', iou)
    
    box1 = torch.tensor([[281, 300, 281+87, 300+275]], dtype=torch.float)
    box2 = torch.tensor([[281, 300, 281+87.5, 300+275]], dtype=torch.float)
    iou = bops.box_iou(box1, box2)
    print('iou=', iou)
    
    box1 = torch.tensor([[281, 300, 281+87, 300+275],
                         [281, 300, 281+87, 300+275],
                         [281, 300, 282, 310]], dtype=torch.float)
    box2 = torch.tensor([[281, 300, 281+87.5, 300+275],
                         [281, 300, 281+87.5, 320+275],
                         [281, 300, 282, 310]], dtype=torch.float)
    iou = bops.box_iou(box1, box2)
    
    diag = torch.diagonal(iou, 0)
    
    print('iou=', iou)
    print('diag=', diag)
    
    bops.box_area

def changeRect(rt): #from x1,y1 w,h --> x1,y1,x2,y2
    rt[:,2] = rt[:,2] + rt[:,0]
    rt[:,3] = rt[:,3] + rt[:,1]
    return rt

def getGroundTrueAndPred(trueFile, predictFile, delimiter=None):
    gt = np.loadtxt(trueFile, delimiter=delimiter, dtype=float, skiprows=0, unpack=False)
    pred = np.loadtxt(predictFile)
    
    #print('before gt=', gt)
    gt = changeRect(gt)
    #print('after gt=', gt)
    pred = changeRect(pred)
    
    #print('gt=', gt.shape)
    #print('pred=', pred.shape)
    assert(gt.shape == pred.shape)
    
    gt = torch.from_numpy(gt)
    pred = torch.from_numpy(pred)
    return gt,pred

def getOTB_GTAndPred():
    trueFile = r'.\data\OTB\Crossing\groundtruth_rect.txt' #
    predictFile = r'.\res\data\OTB\Crossing\CrossingPredict.txt'
    #trueFile = r'.\res\imgRt.txt' #
    #predictFile = r'.\res\imgRt.txt' #
    
    deli = None #',' 
    return getGroundTrueAndPred(trueFile, predictFile, delimiter=deli)

def Tracking_IOU_OTB():
    gt,pred = getOTB_GTAndPred()
    Tracking_IOU(gt,pred)
    
def Tracking_IOU(gt,pred, plot=True): #sucess rate
    iou = bops.box_iou(gt, pred)
    iou = torch.diagonal(iou, 0)
    #print('iou=', iou)
    #print('iou=', iou.shape)
    if plot:
        plotIOU_SR(iou.numpy())
        
    return iou.numpy()

def centerRect(rt): #from x1,y1 w,h --> xc,yc
    xc = (rt[:,0] + rt[:,2])/2
    yc = (rt[:,1] + rt[:,3])/2
    
    xc = xc.reshape([-1,1])
    yc = yc.reshape([-1,1])
    #print(xc)
    #print(yc)
    return torch.cat((xc, yc), 1)

def Tracking_Precision_OTB():
    gt,pred = getOTB_GTAndPred()
    return Tracking_Precision(gt,pred)

def Tracking_Precision(gt, pred, plot=True): #sucess rate
    #print('before gt=', gt)
    gt = centerRect(gt)
    pred = centerRect(pred)
    #print('after gt=', gt)
    #print('after pred=', pred)

    dis = torch.cdist(gt, pred)
    #print('dis=', dis)
    #print('dis=', dis.shape)
    dis = torch.diagonal(dis, 0)
    #print('dis=', dis)
    #print('dis=', dis.shape)
    
    if plot:
        plotPrecision(dis)
    return dis.numpy()

def getIoUXY(iou,N=200):    
    def SuccessRate(k):
        res = np.where(iou > k, 1, 0) 
        return np.sum(res)/len(iou)
    
    x = np.linspace(0,1,num=N)
    y = list(map(SuccessRate, [i for i in x]))
    
    #AUC
    auc = np.sum(np.array(y)/N)
    #print(auc,type(auc))
    return x, y, auc, y[N//2]

def getPrecisionXY(dis,N=200):    
    def Precision(k):
        res = np.where(dis < k, 1, 0) 
        return np.sum(res)/len(dis)
    
    x = np.linspace(0,50,num=N)
    y = list(map(Precision, [i for i in x]))
    pre = y[N*30//50] #get pre when dis=30
    #print('y=', y, ' pre=', pre)
    return x,y,pre

def plotIOU_SR(iou, title='', dstPath=None, show=False):
    def plotXY(x,y,name): #Success rate
        plt.clf()
        plt.title(name)
        plt.plot(x,y,label='SiamFC')
        plt.xlabel('Overlap threshold')
        plt.ylabel('Success rate')
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.grid(linestyle='-.') #'-', '--', '-.', ':', '',
        #plt.legend(loc='lower left')
        if dstPath is None:
            plt.savefig(r'.\res\IoUMetric.png', dpi=300)
        else:
            plt.savefig(dstPath, dpi=300)
            
        if show:
            plt.show()
    
    x,y,_,_ = getIoUXY(iou)
    plotXY(x, y, title) #'Success plots of OPE'
    
def plotPrecision(dis, title='', dstPath=None,show=False):
    def plotXY(x,y,name): #Success rate
        plt.clf()
        plt.title(name)
        plt.plot(x,y,label='SiamFC')
        plt.xlabel('Location error threshold')
        plt.ylabel('Precision')
        plt.xlim((0,50))
        plt.ylim((0,1))
        plt.grid(linestyle='-.') #'-', '--', '-.', ':', '',
        #plt.legend(loc='lower right')
        if dstPath is None:
            plt.savefig(r'.\res\DisMetric.png', dpi=300)
        else:
            plt.savefig(dstPath, dpi=300)
        if show:
            plt.show()
        
    x,y,_ = getPrecisionXY(dis)
    plotXY(x, y, title) #'Precision plots of OPE'
  
def plotIOUs(iou, title='', dstPath=None, show=False):
    def plotXY(x,y,name): #Success rate
        plt.clf()
        plt.title(name)
        plt.plot(x,y,label='SiamFC')
        plt.xlabel('Frames')
        plt.ylabel('IoUs')
        #plt.xlim((0,1))
        #plt.ylim((0,1))
        plt.grid(linestyle='-.') #'-', '--', '-.', ':', '',
        #plt.legend(loc='lower left')
        if dstPath is None:
            plt.savefig(r'.\res\IoUs.png', dpi=300)
        else:
            plt.savefig(dstPath, dpi=300)
            
        if show:
            plt.show()
    
    print('iou=', iou,type(iou),iou.shape)
    
    x,y = np.arange(len(iou)), iou
    plotXY(x, y, title) 
    
def plotDistances(dis, title='', dstPath=None,show=False):
    def plotXY(x,y,name): #Success rate
        plt.clf()
        plt.title(name)
        plt.plot(x,y,label='SiamFC')
        plt.xlabel('Frames')
        plt.ylabel('Distance')
        #plt.xlim((0,50))
        #plt.ylim((0,1))
        plt.grid(linestyle='-.') #'-', '--', '-.', ':', '',
        #plt.legend(loc='lower right')
        if dstPath is None:
            plt.savefig(r'.\res\DisMetric.png', dpi=300)
        else:
            plt.savefig(dstPath, dpi=300)
        if show:
            plt.show()
        
    x,y = np.arange(len(dis)), dis
    plotXY(x, y, title)

##################### MOT7 ###################
def loadLines2Numpy(file):
    data = np.loadtxt(file)
    #print('data=', type(data), data, data.shape)
    
    #data = data[data != 0]
    #print('after data=', type(data), data, data.shape)
    return data

def plotAllIoU_SR(iouFile,dstPath=None,show=False):
    #base = r'.\data\MOT\MOT17_GT_Jason\MOT17-04-FRCNN'
    #iouFile = base + '\\IoU_alexnet_e754\\IoU.txt'
    plotIOU_SR(loadLines2Numpy(iouFile), dstPath=dstPath,show=show)
 
def plotAllPrecision(disFile, dstPath=None,show=False):
    #base = r'.\data\MOT\MOT17_GT_Jason\MOT17-04-FRCNN'
    #disFile = base + '\\IoU_alexnet_e754\\dis.txt'
    plotPrecision(loadLines2Numpy(disFile), dstPath=dstPath,show=show)
       
def plotAllIoUs(iouFile,dstPath=None,show=False):
    plotIOUs(loadLines2Numpy(iouFile), dstPath=dstPath,show=show)
    
def plotAllDis(disFile, dstPath=None,show=False):
    plotDistances(loadLines2Numpy(disFile), dstPath=dstPath,show=show)
    
def testPlot():
    N=500
    #iou = np.random.rand(N)
    iou = np.random.uniform(low=0, high=1, size=(N,))
    plotIOU_SR(iou)
    #plotPrecision()
    
def plotCompareIoU(base):
    iouFile1 = base + '\\IoU_siamfc_alexnet_e754\\IoU.txt'
    label1 = 'SiamFC_Alexnet'
    iouFile2 = base + '\\IoU_siamfc_Con2Net_BalancedLoss_e\\IoU.txt'
    label2 = 'SiamFC_Con2Net'
    
    iouFile3 = base + '\\IoU_siamfc_Sequential_vgg19_BalancedLoss_e\\IoU.txt'
    #iouFile3 = base + '\\IoU_siamfc_Sequential_vgg19_FocalLoss_e\\IoU.txt'
    label3 = 'SiamFC_Vgg19'
    #iouFile4 = base + '\\IoU_siamfc_Sequential_MobileNet_BalancedLoss_e\\IoU.txt'
    iouFile4 = base + '\\IoU_siamfc_Sequential_MobileNet_FocalLoss_e\\IoU.txt'
    label4 = 'SiamFC_MobileNetV2'

    plt.clf()
    ax = plt.subplot(1,1,1)
    
    x,y,auc,sr = getIoUXY(loadLines2Numpy(iouFile1))
    plotSub(x, y, ax, label=label1, color='b')
    print(label1, ' AUC=', auc, ' sr=', sr)
    
    x,y,auc,sr = getIoUXY(loadLines2Numpy(iouFile2))
    plotSub(x, y, ax, label=label2, color='r', linestyle='dashed')
    print(label2, ' AUC=', auc, ' sr=', sr)
    
    x,y,auc,sr = getIoUXY(loadLines2Numpy(iouFile3))
    plotSub(x, y, ax, label=label3, color='g', linestyle='dotted')
    print(label3, ' AUC=', auc, ' sr=', sr)
    
    x,y,auc,sr = getIoUXY(loadLines2Numpy(iouFile4))
    plotSub(x, y, ax, label=label4, color='k', linestyle='dashdot')
    print(label4, ' AUC=', auc, ' sr=', sr)
    
    #plt.axis('square')
    plt.xlabel('Overlap threshold')
    plt.ylabel('Success rate')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.grid(linestyle='-.') #'-', '--', '-.', ':', '',
    plt.legend(loc='upper right')
    #plt.legend()
    plt.savefig(r'.\res\IoUPlot.png', dpi=300)
    plt.show()
    
def plotComparePrecision(base):
    #base = r'.\data\MOT\MOT17_GT_Jason\MOT17-04-FRCNN'
    disFile1 = base + '\\IoU_siamfc_alexnet_e754\\dis.txt'
    label1 = 'SiamFC_Alexnet'
    
    disFile2 = base + '\\IoU_siamfc_Con2Net_BalancedLoss_e\\dis.txt'
    label2 = 'SiamFC_Con2Net'
    
    disFile3 = base + '\\IoU_siamfc_Sequential_vgg19_BalancedLoss_e\\dis.txt'
    label3 = 'SiamFC_Vgg19'
    
    #disFile4 = base + '\\IoU_siamfc_Sequential_MobileNet_BalancedLoss_e\\dis.txt'
    disFile4 = base + '\\IoU_siamfc_Sequential_MobileNet_FocalLoss_e\\dis.txt'
    label4 = 'SiamFC_MobileNetV2'
    
    plt.clf()
    ax = plt.subplot(1,1,1)
    
    x,y,pre = getPrecisionXY(loadLines2Numpy(disFile1))
    plotSub(x, y, ax, label=label1, color='b')
    print(label1, 'final pre=', pre)
    
    x,y,pre = getPrecisionXY(loadLines2Numpy(disFile2))
    plotSub(x, y, ax, label=label2, color='r', linestyle='dashed')
    print(label2, 'final pre=', pre)
    
    x,y,pre = getPrecisionXY(loadLines2Numpy(disFile3))
    plotSub(x, y, ax, label=label3, color='g', linestyle='dotted')
    print(label3, 'final pre=', pre)
    
    x,y,pre = getPrecisionXY(loadLines2Numpy(disFile4))
    plotSub(x, y, ax, label=label4, color='k', linestyle='dashdot')
    print(label4, 'final pre=', pre)
    
    #plt.axis('square')
    plt.xlabel('Location error threshold')
    plt.ylabel('Precision')
    plt.xlim((0,50))
    plt.ylim((0,1))
    plt.grid(linestyle='-.') #'-', '--', '-.', ':', '',
    plt.legend(loc='upper right')
    #plt.legend()
    plt.savefig(r'.\res\PrecisonPlot.png', dpi=300)
    plt.show()

def getMOT_GTAndPredAll(trueFile, predictPath, IoUPath, objIDFilters=[]): #get Iou&Distance to one file from all predciton files
    # base = r'.\data\MOT\MOT17_GT_Jason\MOT17-04-FRCNN'
    # trueFile = base
    # predictPath = base + '\\predBoxFiles'
    # IoUPath = base + '\\IoU'
    createPath(IoUPath)
    ioufilesPath = os.path.join(IoUPath,'resIoU')
    createPath(ioufilesPath)
    disfilesPath = os.path.join(IoUPath,'resDis')
    createPath(disfilesPath)
    
    gtFiles = getFileList(trueFile, 'txt')
    iouAll = None
    disAll = None
    df = pd.DataFrame()
    for i, gtFile in enumerate(gtFiles):
        objId,start,stop = parseFileName(getFileNameNo(gtFile))
        
        if objIDFilters != [] and objId in objIDFilters:
            continue
        
        predictFile = os.path.join(predictPath, getFileNameNo(gtFile) + '_pred.txt')
        print(str(i)+'/'+str(len(gtFiles)), getFileName(gtFile), predictFile, objId,start,stop)
        IouFile = os.path.join(ioufilesPath, getFileNameNo(gtFile) + '_IoU.txt')
        disFile = os.path.join(disfilesPath, getFileNameNo(gtFile) + '_dis.txt')
        
        if not os.path.exists(predictFile):
            print('warning: file not exist! ', predictFile)
            continue 
        
        gt,pred = getGroundTrueAndPred(gtFile, predictFile, delimiter=',')
    
        iou = Tracking_IOU(gt, pred, plot=False)
        if iouAll is None:
            iouAll = iou
        else:
            iouAll = np.hstack((iouAll,iou))
            
        dis = Tracking_Precision(gt,pred,plot=False)
        if disAll is None:
            disAll = dis
        else:
            disAll = np.hstack((disAll,dis))
            
        #torch.save(iou, IouFile)
        np.savetxt(IouFile, iou, fmt='%f')
        np.savetxt(disFile, dis, fmt='%f')
        #print('iou=\n', iou, np.mean(iou))
        #print('dis=\n', dis, np.mean(dis))
        
        line = pd.DataFrame([[objId, np.mean(iou), np.mean(dis)]], columns=['objId','meanIoU','meanDis'])
        df = df.append(line)
        #break
        
    IouFile = os.path.join(IoUPath, 'IoU.txt')
    print('iouAll=', type(iouAll), iouAll.shape)
    np.savetxt(IouFile, iouAll, fmt='%f')
    #plotIOU_SR(iouAll)
    
    disFile = os.path.join(IoUPath, 'dis.txt') 
    print('disAll=', type(disAll), disAll.shape)
    np.savetxt(disFile, disAll, fmt='%f')
    #plotPrecision(disAll)   
    print('df=\n', df)
    df.set_index(["objId"], inplace=True)
    df.to_csv(os.path.join(IoUPath, 'statits.csv'),index=True)
    
def getMOT_GTAndPred():
    base = r'.\res\data\MOT17\MOT17-04-FRCNN_ID_1\\'
    trueFile = base + 'groundtruth_rect.txt' #
    predictFile = base + 'pred.txt'
    return getGroundTrueAndPred(trueFile, predictFile, delimiter=',') 

def removeSomeIoU(base):
    def checkFile(path):
        for f in pathsFiles(path,'txt'):
            objId,start,stop = parseFileName(getFileNameNo(f))
            iou = loadLines2Numpy(f)
            if iou[1] == 0:
                print('pred=0 ', getFileNameNo(f), objId,start,stop)
            #print(f, iou[1])
            
    def removeObjFiles(path,objIds):
        for f in pathsFiles(path,'txt'):
            objId,start,stop = parseFileName(getFileNameNo(f))
            if objId in objIds:
                print('delete', f)
                deleteFile(f)
                
            
    objIds = [107,108,109,110,111,112,113,114,117,118,119,\
        121,124,125,126,127,128,129,130,131,132,133,134,\
        135,138,13,14,15,19,20,2,38,39,41,42,47,51,53,60,\
        63,66,67,68,69,70,71,72,73,75,76,77,78,80,82,84,85,\
        87,88,89,90,91,92,93,94,95,97,98,99]
    
    pathAll =[]
    pathAll.append(os.path.join(base, 'IoU_siamfc_alexnet_e754'))
    pathAll.append(os.path.join(base, 'IoU_siamfc_Con2Net_BalancedLoss_e'))
    pathAll.append(os.path.join(base, 'IoU_siamfc_Sequential_vgg19_BalancedLoss_e'))
    pathAll.append(os.path.join(base, 'IoU_siamfc_Sequential_MobileNet_FocalLoss_e'))
    for i in pathAll:
        disPath = os.path.join(i, 'resDis')
        iouPath = os.path.join(i, 'resIoU')
        #removeObjFiles(disPath,objIds)
        #removeObjFiles(iouPath,objIds)
        print('iouPath=\n', iouPath)
        checkFile(iouPath)
        #break
    
def Tracking_All_MOT7():
    #gt,pred = getMOT_GTAndPred() #one gt and pred
    #Tracking_IOU(gt,pred)
    
    base = os.path.abspath(r'data/MOT/MOT17_GT_Jason/MOT17-04-FRCNN')
    #base = os.path.expanduser(r'data/MOT/MOT17_GT_Jason/MOT17-04-FRCNN')
    #base = os.path.abspath(r'data/MOT/MOT17_GT_Jason/MOT17-13-FRCNN')
    #base = os.path.expanduser(r'data/MOT/MOT17_GT_Jason/MOT17-02-FRCNN')
    trueFile = base
    
    name = 'alexnet_e754'
    
    name = 'siamfc_Sequential_vgg19_BalancedLoss_e'
    name = 'siamfc_Sequential_vgg19_FocalLoss_e'
    name = 'siamfc_Sequential_MobileNet_BalancedLoss_e'
    name = 'siamfc_Sequential_AlexNet_BalancedLoss_e'
    name = 'siamfc_Sequential_MobileNet_FocalLoss_e'
    name = 'siamfc_Con2Net__BalancedLoss_e' #02-frcnn,04
  
  
    # name = 'siamfc_Sequential_AlexNetOO_FocalLoss_e'
    #name = 'siamfc_Con2Net_BalancedLoss_e860'
    #name = 'siamfc_alexnet_e654'
    #name = 'siamfc_AlexNetV1_FocalLoss_e713'
    
    # name = 'siamfc_Sequential_AlexNet_FocalLoss_e'
    # name = 'siamfc_alexnet_e554'
    # #name = 'siamfc_alexnet_e50'
    
    #name = 'old'
    # name = 'siamfc_alexnet_e754'
    # name = 'siamfc_Con2Net_BalancedLoss_e'
    # name = 'siamfc_Sequential_vgg19_BalancedLoss_e'
    # name = 'siamfc_Sequential_MobileNet_FocalLoss_e'
    
    predictPath = os.path.join(base, 'predBoxFiles_'+name)
    IoUPath = os.path.join(base, 'IoU_' + name)
    
    """
    # MOT17-04-FRCNN siamfc_alexnet_e754
    objIds = [107,108,109,110,111,112,113,114,117,118,119,\
        121,124,125,126,127,128,129,130,131,132,133,134,\
        135,138,13,14,15,19,20,2,38,39,41,42,47,51,53,60,\
        63,66,67,68,69,70,71,72,73,75,76,77,78,80,82,84,85,\
        87,88,89,90,91,92,93,94,95,97,98,99]
    
    # MOT17-04-FRCNN siamfc_Con2Net_BalancedLoss_e
    objIds = [115,128]
    
    
    # MOT17-04-FRCNN siamfc_Sequential_vgg19_BalancedLoss_e
    objIds = [107,108,109,110,111,112,113,114,115,117,118,\
        121,124,125,126,127,128,129,130,131,132,133,134,135,\
        138,13,14,15,16,19,20,2,38,40,41,42,47,4,51,53,60,66,\
        67,68,69,70,71,72,73,75,76,77,78,80,84,85,87,88,89,90,\
        91,98,99]
    
    # MOT17-04-FRCNN siamfc_Sequential_MobileNet_FocalLoss_e
    objIds = [102,103,105,107,108,109,110,112,113,114,115,117,\
        118,121,126,127,128,135,138,13,15,21,25,28,33,37,38,39,\
            41,44,4,51,53,5,60,62,66,67,68,70,71,72,73,82,84,85,88,92,9]
    """
    objIds=[]
    #getMOT_GTAndPredAll(trueFile, predictPath, IoUPath, objIDFilters=objIds)#filter excluded objs
    
    #MOT17-13-FRCNN no pedestrain objs
    objIdFilters = [1,3,12,29,30,41,76,80,82,85,104,105,116,\
            106,81,125,92,130101,184,107,108,89,179,90,91,177,93,\
            162,99,182,150,109,181,175,183,138,176,146,96,141,185,\
            75,65,94,95,110,161,103,121,88,97,131,83,87,79,86,111,\
            172,112,174,113,27,117,114,173,118,98,115,77,130,84,78] #no pedestrains
     
    #MOT17-04-FRCNN no pedestrain objs
    objIdFilters = [7,8,9,10,11,12,13,14,15,16,17,18,19,\
            20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,\
            36,37,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,\
                55,56,57,58,59,64,78, 137,] #no pedestrains
    
    #MOT17-02-FRCNN no pedestrain objs
    #objIdFilters = [4,50,51,52,53,79,75] #no pedestrains
    analyseMetrics(base, 'IoU_'+name, noPedObjs=objIdFilters)
    
    #removeSomeIoU(base)
    #plotCompareIoU(base)
    #plotComparePrecision(base)
    
    
def Tracking_Precision_MOT17():
    gt,pred = getMOT_GTAndPred()
    return Tracking_Precision(gt,pred)

def readCsv(file):
    df = pd.read_csv(file)
    #print(df.describe().transpose())
    print(df.head())
    print('df.columns=',df.columns)
    #print('df.dtypes = ',df.dtypes)
    #print('df.dtypes = ',df.dtypes)
    return df
    
def analysisTrecking(df):    
    num = 10
    #best iou
    df = df.sort_values(by=['meanIoU'], ascending=False)
    print('best iou=\n', df[:num])
    #print(df[:num]['objId'])
    
    #best distance
    df = df.sort_values(by=['meanDis'], ascending=True)
    print('best distance=\n', df[:num])
    
    #worst iou
    df = df.sort_values(by=['meanIoU'], ascending=True)
    print('worst iou=\n', df[:num])
    
    #worst distance
    df = df.sort_values(by=['meanDis'], ascending=False)
    print('worst distance=\n', df[:num])
    
def analyseMetrics(base, name, noPedObjs):    
    def filtersNoPed(df, noPedObjs):
        #print('df=', df,  df.shape, '\n')
        #print('noPedObjs=', noPedObjs, len(noPedObjs), '\n')
        for i, row in enumerate(df.iterrows()):
            #print(i, 'row[1]=', row[1][0], type(row[1][0]))
            if int(row[1][0]) in noPedObjs:
                df = df.drop(i)
        #print('df=', df, df.shape, '\n')
        return df
    
    dstPath = os.path.join(base, name)
    iouFile = os.path.join(dstPath, 'IoU.txt')
    disFile = os.path.join(dstPath, 'dis.txt')
    
    plotAllIoU_SR(iouFile, os.path.join(dstPath, 'IoU.png'))
    plotAllPrecision(disFile, os.path.join(dstPath, 'pre.png'))
    
    csvFile = os.path.join(base, name,'statits.csv')
    df = readCsv(csvFile)
    if noPedObjs:
        df = filtersNoPed(df,noPedObjs)
    analysisTrecking(df)
    
def analyseIoU(path, objFilters = None, fromZero=False):
    iouFiles = []
    iouFilesAll=pathsFiles(path,'txt')
    for i in iouFilesAll:
        objId,start,stop = parseFileName(getFileNameNo(i))
        iou = loadLines2Numpy(i)
        #print(i, objId,start,stop, iou.shape)
        if objFilters is None:
            iouFiles.append(i)
        else:
            if objId in objFilters:
                iouFiles.append(i)

    #start draw
    plt.clf()
    ax = plt.subplot(1,1,1)
    
    for i in iouFiles:
        objId,start,stop = parseFileName(getFileNameNo(i))
        iou = loadLines2Numpy(i)
        
        if fromZero:
            x = np.arange(len(iouFiles))
        else:
            x = np.arange(start,stop+1)
        y = iou
        
        ls = 'obj:'+str(objId)+', From '+str(start)+' To '+str(stop)
        plotSub(x, y, ax, label=ls)
            
    #plt.axis('square')
    plt.xlabel('Frames')
    plt.ylabel('IoU')
    #plt.xlim((0,1))
    #plt.ylim((0,1))
    plt.grid(linestyle='-.') #'-', '--', '-.', ':', '',
    plt.legend(loc='lower right') #upper
    #plt.legend()
    #plt.savefig(r'.\res\IoUObjs.png', dpi=300)
    plt.show()
    
def analyseDis(path, objFilters = None, fromZero=False):
    disFiles = []
    disFilesAll=pathsFiles(path,'txt')
    for i in disFilesAll:
        objId,start,stop = parseFileName(getFileNameNo(i))
        iou = loadLines2Numpy(i)
        #print(i, objId,start,stop, iou.shape)
        if objFilters is None:
            disFiles.append(i)
        else:
            if objId in objFilters:
                disFiles.append(i)

    #start draw
    plt.clf()
    ax = plt.subplot(1,1,1)
    
    for i in disFiles:
        objId,start,stop = parseFileName(getFileNameNo(i))
        iou = loadLines2Numpy(i)
        
        if fromZero:
            x = np.arange(len(disFiles))
        else:
            x = np.arange(start,stop+1)
        y = iou
        
        ls = 'obj:'+str(objId)+', From '+str(start)+' To '+str(stop)
        plotSub(x, y, ax, label=ls)
            
    #plt.axis('square')
    plt.xlabel('Frames')
    plt.ylabel('Distance')
    #plt.xlim((0,1))
    #plt.ylim((0,1))
    plt.grid(linestyle='-.') #'-', '--', '-.', ':', '',
    plt.legend(loc='upper right') #
    #plt.legend()
    #plt.savefig(r'.\res\analyseDis.png', dpi=300)
    plt.show()
    
def analyseObjTracking():
    base = r'.\data\MOT\MOT17_GT_Jason\MOT17-04-FRCNN'
    name = 'IoU_siamfc_alexnet_e754'
    iousPath = os.path.join(base, name, 'resIoU')
    disPath = os.path.join(base, name, 'resDis')

    bestIoUObjs = [48,7,55]#None # 116,34,6,16
    bestDisObjs = [9,7,48,25,29,55,116,16,136,34]
    worstIoUObjs = [38,90,53,15,95]
    worstDisObjs = [89,66,2,99,71,80,84,85,111,119,120,121,122,141]
    worstDisObjs = [89,66,2,80,84,120,121,122,141]
    
    #observeObjIds = [6,7,9,12,16,100,101,106,116,123,137]#[1,3,4,5,6,7,9,12,16,100,101,106,116,123,137]
    #observeObjIds = [100,101,106,116,123]
    #observeObjIds = [6,7,9,16,137,139,140,81]
    #observeObjIds = [48,7,9,29,55]
    #observeObjIds = [38,90,53,15,95]
    #observeObjIds = [7,9,48,25,29]
    #observeObjIds = [89,66,2,99,71]
    #analyseIoU(iousPath,objFilters=bestIoUObjs,fromZero=False)
    #analyseIoU(iousPath,objFilters=observeObjIds,fromZero=False)
    
    #analyseDis(disPath,objFilters=bestDisObjs,fromZero=False)
    analyseDis(disPath,objFilters=worstDisObjs,fromZero=False)
    
def analyseObjTracking2():
    base = r'.\data\MOT\MOT17_GT_Jason\MOT17-13-FRCNN'
    name = 'IoU_siamfc_Con2Net__BalancedLoss_e'
    iousPath = os.path.join(base, name, 'resIoU')
    disPath = os.path.join(base, name, 'resDis')

    bestIoUObjs = [175,140,185,108,99,116,44,107,101,109]#None # 116,34,6,16
    bestDisObjs = [107,140,44,175,116,108,101,185,118,99]
    worstIoUObjs = [82,29,130,39,28,41,7,86,85,36]
    worstDisObjs = [125,81,18,139,78,83,84,131,79,20]

    #observeObjIds = [38,90,53,15,95]
    #observeObjIds = [7,9,48,25,29]
    #observeObjIds = [89,66,2,99,71]
    #analyseIoU(iousPath,objFilters=worstIoUObjs,fromZero=False)
    #analyseIoU(iousPath,objFilters=observeObjIds,fromZero=False)
    
    analyseDis(disPath,objFilters=worstDisObjs,fromZero=False)
    #analyseDis(disPath,objFilters=worstDisObjs,fromZero=False)
    
def main():
    #testIOU()
    #Tracking_IOU_OTB()
    #Tracking_Precision_OTB()
    
    #Tracking_Precision_MOT17()
    Tracking_All_MOT7()  #estimate prediction
    
    #analyseObjTracking()
    #analyseObjTracking2()
    #testPlot()
    
if __name__=="__main__":
    main()


"""
SiamFC_Alexnet  AUC= 0.22647158218125962  sr= 0.24445907395216152
SiamFC_Con2Net  AUC= 0.4699662052682746  sr= 0.5392157770473589
SiamFC_Vgg19  AUC= 0.03386087338161071  sr= 0.013605442176870748
SiamFC_MobileNetV2  AUC= 0.040632556736136555  sr= 0.026476863725511324

SiamFC_Alexnet final pre= 0.2869870528856704
SiamFC_Con2Net final pre= 0.5615110411555021
SiamFC_Vgg19 final pre= 0.039850779021285934
SiamFC_MobileNetV2 final pre= 0.03467747149723054

#MOT17-13-FRCNN
best iou=
      objId   meanIoU    meanDis
44     140  0.744940   3.333242
123     44  0.670073   3.573572
1      101  0.612425   7.748514
64     159  0.519107  10.891508
147     66  0.478441  24.586643
156     74  0.421749  13.709427
37     134  0.351134  38.254512
109     31  0.333333  47.203192
22     120  0.329446  27.743734
63     158  0.311724  16.069593
best distance=
      objId   meanIoU    meanDis
44     140  0.744940   3.333242
123     44  0.670073   3.573572
1      101  0.612425   7.748514
64     159  0.519107  10.891508
156     74  0.421749  13.709427
63     158  0.311724  16.069593
147     66  0.478441  24.586643
2      102  0.209580  26.030343
22     120  0.329446  27.743734
148     67  0.075157  29.521161
worst iou=
      objId   meanIoU     meanDis
117     39  0.002525  119.888764
105     28  0.004893  326.000158
162      7  0.005563  158.768482
114     36  0.006944  163.908735
133     53  0.007194  502.518385
65      15  0.007353  182.560378
76      16  0.007634  192.181803
112     34  0.007856  338.214887
54      14  0.008065  274.837277
143     62  0.008323  128.246074
worst distance=
      objId   meanIoU     meanDis
94      18  0.031843  681.014532
42     139  0.011026  642.637283
97      20  0.021491  513.078423
133     53  0.007194  502.518385
100     23  0.032247  483.226457
21      11  0.034483  426.437242
10      10  0.050000  386.852936
51     147  0.049168  377.183754
62     157  0.016165  369.557535
36     133  0.143637  354.565719


#MOT17-02-FRCNN
best iou=
     objId   meanIoU     meanDis
64     69  0.523023   10.284808
63     68  0.420504   10.739003
77     80  0.250000   64.696268
58     63  0.228288   13.149068
76      7  0.141833   93.798218
70     74  0.132100   37.905375
6      16  0.085401  680.402234
73     77  0.083333  105.672327
5      15  0.078630   80.205857
69     73  0.061554   70.271700
best distance=
     objId   meanIoU    meanDis
64     69  0.523023  10.284808
63     68  0.420504  10.739003
58     63  0.228288  13.149068
59     64  0.027940  24.319440
70     74  0.132100  37.905375
49     55  0.023414  63.730792
77     80  0.250000  64.696268
69     73  0.061554  70.271700
67     71  0.017988  74.715707
5      15  0.078630  80.205857
worst iou=
     objId   meanIoU     meanDis
0      10  0.001667  149.903178
65      6  0.001667  262.743045
79      9  0.001667  240.124555
7      17  0.001667  127.141091
23     31  0.001667  215.524640
22     30  0.001667  234.558863
14     23  0.001809  690.585629
32      3  0.001845  756.147893
61     66  0.001880   97.942085
9      19  0.001992  736.207826
worst distance=
     objId   meanIoU      meanDis
21      2  0.017857  1171.805309
29     37  0.003584   763.914744
32      3  0.001845   756.147893
9      19  0.001992   736.207826
14     23  0.001809   690.585629
11     20  0.003106   689.214030
6      16  0.085401   680.402234
33     40  0.002322   608.223009
26     34  0.004032   572.425666
25     33  0.003759   567.760812

#MOT17-04-FRCNN
best iou=
      objId   meanIoU    meanDis
40     139  0.956434   1.611293
31     130  0.874320   4.243345
42     140  0.860584   4.142638
77      45  0.857875   5.294055
16     116  0.780172   7.360580
78      46  0.765151   7.803975
104      6  0.731514   8.564759
111     76  0.696620  22.035732
82       4  0.677235  13.831328
29     129  0.661909  15.556965
best distance=
      objId   meanIoU    meanDis
40     139  0.956434   1.611293
42     140  0.860584   4.142638
31     130  0.874320   4.243345
77      45  0.857875   5.294055
16     116  0.780172   7.360580
78      46  0.765151   7.803975
104      6  0.731514   8.564759
82       4  0.677235  13.831328
117     81  0.613992  15.399903
114     79  0.548472  15.453990
worst iou=
      objId   meanIoU     meanDis
100     66  0.007195  727.575577
12     111  0.007195  331.754863
34     133  0.008089  313.329831
7      107  0.010172  470.197604
110     75  0.010475  575.124912
109     74  0.012657  605.279244
61       2  0.014697  428.098206
135     98  0.016238  436.192008
101     67  0.017782  377.082967
25     124  0.022999  332.224132
worst distance=
      objId   meanIoU     meanDis
100     66  0.007195  727.575577
125     89  0.051858  688.266325
109     74  0.012657  605.279244
110     75  0.010475  575.124912
6      106  0.040529  559.555941
99      65  0.025292  499.430517
136     99  0.115143  496.416125
7      107  0.010172  470.197604
97      63  0.056926  461.098358
24     123  0.040404  456.121002
"""