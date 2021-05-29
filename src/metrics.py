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
from commonPath import pathsFiles,getFileName
from commonPath import getFileNameNo,createPath
from plotCommon import plotSub
import pandas as pd

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
    iouFile1 = base + '\\IoU_alexnet_e754\\IoU.txt'
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
    disFile1 = base + '\\IoU_alexnet_e754\\dis.txt'
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

def getMOT_GTAndPredAll(trueFile, predictPath, IoUPath): #get Iou&Distance to one file from all predciton files
    # base = r'.\data\MOT\MOT17_GT_Jason\MOT17-04-FRCNN'
    # trueFile = base
    # predictPath = base + '\\predBoxFiles'
    # IoUPath = base + '\\IoU'
    createPath(IoUPath)
    ioufilesPath = os.path.join(IoUPath,'res')
    createPath(ioufilesPath)
    
    gtFiles = getFileList(trueFile, 'txt')
    iouAll = None
    disAll = None
    df = pd.DataFrame()
    for i, gtFile in enumerate(gtFiles):
        objId,start,stop = parseFileName(getFileNameNo(gtFile))
        predictFile = os.path.join(predictPath, getFileNameNo(gtFile) + '_pred.txt')
        print(str(i)+'/'+str(len(gtFiles)), getFileName(gtFile), predictFile, objId,start,stop)
        IouFile = os.path.join(ioufilesPath, getFileNameNo(gtFile) + '_IoU.txt')
        
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

def Tracking_All_MOT7():
    #gt,pred = getMOT_GTAndPred() #one gt and pred
    #Tracking_IOU(gt,pred)
    
    #base = os.path.abspath(r'data/MOT/MOT17_GT_Jason/MOT17-04-FRCNN')
    #base = os.path.expanduser(r'data/MOT/MOT17_GT_Jason/MOT17-04-FRCNN')
    base = os.path.abspath(r'data/MOT/MOT17_GT_Jason/MOT17-13-FRCNN')
    #base = os.path.expanduser(r'data/MOT/MOT17_GT_Jason/MOT17-02-FRCNN')
    trueFile = base
    
    name = 'alexnet_e754'
    name = 'siamfc_Con2Net_BalancedLoss_e'
    name = 'siamfc_Sequential_vgg19_BalancedLoss_e'
    name = 'siamfc_Sequential_vgg19_FocalLoss_e'
    name = 'siamfc_Sequential_MobileNet_BalancedLoss_e'
    name = 'siamfc_Sequential_AlexNet_BalancedLoss_e'
    name = 'siamfc_Sequential_MobileNet_FocalLoss_e'
    name = 'siamfc_Con2Net__BalancedLoss_e'
    name = 'siamfc_Sequential_AlexNetOO_FocalLoss_e'
    
    name = 'siamfc_Sequential_AlexNet_FocalLoss_e'
    name = 'siamfc_alexnet_e554'
    #name = 'siamfc_alexnet_e50'
    name = 'siamfc_alexnet_e754'
    
    predictPath = os.path.join(base, 'predBoxFiles_'+name)
    IoUPath = os.path.join(base, 'IoU_' + name)
        
    getMOT_GTAndPredAll(trueFile, predictPath, IoUPath)
    
    #plotCompareIoU(base)
    #plotComparePrecision(base)
    
def Tracking_Precision_MOT17():
    gt,pred = getMOT_GTAndPred()
    return Tracking_Precision(gt,pred)

def analysisTrecking(csvFile):
    def readCsv(file):
        df = pd.read_csv(file)
        #print(df.describe().transpose())
        print(df.head())
        print('df.columns=',df.columns)
        #print('df.dtypes = ',df.dtypes)
        #print('df.dtypes = ',df.dtypes)
        return df
    df = readCsv(csvFile)
    
    num = 5
    #best iou
    df = df.sort_values(by=['meanIoU'], ascending=False)
    print('best iou=\n', df[:num])
    #print(df[:num]['objId'])
    
    #best distance
    df = df.sort_values(by=['meanDis'], ascending=True)
    print('best distance=\n', df[:num])
    
    num = 5
    #worst iou
    df = df.sort_values(by=['meanIoU'], ascending=True)
    print('worst iou=\n', df[:num])
    
    #worst distance
    df = df.sort_values(by=['meanDis'], ascending=False)
    print('worst distance=\n', df[:num])
    
def analyseMetrics():
    base = r'.\data\MOT\MOT17_GT_Jason\MOT17-04-FRCNN'
    #base = r'.\data\MOT\MOT17_GT_Jason\MOT17-13-FRCNN'
    #base = r'.\data\MOT\MOT17_GT_Jason\MOT17-02-FRCNN'

    name = 'IoU_alexnet_e754'
    #name = 'IoU_siamfc_Sequential_vgg19_BalancedLoss_e'
    #name = 'IoU_siamfc_Sequential_MobileNet_FocalLoss_e'
    #name = 'IoU_siamfc_Con2Net_BalancedLoss_e'
    name = 'IoU_siamfc_Sequential_AlexNet_BalancedLoss_e'
    #name = 'IoU_siamfc_Sequential_AlexNetOO_FocalLoss_e'
    #name = 'IoU_siamfc_Con2Net__BalancedLoss_e'
    name = 'IoU_siamfc_Sequential_AlexNet_FocalLoss_e'
    name = 'IoU_siamfc_Sequential_vgg19_FocalLoss_e'
    name = 'IoU_siamfc_alexnet_e554'
    #name = 'IoU_siamfc_alexnet_e50'
    name = 'IoU_siamfc_alexnet_e754'
    
    dstPath = os.path.join(base, name)
    iouFile = os.path.join(dstPath, 'IoU.txt')
    disFile = os.path.join(dstPath, 'dis.txt')
    
    #plotAllIoU_SR(iouFile, os.path.join(dstPath, 'IoU.png'))
    #plotAllPrecision(disFile, os.path.join(dstPath, 'pre.png'))
    
    csvFile = os.path.join(base, name,'statits.csv')
    analysisTrecking(csvFile)
    
def analyseIoU(path):
    objFilters = [100,101,102]
    iouFiles = []
    iouFilesAll=pathsFiles(path,'txt')
    for i in iouFilesAll:
        objId,start,stop = parseFileName(getFileNameNo(i))
        iou = loadLines2Numpy(i)
        #print(i, objId,start,stop, iou.shape)
        if objId in objFilters:
            iouFiles.append(i)
    
    #start draw
    plt.clf()
    ax = plt.subplot(1,1,1)
    
    for i in iouFiles:
        objId,start,stop = parseFileName(getFileNameNo(i))
        iou = loadLines2Numpy(i)
        
        x = np.arange(start,stop+1)
        y = iou
        plotSub(x, y, ax, label='obj:'+str(objId))
            
    #plt.axis('square')
    plt.xlabel('Frames')
    plt.ylabel('IoU')
    #plt.xlim((0,1))
    #plt.ylim((0,1))
    plt.grid(linestyle='-.') #'-', '--', '-.', ':', '',
    plt.legend(loc='upper right')
    #plt.legend()
    #plt.savefig(r'.\res\IoUObjs.png', dpi=300)
    plt.show()
    
def analyseObjTracking():
    base = r'.\data\MOT\MOT17_GT_Jason\MOT17-04-FRCNN'
    name = 'IoU_siamfc_alexnet_e754'
    iousPath = os.path.join(base, name, 'res')
    # iouFile = os.path.join(iousPath, 'IoU.txt')
    # disFile = os.path.join(iousPath, 'dis.txt')
    
    # plotAllIoUs(iouFile, os.path.join(iousPath, 'IoUFrames.png'),show=True)
    # plotAllDis(disFile, os.path.join(iousPath, 'distances.png'),show=True)
    analyseIoU(iousPath)
    
    
def main():
    #testIOU()
    #Tracking_IOU_OTB()
    #Tracking_Precision_OTB()
    
    #Tracking_Precision_MOT17()
    #Tracking_All_MOT7()  #estimate prediction
    analyseMetrics()
    #analyseObjTracking()
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

"""