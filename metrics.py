#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#Description: Tracking metrics, IOU and precision
#Date: 2021/04/11
#Author: Steven Huang, Auckland, NZ
import torch
import torchvision.ops.boxes as bops
import numpy as np
import matplotlib.pyplot as plt
from predict import getFileList
from commonPath import pathsFiles,getFileName
from commonPath import getFileNameNo,createPath

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
        plotIOU(iou.numpy())
        
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
    print('dis=', dis)
    print('dis=', dis.shape)
    
    if plot:
        plotPrecision(dis)
    return dis.numpy()

def plotIOU(iou):
    def plotXY(x,y,name): #Success rate
        plt.title(name)
        plt.plot(x,y,label='SiamFC')
        plt.xlabel('Overlap threshold')
        plt.ylabel('Success rate')
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.grid(linestyle='-.') #'-', '--', '-.', ':', '',
        plt.legend(loc='lower left')
        plt.show()
    
    #print(iou)
    N=200
    x = np.linspace(0,1,num=N)
    
    def SuccessRate(k):
        res = np.where(iou > k, 1, 0) 
        return np.sum(res)/len(iou)
    
    y = list(map(SuccessRate, [i for i in x]))
    #print('y=', y)
    plotXY(x, y, 'Success plots of OPE')
    
def plotPrecision(dis):
    def plotXY(x,y,name): #Success rate
        plt.title(name)
        plt.plot(x,y,label='SiamFC')
        plt.xlabel('Location error threshold')
        plt.ylabel('Precision')
        plt.xlim((0,50))
        plt.ylim((0,1))
        plt.grid(linestyle='-.') #'-', '--', '-.', ':', '',
        plt.legend(loc='lower right')
        plt.show()
        
    #print(dis)
    N=200
    x = np.linspace(0,50,num=N)
    
    def Precision(k):
        res = np.where(dis < k, 1, 0) 
        return np.sum(res)/len(dis)
    
    y = list(map(Precision, [i for i in x]))
    print('y=', y)
    plotXY(x, y, 'Precision plots of OPE')
  
##################### MOT7 ###################
def plotAllIoU():
    base = r'.\data\MOT\MOT17_GT_Jason\MOT17-04-FRCNN'
    iouFile = base + '\\IoU\\IoU.txt'
    
    iouAll = np.loadtxt(iouFile)
    print('iouAll=', type(iouAll), iouAll.shape)
    plotIOU(iouAll)
 
def plotAllPrecision():
    base = r'.\data\MOT\MOT17_GT_Jason\MOT17-04-FRCNN'
    disFile = base + '\\IoU\\dis.txt'
    
    disAll = np.loadtxt(disFile)
    print('disAll=', type(disAll), disAll.shape)
    plotPrecision(disAll)
       
def getMOT_GTAndPredAll():
    base = r'.\data\MOT\MOT17_GT_Jason\MOT17-04-FRCNN'
    trueFile = base
    predictPath = base + '\\predBoxFiles'
    IoUPath = base + '\\IoU'
    createPath(IoUPath)
    
    gtFiles = getFileList(trueFile, 'txt')
    iouAll = None
    disAll = None
    for i, gtFile in enumerate(gtFiles):
        predictFile = predictPath + '\\' +  getFileNameNo(gtFile) + '_pred.txt'
        print(str(i)+'/'+str(len(gtFiles)), gtFile, predictFile)
        #IouFile = IoUPath + '\\' +  getFileNameNo(gtFile) + '_IoU.txt'
        
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
        #np.savetxt(IouFile, iou, fmt='%f')
        #break
        
    IouFile = IoUPath + '\\' +  'IoU.txt'    
    print('iouAll=', type(iouAll), iouAll.shape)
    np.savetxt(IouFile, iouAll, fmt='%f')
    plotIOU(iouAll)
    
    disFile = IoUPath + '\\' +  'dis.txt'    
    print('disAll=', type(disAll), disAll.shape)
    np.savetxt(disFile, disAll, fmt='%f')
    plotPrecision(disAll)    
        
def getMOT_GTAndPred():
    base = r'.\res\data\MOT17\MOT17-04-FRCNN_ID_1\\'
    trueFile = base + 'groundtruth_rect.txt' #
    predictFile = base + 'pred.txt'
    
    return getGroundTrueAndPred(trueFile, predictFile, delimiter=',') 

def Tracking_IOU_MOT7():
    #gt,pred = getMOT_GTAndPred() #one gt and pred
    #Tracking_IOU(gt,pred)
    
    #getMOT_GTAndPredAll()
    #plotAllIoU()
    plotAllPrecision()
    
def Tracking_Precision_MOT17():
    gt,pred = getMOT_GTAndPred()
    return Tracking_Precision(gt,pred)

def main():
    #testIOU()
    #Tracking_IOU_OTB()
    #Tracking_Precision_OTB()
    
    #Tracking_Precision_MOT17()
    Tracking_IOU_MOT7()
    
if __name__=="__main__":
    main()

