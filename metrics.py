#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#Description: Tracking metrics, IOU
#Date: 2021/04/11
#Author: Steven Huang, Auckland, NZ
import torch
import torchvision.ops.boxes as bops
import numpy as np
import matplotlib.pyplot as plt

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

def getOTB_GTAndPred():
    trueFile = r'.\data\OTB\Crossing\groundtruth_rect.txt' #
    predictFile = r'.\res\data\OTB\Crossing\CrossingPredict.txt'
    #trueFile = r'.\res\imgRt.txt' #
    #predictFile = r'.\res\imgRt.txt' #
    
    deli = None #',' 
    gt = np.loadtxt(trueFile, delimiter=deli, dtype=float, skiprows=0, unpack=False)
    pred = np.loadtxt(predictFile)
    
    #print('before gt=', gt)
    gt = changeRect(gt)
    #print('after gt=', gt)
    pred = changeRect(pred)
    
    print('gt=', gt.shape)
    print('pred=', pred.shape)
    
    gt = torch.from_numpy(gt)
    pred = torch.from_numpy(pred)
    return gt,pred

def Tracking_IOU(): #sucess rate
    gt,pred = getOTB_GTAndPred()
    
    iou = bops.box_iou(gt, pred)
    iou = torch.diagonal(iou, 0)
    print('iou=', iou)
    print('iou=', iou.shape)
    
    plotIOU(iou.numpy())
    #return iou.numpy()

def centerRect(rt): #from x1,y1 w,h --> xc,yc
    xc = (rt[:,0] + rt[:,2])/2
    yc = (rt[:,1] + rt[:,3])/2
    
    xc = xc.reshape([-1,1])
    yc = yc.reshape([-1,1])
    #print(xc)
    #print(yc)
    return torch.cat((xc, yc), 1)

def Tracking_Precision(): #sucess rate
    gt,pred = getOTB_GTAndPred()
    #print('before gt=', gt)
    gt = centerRect(gt)
    pred = centerRect(pred)
    print('after gt=', gt)
    print('after pred=', pred)

    dis = torch.cdist(gt, pred)
    #print('dis=', dis)
    #print('dis=', dis.shape)
    dis = torch.diagonal(dis, 0)
    print('dis=', dis)
    print('dis=', dis.shape)
    plotPrecision(dis)

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
    print('y=', y)
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
    
def main():
    #testIOU()
    #Tracking_IOU()
    Tracking_Precision()
    
if __name__=="__main__":
    main()

