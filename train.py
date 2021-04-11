from __future__ import absolute_import

import os
import sys
import argparse
from got10k.datasets import *

from siamfc import TrackerSiamFC
from MOTDatasets import MOTDataset

def getNewestModelCheckpoint(result_dir):
    lists = os.listdir(result_dir)
    lists.sort(key=lambda fn: os.path.getmtime(result_dir + fn))
    file = os.path.join(result_dir, lists[-1])
    return file

def newTrain(tracker,seqs):
    print('New training start...')
    tracker.train_over(seqs) #first training
    
def continueTrain(tracker, seqs):
    file = getNewestModelCheckpoint(r'./pretrained/')
    print('Continue training start, last weights file:', file)
    tracker.train_continue(file, seqs) #continue training
    
def prepareData():
    #root_dir = os.path.expanduser(r'.\data\got10k')
    #seqs = GOT10k(root_dir, subset='train', return_meta=True)
    root_dir = os.path.expanduser(r'./data/MOT/MOT17_dst') #r'./data/MOT/MOT17_dst_test'
    seqs = MOTDataset(root_dir)

    # id = 0
    # print('len=', len(seqs))
    # print('list=',len(seqs[id][0]), seqs[id][0])
    # print('anno=',seqs[id][1], seqs[id][1].shape)
    return seqs

def prepareModel():
    kwargs = dict(arg.split('=') for arg in sys.argv[1:])
    tracker = TrackerSiamFC(**kwargs) #epoch_num=2
    return tracker

def checkData(seqs):
    print('len=', len(seqs))
    for img_files, anno in seqs:
        #print('images=',len(img_files), ',firstImage=',img_files[0],',anno=',anno.shape)
        assert len(img_files) == len(anno)
        assert anno.shape[1] == 4
        
#usage: python train.py ultimate_lr=1e-7 epoch_num=10
def main():
    tracker = prepareModel()
    seqs = prepareData()
    #checkData(seqs)
    newTrain(tracker,seqs)
    
if __name__ == '__main__':
    main()
        