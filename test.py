from __future__ import absolute_import

import os,sys
sys.path.append(r'..\siamfc_model')

from got10k.experiments import *

from siamfc import TrackerSiamFC
from commonPath import pathsFiles
from imageRect import rectImages

def getImageList(path=r'.\res\img'):
    imgs = [i for i in pathsFiles(path,'jpg')]
    return imgs

def writeBoxes(file,boxes):
    with open(file,'w') as f:
        for i in boxes:
            line = "{} {} {} {}\n".format(i[0],i[1],i[2],i[3])
            #print(line)
            f.write(line)

def genPredict():
    if 1:
        path=r'.\res\img_camera'

        # startRect = [370,409,91,247]
        # dstPath = path + '\\' + 'dstImg1'
        # boxFile = path + '\\' + 'dstImg1.txt'

        # startRect = [1104,211,60,179]
        # dstPath = path + '\\' + 'dstImg2'
        # boxFile = path + '\\' + 'dstImg2.txt'
        
        # startRect = [96,544,87,249]
        # dstPath = path + '\\' + 'dstImg3'
        # boxFile = path + '\\' + 'dstImg3.txt'
        
        # startRect = [852,117,114,195]
        # dstPath = path + '\\' + 'dstImg4'
        # boxFile = path + '\\' + 'dstImg4.txt'
        
        startRect = [1190,309,275,272]
        dstPath = path + '\\' + 'dstImg5'
        boxFile = path + '\\' + 'dstImg5.txt'
        
    else:
        path=r'.\res\img_camera2'

        # startRect = [1363,392,74,227]
        # dstPath = path + '\\' + 'dstImg1'
        # boxFile = path + '\\' + 'dstImg1.txt'

        startRect = [661,526,50,58]
        dstPath = path + '\\' + 'dstImg2'
        boxFile = path + '\\' + 'dstImg2.txt'
        
    
    imgs = getImageList(path)
    #print(imgs,len(imgs))
    
    boxes, times = tracker.track(imgs, startRect, visualize=False) #x,y w,h
    print('boxes=',boxes,len(boxes))
    #print('times=',times,len(times))
    
    writeBoxes(boxFile , boxes)
    rectImages(imgs,boxes, dstPath)
    
    
if __name__ == '__main__':
    net_path = 'siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    if 0:
        root_dir = os.path.expanduser(r'.\data\OTB')  #'~/data/OTB'
        e = ExperimentOTB(root_dir, version=2015)
        e.run(tracker)
        e.report([tracker.name])

    #path=r'.\res\img'
    #startRect = [281,137,87,275]
    #boxes, times = tracker.track(imgs, startRect, visualize=False) #x,y w,h
    #print('boxes=',boxes,len(boxes))
    #print('times=',times,len(times))
    
    genPredict()