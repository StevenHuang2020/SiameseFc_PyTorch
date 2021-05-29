from __future__ import absolute_import

import os,sys
sys.path.append(r'..\siamfc_model')

#import argparse 
import torch
import numpy as np
from got10k.experiments import *

from siamfc import TrackerSiamFC
from commonPath import pathsFiles,getFileName,getFileNameNo,createPath
from imageRect import *
from visualModel import visualModel
from siamfc import ops

def getFileList(path=r'.\res\img', fmt='jpg'):
    return [i for i in pathsFiles(path, fmt)]

def writeBoxes(file,boxes):
    with open(file,'w') as f:
        for i in boxes:
            line = "{} {} {} {}\n".format(i[0],i[1],i[2],i[3])
            #print(line)
            f.write(line)

def genPredict(): #predict one obj
    if 1:
        base = r'.\res\data\MOT17\MOT17-04-FRCNN_ID_1'
        path = base + '\\img'   
        gtFile = base + '\\groundtruth_rect.txt'
        
        anno = np.loadtxt(gtFile, delimiter=',')
        startRect = anno[0] #[409,171,31,72]
        
        saveBase = base + '\\pred'
        createPath(saveBase)
        
        dstPath = saveBase
        boxFile = base + '\\pred1.txt'
        
        imgs = getFileList(path)
        imgs=imgs[:2]
    elif 0:
        path=r'.\data\OTB\Crossing\img'       
        startRect = [205,151,17,50]
        
        saveBase = r'.\res\data\OTB\Crossing'
        createPath(saveBase)
        
        dstPath = saveBase
        boxFile = saveBase + '\\' + 'CrossingPredict.txt'
        
    elif 0:
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
        path=r'.\res\pedetrain\drive-download-20210422T012104Z-001\1Copy'
        dstPath = None
        boxFile = path + '\\pred'
        createPath(boxFile)
        
        # startRect = [323,459,28,64] #id1
        # boxFile = boxFile + '\\' + 'pred_1_1_141.txt'
        
        # startRect = [479,460,37,71] #id2
        # boxFile = boxFile + '\\' + 'pred_2_1_115.txt'
        
        startRect = [1349,461,2,2]#[1349,461,61,117] #id3
        boxFile = boxFile + '\\' + 'pred_3_1_85.txt'
        
    #imgs = getFileList(path)
    predictAll(imgs, startRect, boxFile, dstPath)
    
def createOBjDict():
    dictAll={}
    
    id = 1
    dictItem={'startFrame':1, 'stopFrame':34, 'startRect':[146,437,83,227]}
    dictAll.update({id : dictItem});  id+=1  
    dictItem={'startFrame':1, 'stopFrame':30, 'startRect':[225,592,202,144]}
    dictAll.update({id : dictItem});  id+=1  
    dictItem={'startFrame':1, 'stopFrame':66, 'startRect':[777,426,44,116]}
    dictAll.update({id : dictItem});  id+=1  
    dictItem={'startFrame':1, 'stopFrame':60, 'startRect':[850,420,42,113]}
    dictAll.update({id : dictItem});  id+=1  
    dictItem={'startFrame':1, 'stopFrame':43, 'startRect':[970,414,47,136]}
    dictAll.update({id : dictItem});  id+=1  
    dictItem={'startFrame':1, 'stopFrame':62, 'startRect':[917,415,22,66]}
    dictAll.update({id : dictItem});  id+=1  
    dictItem={'startFrame':1, 'stopFrame':40, 'startRect':[956,411,16,58]}
    dictAll.update({id : dictItem});  id+=1  
    dictItem={'startFrame':35, 'stopFrame':60, 'startRect':[1123,365,33,105]}
    dictAll.update({id : dictItem});  id+=1 
    
    dictItem={'startFrame':80, 'stopFrame':140, 'startRect':[588,498,64,168]}
    dictAll.update({id : dictItem});  id+=1 
    dictItem={'startFrame':80, 'stopFrame':156, 'startRect':[847,496,72,130]}
    dictAll.update({id : dictItem});  id+=1
    dictItem={'startFrame':80, 'stopFrame':136, 'startRect':[51,482,39,103]}
    dictAll.update({id : dictItem});  id+=1
    dictItem={'startFrame':80, 'stopFrame':108, 'startRect':[123,481,34,107]}
    dictAll.update({id : dictItem});  id+=1
    dictItem={'startFrame':80, 'stopFrame':166, 'startRect':[219,488,28,77]}
    dictAll.update({id : dictItem});  id+=1
    dictItem={'startFrame':80, 'stopFrame':132, 'startRect':[273,493,35,75]}
    dictAll.update({id : dictItem});  id+=1
    dictItem={'startFrame':80, 'stopFrame':332, 'startRect':[664,493,41,97]}
    dictAll.update({id : dictItem});  id+=1
    dictItem={'startFrame':80, 'stopFrame':203, 'startRect':[413,491,48,128]}
    dictAll.update({id : dictItem});  id+=1
    dictItem={'startFrame':80, 'stopFrame':226, 'startRect':[559,507,38,118]}
    dictAll.update({id : dictItem});  id+=1
    dictItem={'startFrame':80, 'stopFrame':450, 'startRect':[460,502,42,91]}
    dictAll.update({id : dictItem});  id+=1
    dictItem={'startFrame':80, 'stopFrame':450, 'startRect':[507,495,33,88]}
    dictAll.update({id : dictItem});  id+=1
    dictItem={'startFrame':80, 'stopFrame':115, 'startRect':[324,495,21,68]}
    dictAll.update({id : dictItem});  id+=1
    dictItem={'startFrame':80, 'stopFrame':123, 'startRect':[355,492,20,70]}
    dictAll.update({id : dictItem});  id+=1

    return dictAll
       
def genPredictAllObjs():
    imgesPath=r'.\res\pedetrain\outmp4'
    #dstPath = None
    boxFileBase = imgesPath + '\\pred'
    createPath(boxFileBase)

    #textFrameNo2Images(imgesPath) 
        
    #startRect = [323,459,28,64] #id1
    #boxFile = boxFileBase + '\\' + 'pred_1_1_141.txt'
    dictAll = createOBjDict()
    #print('dictAll=\n', dictAll)
    imgs = getFileList(imgesPath)
    for id, dictObj in dictAll.items():
        startFrame,stopFrame,startRect = dictObj['startFrame'], dictObj['stopFrame'], dictObj['startRect']
        
        predFile = 'pred_{}_{}_{}.txt'.format(id,startFrame,stopFrame)
        boxFile = boxFileBase + '\\' + predFile
        print(id, startFrame, stopFrame, startRect, boxFile)
        predictAll(imgs, startRect, boxFile)
    
    ##start gen rect to images
    gtFilesPath = boxFileBase
    delimiter = ' '
    objIdFilters = []
    putRectToImgs(imgesPath, imgesPath, gtFilesPath, objIdFilters, delimiter=delimiter)   
     
    
def predictAll(imgs, startRect, boxFile, dstPath=None):
    #print(imgs,len(imgs))
    boxes, hotMaps, times = tracker.track(imgs, startRect, visualize=False, retHeatmap=False) #x,y w,h
    #print('boxes=',boxes,len(boxes))
    #print('times=',times,len(times))
    writeBoxes(boxFile , boxes)
    #showImgHotmap(imgs, hotMaps, startRect)
    
def showImgHotmap(imgs, hotMaps, boxes):
     for imgFile,hotMap in zip(imgs, hotMaps):
        hotMap = hotMap*10e6
        loc = np.unravel_index(hotMap.argmax(), hotMap.shape)
        print('hotMap.shape=', hotMap.shape, type(hotMap), hotMap, 'loc=', loc)
        
        img = ops.read_image(imgFile)
        print('img.shape=', img.shape, type(img))
        ops.show_image(img, boxes=boxes, delay=0)
        ops.show_image(hotMap, cvt_code=None, delay=0)
        
def parseFileName(fileName):
    str = fileName[fileName.find('_') + 1: ]
    #str = str[: str.rfind('.')]
    str = str.split('_')
    return int(str[0]), int(str[1]), int(str[2])
    
def gen_predictAll(imgesPath,gtFilesPath,predPath):        
    imgsAll = getFileList(imgesPath)
    
    if len(imgsAll)==0:
        print('warning, cant find images,path=', imgesPath)
    #print('len imgsAll=\n', len(imgsAll))

    gtFlies = getFileList(gtFilesPath, 'txt')
    #print('len gtFlies=\n', len(gtFlies))
    if len(imgsAll)==0:
        print('warning, cant find gtfiles,path=', gtFilesPath)
        
    for i, gt in enumerate(gtFlies):
        fileName = getFileNameNo(gt)
        objId,start,stop = parseFileName(fileName)
        print(i,'/',len(gtFlies), ' ', gt, fileName, objId,start,stop )
        startRect = np.loadtxt(gt, delimiter=',')[0]
        
        # if start>=len(imgsAll):
        #     continue 
        
        imgs = imgsAll[start-1:stop]
        
        boxFile = os.path.join(predPath, fileName + '_pred.txt')
        #print(imgs, 'boxFile=', boxFile, 'startRect=', startRect)
        if not os.path.exists(boxFile):
            predictAll(imgs, startRect, boxFile)
        #break
        
def genPredict_MOT17(name):
    #imgesPath = os.path.abspath(r'.\data\MOT\MOT17\train\MOT17-02-FRCNN\img1')
    #base = os.path.abspath(r'.\data\MOT\MOT17_GT_Jason\MOT17-02-FRCNN')
    
    #windows
    imgesPath = os.path.abspath(r'.\data\MOT\MOT17\train\MOT17-04-FRCNN\img1')
    base = os.path.abspath(r'.\data\MOT\MOT17_GT_Jason\MOT17-04-FRCNN')
    # imgesPath = os.path.abspath(r'data/MOT/MOT17/train/MOT17-04-FRCNN/img1')
    # base = os.path.abspath(r'data/MOT/MOT17_GT_Jason/MOT17-04-FRCNN')
    
    #imgesPath = os.path.abspath(r'data/MOT/MOT17/train/MOT17-04-FRCNN/img1')
    #base = os.path.abspath(r'data/MOT/MOT17_GT_Jason/MOT17-04-FRCNN')
    
    #imgesPath = os.path.abspath(r'data/MOT/MOT17/train/MOT17-13-FRCNN/img1')
    #base = os.path.abspath(r'data/MOT/MOT17_GT_Jason/MOT17-13-FRCNN')
    
    gtFilesPath = base
    predPath = os.path.join(base, 'predBoxFiles_' + name) #tracker.saveWeightFileName

    print('imgesPath=', imgesPath)
    print('gtFilesPath=', gtFilesPath)
    print('predPath=', predPath)
    createPath(predPath)
    
    #print('tarckerName=', tracker.saveWeightFileName)
    gen_predictAll(imgesPath,gtFilesPath,predPath)
    
     
def textFrameNo2Images(imgesPath, dstImgPath):
    imgsAll = getFileList(imgesPath)
    #print('imgsAll=\n', imgsAll)
    for i, imgF in enumerate(imgsAll):
        print(i,'/', len(imgsAll), imgF)
        img = loadImg(imgF)
        txt = '#frame{}'.format(i+1)
        
        img = textImg(img, txt, (27,48), color=(0,255,255), fontScale=2, thickness=3, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        #showimage(img)
        
        dst = os.path.join(dstImgPath, getFileName(imgF))
        writeImg(img, dst)
        #break
    
def putRectToImgs(imgesPath, imgesDstPath, rtFilesPath, objIdFilters=[], delimiter=','):
    imgsAll = getFileList(imgesPath)
    
    rtFiles = getFileList(rtFilesPath, 'txt')
    #print('imgsAll=\n', imgsAll)
    for i, rtFile in enumerate(rtFiles):
        objId,start,stop = parseFileName(getFileNameNo(rtFile))
        if objId in objIdFilters:
            continue
        print(str(i) + '/' + str(len(rtFiles)), rtFile, objId, start, stop)
        #predFile = predFilesPath + '\\' + getFileNameNo(rtFile) + '_pred.txt'
        imgs = imgsAll[start-1:stop]
        
        boxes = np.loadtxt(rtFile, delimiter=delimiter)
        
        colr = (np.random.randint(256), np.random.randint(256), np.random.randint(256)) #(100,0,0)#(colr[0],colr[1],colr[2]) #
        #print(colr,type(colr))
        rectImages(imgs,boxes,dst=imgesDstPath, color=colr, str='id:'+str(objId))
        #break

def checkGtObject():
    base = r'.\data\MOT\MOT17_GT_Jason\MOT17-02-FRCNN'
    imgesPath = r'.\data\MOT\MOT17\train\MOT17-02-FRCNN\img1'
    
    #base = r'.\data\MOT\MOT17_GT_Jason\MOT17-04-FRCNN'
    #imgesPath = r'.\data\MOT\MOT17\train\MOT17-04-FRCNN\img1'
    
    # base = r'.\data\MOT\MOT17_GT_Jason\MOT17-05-FRCNN'
    # imgesPath = r'.\data\MOT\MOT17\train\MOT17-05-FRCNN\img1'
    
    # base = r'.\data\MOT\MOT17_GT_Jason\MOT17-09-FRCNN'
    # imgesPath = r'.\data\MOT\MOT17\train\MOT17-09-FRCNN\img1'
    
    base = r'.\data\MOT\MOT17_GT_Jason\MOT17-10-FRCNN'
    imgesPath = r'.\data\MOT\MOT17\train\MOT17-10-FRCNN\img1'
    
    # base = r'.\data\MOT\MOT17_GT_Jason\MOT17-11-FRCNN'
    # imgesPath = r'.\data\MOT\MOT17\train\MOT17-11-FRCNN\img1'
    
    # base = r'.\data\MOT\MOT17_GT_Jason\MOT17-13-FRCNN'
    # imgesPath = os.path.join(base,'gtIndexImages') 
    
    dst = os.path.join(base,'checkDst') 
    createPath(dst)

    rtFilesPath = base
    rtFiles = getFileList(rtFilesPath, 'txt')
    #print('imgsAll=\n', imgsAll)
    color = (0,0,255)
    for i, rtFile in enumerate(rtFiles):
        objId,start,stop = parseFileName(getFileNameNo(rtFile))
        print(str(i) + '/' + str(len(rtFiles)), rtFile, objId, start, stop)
        
        boxes = np.loadtxt(rtFile, delimiter=',')
        
        imgFile = imgesPath + '\\' + '{:06}.jpg'.format(start)
        print(imgFile, boxes[0])
        box = boxes[0]
        x0 = box[0]
        y0 = box[1]
        x1 = x0 + box[2]
        y1 = y0 + box[3]
        img = rectangleImg(loadImg(imgFile),(int(x0),int(y0)),(int(x1),int(y1)),color=color)
        img = textImg(img, str(objId), loc=(int(x0),int(y0)), color=color)
        
        #showimage(img)
        fDst = dst + '\\' + '{:06}_obj_{}.jpg'.format(start,objId)
        #print('fDst=', fDst)
        writeImg(img,fDst)
        #break
    
def genBoxImg_MOT17():
    objIdFilters = []
    delimiter = ' ' #predict is NUll, gt is ','
    if 0:
        base = r'.\res\pedetrain\drive-download-20210422T012104Z-001\1'
        # imgesPath = os.path.join(base,'img1')
        # gtFilesPath = base
        # delimiter = ','
        imgesPath = base
        gtFilesPath = os.path.join(base,'pred')
        
        imgesDstPath = imgesPath
    elif 1:
        base = r'.\data\MOT\MOT17_GT_Jason\MOT17-13-FRCNN'
        #imgesPath = os.path.join(base,'gtIndexImages')
        if 0:
            gtFilesPath = base
            delimiter = ','
            imgesDstPath = os.path.join(base,'gtBoxImages')
            createPath(imgesDstPath)
            imgesPath = imgesDstPath
        else:
            imgesPath = os.path.join(base,'img1_Pred')
            gtFilesPath = os.path.join(base,'predBoxFiles_siamfc_Con2Net__BalancedLoss_e') 
            imgesDstPath = imgesPath
        #imgesIndexDstPath = os.path.join(base,'gtIndexImages')
        #createPath(imgesIndexDstPath)
        
        
        objIdFilters = [1,3,12,29,30,41,76,80,82,85,104,105,116,\
            106,81,125,92,130101,184,107,108,89,179,90,91,177,93,\
            162,99,182,150,109,181,175,183,138,176,146,96,141,185,\
            75,65,94,95,110,161,103,121,88,97,131,83,87,79,86,111,\
            172,112,174,113,27,117,114,173,118,98,115,77,130,84,78] #no pedestrains
    elif 0:
        base = r'.\data\MOT\MOT17_GT_Jason\MOT17-04-FRCNN' #imgesPath == imgesDstPath to overlap preds
        imgesPath = os.path.join(base,'img1_Pred')
        gtFilesPath = os.path.join(base,'predBoxFiles_alexnet_e754') 
        imgesDstPath = imgesPath
              
        #imgesIndexDstPath = os.path.join(base,'gtBoxImages')
        #createPath(imgesIndexDstPath)
       
        objIdFilters = [7,8,9,10,11,12,13,14,15,16,17,18,\
            20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,\
            36,37,38,39,40,41,42,43,44,47,48,49,50,51,52,53,54,\
                55,56,59,64,137] #no pedestrains
    else:
        base = r'.\data\MOT\MOT17_GT_Jason\MOT17-02-FRCNN'
        # imgesPath = os.path.join(base,'img1')
        # gtFilesPath = base
        # delimiter = ','
        imgesPath = os.path.join(base,'gtBoxImages')
        gtFilesPath = os.path.join(base,'predBoxFiles') 
        imgesDstPath = os.path.join(base,'img1_Pred') 
        createPath(imgesDstPath)
        imgesIndexDstPath = os.path.join(base,'gtBoxImages')
        createPath(imgesIndexDstPath)
       
        objIdFilters = [4,50,51,52,53,79,75] #no pedestrains
    
    #checkGtObject()
    #textFrameNo2Images(imgesPath,dstImgPath=imgesIndexDstPath)
    putRectToImgs(imgesPath, imgesDstPath, gtFilesPath, objIdFilters, delimiter=delimiter)
    
def genBoxImgCompare_MOT17():
    base = r'.\data\MOT\MOT17_GT_Jason\MOT17-04-FRCNN'
    imgesPath = base + r'\\img1_Copy'
    
    gtRtFile = base + r'\\objID_2_1_680.txt'
    predRtFile = base + r'\\predBoxFiles\\objID_2_1_680_pred.txt'
    imgesDstPath = base + r'\\img1_PredCompare'
    objId = 2
    
    createPath(imgesDstPath)
       
    imgs = getFileList(imgesPath)
    
    gtBoxes = np.loadtxt(gtRtFile, delimiter=',')
    predBoxes = np.loadtxt(predRtFile, delimiter=' ')
    
    colrGt = (0,0,255)  #bgr red
    colrPred = (0,255,0) #green

    print('start to draw groud true rect images...')
    rectImages(imgs, gtBoxes, dst=imgesDstPath, color=colrGt, str='id:'+str(objId))
    
    print('start to draw pred rect images...')
    imgs = getFileList(imgesDstPath)    
    rectImages(imgs, predBoxes, dst=imgesDstPath, color=colrPred, str='id:'+str(objId))
    
def visualTracker(tracker):
    from siamfc.backbones import AlexNetV1,Con2Net,Con8Net
    # print(tracker)
    # print(tracker.net)
    # print('backbone=\n', tracker.net.backbone)
    # print('head=\n', tracker.net.head)
    
    if 0:   #header network
        model = tracker.net.backbone; name = 'backbone.dot'
        x = torch.randn(1,3,128,128)
        params = dict(list(model.named_parameters()))
        visualModel(model(x), params, file=r'.\res\model' + '\\' + name, view=True)
    elif 1:
        import torchvision.models as models
        
        #model = Con2Net(); name = 'backbone_Con2Net.dot'
        #model = AlexNetV1(); name = 'backboneAlex.dot'
        ##model = models.resnet18(); name = 'backbone_resnet18.dot'
        
        
        #model = models.MobileNetV2().features; name = 'backbone_MobileNetV2.dot'
        #model = models.AlexNet().features; name = 'backbone_alexnet_feature.dot'#5 cnn layers
        model = models.vgg19().features; name = 'backbone_vgg19_feature.dot' #15 cnn
        print(name, '\n', model)
        
        x = torch.randn(99,3,128,128)
        params = dict(list(model.named_parameters()))
        #visualModel(model(x), params, file=r'.\res\model' + '\\' + name, view=True)
    else:   #fc network
        model = tracker.net.head; name = 'fcHead.dot'
        x = torch.randn(1,3,128,128)
        z = torch.randn(1,3,32,32)
        params = dict(list(model.named_parameters()))
        visualModel(model(z,x),params,file=r'.\res\model' + '\\' + name,view=True)


if __name__ == '__main__':
    net_path = os.path.expanduser(r'./pretrained/siamfc_alexnet_e754.pth')
    #net_path = os.path.expanduser(r'./pretrained/siamfc_alexnet_e50.pth')
    #net_path = os.path.expanduser(r'./pretrained/siamfc_AlexNetV1_FocalLoss_e713.pth')
    
    #net_path = os.path.expanduser(r'./pretrained/siamfc_Con2Net_BalancedLoss_e860.pth')
    #net_path = os.path.expanduser(r'./pretrained/siamfc_Sequential_vgg19_BalancedLoss_e801.pth')
    #net_path = os.path.expanduser(r'./pretrained/siamfc_Sequential_vgg19_FocalLoss_e211.pth')
    #net_path = os.path.expanduser(r'./pretrained/siamfc_Sequential_MobileNet_BalancedLoss_e601.pth')
    #net_path = os.path.expanduser(r'./pretrained/siamfc_Sequential_AlexNet_BalancedLoss_e510.pth')
    #net_path = os.path.expanduser(r'./pretrained/new/siamfc_Sequential_vgg19_FocalLoss_e576.pth')
    #net_path = os.path.expanduser(r'./pretrained/new/siamfc_Sequential_MobileNet_FocalLoss_e801.pth')
    
    netWeightsName = getFileNameNo(net_path)
    print('net_path=', net_path, 'netWeightsName=', netWeightsName)
    kwargs = dict(arg.split('=') for arg in sys.argv[1:])
    tracker = TrackerSiamFC(net_path=net_path, **kwargs)
    if 0:
        root_dir = os.path.expanduser(r'.\data\OTB')  #'~/data/OTB'
        e = ExperimentOTB(root_dir, version=2015)
        e.run(tracker)
        e.report([tracker.name])
    
    #visualTracker(tracker)
    
    #genBoxImg_MOT17()  #gen rect to images by predicted box files
    #genBoxImgCompare_MOT17()
    
    #genPredictAllObjs()
    #genPredict()        #gen predict one object boxes
    
    genPredict_MOT17(netWeightsName) #gen predict boxes