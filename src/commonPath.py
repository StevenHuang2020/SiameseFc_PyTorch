#python3 steven
import cv2
import argparse
import numpy as np
import os

def pathsFiles(dir,filter='',subFolder=False): #"cpp h txt jpg"
    def getExtFile(file):
        return file[file.find('.')+1:]
    
    def getFmtFile(path):
        #/home/User/Desktop/file.txt    /home/User/Desktop/file     .txt
        root_ext = os.path.splitext(path) 
        return root_ext[1]

    fmts = filter.split()    
    if fmts:
        for dirpath, dirnames, filenames in os.walk(dir):
            filenames.sort()
            for filename in filenames:
                if getExtFile(getFmtFile(filename)) in fmts:
                    yield dirpath+'\\'+filename
            if not subFolder:
                break
    else:
        for dirpath, dirnames, filenames in os.walk(dir):
            filenames.sort()
            for filename in filenames:
                yield dirpath+'\\'+filename  
            if not subFolder:
                break    

def deleteFile(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)        

def createPath(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def deleteFolder(file_path):
    if os.path.exists(file_path):
        #shutil.rmtree(file_path)
        for lists in os.listdir(file_path):
            f = os.path.join(file_path, lists)
            if os.path.isfile(f):
                os.remove(f)
                       
def getFileName(path):  
    return os.path.basename(path)
   
def getFileNameNo(path):
    base=os.path.basename(path)
    return os.path.splitext(base)[0]
 
def jointImage(img1,img2,hori=True):
    assert(img1.shape == img2.shape)
    H,W = img1.shape[0],img1.shape[1]
    if hori:
        img = np.zeros((H, W*2, 3),dtype=np.uint8)
        img[:, 0:W] = img1
        img[:, W:] = img2
    else:
        img = np.zeros((H*2, W, 3),dtype=np.uint8)
        img[0:H, :] = img1
        img[H:, :] = img2
    return img

def JointImagePath():
    src1=r'E:\python\AI\yolo\darknet-master\video\png'
    src2=r'E:\python\AI\yolo\darknet-master\video\png\segdst'
    dst=r'E:\python\AI\yolo\darknet-master\video\png\segDstVideo'
    
    createPath(dst)
    print(src1,src2)
    for i in pathsFiles(src1,'png'): #png
        fileName = getFileName(i)
        img1 = loadImg(i)
        j = src2 + '\\' + fileName
        img2 = loadImg(j)
        assert(img1 is not None and img2 is not None)
        
        img1 = resizeImg(img1,img1.shape[1]//2,img1.shape[0]//2)
        img2 = resizeImg(img2,img2.shape[1]//2,img2.shape[0]//2)
        dstImg = jointImage(img1,img2)
        
        dstFile = dst + '\\' + fileName
        print('start to joint:',i,j) #dstFile
        writeImg(dstFile,dstImg)
        #break
    
def main():
    JointImagePath()

if __name__ == '__main__':
    main()
