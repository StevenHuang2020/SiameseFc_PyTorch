import cv2
import os 

def createPath(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        
def deleteFile(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def deleteFolder(file_path):
    if os.path.exists(file_path):
        #shutil.rmtree(file_path)
        for lists in os.listdir(file_path):
            f = os.path.join(file_path, lists)
            if os.path.isfile(f):
                os.remove(f)
                
def getFileName(path):  
    return os.path.basename(path)
   
def getImgHW(img):
    return img.shape[0],img.shape[1]

def loadImg(file,mode=cv2.IMREAD_COLOR):
    return cv2.imread(file,mode)
 
def writeImg(img,filePath):
    cv2.imwrite(filePath,img)

def resizeImg(img,NewW,NewH):
    try:
        #INTER_CUBIC INTER_NEAREST INTER_LINEAR INTER_AREA
        return cv2.resize(img, (NewW,NewH), interpolation=cv2.INTER_CUBIC) 
    except:
        print('img.shape,newW,newH',img.shape,NewW,NewH)

def rectangleImg(img,startPt,stopPt,color=(0,0,255),thickness=2):
    return cv2.rectangle(img, startPt, stopPt, color=color, thickness=thickness) 

def rectImages(imgs,boxes,dst):
    deleteFolder(dst)
    createPath(dst)
    for f,box in zip(imgs,boxes):
        img = loadImg(f)
        x0 = box[0]
        y0 = box[1]
        x1 = x0 + box[2]
        y1 = y0 + box[3]
        img = rectangleImg(img,(int(x0),int(y0)),(int(x1),int(y1)))
        
        fDst = dst + '\\' + getFileName(f)
        print(dst,getFileName(f),fDst)
        writeImg(img,fDst)