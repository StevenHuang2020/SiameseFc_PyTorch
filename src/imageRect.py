import cv2
import os 

from commonPath import getFileName  

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

def rectImages(imgs, boxes, dst, color, str, xoffset=0, yoffset=-5):
    for f,box in zip(imgs,boxes):
        img = loadImg(f)
        x0 = box[0]
        y0 = box[1]
        x1 = x0 + box[2]
        y1 = y0 + box[3]
        img = rectangleImg(img,(int(x0),int(y0)),(int(x1),int(y1)), color=color)
        
        loc = (int(x0 + xoffset), int(y0 + yoffset))
        img = textImg(img, str, loc=loc, color=color, fontScale=0.8)
        
        fDst = dst + '\\' + getFileName(f)
        #print(dst,getFileName(f),fDst)
        writeImg(img,fDst)
   
def textImg(img,str,loc=None,color=(0,0,0),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1,thickness = 1):
    newImg = img.copy()
    if loc is None:
        textSize = cv2.getTextSize(str, fontFace, fontScale, thickness)
        #print('textSize=',textSize)
        H,W = getImgHW(img)
        loc = ((W - textSize[0][0])//2, (H + textSize[0][1])//2)
        
    locShadow = list(loc) #add font black shadow
    locShadow[0] = locShadow[0]+2
    locShadow[1] = locShadow[1]+2
    locShadow = tuple(locShadow)
    newImg = cv2.putText(newImg, str,locShadow, fontFace, fontScale, (0,0,0), thickness, cv2.LINE_AA)
        
    return cv2.putText(newImg, str,loc, fontFace, fontScale, color, thickness, cv2.LINE_AA)
    #return cv2.putText(img, str,loc, fontFace, fontScale, color, thickness, cv2.LINE_AA) #color=BGR

def showimage(img,str='image',autoSize=False):
    flag = cv2.WINDOW_NORMAL
    if autoSize:
        flag = cv2.WINDOW_AUTOSIZE

    cv2.namedWindow(str, flag)
    cv2.imshow(str,img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return