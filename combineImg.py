
import numpy as np
from imageRect import *
from commonPath import pathsFiles

def jointImage(img1,img2,hori=True):
    H1,W1 = getImgHW(img1)
    H2,W2 = getImgHW(img2)
    if hori:
        H = np.max(H1,H2)
        img = np.zeros((H, W1+W2, 3),dtype=np.uint8)
        img[:H1, 0:W1] = img1
        img[:H2, W1:] = img2
    else:
        W = np.max(W1,W2)
        img = np.zeros((H1+H2, W, 3),dtype=np.uint8)
        img[:H1, 0:W1] = img1
        img[H1:, 0:W2] = img2
    return img

#ffmpeg -framerate 30 -i ./img_camera/dstImg5/%06d.jpg -c:v libx264 -pix_fmt yuv420p img_camera_5out.mp4
def combineFour(dst=r'.\res\img_camera\combine'):
    
    deleteFolder(dst)
    createPath(dst)
    
    path1=r'.\res\img_camera\dstImg1'
    path2=r'.\res\img_camera\dstImg2'
    path3=r'.\res\img_camera\dstImg3'
    path4=r'.\res\img_camera\dstImg5'
    
    imgs1 = [i for i in pathsFiles(path1,'jpg')]
    imgs2 = [i for i in pathsFiles(path2,'jpg')]
    imgs3 = [i for i in pathsFiles(path3,'jpg')]
    imgs4 = [i for i in pathsFiles(path4,'jpg')]
    
    newW,newH = 960,540
    for f1,f2,f3,f4 in zip(imgs1,imgs2,imgs3,imgs4):
        img1 = loadImg(f1)
        img2 = loadImg(f2)
        img3 = loadImg(f3)
        img4 = loadImg(f4)
        assert(img1 is not None)
        assert(img2 is not None)
        assert(img3 is not None)
        assert(img4 is not None)
        
        img1 = resizeImg(img1,newW,newH)
        img2 = resizeImg(img2,newW,newH)
        img3 = resizeImg(img3,newW,newH)
        img4 = resizeImg(img4,newW,newH)
        
        img = np.zeros((newH*2, newW*2, 3),dtype=np.uint8)
        img[:newH, 0:newW] = img1
        img[:newH, newW:] = img2
        img[newH:, 0:newW] = img3
        img[newH:, newW:] = img4
        
        fDst = dst + '\\' + getFileName(f1)
        writeImg(img,fDst)
        
def main():
    combineFour()

if __name__=='__main__':
    main()
