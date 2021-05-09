#python3 steven
import cv2
import argparse
import numpy as np
import os
from commonPath import *

def main():
    path=r'.\res\img'
    for i in pathsFiles(path,'jpg'):
        fileName = getFileName(i)
        print(fileName,int(fileName[:4]))
        if int(fileName[:4]) % 2 == 0:
            deleteFile(i)
            
if __name__ == '__main__':
    main()
