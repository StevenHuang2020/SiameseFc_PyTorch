from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerSiamFC


if __name__ == '__main__':

    path = r'./data/MOT/MOT17_dst/MOT17-04-FRCNN_ID_2/' #r'./data/OTB/Dancer2/' #r'./data/OTB/Crossing/' #
    seq_dir = os.path.expanduser(path)
    print('seq_dir=',seq_dir)
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt',delimiter=',')    
    
    net_path = 'pretrained/siamfc_alexnet_e554.pth' #'siamfc_alexnet_e50.pth' #
    tracker = TrackerSiamFC(net_path=net_path)
    tracker.track(img_files, anno[0], visualize=True)
