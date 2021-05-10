from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker
import torchvision.models as models

from . import ops
from .backbones import AlexNetV1,Con2Net,Con8Net
from .heads import SiamFC
from .losses import *
from .datasets import Pair
from .transforms import SiamFCTransforms

from .progressBar import SimpleProgressBar

__all__ = ['TrackerSiamFC']


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)

''' #backbones
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()
squeezenet = models.squeezenet1_0()
densenet = models.densenet161()
inception = models.inception_v3(init_weights=True)
googlenet = models.googlenet(init_weights=True)
shufflenet = models.shufflenet_v2_x1_0()
mobilenet = models.mobilenet_v2()
resnext50_32x4d = models.resnext50_32x4d()
wide_resnet50_2 = models.wide_resnet50_2()
mnasnet = models.mnasnet1_0()
'''

class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        seqName = ''
        # bk=AlexNetV1()
        # bk=Con2Net()
        # bk=Con8Net()
        # bk=models.vgg19().features; seqName = 'vgg19'
        bk=models.MobileNetV2().features; seqName = 'MobileNet'

        self.net = Net(
            backbone=bk,
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)        
               
        # load checkpoint if provided
        if net_path is not None:
            #self.net.load_state_dict(torch.load(net_path, map_location=lambda storage, loc: storage))
            self.net.load_state_dict(torch.load(net_path, map_location=lambda storage, loc: storage)['model_state_dict'], strict=False)
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()
        #self.criterion = FocalLoss()
        
        #define save weight file name
        backboneName = type(self.net.backbone).__name__
        lossName = type(self.criterion).__name__
        self.saveWeightFileName = 'siamfc_' + backboneName + '_' + seqName + '_' + lossName + '_e'  #'siamfc_alexnet_e'
        print('fileName=', self.saveWeightFileName)
        
        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

        self.curEpoch = 0 #current epochs, continue training using
        
    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 3, #50,
            'batch_size': 8,
            'num_workers': 32,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}
        
        #print('ultimate_lr=',cfg['ultimate_lr'])
        for key, val in kwargs.items():
            #print('key=',key)
            if key == 'ultimate_lr' or key == 'initial_lr':
                val = float(val)
            elif key == 'epoch_num':
                val = int(val) 
                   
            if key in cfg:
                cfg.update({key: val})
                
        print(cfg)
        return namedtuple('Config', cfg.keys())(**cfg)
    
    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        #print('self.z_sz=', self.z_sz, 'self.x_sz=', self.x_sz)
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)
        
        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone(z)
    
    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
        
        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box
    
    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        bar = SimpleProgressBar(total=frame_num, title='tracking images', width=30)
        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])

            bar.update(f+1)
            
        return boxes, times
    
    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            
            if backward:
                # back propagation
                #self.optimizer.zero_grad()
                loss.backward()
                #self.optimizer.step()
        
        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None, save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)
        
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            t = time.time()
            
            self.optimizer.zero_grad()
            self.optimizer.step()
            # update lr at each epoch
            self.lr_scheduler.step() #epoch=epoch

            epoch = self.curEpoch + epoch
            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                
            #log = 'Epoch({}/{}):total({}) [{}/{}] Loss: {:.5f}'.format(epoch + 1 - self.curEpoch, self.cfg.epoch_num, epoch + 1, it + 1, len(dataloader), loss)
            log = 'Epoch({}/{}):total({})[{}/{}],Loss:{:.5f}, run in {:.2f}s'.format(epoch + 1 - self.curEpoch, self.cfg.epoch_num, epoch + 1, it + 1, len(dataloader), loss, time.time()-t)
            print(log)
            self._writeLog(log + '\n')
            sys.stdout.flush()
            
            # save checkpoint
            if epoch<100 or (epoch<1000 and epoch%100==0) or (epoch>1000 and epoch%300==0):
                self.saveModel(epoch, loss, save_dir)
              
        self.saveModel(epoch, loss, save_dir)
                
    def saveModel(self, epoch, loss, save_dir):
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
        #net_path = os.path.join(save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
        net_path = os.path.join(save_dir, self.saveWeightFileName + '%d.pth' % (epoch + 1))
        
        #print('dict=',self.net.state_dict())
        # for key in self.net.state_dict():
        #     print('key=', key)
        
        torch.save({
        'epoch': epoch+1,
        'model_state_dict': self.net.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'loss': loss,
        }, net_path)
        
        #torch.save(self.net.state_dict(), net_path)
                
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        return self.labels

    def _writeLog(self, log):
        logFile=r'log'
        if not os.path.exists(logFile):
            os.makedirs(logFile)
        
        logFile=logFile + '/' + 'log.txt'
        with open(logFile,'a',newline='\n') as dstF:
            dstF.write(log)
            
    def train_continue(self, modelFile, seqs, save_dir='pretrained'):     
        checkpoint = torch.load(modelFile)
        #for key in checkpoint:
            #print('key=',key)
            
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.curEpoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print('current already epochs:',self.curEpoch,',Loss:', loss)
        self.train_over(seqs, save_dir='pretrained')