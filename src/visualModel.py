#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#Description: Visual pytorch model 
#torchviz install
#1. download & install windows binary installation package
#make sure install with adding to system path
#2. pip install torchviz
#usage: dot -Tpng .\res\model.dot -o .\res\model.png
#Date: 2021/05/06
#Author: Steven Huang, Auckland, NZ
import os
import torch #Version: 1.7.0+cpu
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchviz import make_dot
#from commonTorch import RegressionNet,ClassifierCNN_Net3

def createModel():
    if 0:
        x = torch.zeros(1, 3, 224, 224, dtype=torch.float, requires_grad=False)
        #model = models.resnet()
        model = models.resnet50(pretrained=True)
        name='resnet'
    elif 1:
        model = nn.Sequential()
        model.add_module('W0', nn.Linear(8, 16))
        model.add_module('tanh', nn.Tanh())
        model.add_module('W1', nn.Linear(16, 1))
        
        x = torch.randn(1,8)
        name='seq'
    elif 0:
        model = RegressionNet()
        x = torch.randn(1)
        name='RegressionNet.dot'
    elif 0:
        model = ClassifierCNN_Net3(10)
        x = torch.randn(1,1,28,28)
        name='ClassifierCNN_Net3.dot'
        
    print(model)
    return x, model,name

def visualModel(model,params,file,format='pdf',view=False):
    dot = make_dot(model, params=params) #show_attrs=True, show_saved=True
    dot.format = format #'pdf' #'png' #
    print('format=', dot._format)
    dot.render(filename=file, view=view)
    
    head, tail = os.path.split(file)
    out = head + '\\' + tail+'.png'
    cmd = 'dot -Tpng -Gdpi=300 ' + file + ' -o ' + out
    print('file,head,tail=', file, head, tail, 'cmd=',cmd)
    
    #os.system('dot -Tpng random.dot -o random.png')
    os.system(cmd)
    
def testViz():
    x, model,name = createModel()
    print(x.shape)
    params = dict(list(model.named_parameters()))
    visualModel(model(x),params,file=r'.\res'+'\\'+name,view=False)
    
    # #final color descrption in function make_dot
    # dot = make_dot(y, params=dict(list(model.named_parameters())), show_attrs=True, show_saved=True)
    # dot._format = 'pdf' #'png' #
    # print('format=', dot._format)
    # #dot.render(filename=r".\res\1.pdf",view=True)
    # dot.render(filename=r".\res\1.pdf", view=True)
    
    #make_dot(model, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
    # make_dot(y.mean(), params=dict(model.named_parameters()))


def main():
    testViz()

if __name__=="__main__":
    main()
