#Steven 08/08/2020 
import argparse 
import sys
import matplotlib.pyplot as plt
import os

#----------------------------------------------
#usgae: python plotloss.py .\weights\trainMain.log
#----------------------------------------------

def getLoss(log_file,startIter=0,stopIter=None):
    numbers = {'1','2','3','4','5','6','7','8','9'}
    with open(log_file, 'r') as f:
        lines  = [line.rstrip("\n") for line in f.readlines()]
        
        iters = []
        loss = []
        epoch = 0
        for line in lines:
            iterLine = line.split(',')
            #print(line, iterLine[1])
            if iterLine[0].startswith('Epoch('):
                epoch+=1
                iters.append(epoch)
                
                lossStr = iterLine[1].split(':')[1]
                print(iterLine[1], lossStr, type(lossStr))
                #print('loss,acc=',iterLine[7],iterLine[10],'val_loss,val_acc=',iterLine[13],iterLine[16])
                loss.append(float(lossStr))
               
                if stopIter and epoch>=stopIter:
                    break
                
    return iters,loss

def plotLoss(ax, iters, loss, label=None):
    #ax.set_title(name)
    ax.plot(iters,loss,label=label)
    
def argCmdParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--list', nargs='+', help='path to log file', required=True)
    parser.add_argument('-s', '--start', help = 'startIter')
    parser.add_argument('-t', '--stop', help = 'stopIter')
    
    return parser.parse_args()

def plotCompareLoss(lossFiles, stopEpoch=None):
    print(lossFiles)
    ax = plt.subplot(1,1,1)
    # labels=['BCELoss', 'BalancedLoss', 'FocalLoss']
    # for i,file in enumerate(lossFiles):
    #     iters,loss = getLoss(file,0,None)
    #     plotLoss(ax, iters, loss, label=labels[i])
    
    for file,name in lossFiles:
        iters,loss = getLoss(file,0,stopEpoch)
        plotLoss(ax, iters, loss, label=name)
        
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    #plt.ylim(0, 4)
    #plt.yscale("log")
    #plt.legend()
    plt.grid(linestyle='-.') #'-', '--', '-.', ':', '',
    #plt.legend(loc='lower left')
    #plt.legend()
    plt.savefig(r'.\res\lossCompare.png', dpi=300)
    plt.show()
       
def main():
    args = argCmdParse()
    
    startIter = 0
    stopIter = None
    if args.start:
        startIter = int(args.start)
    if args.stop:
        stopIter = int(args.stop)
        
    logs=[]
    base = r'.\log\new'
    #file = os.path.join(base, 'log_siamfc_Sequential_AlexNet_BCELoss_e.txt')
    #name = 'BCE Loss';      logs.append((file,name))
    
    file = os.path.join(base, 'log_siamfc_Con2Net__BalancedLoss_e.txt')
    name = 'Balanced BCE Loss';     logs.append((file,name))
    
    #file = os.path.join(base, 'log_siamfc_Sequential_AlexNet_FocalLoss_e.txt')
    #name = 'Focal Loss';    logs.append((file,name))
    
    plotCompareLoss(logs, stopEpoch=300)
    # for i in logs:
    #     plotCompareLoss([i],stopEpoch=200)

    
if __name__ == "__main__":
    main()
    