from __future__ import absolute_import

from train import prepareModel,prepareData,continueTrain

def main():
    tracker = prepareModel()
    seqs = prepareData()
    continueTrain(tracker,seqs)
    
if __name__ == '__main__':
    main()
        