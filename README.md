# SiameseFc_PyTorch
Siamese FC network for object/pedestrain tracking.


## References
Dataset benchmark: https://motchallenge.net/data/MOT17/   
Code reference: https://github.com/huanglianghua/siamfc-pytorch
<br/>
FC Siamese network [paper.](https://arxiv.org/pdf/1606.09549.pdf)

## Metrics
 - Intersection over Union(IoU)
 <img src="src/res/IoU.png" width="150" height="150" />
 <img src="https://latex.codecogs.com/svg.latex?IoU%20=%20\frac{S_{gt}%20\bigcap%20S_{pred}%20}{S_{gt}%20\bigcup%20S_{pred}%20}" /> <br/>
 
 - Precision
 <img src="src/res/precision.png" width="150" height="150" />
 <img src="https://latex.codecogs.com/svg.latex?Distance=%20\|C_{gt}-C_{pred}%20\|_{2}" /> <br/>
 <!-- Distance=%20\sqrt{(x_{gt}-x_{pred})^2+(y_{gt}-y_{pred})^2} -->
 <!-- Distance=%20\|C_{gt}-C_{pred} \|_{2} -->
 
## IoU/Precision plot
|||
|---|---|
|<img src="src/res/prAlg.png" width="320" height="240" />|<img src="src/res/srAlg.png" width="320" height="240" />|
|<img src="src/res/PrecisonPlot.png" width="320" height="240" />|<img src="src/res/IoUPlot.png" width="320" height="240" />|

## Tracking effect
|||
|---|---|
|<img src="src/res/Figure_1.png" width="240" height="135" />|<img src="src/res/Figure_2.png" width="240" height="135" />|
|<img src="src/res/Figure_5.png" width="240" height="135" />|<img src="src/res/Figure_6.png" width="240" height="135" />|
|||