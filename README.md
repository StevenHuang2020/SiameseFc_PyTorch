# SiameseFc_PyTorch
Siamese FC network for object/pedestrain tracking.


## References
Dataset benchmark: https://motchallenge.net/data/MOT17/   
Code reference: https://github.com/huanglianghua/siamfc-pytorch
<br/>
FC Siamese network [paper.](https://arxiv.org/pdf/1606.09549.pdf)

## Metrics
 - IoU
 <img src="src/res/IoU.png" width="200" height="200" />
 <img src="https://latex.codecogs.com/svg.latex?Intersection%20over%20Union(IoU)%20=%20\frac{S_{gt}%20\bigcap%20S_{pred}%20}{S_{gt}%20\bigcup%20S_{pred}%20}" /> <br/>
 - Precision
 <img src="src/res/precision.png" width="200" height="200" />
 <img src="https://latex.codecogs.com/svg.latex?Distance=%20\sqrt{(x_{gt}-x_{pred})^2+(y_{gt}-y_{pred})^2}" /> <br/>
 

## IoU/Precision
|||
|---|---|
|<img src="src/res/PrecisonPlot.png" width="320" height="240" />|<img src="src/res/IoUPlot.png" width="320" height="240" />|

## Tracking effect
|||
|---|---|
|<img src="src/res/dstImg1/000001.jpg" width="240" height="135" />|<img src="src/res/dstImg1/000090.jpg" width="240" height="135" />|
|<img src="src/res/dstImg1/000180.jpg" width="240" height="135" />|<img src="src/res/dstImg1/000300.jpg" width="240" height="135" />|
|||
|<img src="src/res/dstImg3/000001.jpg" width="240" height="135" />|<img src="src/res/dstImg3/000090.jpg" width="240" height="135" />|
|<img src="src/res/dstImg3/000180.jpg" width="240" height="135" />|<img src="src/res/dstImg3/000300.jpg" width="240" height="135" />|
|||
|<img src="src/res/dstImg5/000001.jpg" width="240" height="135" />|<img src="src/res/dstImg5/000090.jpg" width="240" height="135" />|
|<img src="src/res/dstImg5/000180.jpg" width="240" height="135" />|<img src="src/res/dstImg5/000300.jpg" width="240" height="135" />|