## ROIPooling

在Fast RCNN网络中，RoI来完成SPP层的作用。RoI指的是在一张图片上完成Selective Search后得到的“候选框”在特征图上的一个映射。

考虑到感兴趣区域（RoI）尺寸不一，但是输入图中后面FC层的大小是一个统一的固定值，因为ROI池化层的作用类似于SPP-net中的SPP层，即将不同尺寸的RoI feature map池化成一个固定大小的feature map。具体操作：假设经过RoI池化后的固定大小为是一个超参数，因为输入的RoI feature map大小不一样，假设为，需要对这个feature map进行池化来减小尺寸，那么可以计算出池化窗口的尺寸为：，即用这个计算出的窗口对RoI feature map做max pooling，Pooling对每一个feature map通道都是独立的。

其次RoI有四个参数![](https://private.codecogs.com/gif.latex?\left %28 r %2Cc%2Ch%2Cw \right %29 "\left \( r ,c,h,w \right \)")除了尺寸参数![](https://private.codecogs.com/gif.latex?h%u3001w "h、w")、![](https://private.codecogs.com/gif.latex?w "w")外，还有两个位置参数![](https://private.codecogs.com/gif.latex?r "r")、![](https://private.codecogs.com/gif.latex?c "c")表示RoI的左上角在整个图片中的坐标。

#### 输入

* 从具有多个卷积核池化的深度网络中获得的固定大小的feature maps；
* 一个表示所有ROI的N\*5的矩阵，其中N表示ROI或者说是region proposal的数目。第一列表示图像index，其余四列表示其余的左上角和右下角坐标。

#### 操作步骤

（1）根据输入image，将ROI映射到feature map对应位置，如image的width和height为\(W,H\)，ROI（x0,y0,w0,h0\)映射到image的feature map（固定的大小）上的区域为（x0/W,y0/H,w0/W,h0/H\)；

（2）将映射后的区域划分为相同大小的sections（sections数量与输出的维度相同）；

（3）对每个sections进行max pooling操作；

