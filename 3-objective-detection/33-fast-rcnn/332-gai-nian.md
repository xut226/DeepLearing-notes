## ROIPooling

在Fast RCNN网络中，RoI来完成SPP层的作用。RoI指的是在一张图片上完成Selective Search后得到的“候选框”在特征图上的一个映射。

考虑到感兴趣区域（RoI）尺寸不一，但是输入图中后面FC层的大小是一个统一的固定值，因为ROI池化层的作用类似于SPP-net中的SPP层，即将不同尺寸的RoI feature map池化成一个固定大小的feature map。具体操作：假设经过RoI池化后的固定大小为是一个超参数，因为输入的RoI feature map大小不一样，需要对这个feature map进行池化来减小尺寸，即用这个计算出的窗口对RoI feature map做max pooling，Pooling对每一个feature map通道都是独立的。

其次RoI有四个参数![](https://private.codecogs.com/gif.latex?\left %28 r %2Cc%2Ch%2Cw \right %29 "\left \( r ,c,h,w \right \)")除了尺寸参数![](https://private.codecogs.com/gif.latex?h%u3001w "h、w")、![](https://private.codecogs.com/gif.latex?w "w")外，还有两个位置参数![](https://private.codecogs.com/gif.latex?r "r")、![](https://private.codecogs.com/gif.latex?c "c")表示RoI的左上角在整个图片中的坐标。

#### 输入

* 从具有多个卷积核池化的深度网络中获得的固定大小的feature maps；
* 一个表示所有ROI的N\*5的矩阵，其中N表示ROI或者说是region proposal的数目。第一列表示图像index，其余四列表示其余的左上角和右下角坐标。

#### 操作步骤

（1）根据输入image，将ROI映射到feature map对应位置，如image的width和height为\(W,H\)，ROI（x0,y0,w0,h0\)映射到image的feature map（固定的大小）上的区域为（x0/W,y0/H,w0/W,h0/H\)；

（2）将映射后的区域划分为相同大小的sections（sections数量与输出的维度相同）；

（3）对每个sections进行max pooling操作；

#### 例子

考虑一个8\*8大小的feature map，一个ROI，以及输出大小为2\*2.

（1）输入的固定大小的feature map

![](/assets/3.3.2 feature map.png)

（2）region proposal 投影之后位置（左上角，右下角坐标）：（0，3），（7，8）。

![](/assets/3.3.2 roi_featuremap.png)

（3）将其划分为（2\*2）个sections（因为输出大小为2\*2），我们可以得到：

![](/assets/3.3.2_featuremap sections.png)

（4）对每个section做max pooling，可以得到：

![](/assets/3.3.2_section_maxpooling.png)

reference：

[https://blog.csdn.net/auto1993/article/details/78514071](https://blog.csdn.net/auto1993/article/details/78514071)

# NMS（non-Maximum Suppression）非极大值抑制

非极大值抑制的方法是：先假设有6个矩形框，根据分类器的类别分类概率做排序，假设从小到大属于车辆的概率 分别为A、B、C、D、E、F。

\(1\)从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;

\(2\)假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。

\(3\)从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，那么就扔掉；并标记E是我们保留下来的第二个矩形框





