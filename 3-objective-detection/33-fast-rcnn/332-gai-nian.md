## ROIPooling

在Fast RCNN网络中，RoI来完成SPP层的作用。RoI指的是在一张图片上完成Selective Search后得到的“候选框”在特征图上的一个映射。

考虑到感兴趣区域（RoI）尺寸不一，但是输入图中后面FC层的大小是一个统一的固定值，因为ROI池化层的作用类似于SPP-net中的SPP层，即将不同尺寸的RoI feature map池化成一个固定大小的feature map。具体操作：假设经过RoI池化后的固定大小为是一个超参数，因为输入的RoI feature map大小不一样，假设为，需要对这个feature map进行池化来减小尺寸，那么可以计算出池化窗口的尺寸为：，即用这个计算出的窗口对RoI feature map做max pooling，Pooling对每一个feature map通道都是独立的。

