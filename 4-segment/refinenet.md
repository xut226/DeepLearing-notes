用于图像分割

RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation

特点：

a. 带有精心设计解码器模块的编码器-解码器结构

b.所有组件遵循残差连接的设计方式。

1.网络结构

![](/assets/RefineNet.png)

每个RefineNet模块包含一个能通过对较低分辨率特征进行上采样来融合多分辨率特征的组件，以及一个能基于步幅为1及5×5大小的重复池化层来获取背景信息的组件。

这些组件遵循恒等映射的思想，采用了残差连接的设计方式。

![](/assets/RefineNet_2.png)

