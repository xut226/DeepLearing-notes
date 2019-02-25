用于图像分割

论文：Large Kernel Matters — Improve Semantic Segmentation by Global Convolutional Network\([https://arxiv.org/abs/1703.02719\](https://arxiv.org/abs/1703.02719%29\)

特点：一种带有大维度卷积核的编码器-解码器结构

1.网络结构

ResNet\\(不带空洞卷积\\)组成了整个结构的编码器部分，同时GCN网络和反卷积层组成了解码器部分。该结构还使用了一种称作边界细化\\(Boundary Refinement，BR\\)的简单残差模块

![](/assets/Large Kernel.png)

