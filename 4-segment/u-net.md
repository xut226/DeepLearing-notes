论文：U-Net: Convolutional Networks for Biomedical Image Segmentation

1.网络结构

![](/assets/U-net.png)

端到端网络（输入是图像，输出也是图像），实际上是一种编码器-解码器结构。

U-net 适用于小数据集

在图像语义分割中使用卷积网络最 大的问题在于池化层，池化层不仅扩大感受野、聚合语境从而造成位置信息的丢失。但是语义分割要求类别图完全贴合，因此需要保留位置信息。

