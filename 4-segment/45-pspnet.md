用于图像分割

论文：Pyramid Scene Parsing Network（[https://arxiv.org/abs/1612.01105）](https://arxiv.org/abs/1612.01105）)

特点：

a.提出了金字塔池化模块来聚合背景信息；

b.使用了附加损失\(auxiliary loss\)

PSPNet也用空洞卷积来改善Resnet结构，并添加了一个金字塔池化模块。该模块将ResNet的特征图谱连接到并行池化层的上采样输出，其中内核分别覆盖了图像的整个区域、半各区域和小块区域。

在ResNet网络的第四阶段\(即输入到金字塔池化模块后\)，除了主分支的损失之外又新增了附加损失，这种思想在其他研究中也被称为中级监督\(intermediate supervision\)。

![](/assets/PSPNet.png)

