去除了最后的全连接层，用全局平均池化层（将图片尺寸变为 1 \* 1）取代。全连接层几乎占据了AlexNet,VGGNet中的90%参数量，而且容易引起过拟合。

卷积层要提升表达能力，主要依靠增加输出通道数，但是副作用是计算量增大和过拟合。

### 2.4.1 Inception V1

Inception Module结构：

![](/assets/Inception Module.png)

```
                                                                             Inception V1
```

1\*1 卷积可以跨通道组织信息，提高网络的表达能力，1\*1 的卷积用很小的计算量就能增加一层特征变换。

早期计算机视觉研究，受视觉神经系统启发，使用不同尺寸的Gabor滤波器处理不同尺寸的照片，扩展网络的深度和宽度。

Inception Module是一种稀疏网络，有利于大型、非常深的神经网络，减少过拟合并降低计算量。

技巧：

Inception V1有22层网络，除最后一层输出分类结果，中间节点的分类效果也很好，作为辅助分类节点（auxiliary classifiers\)

，将中间某一层的输出作为分类，设置较小权重（0.3）加到最后一层分类结果中。**相当于做模型融合，同时给网络增加了反向传播的梯度，也提供额额外的正则化。**

#### General Design Principles

下面的准则来源于大量的实验，因此包含一定的推测，但实际证明基本都是有效的。

1 . 避免表达瓶颈，特别是在网络靠前的地方。 信息流前向传播过程中显然不能经过高度压缩的层，即表达瓶颈。从input到output，feature map的宽和高基本都会逐渐变小，但是不能一下子就变得很小。比如你上来就来个kernel = 7, stride = 5 ,这样显然不合适。  
另外输出的维度channel，一般来说会逐渐增多\(每层的num\_output\)，否则网络会很难训练。（特征维度并不代表信息的多少，只是作为一种估计的手段）

2 . 高维特征更易处理。 高维特征更易区分，会加快训练。

1. 可以在低维嵌入上进行空间汇聚而无需担心丢失很多信息。 比如在进行3x3卷积之前，可以对输入先进行降维而不会产生严重的后果。假设信息可以被简单压缩，那么训练就会加快。

4 . 平衡网络的宽度与深度。

### **2.4.2 Inception V2**

结构：

![](/assets/InceptionV2_struct.png)

特点：

3\*3 取代 5\*5的卷积。使用 BatchNorm

两个3\*3 取代一个5\*5，相比有更少的参数量

### 2.4.3 Inception V3

![](/assets/InceptionV3.png)

用1xn和nx1卷积的串联来代替nxn卷积，计算量可以可以降低为1/n。

![](/assets/InceptionV3_struct1.png)

![](/assets/InceptionV3_struct2.png)

整个InceptionV3的结构

### ![](/assets/InceptionV3_structure.png)

### 2.4.4 Inception V4

在Inception之上加上Resnet

整体结构：

![](/assets/InceptionV4_struct.png)

```
                                                               Inception V4 整体结构
```

![](/assets/InceptionV4_1.png)![](/assets/InceptionV4_2.png)

```
                                                                **stem和Inception-A部分结构图**
```

![](/assets/InceptionV4_reductionA.png)![](/assets/InceptionV4_InceptionB.png)

```
                                                                 **Reduction-A和Inception-B部分结构图**
```

![](/assets/InceptionV4_Reduction-B.png)![](/assets/InceptionV4_C.png)

```
                                                                 **Reduction-B和Inception-C部分结构图**
```

### 2.4.5 Inception-resnet-v1 & Inception-resnet-v2

![](/assets/Inception-resnetv1v2.png)

