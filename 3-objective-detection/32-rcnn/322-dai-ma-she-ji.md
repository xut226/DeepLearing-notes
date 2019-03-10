代码分为以下几个部分设计：

预训练——&gt;微调——&gt;测试

1.预训练（supervised pre-training）

在没有bounding box labels的大数据集（ILSVRC2012）上训练CNN。由于ILSVRC2012太大，训练太耗时，在此阶段，采用ImageNet训练AlexNet网络结构。加载1000类分类的权重

2.微调（Domain-specific fine-tuning）

在关心的几类分类结果中进行微调。

与预训练的网络结构相同，仅仅是将预训练的1000个class替换为21个class（20 VOC class + 1 background），对于是否为背景的那一类如下处理：将所有与ground-truth box 相交的region proposal 的IOU&gt;0.5视为Positive 否则视为negative（all region proposals with  0:5 IoU over-lap with a ground-truth box as positives for that box’s class and the rest as negatives. ）ground-truth为事先标注好的bounding box数据集。,region proposalyou由selective search产生。

selective search 产生多个框，多少个框根据设置的rect，size确定，与N个分类class生成label（n+1维）

超参数：SGD，lr=2e-5,设置不合适loss不降。

数据：batch\_size = 128，32个 positive sample\(覆盖所有class\) 和96个  negative sample

3.分类

SVM训练微调

