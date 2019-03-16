代码分为以下几个部分设计：

预训练——&gt;微调——&gt;分类——&gt;回归——&gt;测试

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

4.框回归

线性回归，

输入：image: finetune 的特征向量4096维，label：region proposal与groud truth 训练的 5维数据

输出：5维数据，类别 1，bounding box 4

5，测试

输入：待预测图片

输入图片用selective search 和wrap操作生成 region proposal数据

导入微调模型，生成4096维特征数据

导入所有类别训练SVM模型，将所有特征数据带入，获取概率值最大的类别预测

导入回归模型，判断类别和bbo 的预测top排序，选择&gt;threshold的留做NMS



