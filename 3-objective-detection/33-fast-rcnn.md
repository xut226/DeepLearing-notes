1.R CNN缺陷

a\) 训练分多步骤（先在分类数据集上预训练，再进行fine-tune训练，然后再针对每个类别都训练一个线性SVM分类器，最后再用regressors对bounding box进行回归，并且bounding box还需要通过selective search生成）

b\) 时间和空间开销大（在训练SVM和回归的时候需要用网络训练的特征作为输入，特征保存在磁盘上再读入的时间开销较大）

c\) 测试比较慢（每张图片的每个region proposal都要做卷积，重复操作太多

2 fast RCNN 结构

![](/assets/Fast RCNN.png)

3 训练过程

* 输入是224×224的固定大小图片

* 输入图片经过5个卷积层+2个降采样层（分别跟在第一和第二个卷积层后面）

* 进入ROIPooling层（其输入是conv5层的输出和region proposal，region proposal个数大约为2000个）

* 再经过两个output都为4096维的全连接层

* 分别经过output各为21和84维的全连接层（并列的，前者是分类输出，后者是回归输出）

* 最后接上两个损失层（分类是softmax，回归是smoothL1）

ROIPooling层:

由于 region proposal的尺度各不相同，而期望提取出来的特征向量维度相同，ROIPooling解决这个问题

将region proposal划分为H \* W大小的网格

对每个网格做MaxPooling（即每个网格对应一个输出）

将所有输出值组合起来形成固定大小为H\*W的feature map

