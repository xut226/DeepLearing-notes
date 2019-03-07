3.2.1结构

![](/assets/RCNN_struct.png)2.extract region proposals 方法：selective search

warped region ：归一化为相同大小的图片，送入CNN网络

3.CNN网络训练

3.1 预训练

如果做的目标定位系统是定位男人、女人、猫、狗这四类目标，那我们将fine tuning的神经网络中的最后一层_num_设置为5（4+1），加的这一类代表背景。

3.2背景获取

首先，提前对图片数据提前标定目标位置，对于每张图可能获得一个或更多的标定矩形框（x，y，w，h分别表示横坐标的最小值，纵坐标的最小值、矩形框宽度、矩形框长度）。

其次，通过Python selectivesearch库中的selectivesearch指令获得多个目标框（Proposals）（selectivesearch指令根据图片的颜色变化、纹理等将多个像素合并为多个选框）。

接着通过定义并计算出的IoU（目标框与标定框的重合程度，即IoU=重合面积/两个矩形所占的面积（其中一个矩形是标定框，另一个矩形是目标框））与阈值比较，若大于这个阈值则表示该目标框标出的是男人、女人、猫或狗四类中的一类，若小于这个阈值则表示该标定框标出的是背景

4.SVM classify region

Fine tuning 阶段将IoU大于0.5的目标框圈定的图片作为正样本，小于0.5的目标框圈定的图片作为负样本。而在对每一类目标分类的SVM训练阶段，我们将标定框圈定的图片作为正样本，IoU小于0.3的目标框圈定的图片作为负样本，其余目标框舍弃。

将图片用selectivesearch指令分为多个矩形选框，用SVM模型对这些选框区域进行分类，即判定该区域中是否包含目标，并将标签为1（即包含人脸的图片）记录下来

5 使用回归器精细修正候选框位置 （box regression）

NMS  非最大值抑制

算法流程：

转载：[https://www.cnblogs.com/zf-blog/p/6740736.html](https://www.cnblogs.com/zf-blog/p/6740736.html)

1 training

a\) supervised pre-training

| 样本 |
| :--- |


|  | 来源 |
| :--- | :--- |
| 正样本 | ILSVRC2012 |
| 负样本 | ILSVRC2012 |

ILSVRC样本集上仅有图像类别标签，没有图像物体位置标注；  
采用AlexNet CNN网络进行有监督预训练，学习率=0.01；  
该网络输入为227×227的ILSVRC训练集图像，输出最后一层为4096维特征-&gt;1000类的映射，训练的是网络参数。

b\) Domain-specific fine-tuning

特定样本下的微调

| 样本 | 来源 |
| :--- | :--- |
| 正样本 | Ground Truth+与Ground Truth相交IoU&gt;0.5的建议框【由于Ground Truth太少了】 |
| 负样本 | 与Ground Truth相交IoU≤0.5的建议框 |

PASCAL VOC 2007样本集上既有图像中物体类别标签，也有图像中物体位置标签；  
采用训练好的AlexNet CNN网络进行PASCAL VOC 2007样本集下的微调，学习率=0.001【0.01/10为了在学习新东西时不至于忘记之前的记忆】；  
mini-batch为32个正样本和96个负样本【由于正样本太少】；  
该网络输入为建议框【由selective search而来】变形后的227×227的图像，修改了原来的1000为类别输出，改为21维【20类+背景】输出，训练的是网络参数。

3）Object category classifiers（SVM训练）

| 样本 |
| :--- |


|  | 来源 |
| :--- | :--- |
| 正样本 | Ground Truth |
| 负样本 | 与Ground Truth相交IoU＜0.3的建议框 |

由于SVM是二分类器，需要为每个类别训练单独的SVM；  
SVM训练时输入正负样本在AlexNet CNN网络计算下的4096维特征，输出为该类的得分，训练的是SVM权重向量；  
由于负样本太多，采用hard negative mining的方法在负样本中选取有代表性的负样本，该方法具体见。

4）Bounding-box regression训练

| 样本 | 来源 |
| :--- | :--- |
| 正样本 | 与Ground Truth相交IoU最大的Region Proposal，并且IoU&gt;0.6的Region Proposal |

输入数据为某类型样本对N个：{\(Pi,Gi\)}i=1⋯N以及Pii=1⋯N所对应的AlexNet CNN网络Pool5层特征ϕ5\(Pi\)i=1⋯N，输出回归后的建议框Bounding-box，训练的是dx\(P\)，dy\(P\)，dw\(P\)，dh\(P\)四种变换操作的权重向量。具体见前面分析。

处理技巧：

hard negatives：在训练的过程中，发现正样本的数量远远小于负样本，训练是将出现很多false negative。一种解决方法是：将正样本与差不多的负样本先加入训练，将训练出来的模型去预测未加入训练的负样本，被预测为正样本时判断为false negative。将其加入训练的负样本集，进行下一次训练。

hard negative mining概念：

难分样本挖掘。参考tensorflow objective detect 源码ssd build\__hard\_negative\_miner_

CVPR2016 Training Region-based Object Detectors with Online Hard Example Mining\(oral\)，将hard negative mining 机制嵌入到SGD 中，fast-RCNN在训练的过程中根据你region proposal的损失自动选取合适的region proposal

作为正负样本训练。

