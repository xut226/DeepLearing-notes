1.1.1 **卷积**

通过卷积运算，可以使原信号特征增强，并且降低噪声。

卷积对于二维图像中的效果就是：对于图像中的每个像素邻域求加权和得到该像素点的输出值.

我们称![](https://www.zhihu.com/equation?tex=%28f*g%29%28n%29 "\(f\*g\)\(n\)")为![](https://www.zhihu.com/equation?tex=f%2Cg "f,g")的卷积

其连续的定义为：

![](https://www.zhihu.com/equation?tex=\displaystyle+%28f*g%29%28n%29%3D\int+_{-\infty+}^{\infty+}f%28\tau+%29g%28n-\tau+%29d\tau+\\ "\displaystyle \(f\*g\)\(n\)=\int \_{-\infty }^{\infty }f\(\tau \)g\(n-\tau \)d\tau \\")

其离散的定义为：

![](https://www.zhihu.com/equation?tex=\displaystyle+%28f*g%29%28n%29%3D\sum+_{\tau+%3D-\infty+}^{\infty+}{f%28\tau+%29g%28n-\tau+%29}\\ "\displaystyle \(f\*g\)\(n\)=\sum \_{\tau =-\infty }^{\infty }{f\(\tau \)g\(n-\tau \)}\\")

一个3\*3的卷积：

![](/assets/Conv.png)

卷积的滑动过程：

![](/assets/Conv_movemet.png)

1.1.2 卷积层

a.用于提取特征。一个n\*n的卷积核可以学习图像的某个特征，多个卷积核可以学习到相对独立的多个特征。

b.超参数：卷积核（filter）个数，卷积核大小，步长，边界填充方式

**通道与通道之间的关系可以如何学习？**

c.卷积核大小:常用 1\*1,3\*3,5\*5,，7\*7

d.卷积对图像边界的处理：

zero padding：四周填充0

same padding：

![](/assets/para_formula.png)

