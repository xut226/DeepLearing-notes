LinkNet是一个轻量级的图像分割网络，速度快

![](/assets/LinkNet_overview.png)

网络结构：

![](/assets/LinkNet.png)

左半部分表示编码，后半部分表示解码。encoder模块包含残差块

![](/assets/LinkNet_encoder.png)

![](/assets/LinkNet_decoder.png)

LinkNet将编码块和解码块相连接。编码器输出连接到解码器输入。

