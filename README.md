## Computer Vision Repo

---

**Star it if it's useful for you! Thanks!**

### Author

- Yang Li 李 阳
- Artificial Intelligence Class of 2019
- School of Computer Science and Technology
- Shandong University

### Update Log

- 2021-9-8 init commit.
- 2021-9-17 experiment 1
- 2021-9-21
  reference [OpenCV-Python Tutorials](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_tutorials.html)
- 2021-9-24 experiment 2
    - Reference1：[理解双线性插值](https://zhuanlan.zhihu.com/p/110754637)
    - Reference2：[理解opencv中的双线性插值](https://www.cnblogs.com/wxl845235800/p/9608736.html)
- 2021-10-10 experiment 3
    - Reference1：[copyMakeBorder函数详解](https://blog.csdn.net/qq_36560894/article/details/105416273)
    - Reference2：[高斯滤波(GaussianFilter)原理及C++实现](https://blog.csdn.net/weixin_40647819/article/details/89742936)
    - Reference3：[积分图实现快速均值滤波](https://blog.csdn.net/weixin_40647819/article/details/88775598)
    - Reference4：[OpenCV图像数据类型](https://www.jianshu.com/p/437c5031615c)
- 2021-10-15 experiment 4-1
    - Reference1：[鼠标响应setMouseCallback的用法](https://blog.csdn.net/qq_29540745/article/details/52562101)
    - Reference2：[基于直方图的目标跟踪](https://github.com/devWangBin/CV-image_processing)
    - Reference3：[RGB三通道直方图的计算与绘制](https://blog.csdn.net/Derical/article/details/108887966)
    - Reference4：[直方图匹配：巴氏系数](https://blog.csdn.net/jameschen9051/article/details/95895256)
- 2021-10-22 experiment 5
    - Reference1：[种子填充算法](https://www.bbsmax.com/A/amd0AVWzge/)
    - Reference2：[连通区域快速标记的two-pass算法及其实现](https://www.cnblogs.com/riddick/p/8280883.html)
    - Reference3：[二值图连通域快速标记算法](https://www.cnblogs.com/ailitao/p/11787513.html)
    - Reference4：[OpenCV连通区域分析](https://blog.csdn.net/icvpr/article/details/10259577)
    - Reference5：[OpenCV距离变换函数：distanceTransform](https://www.jianshu.com/p/77a773d97987)
- 2021-10-29 experiment 6
    - Reference1：[cvtColor和convertTo函数的区别](https://blog.csdn.net/qq_22764813/article/details/52135686)
    - Reference2：[霍夫圆检测](https://zhuanlan.zhihu.com/p/134452506)
    - Reference3：[霍夫梯度法1](https://www.cnblogs.com/bjxqmy/p/12333022.html)
    - Reference4：[霍夫梯度法2](https://blog.csdn.net/qq_41498261/article/details/103104035)
    - Reference5：[HoughCircles源码分析](https://blog.csdn.net/zhaocj/article/details/50454847)
    - Reference6：[OpenCV类型CV_32F和CV_32FC1之间的区别](https://www.askgo.cn/question/1476)
- 2021-11-3 experiment 7
    - Reference1：[为什么OpenCV读取的图像格式是BGR](https://cloud.tencent.com/developer/article/1473677)
    - Reference2：[Harris角点检测原理详解](https://blog.csdn.net/lwzkiller/article/details/54633670)
    - Reference3：[Harris角点检测C++实现](https://www.jianshu.com/p/44e63f7f7f4f)
    - Reference4：[OpenCV角点检测之Harris角点检测](https://blog.csdn.net/poem_qianmo/article/details/29356187)
    - Reference5：[Harris角点检测原理详解及源码分析](https://blog.csdn.net/qq_37059483/article/details/77836239)
- 2021-11-12 experiment 8
    - Reference1：[特征点提取和匹配](https://blog.csdn.net/jiangjiao4726/article/details/78385409)
    - Reference2：[特征检测和匹配方法汇总](https://www.cnblogs.com/skyfsm/p/7401523.html)

### Experiment Content

#### Exp 1 (2021.9.17 -- 2021.9.24)

1. 对比度调整
    - 设计一个Sigmoid函数，实现对图像的对比度调节
    - 使用opencv窗口系统的哦slider控件交互改变Sigmoid函数的参数，实现不同的对比度调整
2. 背景相减
    - 对图像I和对应的背景图B，基于背景相减检测I中的前景区域，并输出前景的mask
    - 分析可能产生误检的情况，设法对背景相减做出改进

#### Exp 2 (2021.9.24 -- 2021.9.30)

1. 图像变形
    - 记 [x’, y’]=f([x, y]) 为像素坐标的一个映射，实现 f 所表示的图像形变，并采用双线性插值进行重采样。f 的逆映射见ppt
    - [x’, y’]和[x, y]都是中心归一化坐标，请先进行转换
2. 仿照实验2.1，自己设计变换函数，对输入视频进行变换，生成哈哈镜的效果。
    - 采用cv::VideoCapture读取摄像头视频，并进行实时处理和显示结果。
    - 优化代码执行效率，改善实时性（不要忘了打开编译优化，vc请用release模式编译）。

#### Exp 3 (2021.10.4 -- 2021.10.12)

1. 高斯滤波
    - 通过调整高斯函数的标准差(sigma)来控制平滑程度
    - 滤波窗口大小取[6*sigma-1], [ ]表示取整
    - 利用二维高斯函数的行列可分性进行加速
2. 快速均值滤波
    - 滤波窗口大小通过参数来指定
    - 采用积分图进行加速，实现与滤波窗口大小无关的效率
    - 与opencv的boxFilter函数比较计算速度，分析差异

#### Exp 4 (2021.10.13 -- 2021.10.19)

（目标跟踪与图像分割可二选一）

1. 基于直方图的目标跟踪
    - 实现基于直方图的目标跟踪：已知第t帧目标的包围矩形，计算第t+1帧目标的矩形区域。
    - 选择适当的测试视频进行测试：给定第1帧目标的矩形框，计算其它帧中的目标区域。

2. 基于颜色分布的交互图像分割
    - 基于由用户交互笔刷标记的前、背景像素（图中黄绿区域），计算前、背景的颜色分布，并用于估计未标记像素属于前景和背景的概率。
    - 颜色分布可以用直方图或者高斯混合模型（GMM）表示。如果用GMM，可以基于OpenCV的实现估计GMM参数。

#### Exp 5 (2021.10.22 -- 2021.10.26)

1. 连通域
    - 实现一个8连通的快速连通域算法，并基于该算法对测试图像进行以下处理：
        - 计算白色连通区域的个数。
        - 删除较小的白色连通域，只保留最大的一个。

2. 距离变换
    - 了解OpenCV的距离变换函数distanceTransform。
    - 使用合适的测试图像进行测试，将距离场可视化输出。

#### Exp 6 (2021.10.29 -- 2021.11.2)

1. 霍夫变换
    - 实现基于霍夫变换的图像圆检测。
    - 边缘检测可以用opencv的canny函数。
    - 尝试对其准确率和效率进行优化实现。

#### Exp 7 (2021.11.5 -- 2021.11.9)

1. Harris角点检测
    - 实现Harris角点检测算法。
    - 与OpenCV的cornerHarris函数的结果和计算速度进行比较。

#### Exp 8 (2021.11.12 -- 2021.11.16)

1. 测试OpenCV中的SIFT, SURF, ORB等特征检测与匹配的方法
    - 将检测到的特征点和匹配关系进行可视化输出。
    - 比较不同方法的效率、效果等。

2. 阅读[论文](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94)