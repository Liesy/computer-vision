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