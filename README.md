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

### Experiment Content

#### Exp 1 (2021.9.17 -- 2021.9.24)

1. 对比度调整
   - 设计一个Sigmoid函数，实现对图像的对比度调节
   - 使用opencv窗口系统的哦slider控件交互改变Sigmoid函数的参数，实现不同的对比度调整
2. 背景相减
   - 对图像I和对应的背景图B，基于背景相减检测I中的前景区域，并输出前景的mask
   - 分析可能产生误检的情况，设法对背景相减做出改进