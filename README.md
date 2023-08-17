# Fish
本项目想要实现在原神内自动钓鱼。

感谢[genshin_auto_fish](https://github.com/7eu7d7/genshin_auto_fish)提供的思路、代码与数据集，
实现过程中还参考了[基于yolov5下的原神鱼群目标识别改进版](https://www.bilibili.com/video/BV1dF411i7d7/?spm_id_from=333.999.0.0&vd_source=e676528ca871aca19979ddeb9404c414)

# Running Fish

本项目暂无二进制包，需要自行配置python环境。

## 安装python环境
建议使用python3.10。本人用pycharm自动建立环境

## 下载源码
```bash
git clone https://github.com/CalibreC/Fish.git
cd Fish
```

## 安装依赖
```bash
python -m pip install -U pip
pip install -r requirements.py
```

## 运行
1.游戏本体不能最小化 \
2.选择dxcam，win32gui中的一种截图方式， \
3.运行win32gui需要管理员权限，否则会报错；运行dxcam需要将本体前置，放到所有窗口最前方


# Goals

- [x] 模型训练
- [x] 高速截图
  - [x] dxcam
  - [x] win32gui
  
  dxcam截图30fps左右，但是有时会截图失败；win32gui稳定，但是帧数低15fps左右
- [x] 实时识别鱼群
- [x] DQN训练
  - [x] 模拟训练
  - [ ] 游戏训练
    - [x] 训练流程
    - [x] 按键点击
    - [ ] dxcam截图失败
- [ ] 数据集拓展
- [ ] 钓鱼
- [ ] PyQt5 GUI

# 训练
## DQN模拟训练
```python
python .\\DQN\\train_simulation.py
```

## DQN游戏训练
```python
python .\\DQN\\train_genshin.py
```
### 原项目

