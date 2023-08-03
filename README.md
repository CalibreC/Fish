# Fish
本项目想要实现在原神内自动钓鱼。

感谢[genshin_auto_fish](https://github.com/7eu7d7/genshin_auto_fish)提供的思路与数据集，
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
python requirements.py
```

## 运行
运行时，请以管理员权限运行脚本，否则无法截图。

# Goals

- [x] 模型训练
- [x] 高速截图
- [ ] 识别鱼群
- [ ] DQN训练
- [ ] 数据集拓展
- [ ] 钓鱼
