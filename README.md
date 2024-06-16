# 2024 IESD智能嵌入式系统设计大赛

> 参赛学校：华东师范大学
>
> 参赛队名：ECNUer
> 
> 指导教师：沙行勉老师
> 
> 参赛队员：李琳、徐珑珊

## 工作概述

[本仓库](https://github.com/lin772021/IESD-CONTEST-TF)是[2024 IESD大赛](https://iesdcontest.github.io/iesd-2024/index.html)的提交。本项目是基于人工智能算法的房颤检测模型的设计与实现，整体完成情况如下。

- 探索了**经典-量子混合神经网络模型**在该问题上的应用潜力；

- 通过调研，设计了基于**卷积神经网络**的经典模型，包括CNN、CNN-LSTM、CNN-RNN；

- 完成了CNN模型与CNN-LSTM模型在龙芯2K500先锋板上的部署；

- 开发了**测试数据可视化工具**，可视化心电图、心电数据医学特征和模型测试结果，可用于协助模型调试。

按照大赛评分标准，本项目所提交的模型得分为：

| 排名   | 综合得分   | $F_\beta$   | G Score   | 推理延迟（ms）   | 存储占用（kb）   |
|-------|-------|-------|-------|-------|-------|
| 8 | 60.42493328 | 0.941 | 0.833 | 13.789 | 471 |

## 目录说明

本仓库的目录说明如下。

```
2024IESD-ECNUer
├── deploy                   模型部署上板代码
│
├── train                    模型训练代码
│   ├── data_indices         数据条目索引
│   ├── models               模型架构定义
│   │   ├── best_model.py    提交的CNN模型（tensorflow）
│   │   ├── LSTM.py          经典CNN-LSTM模型（tensorflow）
│   │   ├── RNN.py           经典CNN-RNN模型（tensorflow）
│   │   └── QNN.py           经典-量子混合模型定义（pytorch）
│   │
│   ├── train(_QNN).py       训练经典（量子）模型的代码
│   ├── test(_QNN).py        测试经典（量子）模型的代码
│   ├── keras2tflite.py      转换CNN模型
│   └── LSTM2tflite.py       转换LSTM模型
│
├── visualize                测试数据可视化代码
└── README.md                本文档
```

数据集需放在`../data/training_dataset/`路径下。

## 代码运行

### 模型训练

数据集放置路径为`../data/training_dataset/`。环境配置方法如下。

    pip install requirements.txt

配置完毕后，进入训练文件夹`./train`。

    cd train 
    python train.py
    python test.py

### 模型部署

模型部署所需的文件位于`./deploy`文件夹下，使用方式可以参考[iesdcontest2024_demo_example_deployment](https://github.com/iesdcontest/iesdcontest2024_demo_example_deployment)中的介绍。

### 测试数据可视化

进入`./visualize/Web`文件夹，并运行app.py文件，即可看到可视化结果。

```
cd visualize/Web
python app.py
```
