#### 训练环境：

```
Python==3.8.10
numpy==1.21.4
matplotlib==3.5.0
torch==1.10.0+cu113
torchvision==0.11.1+cu113
argparse==1.1
sklearn==1.1.1
pandas==1.4.3
PIL==8.4.0
transformers==4.20.1
```

#### 文件结构：

```
├─data
│  └─实验五数据
│      ├─data
│      ├─test_without_label.txt
│      └─train.txt
├─model	# 保存模型架构文件，训练完成后的模型参数和结构
│  ├─bert.py
│  ├─Multimodal.py
│  └─VGG16.py
├─result	# 保存训练过程的细节，包括loss和accuracy，以及对数据绘制的折线图
├─lab5.ipynb	# 运行项目，绘制实验结果折线图
├─main.py	# 项目主函数
├─make_dataloader.py	# 数据预处理
├─requirements.txt	# 项目配置文件
├─run_multimodal.py	# 训练多模态模型并测试
├─run_pictrue.py	# 只输入图像特征的消融实验训练函数
└─run_text.py	# 只输入文本特征的消融实验训练函数
```

#### 运行说明：

1.参数设置：

```
--model_type	# 选择训练模式，输入Multimodal训练多模态模型，输入Picture_model进行保留图像特征的消融实验，输入Text_model进行保留文本特征的消融实验。默认为Multimodal
--lr	# 设置学习率，默认为0.005
--momentum	# 设置优化器动量，默认为0.90
--batch_size	#设置batch_size，默认为32
```

2.运行项目：

训练多模态模型并保存模型结构和参数，输入：

```
python main.py --model_type Multimodal
```

进行保留图像特征的消融实验，输入：

```
python main.py --model_type Picture_model
```

进行保留文本特征的消融实验，输入：

```
python main.py --model_type Text_model
```

以上命令均可在lab5.ipynb文件中运行对应代码块。
