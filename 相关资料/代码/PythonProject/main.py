#!/usr/bin/env python
# coding: utf-8

"""
导入实验数据并训练模型
训练数据的产生：
1. 将数据按时间步划分
2. 归一化
模型输入的shape为：
[batch_size, seq_len, input_size]
batch_size:训练包的大小
seq_len:序列的长度
input_size:特征的维度
author: yiwen
time: 2023-12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from norm import norm
import torch
import torch.nn as nn
from fit import fit
from split import split
from torchinfo import summary
from mydataloader import my_dataloder
from lsnet import LsNet
from vnet import Vnet
from lnet import LNet

# 定义文件存放路径
data_path = '../data/'
save_path = '../save/'
seq_len = 30


# ------------------------数据处理部分----------------------------------
# 导入数据
data1 = pd.read_excel(data_path + 'SP2_25C_FUDS/fuds_80.xlsx').values
data2 = pd.read_excel(data_path + 'SP2_25C_DST/dst_80.xlsx').values
data3 = pd.read_excel(data_path + 'SP2_25C_BJDST/bjdst_80.xlsx').values
data4 = pd.read_excel(data_path + 'SP2_25C_US06/us06_80.xlsx').values
data5 = pd.read_excel(data_path + 'OTHER_DATA/incre.xlsx').values

#将数据按时间步划分
data_dict1 = split(data1, seq_len=seq_len)
data_dict2 = split(data2, seq_len=seq_len)
data_dict3 = split(data3, seq_len=seq_len)
data_dict4 = split(data4, seq_len=seq_len)
data_dict5 = split(data5, seq_len=seq_len)

# 合并数据
data_dict = {}
for key in data_dict1.keys():
    data_dict[key] = np.concatenate((data_dict1[key], data_dict2[key], \
                                     data_dict3[key], data_dict4[key], \
                                     data_dict5[key]), axis=0)

# 划分训练集和测试集
y = np.repeat(data_dict['soc_real'].reshape(-1, 1), repeats=seq_len, axis=1)
y = y.reshape(-1, seq_len, 1)
xy = np.concatenate((data_dict['iu'], y), axis=2)
np.random.shuffle(xy)
split_p = int(xy.shape[0] * 0.75)
train_x = xy[:split_p, :, :-1]
train_y = xy[:split_p, 0, -1]
test_x = xy[split_p:, :, :-1]
test_y = xy[split_p:, 0, -1]

# 归一化
maxi = np.max(train_x, axis=(0, 1))
mini = np.min(train_x, axis=(0, 1))
norm_dict = {'max': maxi, 'min': mini}
train_x_nm = norm(train_x, norm_dict)
test_x_nm = norm(test_x, norm_dict)
xy_dict = {'train_x': train_x_nm, 'train_y': train_y, \
           'test_x': test_x_nm, 'test_y': test_y}

# 将本次训练的关键数据保存下来
np.save(save_path + 'norm_dict.npy', norm_dict)
np.save(save_path + 'xy_dict.npy', xy_dict)


# --------------------------训练模型部分-----------------------------------
train_x = xy_dict['train_x'].astype(np.float32)
train_y = xy_dict['train_y'].astype(np.float32)
test_x = xy_dict['test_x'].astype(np.float32)
test_y = xy_dict['test_y'].astype(np.float32)

# 设置超参数
seq_len = 30
bs = 128
epochs = 100
hidden_size = 64
num_layers = 2
input_size = train_x.shape[-1]

# 数据打包
train_dl, test_dl = my_dataloder(train_x, train_y, test_x, test_y, bs, shuffle=True)

#选择要使用的模型
# model = LsNet(hidden_size, input_size, num_layers)
# model = Vnet(input_size)
model = LNet(seq_len)

# 设置学习方式
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
summary(model, input_size=(bs, seq_len, input_size))

# 训练
train_loss = []
test_loss = []
for epoch in range(epochs):
    epoch_loss, epoch_test_loss = fit(epoch, model, train_dl, test_dl, loss_fn, optimizer)
    train_loss.append(epoch_loss)
    test_loss.append(epoch_test_loss)
print('Done!')

# 绘图
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['PingFang']
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.title('训练loss曲线')
plt.plot(train_loss)
plt.subplot(122)
plt.title('测试集loss曲线')
plt.plot(test_loss)
plt.show()

# 保存训练结果
torch.save(model, save_path + 'model.pth')








