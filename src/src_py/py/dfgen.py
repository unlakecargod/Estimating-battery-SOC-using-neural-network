#!/usr/bin/env python
# coding: utf-8

# 功能描述：  
# 输入ui与soc_ini，生成安时积分的soc_pre以及对应的模型参数。  
# 输入单位为mV和mA，时间单位为s。

# 以下为模块的主体部分


import numpy as np

def ah(soc_ini, i, cap):
    soc_pre = soc_ini + i / (cap * 3600)
    return soc_pre


# In[34]:


def dfgen(input_dict):
    """加载电流电压，初始soc以及参数查找表三个变量"""
    ui = np.zeros(input_dict['iu'].shape)
    # 电压放前面并取整数
    ui[:, 0] = np.round(input_dict['iu'][:, 1])
    # 电流放后面并取整数
    ui[:, 1] = np.round(input_dict['iu'][:, 0])
    soc_ah = input_dict['soc_ini']
    param = input_dict['param']
    cap = input_dict['cap']
    """逐时间步生成特征"""
    df = []
    for i in range(ui.shape[0]):
        soc_ah = ah(soc_ah, ui[i, 1], cap)
        index = int(np.round(soc_ah * 10))
        feature = param[index]
        features = np.concatenate((ui[i, :], soc_ah.reshape(-1), feature))
        df.append(features)
    df_out = np.array(df)
    return df_out

