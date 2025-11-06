#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[1]:


def add_noise_s(data, sigma_p, adp_noise):
    data_mea = np.zeros(data.shape)
    if (adp_noise):
        """模式1: 输入百分比, 数据本身的大小 * 百分比 = sigma值"""
        for j in range(len(data)):
            sigma = abs(data[j] * sigma_p)
            data_mea[j] = data[j] + np.random.normal(0, sigma)
    else:
        """模式0: 直接输入sigma值"""
        data_mea = data + np.random.normal(0, sigma_p, size=data.shape)
    return data_mea


# In[ ]:


def add_noise_iu(iu, sigma_iu, adp_iu):
    iu_noise = np.zeros(iu.shape)
    for j in range(2):
        iu_noise[:, j] = add_noise_s(iu[:, j], sigma_p=sigma_iu, adp_noise=adp_iu)
    return iu_noise

