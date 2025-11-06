#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import matplotlib.pyplot as plt


# 此函数表达式为：$hardtanh = \left\{\begin{matrix}
# -1 & x<-1\\
# x  & -1 \le x \le 1\\
# 1 & x>1 
# \end{matrix}\right.$

# In[2]:


#反正切函数
def vtanh(xin):
    yout = np.zeros(xin.shape)
    for i in range(xin.shape[0]):
        x = xin[i]
        if x < 0:
            if x < -16**2:
                y = -16**2
            else:
                y = x
        else:
            if x > 16**2:
                y = 16**2
            else:
                y = x
        yout[i] = y
    return yout    


# ### 以下为测试部分

# In[3]:


#定义浮点sigmoid模型
def hardtanh(x):
    x_temp = x.reshape(-1)
    if type(x) is np.ndarray:
        y_temp = x_temp.copy()
    else:
        y_temp = x_temp.clone()
    for i in range(len(x_temp)):
        if x_temp[i]<-1:
            y_temp[i] = -1
        else:
            if x_temp[i]>1:
                y_temp[i] = 1
            else:
                y_temp[i] = x_temp[i]
    y = y_temp.reshape(x.shape)
    return y


def hardtanh2(x):
    b1 = x < -1
    b2 = x > 1
    b3 = ~b1 & ~b2
    a1 = - torch.ones(x.shape)
    a2 = torch.ones(x.shape)
    a3 = x
    y = a1 * b1 + a2 * b2 + a3 * b3
    return y

def hardtanh3(x, scale):
    b1 = x < -1*scale
    b2 = x > 1*scale
    b3 = ~b1 & ~b2
    a1 = - np.ones(x.shape) * scale
    a2 = np.ones(x.shape) * scale
    a3 = x
    y = a1 * b1 + a2 * b2 + a3 * b3
    return y
# In[4]:


#定义缩放因子
scale = 2**8


# In[5]:


#生成一组浮点数自变量并定点化
x_flo = np.linspace(-10, 10, 2001)
x_fix = np.floor(x_flo * scale)


# In[6]:


#运行浮点模型
y = hardtanh2(x_flo)


# In[7]:


#运行定点模型并恢复浮点结果
yv = vtanh(x_fix)
yr = yv / scale


# In[8]:


#结果对比
#plt.figure()
#plt.plot(x_flo, yr)
#plt.plot(x_flo, y)
#plt.show()


# In[ ]:




