#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import torch
import matplotlib.pyplot as plt


# 此函数表达式为：$hardsigmoid = \left\{\begin{matrix}
# 0 & x<-2.5\\
# 0.2 \times x + 0.5  & -2.5 \le x \le 2.5\\
# 1 & x>2.5 
# \end{matrix}\right.$

# In[16]:


#定义verilog中的sigmoid函数
def vsigmoid(xin):
    yout = np.zeros(xin.shape)
    for i in range(xin.shape[0]):
        x = xin[i]
        s1 = x + 2*(16**2)
        if x < 0:
            if x < -2*(16**2):
                y = 0
            else:
                y = np.floor(s1 / 4)
        else:
            if x > 2*(16**2):
                y = 16**2
            else:
                y = np.floor(s1 / 4)
        yout[i] = y
    return yout


# ### 以下为测试部分

# In[3]:


#定义浮点sigmoid模型
def hardsigmoid(x):
    x_temp = x.reshape(-1)
    if type(x) is np.ndarray:
        y_temp = x_temp.copy()
    else:
        y_temp = x_temp.clone()
    for j in range(len(x_temp)):
            if x_temp[j]<-2.5:
                y_temp[j] = 0
            else:
                if x_temp[j]>2.5:
                    y_temp[j] = 1
                else:
                    y_temp[j] = 0.2*x_temp[j] + 0.5
    y = y_temp.reshape(x.shape)
    return y


def hardsigmoid2(x):
    b1 = x < -2
    b2 = x > 2
    b3 = ~b1 & ~b2
    a1 = torch.zeros(x.shape)
    a2 = torch.ones(x.shape)
    a3 = 0.25 * x + 0.5
    y = a1 * b1 + a2 * b2 + a3 * b3
    return y

def hardsigmoid3(x, scale):
    b1 = x < -2*scale
    b2 = x > 2*scale
    b3 = ~b1 & ~b2
    a1 = np.zeros(x.shape) * scale
    a2 = np.ones(x.shape) * scale
    a3 = np.floor((x + 2 * scale) / 4)
    y = a1 * b1 + a2 * b2 + a3 * b3
    return y
# In[14]:



#定义缩放因子
scale = 2**8


# In[11]:


#生成一组浮点数自变量并定点化
x_flo = np.linspace(-10, 10, 2001)
x_fix = np.floor(x_flo * scale)


# In[12]:


#运行浮点模型
y = hardsigmoid2(x_flo)


# In[17]:


#运行定点模型并恢复浮点结果
yv = vsigmoid(x_fix)
yr = yv / scale


# In[24]:


#结果对比
#plt.figure()
#plt.plot(x_flo, yr)
#plt.plot(x_flo, y)
#plt.show()


# In[ ]:




