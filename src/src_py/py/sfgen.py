#!/usr/bin/env python
# coding: utf-8

# 特征由两部分组成[u, i][features]  
# ui会随着时间步变化，而后面的特征仅仅重复时间步的次数  
# 生成为归一化的几个静态特征


import numpy as np
import pandas as pd

def sfgen(soc, param_path, seq_len):
    param = pd.read_excel(param_path, header=None).values
    index = int(np.round(soc*10))
    feature = param[index, :]
    feature_ = np.insert(feature, 0, soc, axis=0)
    features = np.repeat(feature_.reshape(1, -1), seq_len, axis=0)
    return features




