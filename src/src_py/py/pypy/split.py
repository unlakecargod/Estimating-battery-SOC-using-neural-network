import numpy as np


def split(data, seq_len):
    """交替划分"""
    data1 = []
    for i in range(len(data) - seq_len):
        data1.append(data[i: i + seq_len, :])
    data2 = np.array(data1)
    """划分x和y"""
    x = np.round(data2[:, :, 0:-1] * 1000)
    y = data2[:, -1, -1]
    data_dict = {'iu': x, 'soc_real': y}
    return data_dict
