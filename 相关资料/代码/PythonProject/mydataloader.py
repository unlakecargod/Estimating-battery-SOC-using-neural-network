import torch.utils.data
from mydataset import Mydataset


def my_dataloder(train_x, train_y, test_x, test_y, bs, shuffle):
    """输入x,y,返回dl"""
    train_ds = Mydataset(train_x, train_y)
    test_ds = Mydataset(test_x, test_y)
    train_dl = torch.utils.data.DataLoader(train_ds,
                                           batch_size = bs,
                                           shuffle = shuffle)
    test_dl = torch.utils.data.DataLoader(test_ds,
                                           batch_size=bs,
                                           shuffle=shuffle)
    return train_dl, test_dl