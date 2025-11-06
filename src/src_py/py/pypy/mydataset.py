import torch.utils.data

class Mydataset(torch.utils.data.Dataset):
    """输入x,y,返回ds"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __getitem__(self, item):
        return self.features[item], self.labels[item]
    def __len__(self):
        return len(self.features)