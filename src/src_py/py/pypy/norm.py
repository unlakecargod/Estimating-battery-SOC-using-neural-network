def norm(x, norm_dict):
    """归一化"""
    x_n = (x - norm_dict['min']) / (norm_dict['max'] - norm_dict['min'])
    return x_n
