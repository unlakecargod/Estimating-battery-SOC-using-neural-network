def norm(x, norm_dict):
    x_n = (x - norm_dict['min']) / (norm_dict['max'] - norm_dict['min'])
    return x_n
