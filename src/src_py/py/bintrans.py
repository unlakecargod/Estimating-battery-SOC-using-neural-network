#!/usr/bin/env python
# coding: utf-8

# In[2]:


def bnr2dec(data: str) -> int:
    """二进制补码(BNR)转十进制数

    Args:
        data (str): 二进制补码字符串，如"100101100"

    Raises:
        TypeError: 输入非字符串！
        ValueError: 输入非二进制字符串！

    Returns:
        int: 十进制数
    """

    if not isinstance(data, str):
        raise TypeError("输入非字符串！")

    for num in data:
        if num not in ["0", "1"]:
            raise ValueError("输入非二进制字符串！")

    # 正整数原码与补码相同
    if data.startswith("0"):
        dec = int(data, 2)
    else:
        # 补码-->反码-->原码
        dec = int(data[1:], 2) - 0x01
        dec = -(~dec & int("0b" + "1" * (len(data) - 1), 2))

        # Note: 整数在计算机中以补码的形式存储，所以按位取反运算符(~)会将补码的符号位也取反，故用&运算符清零特性(任何数与0相与都为0，与1相与保持不变)
        #       即dec & 0b1111...(位数取决于dec的位数)，0b1111在计算机中的存储的补码为01111，dec与之相与后符号位被清零，即可实现非计算机层面的按位取反

    return dec


# In[3]:


def dec2bnr(dec: int, lenth: int = 16) -> str:
    """十进制数转指定长度二进制补码(BNR)

    Args:
        dec (int): 十进制数
        lenth (int, optional): 指定长度(正数高位补0，负数高位补1). Defaults to 16.

    Raises:
        TypeError: 输入非十进制整数！
        OverflowError: 输入十进制整数过大，超过指定补码长度

    Returns:
        str: 返回二进制补码字符串
    """

    if not isinstance(dec, int):
        raise TypeError("输入非十进制整数！")

    # 计算十进制转化为二进制后的位数
    digits = (len(bin(dec)) - 3) if dec < 0 else (len(bin(dec)) - 2)

    if digits >= lenth:
        raise OverflowError("输入十进制整数过大，超过指定补码长度")

    # Note: dec & 相同位数的0b111...强制转换为补码形式
    pattern = f"{dec & int('0b' + '1' * lenth, 2):0{lenth}b}"

    return pattern


# In[4]:


def flo2brn(flo, scale):
    """浮点数转换成二进制补码"""
    temp = int(flo * scale)
    stri = dec2bnr(temp)
    print(stri)
    return stri


# In[5]:


def any2bin(hex, num):
    """任意进制转换成二进制"""
    num = int(hex, num)
    stri = bin(num)[2:]
    print(stri)
    return stri


# In[ ]:


def any2dec(hex, num, wid):
    stri = any2bin(hex, num)
    if len(stri) <= wid:
        stri_long = '0'*(wid-len(stri)) + stri
    else:
        print('too long!')
    dec = bnr2dec(stri_long)
    print(dec)
    return dec

