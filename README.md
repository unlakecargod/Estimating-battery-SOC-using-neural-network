# Estimating-battery-SOC-using-neural-network
一个使用神经网络估计电池剩余电量的数字芯片设计

Author: pengyiwen  
Email: 1125605564@qq.com  
Description: 这是我2024年的毕业设计项目，不小心把虚拟机删了，导致之前在eda仿真的文件全都无了，只剩下源码在主力机上的备份，不确定保存的是不是最终的版本，也没啥技术含量，都上传到github。

整体架构图：  
![lstm](https://github.com/user-attachments/assets/1efd086a-5913-4fc9-a2dc-6b9898e5b1e1)

本仓库包括整个设计：  
1. python源码，包括多种神经网络及混合算法的训练及推理模型，分别是LSTM网络、VGG16网络和全连接网络，当时不会用git管理版本，所以很乱，有用没用的现在也搞不清了，后面有空再重新整理  
2. verilog源码，在vcs+verdi仿真验证过，试了下在fpga导入可以分析出schematic
3. fpga项目，verilog源码移植到zynq7020开发板上验证过的，打开发现有些ip找不到的报错，移植的时候将源码中的ram替换为fpga的blockram ip了，应该是换个vivado版本ip找不到了？  
4. 毕业论文，当时怎么跑的有些记不清了，论文应该有详细说明 = =

文档下载来源：  
https://www.ti.com.cn  
训练数据下载来源：  
https://calce.umd.edu/battery-data



