import torch
import torch.nn as nn
import math
from vsigmoid import hardsigmoid2
from vtanh import hardtanh2


class Lstm_2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Lstm_2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 输入门i_t
        # 第一层
        self.U_i1 = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_i1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i1 = nn.Parameter(torch.Tensor(hidden_size))
        # 第二层
        self.U_i2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V_i2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i2 = nn.Parameter(torch.Tensor(hidden_size))

        # f_t
        # 第一层
        self.U_f1 = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_f1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f1 = nn.Parameter(torch.Tensor(hidden_size))
        # 第二层
        self.U_f2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V_f2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f2 = nn.Parameter(torch.Tensor(hidden_size))

        # c_t
        # 第一层
        self.U_c1 = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_c1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c1 = nn.Parameter(torch.Tensor(hidden_size))
        # 第二层
        self.U_c2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V_c2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c2 = nn.Parameter(torch.Tensor(hidden_size))

        # o_t
        # 第一层
        self.U_o1 = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_o1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o1 = nn.Parameter(torch.Tensor(hidden_size))
        # 第二层
        self.U_o2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V_o2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o2 = nn.Parameter(torch.Tensor(hidden_size))

        #线性层
        self.fc = nn.Linear(hidden_size, 1)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, lstm_input, init_states=None):

        batch_size, seq_len = lstm_input.size(0), lstm_input.size(1)

        if init_states is None:
            h_t1, c_t1, h_t2, c_t2 = (
                torch.zeros(batch_size, self.hidden_size),
                torch.zeros(batch_size, self.hidden_size),
                torch.zeros(batch_size, self.hidden_size),
                torch.zeros(batch_size, self.hidden_size)
            )
        else:
            h_t1, c_t1, h_t2, c_t2 = init_states

        for t in range(seq_len):
            # 更新门组件及内部候选状态（Tips:Pytorch中@用于矩阵相乘，*用于逐个元素相乘）
            # 第一层计算
            x_t = lstm_input[:, t, :]
            i_t1 = hardsigmoid2(x_t @ self.U_i1 + h_t1 @ self.V_i1 + self.b_i1)
            f_t1 = hardsigmoid2(x_t @ self.U_f1 + h_t1 @ self.V_f1 + self.b_f1)
            g_t1 = hardtanh2(x_t @ self.U_c1 + h_t1 @ self.V_c1 + self.b_c1)
            o_t1 = hardsigmoid2(x_t @ self.U_o1 + h_t1 @ self.V_o1 + self.b_o1)
            c_t1 = f_t1 * c_t1 + i_t1 * g_t1
            h_t1 = o_t1 * hardtanh2(c_t1)
            # 第二层计算
            i_t2 = hardsigmoid2(h_t1 @ self.U_i2 + h_t2 @ self.V_i2 + self.b_i2)
            f_t2 = hardsigmoid2(h_t1 @ self.U_f2 + h_t2 @ self.V_f2 + self.b_f2)
            g_t2 = hardtanh2(h_t1 @ self.U_c2 + h_t2 @ self.V_c2 + self.b_c2)
            o_t2 = hardsigmoid2(h_t1 @ self.U_o2 + h_t2 @ self.V_o2 + self.b_o2)
            c_t2 = f_t2 * c_t2 + i_t2 * g_t2
            h_t2 = o_t2 * hardtanh2(c_t2)

        output = self.fc(h_t2)

        return output.squeeze()
