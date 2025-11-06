import numpy as np
import vtanh
import vsigmoid

class Vlstm():
    def __init__(self, x, parameters, scale):
        self.scale = scale
        self.x = x
        self.param = parameters
        self.xfix = np.trunc(x * scale)
        self.U_i1 = np.trunc(parameters['U_i1'] * scale)
        self.V_i1 = np.trunc(parameters['V_i1'] * scale)
        self.b_i1 = np.trunc(parameters['b_i1'] * scale)
        self.U_i2 = np.trunc(parameters['U_i2'] * scale)
        self.V_i2 = np.trunc(parameters['V_i2'] * scale)
        self.b_i2 = np.trunc(parameters['b_i2'] * scale)
        self.U_f1 = np.trunc(parameters['U_f1'] * scale)
        self.V_f1 = np.trunc(parameters['V_f1'] * scale)
        self.b_f1 = np.trunc(parameters['b_f1'] * scale)
        self.U_f2 = np.trunc(parameters['U_f2'] * scale)
        self.V_f2 = np.trunc(parameters['V_f2'] * scale)
        self.b_f2 = np.trunc(parameters['b_f2'] * scale)
        self.U_c1 = np.trunc(parameters['U_c1'] * scale)
        self.V_c1 = np.trunc(parameters['V_c1'] * scale)
        self.b_c1 = np.trunc(parameters['b_c1'] * scale)
        self.U_c2 = np.trunc(parameters['U_c2'] * scale)
        self.V_c2 = np.trunc(parameters['V_c2'] * scale)
        self.b_c2 = np.trunc(parameters['b_c2'] * scale)
        self.U_o1 = np.trunc(parameters['U_o1'] * scale)
        self.V_o1 = np.trunc(parameters['V_o1'] * scale)
        self.b_o1 = np.trunc(parameters['b_o1'] * scale)
        self.U_o2 = np.trunc(parameters['U_o2'] * scale)
        self.V_o2 = np.trunc(parameters['V_o2'] * scale)
        self.b_o2 = np.trunc(parameters['b_o2'] * scale)
        self.fcweight = np.trunc(parameters['fc.weight'] * scale).reshape(-1)
        self.fcbias = np.trunc(parameters['fc.bias'] * scale)
        self.h1 = np.trunc(np.zeros((self.V_i1.shape[0])))
        self.c1 = np.trunc(np.zeros((self.V_i1.shape[0])))
        self.h2 = np.trunc(np.zeros((self.V_i1.shape[0])))
        self.c2 = np.trunc(np.zeros((self.V_i1.shape[0])))

    def gate(self, x_fix, wx_fix, h_fix, wh_fix, b_fix):
        xw_fix_temp = np.zeros(h_fix.shape)
        hw_fix_temp = np.zeros(h_fix.shape)
        for i in range(wx_fix.shape[1]):
            for j in range(x_fix.shape[0]):
                xw_temp = np.floor(wx_fix.T[i, j] * x_fix[j] / self.scale)
                xw_fix_temp[i] = xw_fix_temp[i] + xw_temp
            for k in range(wh_fix.shape[1]):
                hw_temp = np.floor(wh_fix.T[i, k] * h_fix[k] / self.scale)
                hw_fix_temp[i] = hw_fix_temp[i] + hw_temp
        gate_o = xw_fix_temp + hw_fix_temp + b_fix
        return gate_o

    def lstm1(self, x):
        gate_i = self.gate(x, self.U_i1, self.h1, self.V_i1, self.b_i1)
        gate_f = self.gate(x, self.U_f1, self.h1, self.V_f1, self.b_f1)
        gate_g = self.gate(x, self.U_c1, self.h1, self.V_c1, self.b_c1)
        gate_o = self.gate(x, self.U_o1, self.h1, self.V_o1, self.b_o1)
        gate_i_act = vsigmoid.hardsigmoid3(gate_i, self.scale)
        gate_f_act = vsigmoid.hardsigmoid3(gate_f, self.scale)
        gate_g_act = vtanh.hardtanh3(gate_g, self.scale)
        gate_o_act = vsigmoid.hardsigmoid3(gate_o, self.scale)
        ixg = np.floor(gate_i_act * gate_g_act / self.scale)
        fxc = np.floor(gate_f_act * self.c1 / self.scale)
        c_new = ixg + fxc
        c_nwe_act = vtanh.hardtanh3(c_new, self.scale)
        h_new = np.floor(gate_o_act * c_nwe_act / self.scale)
        lstm_result = locals()
        return lstm_result

    def lstm2(self):
        gate_i = self.gate(self.h1, self.U_i2, self.h2, self.V_i2, self.b_i2)
        gate_f = self.gate(self.h1, self.U_f2, self.h2, self.V_f2, self.b_f2)
        gate_g = self.gate(self.h1, self.U_c2, self.h2, self.V_c2, self.b_c2)
        gate_o = self.gate(self.h1, self.U_o2, self.h2, self.V_o2, self.b_o2)
        gate_i_act = vsigmoid.hardsigmoid3(gate_i, self.scale)
        gate_f_act = vsigmoid.hardsigmoid3(gate_f, self.scale)
        gate_g_act = vtanh.hardtanh3(gate_g, self.scale)
        gate_o_act = vsigmoid.hardsigmoid3(gate_o, self.scale)
        ixg = np.floor(gate_i_act * gate_g_act / self.scale)
        fxc = np.floor(gate_f_act * self.c2 / self.scale)
        c_new = ixg + fxc
        c_nwe_act = vtanh.hardtanh3(c_new, self.scale)
        h_new = np.floor(gate_o_act * c_nwe_act / self.scale)
        lstm_result = locals()
        return lstm_result

    def linear(self):
        xw = self.fcbias
        for i in range(self.h2.shape[0]):
            wxh = np.floor(self.h2[i] * self.fcweight[i] / self.scale)
            xw = xw + wxh
        return xw

    def forward(self, seq_len):
        # 用一个gate保存所有的gate变量，维度从左至右依次是：
        # 30时间步、2lstm层、4个门ifgo、64个隐藏单元
        gate = np.zeros((30, 2, 4, 64))

        c = np.zeros((30, 2, 64))
        h = np.zeros((30, 2, 64))

        for i in range(seq_len):
            result_l1 = self.lstm1(self.xfix[i, :])
            self.h1 = result_l1['h_new']
            self.c1 = result_l1['c_new']
            result_l2 = self.lstm2()
            self.h2 = result_l2['h_new']
            self.c2 = result_l2['c_new']
            # 记录门单元的值
            gate[i, 0, 0, :] = result_l1['gate_i']
            gate[i, 0, 1, :] = result_l1['gate_f']
            gate[i, 0, 2, :] = result_l1['gate_g']
            gate[i, 0, 3, :] = result_l1['gate_o']
            gate[i, 1, 0, :] = result_l2['gate_i']
            gate[i, 1, 1, :] = result_l2['gate_f']
            gate[i, 1, 2, :] = result_l2['gate_g']
            gate[i, 1, 3, :] = result_l2['gate_o']
            # 记录中间变量的值
            h[i, 0, :] = result_l1['h_new']
            c[i, 0, :] = result_l1['c_new']
            h[i, 1, :] = result_l2['h_new']
            c[i, 1, :] = result_l2['c_new']
            print(f'时间步：{i}')
        result = self.linear()
        all_result = locals()
        return all_result


class Plstm():
    def __init__(self, x, parameters):
        self.x = x
        self.param = parameters
        self.U_i1 = parameters['U_i1']
        self.V_i1 = parameters['V_i1']
        self.b_i1 = parameters['b_i1']
        self.U_i2 = parameters['U_i2']
        self.V_i2 = parameters['V_i2']
        self.b_i2 = parameters['b_i2']
        self.U_f1 = parameters['U_f1']
        self.V_f1 = parameters['V_f1']
        self.b_f1 = parameters['b_f1']
        self.U_f2 = parameters['U_f2']
        self.V_f2 = parameters['V_f2']
        self.b_f2 = parameters['b_f2']
        self.U_c1 = parameters['U_c1']
        self.V_c1 = parameters['V_c1']
        self.b_c1 = parameters['b_c1']
        self.U_c2 = parameters['U_c2']
        self.V_c2 = parameters['V_c2']
        self.b_c2 = parameters['b_c2']
        self.U_o1 = parameters['U_o1']
        self.V_o1 = parameters['V_o1']
        self.b_o1 = parameters['b_o1']
        self.U_o2 = parameters['U_o2']
        self.V_o2 = parameters['V_o2']
        self.b_o2 = parameters['b_o2']
        self.fcweight = parameters['fc.weight'].reshape(-1)
        self.fcbias = parameters['fc.bias']
        self.h1 = np.zeros((self.V_i1.shape[0]))
        self.c1 = np.zeros((self.V_i1.shape[0]))
        self.h2 = np.zeros((self.V_i1.shape[0]))
        self.c2 = np.zeros((self.V_i1.shape[0]))

    def gate(self, x, wx, h, wh, b):
        xws = np.zeros(h.shape)
        hws = np.zeros(h.shape)
        for i in range(wx.shape[1]):
            for j in range(x.shape[0]):
                xw = wx.T[i, j] * x[j]
                xws[i] = xws[i] + xw
            for k in range(wh.shape[1]):
                hw = wh.T[i, k] * h[k]
                hws[i] = hws[i] + hw
        gate_o = xws + hws + b
        return gate_o

    def lstm1(self, x):
        gate_i = self.gate(x, self.U_i1, self.h1, self.V_i1, self.b_i1)
        gate_f = self.gate(x, self.U_f1, self.h1, self.V_f1, self.b_f1)
        gate_g = self.gate(x, self.U_c1, self.h1, self.V_c1, self.b_c1)
        gate_o = self.gate(x, self.U_o1, self.h1, self.V_o1, self.b_o1)
        gate_i_act = vsigmoid.hardsigmoid2(gate_i).numpy()
        gate_f_act = vsigmoid.hardsigmoid2(gate_f).numpy()
        gate_g_act = vtanh.hardtanh2(gate_g).numpy()
        gate_o_act = vsigmoid.hardsigmoid2(gate_o).numpy()
        ixg = gate_i_act * gate_g_act
        fxc = gate_f_act * self.c1
        c_new = ixg + fxc
        c_nwe_act = vtanh.hardtanh2(c_new).numpy()
        h_new = gate_o_act * c_nwe_act
        lstm_result = locals()
        return lstm_result

    def lstm2(self):
        gate_i = self.gate(self.h1, self.U_i2, self.h2, self.V_i2, self.b_i2)
        gate_f = self.gate(self.h1, self.U_f2, self.h2, self.V_f2, self.b_f2)
        gate_g = self.gate(self.h1, self.U_c2, self.h2, self.V_c2, self.b_c2)
        gate_o = self.gate(self.h1, self.U_o2, self.h2, self.V_o2, self.b_o2)
        gate_i_act = vsigmoid.hardsigmoid2(gate_i).numpy()
        gate_f_act = vsigmoid.hardsigmoid2(gate_f).numpy()
        gate_g_act = vtanh.hardtanh2(gate_g).numpy()
        gate_o_act = vsigmoid.hardsigmoid2(gate_o).numpy()
        ixg = gate_i_act * gate_g_act
        fxc = gate_f_act * self.c2
        c_new = ixg + fxc
        c_nwe_act = vtanh.hardtanh2(c_new).numpy()
        h_new = gate_o_act * c_nwe_act
        lstm_result = locals()
        return lstm_result

    def linear(self):
        xw = self.fcbias
        for i in range(self.h2.shape[0]):
            wxh = self.h2[i] * self.fcweight[i]
            xw = xw + wxh
        return xw

    def forward(self, seq_len):

        # 用一个gate保存所有的gate变量，维度从左至右依次是：
        # 30时间步、2lstm层、4个门ifgo、64个隐藏单元
        gate = np.zeros((30, 2, 4, 64))

        c = np.zeros((30, 2, 64))
        h = np.zeros((30, 2, 64))

        for i in range(seq_len):
            result_l1 = self.lstm1(self.x[i, :])
            self.h1 = result_l1['h_new']
            self.c1 = result_l1['c_new']
            result_l2 = self.lstm2()
            self.h2 = result_l2['h_new']
            self.c2 = result_l2['c_new']
            # 记录门单元的值
            gate[i, 0, 0, :] = result_l1['gate_i']
            gate[i, 0, 1, :] = result_l1['gate_f']
            gate[i, 0, 2, :] = result_l1['gate_g']
            gate[i, 0, 3, :] = result_l1['gate_o']
            gate[i, 1, 0, :] = result_l2['gate_i']
            gate[i, 1, 1, :] = result_l2['gate_f']
            gate[i, 1, 2, :] = result_l2['gate_g']
            gate[i, 1, 3, :] = result_l2['gate_o']
            # 记录中间变量的值
            h[i, 0, :] = result_l1['h_new']
            c[i, 0, :] = result_l1['c_new']
            h[i, 1, :] = result_l2['h_new']
            c[i, 1, :] = result_l2['c_new']

            print(f'时间步：{i}')
        result = self.linear()
        all_result = locals()
        return all_result