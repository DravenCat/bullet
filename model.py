import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(DQNModel, self).__init__()
        self.hid1_size = 128
        self.hid2_size = 128

        # 定义三层全连接层 (fc)
        self.fc1 = nn.Linear(obs_dim, self.hid1_size)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(self.hid1_size, self.hid2_size)  # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(self.hid2_size, act_dim)  # 隐藏层2到输出层

    def forward(self, obs):
        # 前向传播，对应PARL中的value方法
        h1 = F.relu(self.fc1(obs))  # 第一层 + ReLU激活
        h2 = F.relu(self.fc2(h1))  # 第二层 + ReLU激活
        Q = self.fc3(h2)  # 第三层输出，无激活函数
        return Q