import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from model import DQNModel


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # 将 dones 转换为整数类型 (0 或 1)
        dones_int = [int(d) for d in dones]

        return (
            torch.tensor(np.array(states)),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states)),
            torch.tensor(dones_int, dtype=torch.uint8)
        )

    def __len__(self):
        return len(self.buffer)


class FishDQNAgent:
    def __init__(self, obs_dim, act_dim,
                 lr=0.001, gamma=0.99,
                 e_greed=0.1, e_greed_decrement=0,
                 target_update_freq=200,
                 buffer_size=10000,
                 batch_size=64):

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        # 上一步动作（初始化为 None，表示不限）
        self.prev_action_idx = None
        # 创建Q网络和目标网络
        self.q_net = DQNModel(obs_dim, act_dim)
        self.target_net = DQNModel(obs_dim, act_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # 目标网络不计算梯度

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.global_step = 0

    def sync_target(self):
        """同步目标网络参数"""
        self.target_net.load_state_dict(self.q_net.state_dict())
    def _decompose(self, action_idx):
        """把动作索引拆成四个 0-8 的子索引"""
        speed = action_idx // (9*9*9)
        rem = action_idx % (9*9*9)
        steer = rem // (9*9)
        rem = rem % (9*9)
        left = rem // 9
        right = rem % 9
        return speed, steer, left, right
    def _is_adjacent(self, a1, a2, dims_to_restrict=[0,1,2,3]):
        """仅当子索引在规定维度上相差 ≤1 时才视作相邻"""
        idx1 = self._decompose(a1)
        idx2 = self._decompose(a2)
        for d in dims_to_restrict:
           if abs(idx1[d] - idx2[d]) > 1:
                return False
        return True
    def _valid_actions(self):
        """返回与上一步相邻的所有合法动作索引"""
        if self.prev_action_idx is None:
            return list(range(self.act_dim))
        return [
            a for a in range(self.act_dim)
            if self._is_adjacent(a, self.prev_action_idx)
       ]
    def sample(self, state):
        """ε-greedy策略选择动作"""
        valid = self._valid_actions()
        if np.random.uniform(0, 1) < self.e_greed:
            action = random.choice(valid)
        else:
            # 在 valid 动作里做贪心选
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q = self.q_net(state_t).squeeze(0).cpu().numpy()
            # 把不合法的位置设为 -inf
            mask = np.full_like(q, -np.inf)
            mask[valid] = q[valid]
            action = int(mask.argmax())
        # 逐渐减小探索率
        if self.e_greed_decrement > 0:
            self.e_greed = max(0.01, self.e_greed - self.e_greed_decrement)
        # 更新 prev_action_idx
        self.prev_action_idx = action
        return action

    def predict(self, state):
        """预测最佳动作"""
        valid = self._valid_actions()
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(state_t).squeeze(0).cpu().numpy()
        mask = np.full_like(q, -np.inf)
        mask[valid] = q[valid]
        action = int(mask.argmax())
        self.prev_action_idx = action
        return action

    def store_experience(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self):
        """从经验回放中学习"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # 从缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 计算当前Q值
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q_values * (1 - dones)

        # 计算损失
        loss = nn.MSELoss()(q_values, target_q)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期同步目标网络
        if self.global_step % self.target_update_freq == 0:
            self.sync_target()
        self.global_step += 1

        return loss.item()