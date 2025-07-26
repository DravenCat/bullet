import numpy as np
import math

# 状态维度：8个特征
STATE_DIM = 11
# 更新后的动作维度
ACTION_DIM = 6561

# 目标位置 [x, y, z] - 示例值，实际应在训练循环中设置
TARGET_POSITION = np.array([5.0, 5.0, 2.5])


def action_mapping(action_idx):
    """将离散动作索引映射到控制参数（每个参数9种选择）"""
    # 动作索引分解为四个0-8的索引
    speed_idx = action_idx // (9 * 9 * 9)  # [0-8]
    action_idx %= (9 * 9 * 9)

    steer_idx = action_idx // (9 * 9)  # [0-8]
    action_idx %= (9 * 9)

    left_idx = action_idx // 9  # [0-8]
    right_idx = action_idx % 9  # [0-8]

    # 动作取值列表（每个参数9个值）
    speed_factor = np.linspace(0.75, 1.25, 9)[speed_idx]
    steer_angle = np.linspace(-0.05, 0.05, 9)[steer_idx]
    angle_left = [-0.05, -0.025, -0.0125, 0, 0.075, 0.135, 0.15, 0.165, 0.18][left_idx]
    angle_right = [-0.05, -0.025, -0.0125, 0, 0.075, 0.135, 0.15, 0.165, 0.18][right_idx]

    return speed_factor, steer_angle, angle_left, angle_right


# 获取当前状态 (包括目标位置信息)
def get_state(robot_id, p, target_pos):
    # 获取位置和方向
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    # 获取速度
    lin_vel, ang_vel = p.getBaseVelocity(robot_id)
    # 获取欧拉角
    euler = p.getEulerFromQuaternion(orn)

    # 计算目标差值
    dx = target_pos[0] - pos[0]
    dy = target_pos[1] - pos[1]
    dz = target_pos[2] - pos[2]

    # 计算到目标的距离
    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # 状态向量
    state = np.array([
        pos[2],  # Z位置（深度）
        lin_vel[0],  # X速度
        lin_vel[1],  # Y速度
        lin_vel[2],  # Z速度
        euler[0],  # 翻滚角 (roll)
        euler[1],  # 俯仰角 (pitch)
        euler[2],  # 偏航角 (yaw)
        dz,  # 与目标深度的差值
        dx,  # X方向目标差值
        dy,  # Y方向目标差值
        distance  # 到目标的欧氏距离
    ], dtype=np.float32)

    return state


# 计算奖励 (考虑目标位置)
def calculate_reward(state, prev_state):
    # 距离减少奖励：鼓励向目标移动
    if prev_state is not None:
        distance_reward = (prev_state[10] - state[10]) * 20  # 距离减少的奖励
    else:
        distance_reward = 0

    # 方向对齐奖励：鼓励面向目标
    target_direction = math.atan2(state[9], state[8])  # atan2(dy, dx)
    direction_diff = abs(state[6] - target_direction)  # 当前偏航角与目标方向的差值

    # 将方向差映射到[0, π]范围内
    if direction_diff > math.pi:
        direction_diff = 2 * math.pi - direction_diff

    direction_reward = -direction_diff * 5  # 方向差越小奖励越大

    # 姿态稳定奖励：减少翻滚和俯仰
    stability_reward = - (abs(state[4]) + abs(state[5])) * 2  # roll + pitch

    # 总奖励
    total_reward = distance_reward + direction_reward + stability_reward

    # 终止条件：到达目标或姿态失控
    done = (state[10] < 0.3) or (abs(state[4]) > (math.pi) / 2) or (abs(state[5]) > (math.pi) / 2)

    return total_reward, done