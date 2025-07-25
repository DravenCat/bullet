import numpy as np


# 状态维度：8个特征
STATE_DIM = 8
# 动作维度：离散动作空间大小（4个动作，每个动作3个选择 = 81种组合）
ACTION_DIM = 81


# 动作映射：将离散动作索引映射到具体的控制参数
def action_mapping(action_idx):
    """将离散动作索引映射到控制参数"""
    # 动作空间分解：
    # speed_factor: [0.5, 1.0, 1.5] - 3种
    # steer_angle: [-0.1, 0, 0.1] - 3种
    # angle_left: [-0.1, 0.15, 0.2] - 3种
    # angle_right: [-0.1, 0.15, 0.2] - 3种
    # 总动作数 = 3*3*3*3 = 81

    # 计算各个动作的索引
    speed_idx = action_idx // 27
    action_idx %= 27
    steer_idx = action_idx // 9
    action_idx %= 9
    left_idx = action_idx // 3
    right_idx = action_idx % 3

    # 映射到具体值
    speed_factor = [0.5, 1.0, 1.5][speed_idx]
    steer_angle = [-0.1, 0, 0.1][steer_idx]
    angle_left = [-0.1, 0.15, 0.2][left_idx]
    angle_right = [-0.1, 0.15, 0.2][right_idx]

    return speed_factor, steer_angle, angle_left, angle_right


# 获取当前状态
def get_state(robot_id, p):
    # 获取位置和方向
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    # 获取速度
    lin_vel, ang_vel = p.getBaseVelocity(robot_id)
    # 获取欧拉角
    euler = p.getEulerFromQuaternion(orn)

    # 状态向量
    state = np.array([
        pos[2],  # Z位置（深度）
        lin_vel[0],  # X速度
        lin_vel[1],  # Y速度
        lin_vel[2],  # Z速度
        euler[0],  # 翻滚角 (roll)
        euler[1],  # 俯仰角 (pitch)
        euler[2],  # 偏航角 (yaw)
        pos[2] - 1.5  # 与目标深度的差值
    ], dtype=np.float32)

    return state


# 计算奖励
def calculate_reward(state, prev_state):
    # 基础奖励：前进速度（鼓励向前移动）
    forward_reward = state[1] * 10  # X轴速度

    # 深度惩罚：与目标深度(1.5m)的差值
    depth_penalty = -abs(state[0] - 1.5) * 5

    # 姿态稳定奖励：减少翻滚和俯仰
    stability_reward = - (abs(state[4]) + abs(state[5])) * 2  # roll + pitch

    # 方向稳定惩罚：偏航角变化
    if prev_state is not None:
        yaw_change = abs(state[6] - prev_state[6])
        direction_penalty = -yaw_change * 3
    else:
        direction_penalty = 0

    # 总奖励
    total_reward = forward_reward + depth_penalty + stability_reward + direction_penalty

    # 终止条件：如果深度偏差太大或姿态失控
    done = (abs(state[0] - 1.5) > 0.5) or (abs(state[4]) > 0.8) or (abs(state[5]) > 0.8)

    return total_reward, done