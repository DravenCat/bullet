import numpy as np
import math


EST_VOL = 6.85e-5  # m³  – rough
BUOYANCY = 1000 * 9.81 * EST_VOL  # N   – ρ g V


def get_buoyancy_force(p, body_id):
    """动态计算浮力，基于物体质量和重力加速度"""
    # 获取物体质量
    mass = p.getDynamicsInfo(body_id, -1)[0]

    # 获取重力加速度
    gravity = -9.81  # 通常为-9.8或-10

    # 计算浮力 (ρ_water * g * V_displaced)
    # 假设鱼体密度接近水密度，浮力≈重力
    buoyancy = -mass * gravity * 2.2  # 1.0表示中性浮力

    return buoyancy


def apply_buoyancy(p, box_id):
    """ Buoyancy helper now has a valid boxId to work with"""
    buoyancy = get_buoyancy_force(p, box_id)
    com_pos, _ = p.getBasePositionAndOrientation(box_id)
    p.applyExternalForce(box_id, -1, [0, 0, buoyancy],
                         com_pos, p.WORLD_FRAME)
    return buoyancy


def apply_water_drag(p, box_id):
    """Optional quadratic drag helper"""
    lin_vel, ang_vel = p.getBaseVelocity(box_id)
    v = np.array(lin_vel)
    w = np.array(ang_vel)
    # translational drag
    speed = np.linalg.norm(v)
    if speed > 1e-6:
        F = -0.5 * 1000 * 0.8 * 0.005 * speed * v
        p.applyExternalForce(box_id, -1, F.tolist(), [0, 0, 0], p.WORLD_FRAME)
    # rotational drag
    rate = np.linalg.norm(w)
    if rate > 1e-6:
        T = -0.5 * 1000 * 0.04 * 1e-4 * rate * w
        p.applyExternalTorque(box_id, -1, T.tolist(), p.WORLD_FRAME)


def apply_tail_thrust(p, body_id, joint_id,
                      fin_len=0.08,      # distance hinge → tip  (m)
                      chord=0.025,       # fin depth            (m)
                      rho=1000,          # water density        (kg/m³)
                      C_T=2000,          # thrust coefficient   (‑)
                      C_T_steer=300,     # steering coefficient (‑)
                      damping_factor=0.3):  # damping factor for sway reduction
    """ Rear fin thrust helper"""
    # joint kinematics
    theta, omega = p.getJointState(body_id, joint_id)[:2]
    v_tip = abs(omega) * fin_len  # lateral speed at tip

    # 计算基本推力大小
    thrust_base = -0.5 * rho * chord * fin_len * v_tip ** 2

    # 分解为前进推力和转向分量
    # 前进推力与尾鳍偏角余弦成正比
    forward_thrust = C_T * thrust_base * abs(math.sin(theta)) * math.cos(theta)

    # 转向推力与尾鳍偏角正弦成正比
    steer_thrust = C_T_steer * thrust_base * math.sin(theta)

    # body‑x axis in world frame
    _, orn = p.getBasePositionAndOrientation(body_id)
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    forward_vector = rotation_matrix[:, 0]  # 第一列为前进方向
    lateral_vector = rotation_matrix[:, 1]  # 侧向方向 (y轴)

    # 计算推力向量
    forward_force = forward_thrust * forward_vector
    lateral_force = steer_thrust * lateral_vector
    total_force = forward_force + lateral_force

    # 施加推力 - 在尾鳍位置而非重心
    link_state = p.getLinkState(body_id, joint_id)
    tail_pos = link_state[0]  # 尾鳍位置

    # 计算转向力矩 (绕z轴旋转)
    # 力臂长度：从质心到尾鳍位置的距离
    com_pos, _ = p.getBasePositionAndOrientation(body_id)
    moment_arm = np.array(tail_pos) - np.array(com_pos)
    moment_arm_xy = moment_arm.copy()
    moment_arm_xy[2] = 0  # 只考虑xy平面内的力臂

    # 计算力矩 (力矩 = 力臂 × 力)
    torque = np.cross(moment_arm_xy, lateral_force)
    steering_torque = np.array([0, 0, torque[2]])  # 只保留z轴力矩

    # 施加力和力矩
    p.applyExternalForce(body_id, -1,
                         total_force.tolist(),
                         tail_pos,
                         p.WORLD_FRAME)

    # 添加侧向阻尼减少摇摆
    lin_vel, ang_vel = p.getBaseVelocity(body_id)
    lateral_vel = np.dot(lin_vel, lateral_vector)
    damping_force = -damping_factor * lateral_vel * lateral_vector
    p.applyExternalForce(body_id, -1, damping_force.tolist(), tail_pos, p.WORLD_FRAME)

    return forward_thrust, steer_thrust


def apply_fin_lift(p, body_id, joint_id,
                   fin_len=0.04,  # 鳍的长度 (m)
                   chord=0.025,  # 鳍的弦长 (m)
                   rho=1000,  # 水的密度 (kg/m³)
                   C_L_base=400,  # 基础升力系数
                   max_angle=0.2618):  # 最大攻角 (15度≈0.2618弧度)
    """
    为胸鳍(左鳍/右鳍)提供升力（忽略关节角速度）
    - 升力与鱼的移动速度和鳍的攻角相关
    - 恒速前进时鳍保持固定角度
    - 升力方向主要沿鱼体垂直方向
    """
    # Get current angle of the fin
    theta = p.getJointState(body_id, joint_id)[0]

    # -15 < effective angle /, 15
    effective_angle = max(min(theta, max_angle), -max_angle)

    # Get the robot speed
    lin_vel, _ = p.getBaseVelocity(body_id)
    fish_speed = np.linalg.norm(lin_vel)

    # 获取鱼体方向
    _, quat = p.getBasePositionAndOrientation(body_id)
    rot_matrix = p.getMatrixFromQuaternion(quat)
    body_x = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])  # 前进方向
    body_z = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])  # 垂直方向

    # 计算前进方向的速度分量（投影到鱼体x轴）
    forward_speed = np.dot(lin_vel, body_x)

    # 升力大小计算（与速度平方和攻角正弦成正比）
    lift_magnitude = 0.5 * rho * C_L_base * (fin_len * chord) * \
                     (forward_speed ** 2) * math.sin(abs(effective_angle))

    # 升力方向主要沿鱼体垂直方向
    lift_direction = body_z.flatten()

    # 根据攻角符号调整方向（负角度产生正升力）
    lift_sign = -1 if effective_angle > 0 else 1
    lift_force = lift_sign * lift_magnitude * lift_direction

    # 施加升力（在鳍的关节位置）
    joint_pos = p.getLinkState(body_id, joint_id)[0]
    p.applyExternalForce(body_id, joint_id,
        forceObj=lift_force.tolist(),
        posObj=joint_pos,
        flags=p.WORLD_FRAME
    )

    return lift_magnitude