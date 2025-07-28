# encoding:utf-8
# test.py
from pybullet_envs.bullet import CartPoleBulletEnv
from stable_baselines3.dqn import DQN
from time import sleep
import pybullet as p

env = CartPoleBulletEnv(renders=True, discrete_actions=True)

model = DQN(policy="MlpPolicy", env=env)
model.load(
    path="./models/test_models",
    env=env
)



def evaluate_agent(agent, env, num_episodes=5):
    """
    评估智能体性能
    :param agent: 智能体实例
    :param env: 环境实例
    :param num_episodes: 评估回合数
    :return: 平均奖励
    """
    total_reward = 0.0
    agent.q_net.eval()  # 设置评估模式

    for _ in range(num_episodes):
        # 重置环境
        p.resetSimulation()
        target_pos = generate_random_target()
        plane, robot_id, rear_fin_id, left_fin_id, right_fin_id = load_environment(p)
        p.setRealTimeSimulation(0)

        # 初始化
        for _ in range(20):
            p.stepSimulation()
            time.sleep(1 / 240)

        p.resetBasePositionAndOrientation(robot_id, [0, 0, 1.5], p.getQuaternionFromEuler([0, 0, 0]))
        state = get_state(robot_id, p, target_pos)
        episode_reward = 0
        done = False
        step_count = 0

        while not done and step_count < 1000:
            action_idx = agent.predict(state)
            speed_factor, steer_angle, angle_left, angle_right = action_mapping(action_idx)

            # 执行动作
            angle_rear = steer_angle + 0.5 * speed_factor * math.sin(2 * math.pi * step_count / 100)
            p.setJointMotorControl2(robot_id, rear_fin_id, p.POSITION_CONTROL,
                                    targetPosition=angle_rear, positionGain=1.0, force=10)
            p.setJointMotorControl2(robot_id, left_fin_id, p.POSITION_CONTROL,
                                    targetPosition=angle_left, positionGain=1.0, force=10)
            p.setJointMotorControl2(robot_id, right_fin_id, p.POSITION_CONTROL,
                                    targetPosition=angle_right, positionGain=1.0, force=10)

            # 应用流体力学
            underwater.apply_buoyancy(p, robot_id)
            forward_thrust, steer_thrust = underwater.apply_tail_thrust(p, robot_id, rear_fin_id)
            left_lift = underwater.apply_fin_lift(p, robot_id, left_fin_id)
            right_lift = underwater.apply_fin_lift(p, robot_id, right_fin_id)
            p.stepSimulation()

            # 获取新状态和奖励
            next_state = get_state(robot_id, p, target_pos)
            reward, done = calculate_reward(next_state, state)
            episode_reward += reward

            state = next_state
            step_count += 1

        total_reward += episode_reward

    return total_reward / num_episodes


obs = env.reset()
while True:
    sleep(1 / 60)
    action, state = model.predict(observation=obs)
    print(action)
    obs, reward, done, info = env.step(action)
    if done:
        break