import pybullet as p
import pybullet_data
import math
import time
from fishAgent import *
from trainFish import *
import underwater
import debug_params


load_existing_model = False

    
def setup_physical_engine(useGUI):
    """Set up and return a new physical engine"""
    if useGUI:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=1 / 240, numSolverIterations=100)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # GUI for debug mode
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)  # Disable CPU rendering
    return physicsClient


def camera_follow(robot_id):
    """Enable camera to follow the robot"""
    # aabb_min, aabb_max = p.getAABB(robot_id)
    # model_center = [(aabb_min[i] + aabb_max[i]) / 2 for i in range(3)]
    # model_size = [aabb_max[i] - aabb_min[i] for i in range(3)]
    # max_dim = max(model_size)
    #
    # camera_distance = max_dim * 1
    #
    # camera_position = [
    #     model_center[0],
    #     model_center[1] - camera_distance,
    #     model_center[2] + max_dim * 0.5
    # ]

    camera_target, _ = p.getBasePositionAndOrientation(robot_id)

    # Set Camera params
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=camera_target
    )


def load_environment(p):
    # --------------Load and render the robots ------------------------------------------
    # Load ground and fish BEFORE touching boxId‑dependent stuff
    plane = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("robots/Biomimetic_Fish_v8.urdf",
                          basePosition=[0, 0, 1.5],
                          baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))

    # Set linear / angular damping (simple “viscosity”)
    p.changeDynamics(robot_id, -1,
                     linearDamping=7.0,
                     angularDamping=10.0)

    # Get all the joints id
    available_joints_indexes = [i for i in range(p.getNumJoints(robot_id)) if
                                p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
    rear_fin_id = available_joints_indexes[0]
    left_fin_id = available_joints_indexes[1]
    right_fin_id = available_joints_indexes[2]

    # Initialize rear fin position
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=rear_fin_id,
        controlMode=p.POSITION_CONTROL,
        targetPosition=0,
        positionGain=1.0,
        force=10
    )

    # Initialize left fin position
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=left_fin_id,
        controlMode=p.POSITION_CONTROL,
        targetPosition=0,
        positionGain=1.0,
        force=10
    )

    # Initialize right fin position
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=right_fin_id,
        controlMode=p.POSITION_CONTROL,
        targetPosition=0,
        positionGain=1.0,
        force=10
    )

    return plane, robot_id, rear_fin_id, left_fin_id, right_fin_id


def main():
    # -------------Set up the physical engine -----------------------------------------
    physicalClient = setup_physical_engine(useGUI=True)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # Disable rendering before all the models being loaded
    # ------------------------------------------------------------------------
    # --------------Load and render the robots ------------------------------------------
    max_rear_radius = 0.5236  # radius limit for the rear fin (30 * π/180)
    max_front_radius = 0.2618  # radius limit for the front fin (15 * π/180)
    period_steps_rear = 100
    step_counter = 0

    plane, robot_id, rear_fin_id, left_fin_id, right_fin_id = load_environment(p)
    # Add reset button
    btn = p.addUserDebugParameter(
        paramName="reset",
        rangeMin=1,
        rangeMax=0,
        startValue=0
    )

    previous_btn_value = p.readUserDebugParameter(btn)
    # ------------------------------------------------------------------------

    # --------------- Initialization ---------------------------------------------
    # Run 10 steps to ensure the initialization
    p.setRealTimeSimulation(0)  # Disable real time simulation
    for _ in range(20):
        p.stepSimulation()
        time.sleep(1 / 240)

    # --------------- Simulation ---------------------------------------------
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # Rendering after all the models have been loaded
    # log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "log/fish_move.mp4")  # Start recording

    if load_existing_model:
        # Reset the position and orientation
        p.resetBasePositionAndOrientation(robot_id, [0, 0, 1.5], p.getQuaternionFromEuler([0, 0, 0]))

        # Run the simulation
        for step_i in range(10000):
            p.stepSimulation()

            # Set underwater environment
            underwater.apply_buoyancy(p, robot_id)
            # underwater.apply_water_drag(p, robot_id)  # optional

            # Rear fin forward control
            speed_factor = 1  # Forward speed control
            steer_angle = 0.0  # Steer control
            angle_rear = max_rear_radius * math.sin(
                speed_factor * 2 * math.pi * step_counter / period_steps_rear) + steer_angle
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=rear_fin_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle_rear,
                positionGain=1.0,
                force=10
            )
            # Rear fin thrust
            forward_thrust, steer_thrust = underwater.apply_tail_thrust(p, robot_id, rear_fin_id)

            # Left fin control
            angle_left = 0.15  # Left fin control
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=left_fin_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle_left,
                positionGain=1.0,
                force=10
            )
            left_lift = underwater.apply_fin_lift(p, robot_id, left_fin_id)

            # Right fin control
            angle_right = 0.15  # Right fin control
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=right_fin_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle_right,
                positionGain=1.0,
                force=10
            )
            right_lift = underwater.apply_fin_lift(p, robot_id, right_fin_id)

            # Camera control
            # camera_follow(robot_id)

            # Click the reset button
            if p.readUserDebugParameter(btn) != previous_btn_value:
                previous_btn_value = debug_params.reset_env(btn, p, robot_id)

            step_counter += 1
            time.sleep(1 / 240)

        # p.stopStateLogging(log_id)  # Stop recording
        # -------------------------------------------------------------------------

        print("Final pose:", p.getBasePositionAndOrientation(robot_id))
        p.disconnect()

    else:
        # 创建智能体
        agent = FishDQNAgent(
            obs_dim=STATE_DIM,
            act_dim=ACTION_DIM,
            lr=0.001,
            gamma=0.99,
            e_greed=0.2,
            e_greed_decrement=1e-5,
            target_update_freq=200,
            buffer_size=10000,
            batch_size=64
        )

        # 训练参数
        num_episodes = 1000
        max_steps_per_episode = 1000
        save_interval = 50

        # 训练循环
        for episode in range(num_episodes):
            # 重置环境
            p.resetSimulation()
            # --------------- Reload environment ---------------------------------------------
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,
                                       0)  # Disable rendering before all the models being loaded
            plane, robot_id, rear_fin_id, left_fin_id, right_fin_id = load_environment(p)

            # Run 10 steps to ensure the initialization
            p.setRealTimeSimulation(0)  # Disable real time simulation
            for _ in range(20):
                p.stepSimulation()
                time.sleep(1 / 240)
            # ------------------------------------------------------------------------
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,
                                       1)  # Disable rendering before all the models being loaded
            p.resetBasePositionAndOrientation(robot_id, [0, 0, 1.5], p.getQuaternionFromEuler([0, 0, 0]))

            # 获取初始状态
            state = get_state(robot_id, p)
            prev_state = None
            episode_reward = 0

            # 单回合循环
            for step in range(max_steps_per_episode):
                # 选择动作
                action_idx = agent.sample(state)
                speed_factor, steer_angle, angle_left, angle_right = action_mapping(action_idx)

                # 执行动作
                # 尾鳍控制
                angle_rear = steer_angle + 0.5 * speed_factor * math.sin(2 * math.pi * step / 100)
                p.setJointMotorControl2(robot_id, rear_fin_id, p.POSITION_CONTROL,
                                        targetPosition=angle_rear, positionGain=1.0, force=10)

                # 左鳍控制
                p.setJointMotorControl2(robot_id, left_fin_id, p.POSITION_CONTROL,
                                        targetPosition=angle_left, positionGain=1.0, force=10)

                # 右鳍控制
                p.setJointMotorControl2(robot_id, right_fin_id, p.POSITION_CONTROL,
                                        targetPosition=angle_right, positionGain=1.0, force=10)

                # 应用流体力学
                underwater.apply_buoyancy(p, robot_id)
                forward_thrust, steer_thrust = underwater.apply_tail_thrust(p, robot_id, rear_fin_id)
                left_lift = underwater.apply_fin_lift(p, robot_id, left_fin_id)
                right_lift = underwater.apply_fin_lift(p, robot_id, right_fin_id)

                # 物理步进
                p.stepSimulation()

                # 获取新状态
                next_state = get_state(robot_id, p)

                # 计算奖励
                reward, done = calculate_reward(next_state, state)
                episode_reward += reward

                # 存储经验
                agent.store_experience(state, action_idx, reward, next_state, done)

                # 学习
                loss = agent.learn()

                # 更新状态
                prev_state = state
                state = next_state

                # 检查终止条件
                if done:
                    break

                # 控制模拟速度
                time.sleep(1 / 240)

            # 打印回合信息
            print(
                f"Episode: {episode + 1}, Reward: {episode_reward:.2f}, Steps: {step + 1}, Epsilon: {agent.e_greed:.4f}")

            # 定期保存模型
            if (episode + 1) % save_interval == 0:
                torch.save(agent.q_net.state_dict(), f"./models/fish_model_ep{episode + 1}.pth")
                print(f"Model saved at episode {episode + 1}")

        # 保存最终模型
        torch.save(agent.q_net.state_dict(), "./models/fish_model_final.pth")
        p.disconnect()


if __name__ == '__main__':
    main()