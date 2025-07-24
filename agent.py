import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

import underwater    # your existing buoyancy/drag/thrust helpers
import test_object   # if you want obstacles (optional)

class FishGymEnv(gym.Env):
    """Gym wrapper around your Biomimetic_Fish_v8 simulation."""
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False, sim_steps=50):
        super().__init__()
        self.render_mode = render
        self.sim_steps = sim_steps

        # Physics client
        flags = p.GUI if render else p.DIRECT
        self._client = p.connect(flags)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load plane once
        self.plane = p.loadURDF("plane.urdf")

        # Joint limits from your main.py
        self.max_rear  = 0.5236
        self.max_front = 0.2618

        # Action space: target angles for [rear, left, right] fins
        self.action_space = spaces.Box(
            low=np.array([-self.max_rear, -self.max_front, -self.max_front]),
            high=np.array([ self.max_rear,  self.max_front,  self.max_front]),
            dtype=np.float32
        )

        # Observation: fish pos (3), fish lin vel (3), target pos (3)
        high = np.array([np.inf]*9, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.robot_id = None
        self.step_counter = 0
        self.max_steps = 200  # per episode

    def reset(self):
        p.resetSimulation(self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self._client)

        # Load fish
        self.robot_id = p.loadURDF(
            "robots/Biomimetic_Fish_v8.urdf",
            basePosition=[0,0,1.5],
            physicsClientId=self._client
        )
        p.changeDynamics(self.robot_id, -1, linearDamping=7, angularDamping=10)

        # Joints
        joints = [i for i in range(p.getNumJoints(self.robot_id))
                  if p.getJointInfo(self.robot_id,i)[2] != p.JOINT_FIXED]
        self.rear, self.left, self.right = joints

        # Random target in XY plane
        self.target = np.random.uniform(-2, 2, size=2).tolist() + [1.0]

        self.step_counter = 0
        return self._get_obs()

    def _get_obs(self):
        pos, ori = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._client)
        lin_vel, _ = p.getBaseVelocity(self.robot_id, physicsClientId=self._client)
        return np.array(pos + lin_vel + list(self.target), dtype=np.float32)

    def step(self, action):
        # Apply fin positions
        p.setJointMotorControl2(self.robot_id, self.rear,
            p.POSITION_CONTROL, targetPosition=action[0], force=10,
            physicsClientId=self._client)
        p.setJointMotorControl2(self.robot_id, self.left,
            p.POSITION_CONTROL, targetPosition=action[1], force=10,
            physicsClientId=self._client)
        p.setJointMotorControl2(self.robot_id, self.right,
            p.POSITION_CONTROL, targetPosition=action[2], force=10,
            physicsClientId=self._client)

        # Simulate forward
        for _ in range(self.sim_steps):
            underwater.apply_buoyancy(p, self.robot_id)
            # underwater.apply_water_drag(p, self.robot_id)
            underwater.apply_tail_thrust(p, self.robot_id, self.rear)
            underwater.apply_fin_lift(p, self.robot_id, self.left)
            underwater.apply_fin_lift(p, self.robot_id, self.right)
            p.stepSimulation(physicsClientId=self._client)
            if self.render_mode:
                time.sleep(1/240)

        obs = self._get_obs()
        fish_pos = obs[0:3]
        dist = np.linalg.norm(np.array(fish_pos) - np.array(self.target))

        # Reward: negative distance, plus bonus for reaching target
        reward = -dist
        done = False
        if dist < 0.2:
            reward += 100
            done = True

        self.step_counter += 1
        if self.step_counter >= self.max_steps:
            done = True

        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self._client)


from stable_baselines3 import PPO

def make_agent(env):
    """Instantiate and return the RL agent."""
    model = PPO("MlpPolicy", env,
                verbose=1,
                tensorboard_log="./logs/fish_swim")
    return model