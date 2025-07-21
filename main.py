import pybullet as p
import pybullet_data
import math
import time
import underwater


def setup_physical_engine(connection_mode):
    """Set up and return a new physical engine"""
    physicsClient = p.connect(connection_mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)  # 10 for easy calculation
    p.setPhysicsEngineParameter(fixedTimeStep=1 / 240, numSolverIterations=50)
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


def main():
    # -------------Set up the physical engine -----------------------------------------
    physicsClient = setup_physical_engine(p.GUI)
    # ------------------------------------------------------------------------

    # --------------Load and render the model ------------------------------------------
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # Disable rendering before all the models being loaded

    # Load ground and fish BEFORE touching boxId‑dependent stuff
    plane = p.loadURDF("plane.urdf")
    startPos = [0, 0, 1]
    startOri = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF("model/Biomimetic_Fish_v7.urdf", startPos, startOri)

    # Set linear / angular damping (simple “viscosity”)
    p.changeDynamics(robot_id, -1,
                     linearDamping=5.0,
                     angularDamping=2.0)

    # Set underwater environment
    underwater.apply_buoyancy(p, robot_id)
    underwater.apply_water_drag(p, robot_id)  # optional

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # Rendering after all the models have been loaded
    # ------------------------------------------------------------------------

    # --------------- Initialization ---------------------------------------------
    # Get all the joints id
    available_joints_indexes = [i for i in range(p.getNumJoints(robot_id)) if
                                p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
    rear_fin_id = available_joints_indexes[0]
    front_fin_id = available_joints_indexes[1]

    p.setRealTimeSimulation(0)  # Disable real time simulation

    max_rear_radius = 0.5236  # radius limit for the rear fin (30 * π/180)
    max_front_radius = 0.2618 # radius limit for the front fin (15 * π/180)
    period_steps_rear = 100
    period_steps_front = 1000
    step_counter = 0

    # Initialize rear fin position
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=rear_fin_id,
        controlMode=p.POSITION_CONTROL,
        targetPosition=0,
        positionGain=1.0,
        force=10
    )

    # Initialize front fin position
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=front_fin_id,
        controlMode=p.POSITION_CONTROL,
        targetPosition=0,
        positionGain=1.0,
        force=10
    )

    # Run 10 steps to ensure the initialization
    for _ in range(10):
        p.stepSimulation()
        time.sleep(1 / 240)
    # ------------------------------------------------------------------------

    # --------------- Simulation ---------------------------------------------
    # log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "log/fish_move.mp4")  # Start recording

    # Run the simulation
    for _ in range(2000):
        p.stepSimulation()

        # Rear fin control
        angle_rear = max_rear_radius * math.sin(2 * math.pi * step_counter / period_steps_rear)
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=rear_fin_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=angle_rear,
            positionGain=1.0,
            force=10
        )

        # Front fin control
        angle_front = max_front_radius * math.sin(2 * math.pi * step_counter / period_steps_front)
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=front_fin_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=angle_front,
            positionGain=1.0,
            force=10
        )

        # camera_follow(robot_id)

        step_counter += 1
        time.sleep(1/240)

    # p.stopStateLogging(log_id)  # Stop recording
    # -------------------------------------------------------------------------

    print("Final pose:", p.getBasePositionAndOrientation(robot_id))
    p.disconnect()


if __name__ == '__main__':
    main()