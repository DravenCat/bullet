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


def setup_camera_for_large_model(obj_id):
    """Set the camera position and size"""

    aabb_min, aabb_max = p.getAABB(obj_id)
    model_center = [(aabb_min[i] + aabb_max[i]) / 2 for i in range(3)]
    model_size = [aabb_max[i] - aabb_min[i] for i in range(3)]
    max_dim = max(model_size)

    camera_distance = max_dim * 1

    camera_position = [
        model_center[0],
        model_center[1] - camera_distance,
        model_center[2] + max_dim * 0.5
    ]

    camera_target = model_center

    # Set Camera params
    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=camera_target
    )

    # Set far plane
    far_plane = max_dim * 2

    # Set project matrix
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=1.0,
        nearVal=0.1,
        farVal=far_plane
    )

    # Set view matrix
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target,
        distance=camera_distance,
        yaw=45,
        pitch=-30,
        roll=0,
        upAxisIndex=2
    )

    # Get camera image
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=1920,
        height=1080,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix
    )

    return camera_position, camera_target, far_plane


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
    # Run the simulation
    for _ in range(10000):
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

        step_counter += 1
        time.sleep(1/240)
    # -------------------------------------------------------------------------

    print("Final pose:", p.getBasePositionAndOrientation(robot_id))
    p.disconnect()


if __name__ == '__main__':
    main()