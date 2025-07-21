import pybullet as p
import pybullet_data
import os
import time
import underwater


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
    # ------------------------------------------------------------------------
    # Set up the physical engine
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)  # 10 for easy calculation
    p.setPhysicsEngineParameter(fixedTimeStep=1/240, numSolverIterations=50)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # GUI for debug mode
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # Disable rendering before all the models being loaded
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)  # Disable CPU rendering
    # ------------------------------------------------------------------------

    # Load ground and fish BEFORE touching boxId‑dependent stuff
    p.loadURDF("plane.urdf")
    startPos = [0, 0, 1]
    startOri = p.getQuaternionFromEuler([0, 0, 0])
    box_id = p.loadURDF("model/Biomimetic_Fish_v7.urdf", startPos, startOri)

    # Set linear / angular damping (simple “viscosity”)
    p.changeDynamics(box_id, -1,
                     linearDamping=5.0,
                     angularDamping=2.0)

    # Set underwater environment
    underwater.apply_buoyancy(p, box_id)
    underwater.apply_water_drag(p, box_id)  # optional

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # Rendering after all the models have been loaded

    # Run the simulation
    for _ in range(10000):
        p.stepSimulation()
        time.sleep(1/240)

    print("Final pose:", p.getBasePositionAndOrientation(box_id))
    p.disconnect()


if __name__ == '__main__':
    main()