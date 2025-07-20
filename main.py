import pybullet as p
import pybullet_data
import os
import numpy as np


def get_model_file(filename: str) -> str:
    """Get the stl model file name"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    stl_path = os.path.join(current_dir, "model/" + filename + ".stl")
    return stl_path


def load_stl_model(stl_path: str):
    """Load the stl model. Return the model id, center, size and dimension"""
    # create collision shape
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=stl_path,
        meshScale=[1, 1, 1]  # scaling
    )

    # create visual shape
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=stl_path,
        meshScale=[1, 1, 1],
        # rgbaColor=[0.7, 0.7, 0.9, 1]
    )

    # create object
    obj_id = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[0, 0, 0],
        # baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
    )

    aabb_min, aabb_max = p.getAABB(obj_id)
    model_center = [(aabb_min[i] + aabb_max[i]) / 2 for i in range(3)]
    model_size = [aabb_max[i] - aabb_min[i] for i in range(3)]
    max_dim = max(model_size)

    return obj_id, model_center, model_size, max_dim


def setup_camera_for_large_model(model_center, max_dim):
    """Set the camera position and size"""
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
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)

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
    # Connect to physical client and set the client params
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    # p.setTimeStep(1 / 240)

    # Load the model
    planeId = p.loadURDF("plane.urdf")

    # startPos = [0, 0, 1]
    # startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    # boxId = p.loadURDF("r2d2.urdf", startPos, startOrientation)

    stl_path = get_model_file("Biomimetic_Fish_v6")
    obj_id, model_center, model_size, max_dim= load_stl_model(stl_path)
    print(f"Model loaded successfully! ID: {obj_id}")

    # Set the camera
    cam_pos, cam_target, far_plane = setup_camera_for_large_model(model_center, max_dim)

    # Run simulation
    while True:
        p.stepSimulation()


if __name__ == '__main__':
    main()