


def reset_env(btn, p, robot_id):
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 1], p.getQuaternionFromEuler([0, 0, 0]))
    return p.readUserDebugParameter(btn)