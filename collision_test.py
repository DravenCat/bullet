import math


def ray_test(robot_id, p, rayNum, rayLength, useDebugLine=False,
             missRayColor=[1, 0, 0], # Rad
             hitRayColor=[0, 1, 0]  # Green
             ):
    # Get ray froms and rat tos
    begins, _ = p.getBasePositionAndOrientation(robot_id)
    rayFroms = [begins for _ in range(rayNum)]
    rayTos = [
        [
            begins[0] + rayLength * math.cos(2 * math.pi * float(i) / rayNum),
            begins[1] + rayLength * math.sin(2 * math.pi * float(i) / rayNum),
            begins[2]
        ] for i in range(rayNum)]
    results = p.rayTestBatch(rayFroms, rayTos)

    p.removeAllUserDebugItems()

    if useDebugLine:
        # Color the results
        for index, result in enumerate(results):
            if result[0] == -1:
                p.addUserDebugLine(rayFroms[index], rayTos[index], missRayColor)
            else:
                p.addUserDebugLine(rayFroms[index], rayTos[index], hitRayColor)


def get_robot_overlapping(robot_id, p):
    """Return the object id that the robot is overlapping"""
    arr = []
    P_min, P_max = p.getAABB(robot_id)
    id_tuples = p.getOverlappingObjects(P_min, P_max)
    if len(id_tuples) > 1:
        for ID, _ in id_tuples:
            if ID == robot_id:
                continue
            else:
                # print(f"hit happen! hit object is {p.getBodyInfo(ID)}")
                arr.append(ID)
    return arr