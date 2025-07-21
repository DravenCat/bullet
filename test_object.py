
def create_test_wall(p):
    # 创建一面墙
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[60, 5, 5]
    )

    collison_box_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[60, 5, 5]
    )

    wall_id = p.createMultiBody(
        baseMass=10000,
        baseCollisionShapeIndex=collison_box_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[0, 10, 5],
        useMaximalCoordinates=True
    )