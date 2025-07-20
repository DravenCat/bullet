import numpy as np


EST_VOL = 6.85e-5  # m³  – rough
BUOYANCY = 1000 * 9.81 * EST_VOL  # N   – ρ g V

# Buoyancy helper now has a valid boxId to work with
def apply_buoyancy(p, box_id):
    p.applyExternalForce(box_id, -1, [0, 0, BUOYANCY],
                         [0, 0, 0], p.WORLD_FRAME)


# Optional quadratic drag helper (from earlier message)
def apply_water_drag(p, box_id):
    lin_vel, ang_vel = p.getBaseVelocity(box_id)
    v = np.array(lin_vel)
    w = np.array(ang_vel)
    # translational drag
    speed = np.linalg.norm(v)
    if speed > 1e-6:
        F = -0.5 * 1000 * 0.8 * 0.005 * speed * v
        p.applyExternalForce(box_id, -1, F.tolist(), [0, 0, 0], p.WORLD_FRAME)
    # rotational drag
    rate = np.linalg.norm(w)
    if rate > 1e-6:
        T = -0.5 * 1000 * 0.04 * 1e-4 * rate * w
        p.applyExternalTorque(box_id, -1, T.tolist(), p.WORLD_FRAME)