import numpy as np
import math


EST_VOL = 6.85e-5  # m³  – rough
BUOYANCY = 1000 * 9.81 * EST_VOL  # N   – ρ g V

def apply_buoyancy(p, box_id):
    """ Buoyancy helper now has a valid boxId to work with"""
    p.applyExternalForce(box_id, -1, [0, 0, BUOYANCY],
                         [0, 0, 0], p.WORLD_FRAME)


def apply_water_drag(p, box_id):
    """Optional quadratic drag helper"""
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


def apply_tail_thrust(p, body_id, joint_id,
                      fin_len=0.08,      # distance hinge → tip  (m)
                      chord=0.025,       # fin depth            (m)
                      rho=1000,          # water density        (kg/m³)
                      C_T=2000):          # thrust coefficient   (‑)
    """ Rear fin thrust helper"""
    # joint kinematics
    theta, omega = p.getJointState(body_id, joint_id)[:2]
    v_tip = abs(omega) * fin_len  # lateral speed at tip

    # simplified thrust from elongated‑body theory
    thrust = -0.5 * rho * C_T * chord * fin_len * v_tip ** 2 * abs(math.sin(theta))

    # body‑x axis in world frame
    _, quat = p.getBasePositionAndOrientation(body_id)
    r = p.getMatrixFromQuaternion(quat)
    fwd = [r[0], r[3], r[6]]

    p.applyExternalForce(body_id, -1,
                         [thrust * c for c in fwd],  # world‑space force
                         [0, 0, 0],  # at COM
                         p.WORLD_FRAME)