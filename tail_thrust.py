def apply_tail_thrust(p, body_id, joint_id,
                      fin_len=0.08,      # distance hinge → tip  (m)
                      chord=0.025,       # fin depth            (m)
                      rho=1000,          # water density        (kg/m³)
                      C_T=0.8):          # thrust coefficient   (‑)

    # joint kinematics
    theta, omega = p.getJointState(body_id, joint_id)[:2]
    v_tip = abs(omega) * fin_len           # lateral speed at tip

    # simplified thrust from elongated‑body theory
    thrust = 0.5 * rho * C_T * chord * fin_len * v_tip**2 * abs(math.sin(theta))

    # body‑x axis in world frame
    _, quat = p.getBasePositionAndOrientation(body_id)
    r = p.getMatrixFromQuaternion(quat)
    fwd = [r[0], r[3], r[6]]

    p.applyExternalForce(body_id, -1,
                         [thrust * c for c in fwd],   # world‑space force
                         [0, 0, 0],                   # at COM
                         p.WORLD_FRAME)