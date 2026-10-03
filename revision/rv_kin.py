import math

V_MAX = 1.0
A_MAX = 0.5
W_MAX = 1.0
AW_MAX = 1.0
MASS = 1.0
INERTIA = 0.5
C_ROLL = 0.1
C_ROT = 0.05
ANGLE_EPS = 1e-9


def trapezoid(dist, vmax, amax):
    if dist < 0 or not math.isfinite(dist):
        raise ValueError("distance must be finite and non-negative")
    if dist == 0.0:
        return 0.0, 0.0
    if dist >= vmax * vmax / amax:
        return dist / vmax + vmax / amax, vmax
    vp = math.sqrt(amax * dist)
    return 2.0 * vp / amax, vp


def heading_change(p0, p1, p2):
    a = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    b = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    d = b - a
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return abs(d)


def profile(path):
    if not path or len(path) < 2:
        raise ValueError("path needs at least two waypoints")
    pts = [path[0]]
    for p in path[1:]:
        if p[0] != pts[-1][0] or p[1] != pts[-1][1]:
            pts.append(p)
    if len(pts) < 2:
        raise ValueError("degenerate path")
    t_drive = t_rot = 0.0
    e_acc = e_rot = e_roll = 0.0
    rot_total = 0.0
    stops = 0
    for k in range(len(pts) - 1):
        L = math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1])
        t, vp = trapezoid(L, V_MAX, A_MAX)
        t_drive += t
        e_acc += MASS * vp * vp
        e_roll += C_ROLL * MASS * L
    for k in range(1, len(pts) - 1):
        th = heading_change(pts[k - 1], pts[k], pts[k + 1])
        if th > ANGLE_EPS:
            stops += 1
            rot_total += th
            t, wp = trapezoid(th, W_MAX, AW_MAX)
            t_rot += t
            e_rot += INERTIA * wp * wp + C_ROT * th
    return {"kin_time": t_drive + t_rot, "kin_t_drive": t_drive, "kin_t_rot": t_rot,
            "kin_stops": stops, "kin_rot_rad": rot_total,
            "kin_energy": e_acc + e_rot + e_roll, "kin_e_acc": e_acc,
            "kin_e_rot": e_rot, "kin_e_roll": e_roll}


def self_test():
    t, vp = trapezoid(10.0, 1.0, 0.5)
    assert abs(t - 12.0) < 1e-12 and vp == 1.0
    t, vp = trapezoid(1.0, 1.0, 0.5)
    assert abs(vp - math.sqrt(0.5)) < 1e-12 and abs(t - 2.0 * math.sqrt(0.5) / 0.5) < 1e-12
    r = profile([(0, 0), (0, 10)])
    assert r["kin_stops"] == 0 and abs(r["kin_time"] - 12.0) < 1e-12
    r = profile([(0, 0), (0, 5), (5, 5)])
    assert r["kin_stops"] == 1 and abs(r["kin_rot_rad"] - math.pi / 2) < 1e-12
    r2 = profile([(0, 0), (0, 5), (0, 5), (5, 5)])
    assert abs(r2["kin_time"] - r["kin_time"]) < 1e-12
    r3 = profile([(0, 0), (0, 5), (0, 10)])
    assert r3["kin_stops"] == 0
    return True


if __name__ == "__main__":
    print("rv_kin self test", self_test())
