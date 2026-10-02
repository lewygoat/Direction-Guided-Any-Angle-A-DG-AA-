import hashlib
import math
import time

import mujoco
import numpy as np

import rv_geom as G
import rv_kin as K

DT = 0.002
CTRL_EVERY = 10
DTC = DT * CTRL_EVERY
WHEEL_R = 0.1
KV = 20.0
TAU_MAX = 10.0
MU_NOM = 0.8
V_MAX, A_MAX, W_MAX, AW_MAX = 1.0, 0.5, 1.0, 1.0
K_CT, K_H = 1.5, 2.0
ROT_TOL = 0.05
WP_TOL = 0.05
GOAL_TOL = 0.2
QVEL_MAX = 100.0
SIG_XY, SIG_YAW, SIG_ACT = 0.02, 0.01, 0.05
MU_LO, MU_HI = 0.6, 1.0
WALL_T = 0.5


class SimError(Exception):
    pass


def seed_for(key, k):
    return int(hashlib.sha256(("%s|%d" % (key, k)).encode()).hexdigest()[:8], 16)


def _row_runs(grid):
    out = []
    R, C = grid.shape
    for r in range(R):
        c = 0
        while c < C:
            if grid[r, c] == G.OBST:
                c0 = c
                while c + 1 < C and grid[r, c + 1] == G.OBST:
                    c += 1
                out.append((r, c0, c))
            c += 1
    return out


def build_xml(grid, r_body):
    R, C = grid.shape
    off = 0.7 * r_body
    obs = []
    for i, (r, c0, c1) in enumerate(_row_runs(grid)):
        obs.append('<geom name="o%d" type="box" pos="%.6f %.6f 0.5" size="0.5 %.6f 0.5" contype="0" conaffinity="4"/>'
                   % (i, r, 0.5 * (c0 + c1), 0.5 * (c1 - c0 + 1)))
    t = WALL_T
    walls = [(-0.5 - t / 2, (C - 1) / 2, t / 2, C / 2 + t), (R - 0.5 + t / 2, (C - 1) / 2, t / 2, C / 2 + t),
             ((R - 1) / 2, -0.5 - t / 2, R / 2 + t, t / 2), ((R - 1) / 2, C - 0.5 + t / 2, R / 2 + t, t / 2)]
    for i, (x, y, hx, hy) in enumerate(walls):
        obs.append('<geom name="w%d" type="box" pos="%.6f %.6f 0.5" size="%.6f %.6f 0.5" contype="0" conaffinity="4"/>'
                   % (i, x, y, hx, hy))
    wheel = ('<body name="%s" pos="0 %.6f 0"><joint name="%s" type="hinge" axis="0 1 0" armature="0.01"/>'
             '<geom name="%s" type="cylinder" size="%.6f 0.02" zaxis="0 1 0" mass="1" condim="3" '
             'friction="%.6f 0.005 0.0001" contype="2" conaffinity="0"/></body>')
    return """<mujoco model="agv">
<option timestep="%.6f" integrator="implicitfast"/>
<worldbody>
<geom name="floor" type="plane" size="0 0 1" condim="1" friction="0.1 0.005 0.0001" contype="1" conaffinity="2"/>
%s
<body name="agv" pos="0 0 0.1">
<freejoint name="root"/>
<geom name="chassis" type="cylinder" size="%.6f 0.15" pos="0 0 0.1" mass="96" contype="4" conaffinity="0"/>
<geom name="cf" type="sphere" size="0.05" pos="%.6f 0 -0.05" mass="1" condim="1" contype="2" conaffinity="0"/>
<geom name="cb" type="sphere" size="0.05" pos="%.6f 0 -0.05" mass="1" condim="1" contype="2" conaffinity="0"/>
%s
%s
</body>
</worldbody>
<actuator>
<velocity name="al" joint="jl" kv="%.6f" forcelimited="true" forcerange="%.6f %.6f"/>
<velocity name="ar" joint="jr" kv="%.6f" forcelimited="true" forcerange="%.6f %.6f"/>
</actuator>
</mujoco>""" % (DT, "\n".join(obs), r_body, 0.6 * r_body, -0.6 * r_body,
                wheel % ("wl", off, "jl", "gl", WHEEL_R, MU_NOM),
                wheel % ("wr", -off, "jr", "gr", WHEEL_R, MU_NOM),
                KV, -TAU_MAX, TAU_MAX, KV, -TAU_MAX, TAU_MAX)


def _prep_path(grid, path):
    pts = []
    for p in path:
        q = (float(p[0]), float(p[1]))
        if not all(math.isfinite(v) for v in q):
            raise SimError("non-finite waypoint")
        if not pts or q != pts[-1]:
            pts.append(q)
    if len(pts) < 2:
        raise SimError("path needs two distinct waypoints")
    R, C = grid.shape
    for x, y in (pts[0], pts[-1]):
        r, c = int(round(x)), int(round(y))
        if not (0 <= r < R and 0 <= c < C) or grid[r, c] != G.FREE:
            raise SimError("endpoint not on a free cell")
    return np.array(pts, dtype=np.float64)


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _xtrack(P, p):
    A = P[:-1]
    AB = P[1:] - A
    t = np.clip(((p - A) * AB).sum(1) / (AB * AB).sum(1), 0.0, 1.0)
    q = A + t[:, None] * AB - p
    return float(np.min(np.hypot(q[:, 0], q[:, 1])))


def default_limit(path):
    return 3.0 * K.profile([tuple(p) for p in path])["kin_time"] + 30.0


def simulate(grid, path, r_body, seed=None, noise=True, t_limit=None):
    if not (isinstance(r_body, (int, float)) and math.isfinite(r_body) and r_body > 0):
        raise SimError("r_body must be positive and finite")
    if noise and seed is None:
        raise SimError("noisy run needs an explicit seed")
    P = _prep_path(grid, path)
    if t_limit is None:
        t_limit = default_limit(P)
    if not (math.isfinite(t_limit) and t_limit > 0):
        raise SimError("invalid time limit")
    m = mujoco.MjModel.from_xml_string(build_xml(grid, r_body))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    mu = float(rng.uniform(MU_LO, MU_HI)) if noise else MU_NOM

    def gid(n):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        if i < 0:
            raise SimError("missing geom " + n)
        return i

    for n in ("gl", "gr"):
        m.geom_friction[gid(n), 0] = mu
    cid = gid("chassis")
    dof = [int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in ("jl", "jr")]
    off = 0.7 * r_body
    yaw0 = math.atan2(P[1, 1] - P[0, 1], P[1, 0] - P[0, 0])
    d.qpos[0:3] = (P[0, 0], P[0, 1], 0.1)
    d.qpos[3:7] = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))
    mujoco.mj_forward(m, d)

    last = len(P) - 2
    k, mode = 0, "rot"
    v = w = 0.0
    energy = ctime = pen = 0.0
    episodes = 0
    in_c = False
    clear_min = math.inf
    xt_sum = xt_max = 0.0
    n_ctrl = 0
    reached = False
    error = None
    step = 0
    nmax = int(math.ceil(t_limit / DT))
    goal = P[-1]
    while step < nmax:
        if step % CTRL_EVERY == 0:
            if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))) or np.max(np.abs(d.qvel)) > QVEL_MAX:
                error = "diverged"
                break
            if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
                error = "badqacc"
                break
            x, y = float(d.qpos[0]), float(d.qpos[1])
            qw, qx, qy, qz = (float(z) for z in d.qpos[3:7])
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            clear_min = min(clear_min, G.point_clearance(grid, x, y))
            e = _xtrack(P, np.array([x, y]))
            xt_sum += e
            xt_max = max(xt_max, e)
            n_ctrl += 1
            if math.hypot(x - goal[0], y - goal[1]) < GOAL_TOL:
                reached = True
                break
            if noise:
                xs, ys, ys_yaw = x + rng.normal(0.0, SIG_XY), y + rng.normal(0.0, SIG_XY), yaw + rng.normal(0.0, SIG_YAW)
            else:
                xs, ys, ys_yaw = x, y, yaw

            def seg(i):
                a = P[i]
                dv = P[i + 1] - a
                L = math.hypot(dv[0], dv[1])
                return a, dv[0] / L, dv[1] / L, L

            a, ux, uy, L = seg(k)
            if mode == "track" and k < last and L - ((xs - a[0]) * ux + (ys - a[1]) * uy) <= WP_TOL:
                k += 1
                mode = "rot"
                a, ux, uy, L = seg(k)
            eh = _wrap(math.atan2(uy, ux) - ys_yaw)
            if mode == "rot" and abs(eh) < ROT_TOL:
                mode = "track"
            if mode == "rot":
                wd = math.copysign(min(W_MAX, math.sqrt(2.0 * AW_MAX * abs(eh))), eh)
                v = max(0.0, v - A_MAX * DTC)
            else:
                s = (xs - a[0]) * ux + (ys - a[1]) * uy
                vd = min(V_MAX, math.sqrt(2.0 * A_MAX * max(L - s, 0.0)))
                v = min(vd, v + A_MAX * DTC)
                ect = ux * (ys - a[1]) - uy * (xs - a[0])
                wd = max(-W_MAX, min(W_MAX, K_H * eh - K_CT * ect))
            w += max(-AW_MAX * DTC, min(AW_MAX * DTC, wd - w))
            vl = (v - w * off) / WHEEL_R
            vr = (v + w * off) / WHEEL_R
            if noise:
                vl *= 1.0 + rng.normal(0.0, SIG_ACT)
                vr *= 1.0 + rng.normal(0.0, SIG_ACT)
            d.ctrl[0] = vl
            d.ctrl[1] = vr
        mujoco.mj_step(m, d)
        step += 1
        tau = d.actuator_force
        energy += (abs(tau[0] * d.qvel[dof[0]]) + abs(tau[1] * d.qvel[dof[1]])) * DT
        hit = False
        if d.ncon:
            c = d.contact
            msk = (c.geom == cid).any(axis=1)
            if msk.any():
                hit = True
                pen = max(pen, float(np.max(-c.dist[msk])))
        if hit:
            ctime += DT
            if not in_c:
                episodes += 1
        in_c = hit
    return {
        "reached": bool(reached),
        "clean": bool(reached and episodes == 0),
        "contacts": int(episodes),
        "contact_time": float(ctime),
        "max_pen": float(pen),
        "min_clear": float(clear_min - r_body),
        "exec_time": float(d.time) if reached else None,
        "energy": float(energy),
        "xt_mean": float(xt_sum / max(n_ctrl, 1)),
        "xt_max": float(xt_max),
        "t_limit": float(t_limit),
        "mu": float(mu),
        "steps": int(step),
        "error": error,
    }


def reach_time(L, vmax=V_MAX, amax=A_MAX, tol=GOAL_TOL):
    s = L - tol
    if L >= vmax * vmax / amax:
        da, ta = vmax * vmax / (2 * amax), vmax / amax
        if s <= da:
            return math.sqrt(2 * s / amax)
        if s <= L - da:
            return ta + (s - da) / vmax
        ve = math.sqrt(2 * amax * (L - s))
        return ta + (L - 2 * da) / vmax + (vmax - ve) / amax
    vp = math.sqrt(amax * L)
    if s <= L / 2:
        return math.sqrt(2 * s / amax)
    ve = math.sqrt(2 * amax * (L - s))
    return vp / amax + (vp - ve) / amax


def self_test(verbose=True):
    fails = []

    def chk(name, ok, info):
        if verbose:
            print("%-26s %s  %s" % (name, "PASS" if ok else "FAIL", info))
        if not ok:
            fails.append(name)

    t0 = time.perf_counter()
    g = np.zeros((12, 7), dtype=np.int8)
    r = simulate(g, [(3, 3), (8, 3)], 0.3, noise=False)
    exp = reach_time(5.0)
    chk("straight_trapezoid", r["reached"] and r["contacts"] == 0 and r["exec_time"] is not None
        and abs(r["exec_time"] - exp) / exp < 0.10, "t=%s exp=%.3f" % (r["exec_time"], exp))

    g2 = np.zeros((12, 12), dtype=np.int8)
    pth = [(3, 3), (8, 3), (8, 8)]
    r = simulate(g2, pth, 0.3, noise=False)
    exp = K.profile(pth)["kin_time"] - (K.trapezoid(5.0, V_MAX, A_MAX)[0] - reach_time(5.0))
    chk("turn_90deg", r["reached"] and r["contacts"] == 0 and r["exec_time"] is not None
        and abs(r["exec_time"] - exp) / exp < 0.15, "t=%s exp=%.3f xt_max=%.3f" % (r["exec_time"], exp, r["xt_max"]))

    g3 = np.zeros((12, 9), dtype=np.int8)
    g3[5, 5] = G.OBST
    r_big = simulate(g3, [(3, 4), (8, 4)], 0.8, noise=False)
    r_small = simulate(g3, [(3, 4), (8, 4)], 0.3, noise=False)
    chk("wall_contact", r_big["contacts"] >= 1 and r_big["max_pen"] > 0 and r_small["contacts"] == 0
        and 0.1 < r_small["min_clear"] < 0.3,
        "big contacts=%d pen=%.4f small contacts=%d min_clear=%.3f"
        % (r_big["contacts"], r_big["max_pen"], r_small["contacts"], r_small["min_clear"]))

    a = simulate(g2, pth, 0.3, seed=7, noise=True)
    b = simulate(g2, pth, 0.3, seed=7, noise=True)
    c = simulate(g2, pth, 0.3, seed=8, noise=True)
    chk("determinism", a == b and a != c, "same=%s diff_seed_differs=%s" % (a == b, a != c))

    bad = 0
    for args in ([(3, 3)], [(3, 3), (3, 3)], [(3, 3), (5, 5)]):
        try:
            simulate(g3 if args[-1] == (5, 5) else g2, args, 0.3, noise=False)
        except SimError:
            bad += 1
    for kw in ({"r_body": -1.0, "noise": False}, {"r_body": 0.3, "noise": True}):
        try:
            simulate(g2, pth, **kw)
        except SimError:
            bad += 1
    chk("input_validation", bad == 5, "rejected %d/5" % bad)
    if verbose:
        print("self_test failures=%d, %.1fs" % (len(fails), time.perf_counter() - t0))
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if self_test() else 0)
