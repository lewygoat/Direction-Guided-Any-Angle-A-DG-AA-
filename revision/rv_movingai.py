import functools
import os
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "movingai")
RAW = os.path.join(DATA, "raw")
MAPS = ("warehouse-10-20-10-2-1", "warehouse-10-20-10-2-2",
        "warehouse-20-40-10-2-1", "warehouse-20-40-10-2-2")
N_SCEN = 25
PER_SCEN = 4
PASSABLE = set(".GS")
BLOCKED = set("@OTW")


def ensure_extracted():
    os.makedirs(os.path.join(DATA, "maps"), exist_ok=True)
    os.makedirs(os.path.join(DATA, "scen"), exist_ok=True)
    with zipfile.ZipFile(os.path.join(RAW, "mapf-map.zip")) as z:
        names = {os.path.basename(n): n for n in z.namelist()}
        for m in MAPS:
            dst = os.path.join(DATA, "maps", m + ".map")
            if not os.path.exists(dst):
                if m + ".map" not in names:
                    raise FileNotFoundError(m + ".map missing from mapf-map.zip")
                with open(dst, "wb") as f:
                    f.write(z.read(names[m + ".map"]))
    with zipfile.ZipFile(os.path.join(RAW, "mapf-scen-random.zip")) as z:
        names = {os.path.basename(n): n for n in z.namelist()}
        for m in MAPS:
            for k in range(1, N_SCEN + 1):
                fn = "%s-random-%d.scen" % (m, k)
                dst = os.path.join(DATA, "scen", fn)
                if not os.path.exists(dst):
                    if fn not in names:
                        raise FileNotFoundError(fn + " missing from mapf-scen-random.zip")
                    with open(dst, "wb") as f:
                        f.write(z.read(names[fn]))


@functools.lru_cache(maxsize=None)
def load_map(name):
    if name not in MAPS:
        raise ValueError("unknown MovingAI map " + name)
    with open(os.path.join(DATA, "maps", name + ".map")) as f:
        lines = [l.rstrip("\r\n") for l in f]
    if not lines[0].startswith("type ") or lines[3] != "map":
        raise ValueError("bad MovingAI header in " + name)
    H = int(lines[1].split()[1])
    W = int(lines[2].split()[1])
    rows = lines[4:4 + H]
    if len(rows) != H or any(len(r) != W for r in rows):
        raise ValueError("MovingAI map %s does not match declared %dx%d" % (name, H, W))
    grid = np.zeros((H, W), dtype=np.int8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in BLOCKED:
                grid[y, x] = 1
            elif ch not in PASSABLE:
                raise ValueError("unexpected tile %r in %s" % (ch, name))
    grid.setflags(write=False)
    return grid


@functools.lru_cache(maxsize=None)
def load_queries(name):
    grid = load_map(name)
    H, W = grid.shape
    out = []
    for k in range(1, N_SCEN + 1):
        with open(os.path.join(DATA, "scen", "%s-random-%d.scen" % (name, k))) as f:
            lines = [l.rstrip("\r\n") for l in f if l.strip()]
        if not lines[0].startswith("version"):
            raise ValueError("bad scen header %s #%d" % (name, k))
        body = lines[1:1 + PER_SCEN]
        if len(body) < PER_SCEN:
            raise ValueError("scen %s #%d has fewer than %d queries" % (name, k, PER_SCEN))
        for j, l in enumerate(body):
            p = l.split("\t")
            if len(p) != 9:
                raise ValueError("bad scen line %s #%d:%d" % (name, k, j))
            if p[1] != name + ".map" or int(p[2]) != W or int(p[3]) != H:
                raise ValueError("scen %s #%d refers to another map or size" % (name, k))
            sx, sy, gx, gy = int(p[4]), int(p[5]), int(p[6]), int(p[7])
            s, g = (sy, sx), (gy, gx)
            for pt in (s, g):
                if not (0 <= pt[0] < H and 0 <= pt[1] < W) or grid[pt] != 0:
                    raise ValueError("scen %s #%d:%d endpoint invalid" % (name, k, j))
            out.append({"scen": k, "line": j, "start": s, "goal": g, "opt": float(p[8])})
    return tuple(out)
