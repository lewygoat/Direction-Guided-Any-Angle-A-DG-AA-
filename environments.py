import numpy as np

FREE = 0
OBSTACLE = 1


def _first_free_near(grid, corner):
    R, C = grid.shape
    if corner == "tl":
        order = [(r, c) for s in range(R + C) for r in range(s + 1) for c in [s - r] if r < R and c < C]
    else:
        order = [(R - 1 - r, C - 1 - c) for s in range(R + C) for r in range(s + 1) for c in [s - r] if r < R and c < C]
    for r, c in order:
        if grid[r, c] == FREE:
            return (r, c)
    return None


def has_solution(grid, start, goal):
    from collections import deque
    R, C = grid.shape
    if grid[start] == OBSTACLE or grid[goal] == OBSTACLE:
        return False
    seen = np.zeros_like(grid, dtype=bool)
    dq = deque([start])
    seen[start] = True
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while dq:
        r, c = dq.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and not seen[nr, nc] and grid[nr, nc] == FREE:
                if dr != 0 and dc != 0:
                    if grid[r + dr, c] == OBSTACLE and grid[r, c + dc] == OBSTACLE:
                        continue
                seen[nr, nc] = True
                dq.append((nr, nc))
    return False


def make_general(seed, R=50, C=50, obstacle_ratio=0.20):
    for attempt in range(1000):
        rng = np.random.default_rng(seed * 1000 + attempt)
        grid = (rng.random((R, C)) < obstacle_ratio).astype(np.int8)
        start = _first_free_near(grid, "tl")
        goal = _first_free_near(grid, "br")
        if start is None or goal is None or start == goal:
            continue
        if has_solution(grid, start, goal):
            return grid, start, goal, None
    raise RuntimeError(f"no solvable general map for seed {seed}")


def make_general_density(seed, ratio, R=50, C=50):
    for attempt in range(1000):
        rng = np.random.default_rng(seed * 1000 + attempt + int(ratio * 1e6))
        grid = (rng.random((R, C)) < ratio).astype(np.int8)
        start = _first_free_near(grid, "tl")
        goal = _first_free_near(grid, "br")
        if start is None or goal is None or start == goal:
            continue
        if has_solution(grid, start, goal):
            return grid, start, goal, None
    raise RuntimeError("no solvable map seed %d ratio %.2f" % (seed, ratio))


def make_general_size(seed, S, ratio=0.20):
    return make_general_density(seed, ratio, R=S, C=S)


def random_query(grid, seed):
    rng = np.random.default_rng(seed + 777777)
    free = np.argwhere(grid == FREE)
    for _ in range(3000):
        i = rng.integers(len(free))
        j = rng.integers(len(free))
        s = (int(free[i][0]), int(free[i][1]))
        g = (int(free[j][0]), int(free[j][1]))
        if s != g and abs(s[0] - g[0]) + abs(s[1] - g[1]) > (grid.shape[0] + grid.shape[1]) // 2:
            if has_solution(grid, s, g):
                return s, g
    raise RuntimeError("no query for seed %d" % seed)


def make_warehouse(seed, R=50, C=50):
    rng = np.random.default_rng(seed + 100000)
    grid = np.zeros((R, C), dtype=np.int8)
    shelf_w = 2
    shelf_len = 6
    aisle = 3
    border = 3
    shelf_cells = []
    row_block = shelf_len + aisle
    col_block = shelf_w + aisle
    r = border
    while r + shelf_len <= R - border:
        c = border
        while c + shelf_w <= C - border:
            for rr in range(r, r + shelf_len):
                for cc in range(c, c + shelf_w):
                    grid[rr, cc] = OBSTACLE
                    shelf_cells.append((rr, cc))
            c += col_block
        r += row_block

    n_hot = max(1, int(0.15 * len(shelf_cells)))
    idx = rng.choice(len(shelf_cells), size=n_hot, replace=False)
    hotspots = []
    for i in idx:
        sr, sc = shelf_cells[i]
        cand = [(sr, sc - 1), (sr, sc + shelf_w), (sr - 1, sc), (sr + shelf_len, sc)]
        cand = [(a, b) for a, b in cand if 0 <= a < R and 0 <= b < C and grid[a, b] == FREE]
        if cand:
            hotspots.append(cand[rng.integers(len(cand))])
    hotspots = list(set(hotspots))

    tl_free = [(r, c) for r in range(border) for c in range(C) if grid[r, c] == FREE]
    br_free = [(r, c) for r in range(R - border, R) for c in range(C) if grid[r, c] == FREE]
    for _ in range(2000):
        start = tuple(int(x) for x in tl_free[rng.integers(len(tl_free))])
        goal = tuple(int(x) for x in br_free[rng.integers(len(br_free))])
        if start != goal and has_solution(grid, start, goal):
            return grid, start, goal, hotspots
    raise RuntimeError(f"warehouse seed {seed} unsolvable")
