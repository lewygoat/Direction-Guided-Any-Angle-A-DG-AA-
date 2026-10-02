import functools
import math

import numpy as np
from numba import njit

import rv_planners as RP

INF = np.inf
MOVES8 = np.array(RP.NB8, dtype=np.int64)
MOVES16 = np.array(RP.NB16, dtype=np.int64)
ST_OK = 1
ST_NOPATH = 0
ST_BUDGET = -1


@njit(cache=True)
def _passable(grid, r, c):
    if r < 0 or c < 0 or r >= grid.shape[0] or c >= grid.shape[1]:
        return False
    return grid[r, c] == 0


@njit(cache=True)
def _fsum(xs, n):
    p = np.empty(n + 1)
    m = 0
    for i in range(n):
        x = xs[i]
        k = 0
        for j in range(m):
            y = p[j]
            if abs(x) < abs(y):
                t = x
                x = y
                y = t
            hi = x + y
            yr = hi - x
            lo = y - yr
            if lo != 0.0:
                p[k] = lo
                k += 1
            x = hi
        m = k
        if x != 0.0:
            p[m] = x
            m += 1
    hi = 0.0
    lo = 0.0
    if m > 0:
        m -= 1
        hi = p[m]
        while m > 0:
            x = hi
            m -= 1
            y = p[m]
            hi = x + y
            yr = hi - x
            lo = y - yr
            if lo != 0.0:
                break
        if m > 0 and ((lo < 0.0 and p[m - 1] < 0.0) or (lo > 0.0 and p[m - 1] > 0.0)):
            y = lo * 2.0
            x = hi + y
            yr = x - hi
            if y == yr:
                hi = x
    return hi


@njit(cache=True)
def step_cost(grid, w, hyp, ar, ac, br, bc):
    dr = br - ar
    dc = bc - ac
    if dr == 0 and dc == 0:
        return 0.0
    if grid[ar, ac] != 0 or grid[br, bc] != 0:
        return INF
    if dr != 0 and dc != 0 and grid[ar + dr, ac] != 0 and grid[ar, ac + dc] != 0:
        return INF
    return 0.5 * hyp[abs(dr), abs(dc)] * (w[ar, ac] + w[br, bc])


@njit(cache=True)
def walk_cost(grid, w, hyp, a0, a1, b0, b1):
    d0 = b0 - a0
    d1 = b1 - a1
    L = hyp[abs(d0), abs(d1)]
    if L == 0.0:
        return 0.0
    n0 = abs(d0)
    n1 = abs(d1)
    t0s = np.empty(n0)
    t1s = np.empty(n1)
    for i in range(n0):
        k = a0 + i if d0 > 0 else a0 - 1 - i
        t0s[i] = (k + 0.5 - a0) / d0
    for i in range(n1):
        k = a1 + i if d1 > 0 else a1 - 1 - i
        t1s[i] = (k + 0.5 - a1) / d1
    n = n0 + n1 + 1
    et = np.empty(n)
    ea = np.empty(n, np.int64)
    i0 = 0
    i1 = 0
    for m in range(n - 1):
        if i1 >= n1 or (i0 < n0 and t0s[i0] <= t1s[i1]):
            et[m] = t0s[i0]
            ea[m] = 0
            i0 += 1
        else:
            et[m] = t1s[i1]
            ea[m] = 1
            i1 += 1
    et[n - 1] = 1.0
    ea[n - 1] = 2
    s0 = 1 if d0 > 0 else -1
    s1 = 1 if d1 > 0 else -1
    terms = np.empty(n)
    nt = 0
    prev = 0.0
    i = 0
    while i < n:
        t = et[i]
        k0 = False
        k1 = False
        j = i
        while j < n and et[j] - t < 1e-9:
            if ea[j] == 0:
                k0 = True
            elif ea[j] == 1:
                k1 = True
            j += 1
        if t - prev > 1e-12:
            tm = 0.5 * (prev + t)
            r = int(math.floor(a0 + tm * d0 + 0.5))
            c = int(math.floor(a1 + tm * d1 + 0.5))
            if grid[r, c] != 0:
                return INF
            terms[nt] = ((t - prev) * L) * w[r, c]
            nt += 1
        if k0 and k1:
            x = a0 + t * d0
            y = a1 + t * d1
            ra = int(np.rint(x - 0.5 * s0))
            ca = int(np.rint(y + 0.5 * s1))
            rb = int(np.rint(x + 0.5 * s0))
            cb = int(np.rint(y - 0.5 * s1))
            if grid[ra, ca] != 0 and grid[rb, cb] != 0:
                return INF
        prev = t
        i = j
    return _fsum(terms, nt)


@njit(cache=True)
def seg_cost(grid, w, hyp, ar, ac, br, bc):
    if abs(ar - br) <= 1 and abs(ac - bc) <= 1:
        return step_cost(grid, w, hyp, ar, ac, br, bc)
    return walk_cost(grid, w, hyp, ar, ac, br, bc)


@njit(cache=True)
def _less(f1, g1, c1, f2, g2, c2):
    if f1 != f2:
        return f1 < f2
    if g1 != g2:
        return g1 < g2
    return c1 < c2


@njit(cache=True)
def _sift_up(hf, hg, hc, hn, i):
    while i > 0:
        p = (i - 1) >> 1
        if _less(hf[i], hg[i], hc[i], hf[p], hg[p], hc[p]):
            hf[i], hf[p] = hf[p], hf[i]
            hg[i], hg[p] = hg[p], hg[i]
            hc[i], hc[p] = hc[p], hc[i]
            hn[i], hn[p] = hn[p], hn[i]
            i = p
        else:
            break


@njit(cache=True)
def _sift_down(hf, hg, hc, hn, size):
    i = 0
    while True:
        l = 2 * i + 1
        if l >= size:
            break
        m = l
        r = l + 1
        if r < size and _less(hf[r], hg[r], hc[r], hf[l], hg[l], hc[l]):
            m = r
        if _less(hf[m], hg[m], hc[m], hf[i], hg[i], hc[i]):
            hf[i], hf[m] = hf[m], hf[i]
            hg[i], hg[m] = hg[m], hg[i]
            hc[i], hc[m] = hc[m], hc[i]
            hn[i], hn[m] = hn[m], hn[i]
            i = m
        else:
            break


@njit(cache=True)
def _key(hyp, nr, nc, ar, ac, gn, gr, gc, d0, eps, beta, adaptive):
    hn = hyp[abs(gr - nr), abs(gc - nc)]
    if hn == 0.0:
        return gn
    e = eps
    if adaptive:
        e = 1.0 + (eps - 1.0) * min(1.0, hn / d0)
    delta = 0.0
    if beta > 0.0 and (ar != nr or ac != nc):
        mr = nr - ar
        mc = nc - ac
        ml = hyp[abs(mr), abs(mc)]
        a = (mr * (gr - nr) + mc * (gc - nc)) / (ml * hn)
        delta = beta * max(0.0, 1.0 - a)
    return gn + e * hn * (1.0 + delta)


@njit(cache=True)
def _reconstruct(par, s, t, C, limit):
    buf = np.empty(limit, np.int64)
    n = 0
    cur = t
    buf[n] = cur
    n += 1
    while cur != s:
        cur = par[cur]
        if cur < 0 or n >= limit:
            return np.empty((0, 2), np.int64)
        buf[n] = cur
        n += 1
    out = np.empty((n, 2), np.int64)
    for i in range(n):
        v = buf[n - 1 - i]
        out[i, 0] = v // C
        out[i, 1] = v % C
    return out


@njit(cache=True)
def search_nb(grid, w, hyp, sr, sc, gr, gc, anyangle, eps, beta, adaptive, reopen, moves, max_exp):
    R, C = grid.shape
    N = R * C
    g = np.full(N, INF)
    par = np.full(N, -1, np.int64)
    closed = np.zeros(N, np.bool_)
    s = sr * C + sc
    t = gr * C + gc
    empty = np.empty((0, 2), np.int64)
    if s == t:
        one = np.empty((1, 2), np.int64)
        one[0, 0] = sr
        one[0, 1] = sc
        return ST_OK, 0.0, 0, 0, 0, one
    d0 = hyp[abs(sr - gr), abs(sc - gc)]
    cap = 4096
    hf = np.empty(cap)
    hg = np.empty(cap)
    hc = np.empty(cap, np.int64)
    hn = np.empty(cap, np.int64)
    g[s] = 0.0
    par[s] = s
    hf[0] = _key(hyp, sr, sc, sr, sc, 0.0, gr, gc, d0, eps, beta, adaptive)
    hg[0] = 0.0
    hc[0] = 0
    hn[0] = s
    size = 1
    cnt = 0
    expanded = 0
    reopened = 0
    los = 0
    while size > 0:
        neg_g = hg[0]
        cur = hn[0]
        size -= 1
        if size > 0:
            hf[0] = hf[size]
            hg[0] = hg[size]
            hc[0] = hc[size]
            hn[0] = hn[size]
            _sift_down(hf, hg, hc, hn, size)
        gcur = -neg_g
        if gcur > g[cur] + 1e-12 or closed[cur]:
            continue
        closed[cur] = True
        expanded += 1
        if expanded > max_exp:
            return ST_BUDGET, INF, expanded, reopened, los, empty
        if cur == t:
            return ST_OK, gcur, expanded, reopened, los, _reconstruct(par, s, t, C, N + 1)
        r = cur // C
        c = cur % C
        p = par[cur]
        use_p = anyangle and p != cur
        gp = g[p] if use_p else 0.0
        pr = p // C
        pc = p % C
        for k in range(moves.shape[0]):
            dr = moves[k, 0]
            dc = moves[k, 1]
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= R or nc < 0 or nc >= C or grid[nr, nc] != 0:
                continue
            nb = nr * C + nc
            if closed[nb] and not reopen:
                continue
            if -1 <= dr <= 1 and -1 <= dc <= 1:
                c1 = step_cost(grid, w, hyp, r, c, nr, nc)
            else:
                c1 = seg_cost(grid, w, hyp, r, c, nr, nc)
            if not math.isfinite(c1):
                continue
            cand = gcur + c1
            anchor = cur
            ar = r
            ac = c
            if use_p and nb != p:
                los += 1
                c2 = seg_cost(grid, w, hyp, pr, pc, nr, nc)
                if gp + c2 < cand - 1e-12:
                    cand = gp + c2
                    anchor = p
                    ar = pr
                    ac = pc
            if cand < g[nb] - 1e-12:
                if closed[nb]:
                    closed[nb] = False
                    reopened += 1
                g[nb] = cand
                par[nb] = anchor
                cnt += 1
                if size == cap:
                    cap *= 2
                    hf2 = np.empty(cap)
                    hg2 = np.empty(cap)
                    hc2 = np.empty(cap, np.int64)
                    hn2 = np.empty(cap, np.int64)
                    hf2[:size] = hf[:size]
                    hg2[:size] = hg[:size]
                    hc2[:size] = hc[:size]
                    hn2[:size] = hn[:size]
                    hf, hg, hc, hn = hf2, hg2, hc2, hn2
                hf[size] = _key(hyp, nr, nc, ar, ac, cand, gr, gc, d0, eps, beta, adaptive)
                hg[size] = -cand
                hc[size] = cnt
                hn[size] = nb
                _sift_up(hf, hg, hc, hn, size)
                size += 1
    return ST_NOPATH, INF, expanded, reopened, los, empty


@njit(cache=True)
def _corner_ok(grid, r, c, dr, dc):
    if dr != 0 and dc != 0:
        if grid[r + dr, c] != 0 and grid[r, c + dc] != 0:
            return False
    return True


@njit(cache=True)
def _jump_straight(grid, r, c, dr, dc, gr, gc):
    while True:
        nr = r + dr
        nc = c + dc
        if not _passable(grid, nr, nc):
            return -1, -1
        if nr == gr and nc == gc:
            return nr, nc
        if dr != 0:
            if (not _passable(grid, nr, nc + 1)) and _passable(grid, nr + dr, nc + 1):
                return nr, nc
            if (not _passable(grid, nr, nc - 1)) and _passable(grid, nr + dr, nc - 1):
                return nr, nc
        else:
            if (not _passable(grid, nr + 1, nc)) and _passable(grid, nr + 1, nc + dc):
                return nr, nc
            if (not _passable(grid, nr - 1, nc)) and _passable(grid, nr - 1, nc + dc):
                return nr, nc
        r = nr
        c = nc


@njit(cache=True)
def _jump(grid, r, c, dr, dc, gr, gc):
    if dr == 0 or dc == 0:
        return _jump_straight(grid, r, c, dr, dc, gr, gc)
    while True:
        nr = r + dr
        nc = c + dc
        if not _passable(grid, nr, nc):
            return -1, -1
        if not _corner_ok(grid, r, c, dr, dc):
            return -1, -1
        if nr == gr and nc == gc:
            return nr, nc
        if (not _passable(grid, nr - dr, nc)) and _passable(grid, nr - dr, nc + dc):
            return nr, nc
        if (not _passable(grid, nr, nc - dc)) and _passable(grid, nr + dr, nc - dc):
            return nr, nc
        a, _ = _jump_straight(grid, nr, nc, dr, 0, gr, gc)
        if a >= 0:
            return nr, nc
        a, _ = _jump_straight(grid, nr, nc, 0, dc, gr, gc)
        if a >= 0:
            return nr, nc
        r = nr
        c = nc


@njit(cache=True)
def _prune(grid, r, c, pr, pc, out):
    n = 0
    if pr < 0:
        for k in range(MOVES8.shape[0]):
            dr = MOVES8[k, 0]
            dc = MOVES8[k, 1]
            if _passable(grid, r + dr, c + dc) and _corner_ok(grid, r, c, dr, dc):
                out[n, 0] = dr
                out[n, 1] = dc
                n += 1
        return n
    dr = 1 if r > pr else (-1 if r < pr else 0)
    dc = 1 if c > pc else (-1 if c < pc else 0)
    if dr != 0 and dc != 0:
        if _passable(grid, r, c + dc):
            out[n, 0] = 0
            out[n, 1] = dc
            n += 1
        if _passable(grid, r + dr, c):
            out[n, 0] = dr
            out[n, 1] = 0
            n += 1
        if _passable(grid, r + dr, c + dc) and _corner_ok(grid, r, c, dr, dc):
            out[n, 0] = dr
            out[n, 1] = dc
            n += 1
        if (not _passable(grid, r - dr, c)) and _passable(grid, r - dr, c + dc):
            out[n, 0] = -dr
            out[n, 1] = dc
            n += 1
        if (not _passable(grid, r, c - dc)) and _passable(grid, r + dr, c - dc):
            out[n, 0] = dr
            out[n, 1] = -dc
            n += 1
    elif dr != 0:
        if _passable(grid, r + dr, c):
            out[n, 0] = dr
            out[n, 1] = 0
            n += 1
        if (not _passable(grid, r, c + 1)) and _passable(grid, r + dr, c + 1):
            out[n, 0] = dr
            out[n, 1] = 1
            n += 1
        if (not _passable(grid, r, c - 1)) and _passable(grid, r + dr, c - 1):
            out[n, 0] = dr
            out[n, 1] = -1
            n += 1
    else:
        if _passable(grid, r, c + dc):
            out[n, 0] = 0
            out[n, 1] = dc
            n += 1
        if (not _passable(grid, r + 1, c)) and _passable(grid, r + 1, c + dc):
            out[n, 0] = 1
            out[n, 1] = dc
            n += 1
        if (not _passable(grid, r - 1, c)) and _passable(grid, r - 1, c + dc):
            out[n, 0] = -1
            out[n, 1] = dc
            n += 1
    return n


@njit(cache=True)
def jps_nb(grid, hyp, sr, sc, gr, gc, max_exp):
    R, C = grid.shape
    N = R * C
    g = np.full(N, INF)
    came = np.full(N, -1, np.int64)
    ppar = np.full(N, -1, np.int64)
    closed = np.zeros(N, np.bool_)
    s = sr * C + sc
    t = gr * C + gc
    empty = np.empty((0, 2), np.int64)
    if s == t:
        one = np.empty((1, 2), np.int64)
        one[0, 0] = sr
        one[0, 1] = sc
        return ST_OK, 0.0, 0, 0, 0, one
    cap = 4096
    hf = np.empty(cap)
    hg = np.empty(cap)
    hc = np.empty(cap, np.int64)
    hn = np.empty(cap, np.int64)
    g[s] = 0.0
    came[s] = s
    hf[0] = hyp[abs(sr - gr), abs(sc - gc)]
    hg[0] = 0.0
    hc[0] = 0
    hn[0] = s
    size = 1
    cnt = 0
    expanded = 0
    dirs = np.empty((8, 2), np.int64)
    while size > 0:
        neg_g = hg[0]
        cur = hn[0]
        size -= 1
        if size > 0:
            hf[0] = hf[size]
            hg[0] = hg[size]
            hc[0] = hc[size]
            hn[0] = hn[size]
            _sift_down(hf, hg, hc, hn, size)
        gcur = -neg_g
        if gcur > g[cur] + 1e-12 or closed[cur]:
            continue
        closed[cur] = True
        expanded += 1
        if expanded > max_exp:
            return ST_BUDGET, INF, expanded, 0, 0, empty
        if cur == t:
            return ST_OK, gcur, expanded, 0, 0, _reconstruct(came, s, t, C, N + 1)
        r = cur // C
        c = cur % C
        pp = ppar[cur]
        pr = pp // C if pp >= 0 else -1
        pc = pp % C if pp >= 0 else -1
        nd = _prune(grid, r, c, pr, pc, dirs)
        for k in range(nd):
            jr, jc = _jump(grid, r, c, dirs[k, 0], dirs[k, 1], gr, gc)
            if jr < 0:
                continue
            jp = jr * C + jc
            if closed[jp]:
                continue
            ng = gcur + hyp[abs(r - jr), abs(c - jc)]
            if ng < g[jp] - 1e-12:
                g[jp] = ng
                came[jp] = cur
                ppar[jp] = cur
                cnt += 1
                if size == cap:
                    cap *= 2
                    hf2 = np.empty(cap)
                    hg2 = np.empty(cap)
                    hc2 = np.empty(cap, np.int64)
                    hn2 = np.empty(cap, np.int64)
                    hf2[:size] = hf[:size]
                    hg2[:size] = hg[:size]
                    hc2[:size] = hc[:size]
                    hn2[:size] = hn[:size]
                    hf, hg, hc, hn = hf2, hg2, hc2, hn2
                hf[size] = ng + hyp[abs(jr - gr), abs(jc - gc)]
                hg[size] = -ng
                hc[size] = cnt
                hn[size] = jp
                _sift_up(hf, hg, hc, hn, size)
                size += 1
    return ST_NOPATH, INF, expanded, 0, 0, empty


@functools.lru_cache(maxsize=None)
def hyp_table(n):
    t = np.array([[math.hypot(i, j) for j in range(n)] for i in range(n)], dtype=np.float64)
    t.setflags(write=False)
    return t


def run_nb(kw, grid, w, s, g, max_exp=None):
    grid = np.ascontiguousarray(grid, dtype=np.int8)
    hyp = hyp_table(max(grid.shape))
    if max_exp is None:
        max_exp = 50 * grid.size
    if kw is None:
        return jps_nb(grid, hyp, int(s[0]), int(s[1]), int(g[0]), int(g[1]), int(max_exp))
    w = np.ascontiguousarray(w, dtype=np.float64)
    moves = MOVES16 if kw.get("moves") is RP.NB16 else MOVES8
    return search_nb(grid, w, hyp, int(s[0]), int(s[1]), int(g[0]), int(g[1]),
                     bool(kw.get("anyangle", False)), float(kw.get("eps", 1.0)),
                     float(kw.get("beta", 0.0)), bool(kw.get("adaptive", False)),
                     bool(kw.get("reopen", True)), moves, int(max_exp))


def warmup():
    grid = np.zeros((6, 6), dtype=np.int8)
    grid[2, 1:5] = 1
    w = np.ones((6, 6))
    run_nb({}, grid, w, (0, 0), (5, 5))
    run_nb({"anyangle": True, "eps": 1.3, "beta": 0.05, "adaptive": True}, grid, w, (0, 0), (5, 5))
    run_nb({"moves": RP.NB16}, grid, w, (0, 0), (5, 5))
    run_nb(None, grid, w, (0, 0), (5, 5))
