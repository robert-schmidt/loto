"""The algorithm. The physical draws are independent and uniform, so nothing here can predict the next
numbers. What it does:
  1. Describe: frequencies, gaps, pairs, splits, each next to its exact theoretical value.
  2. Test fairness: chi-square on the ball counts, all time and recent (would catch a biased machine).
  3. Test prediction honestly: walk-forward backtest of classic "hot/cold/overdue" models scored with
     Benter's metric, log-likelihood vs the uniform baseline (>0 means it beats chance).
  4. Exploit the one real edge (Benter's insight applied to parimutuel prizes): you can't change your
     odds, but you can avoid combinations other players also pick, so a jackpot win is less likely split.
"""
import datetime as dt
import json
import math
import random
import statistics
from itertools import combinations
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Bucharest")
DRAW_HOUR = (18, 30)  # ponytail: regular draws Thu+Sun ~18:30; extra draws only come via announcements
BACKTEST_DRAWS = 1000
RECENT = 100


def prize_category(game, hits, bonus):
    """bonus: Joker ball matched (Joker) / your 5 are exactly the first 5 balls drawn (5/40)."""
    if game == "649":
        return {6: "I", 5: "II", 4: "III", 3: "IV"}.get(hits)
    if game == "540":  # I = 5/5, II = 5/6, III = 4/6 (press reports; no 3/6 category seen)
        return "I" if hits == 5 and bonus else {5: "II", 4: "III"}.get(hits)
    if bonus:
        return {5: "I", 4: "III", 3: "V", 2: "VII", 1: "VIII"}.get(hits)
    return {5: "II", 4: "IV", 3: "VI"}.get(hits)


def hypergeom(n, d, pick, k):
    """P(k of your `pick` numbers are among the `d` drawn from n)."""
    return math.comb(d, k) * math.comb(n - d, pick - k) / math.comb(n, pick)


def chi2_sf(x, df):
    # ponytail: Wilson-Hilferty approximation, accurate to ~3 digits for df >= 19 (all our cases)
    z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def chi2_uniform(counts):
    exp = sum(counts) / len(counts)
    stat = sum((c - exp) ** 2 / exp for c in counts)
    return dict(stat=round(stat, 1), df=len(counts) - 1, p=round(chi2_sf(stat, len(counts) - 1), 4))


def number_stats(draws, n, d, key="nums"):
    T = len(draws)
    count, last, maxgap, recent = [0] * (n + 1), [-1] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for t, dr in enumerate(draws):
        for x in (dr[key] if key == "nums" else [dr[key]]):
            count[x] += 1
            maxgap[x] = max(maxgap[x], t - last[x] - 1)
            last[x] = t
            if t >= T - RECENT:
                recent[x] += 1
    p = d / n
    out = []
    for x in range(1, n + 1):
        gap = T - 1 - last[x]
        a, b = count[x] + 1, T - count[x] + 1  # Beta posterior on this ball's rate, uniform prior
        mean, sd = a / (a + b), math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        out.append(dict(n=x, count=count[x], rate=round(count[x] / T, 4), recent=recent[x],
                        recent_rate=round(recent[x] / min(T, RECENT), 4), gap=gap, max_gap=max(maxgap[x], gap),
                        p_gap=round((1 - p) ** gap, 4), lo=round(mean - 2 * sd, 4), hi=round(mean + 2 * sd, 4)))
    return out


class Predictors:
    """Every system sees only past draws. scores() -> raw score per number (higher = more likely, per the system).
    update(draw) is called after each draw is revealed."""

    def __init__(self, n, d, rnd):
        self.n, self.d, self.rnd, self.t = n, d, rnd, 0
        self.w10, self.w50, self.cnt = [0.0] * (n + 1), [0.0] * (n + 1), [0] * (n + 1)
        self.c10, self.c50, self.c200 = [0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
        self.last_seen, self.hist, self.prev = [0] * (n + 1), [], []
        self.follow = [[0] * (n + 1) for _ in range(n + 1)]  # follow[y][x]: x drawn right after y was drawn
        self.pair = [[0] * (n + 1) for _ in range(n + 1)]    # pair[y][x]: drawn together
        self.lr = [0.0] * 7                                    # online logistic regression weights
        # sliding DFT per number (window W) for spectral extrapolation
        self.W = 128
        self.K = list(range(1, self.W // 2))
        self.tw = [complex(math.cos(2 * math.pi * k / self.W), math.sin(2 * math.pi * k / self.W)) for k in self.K]
        self.X = [[0j] * len(self.K) for _ in range(n + 1)]
        self.bits = [[] for _ in range(n + 1)]
        # mixture of context models (depth 0..6) per number, Krichevsky-Trofimov counts, Bayesian weights
        self.D = 6
        self.ctx = [[{} for _ in range(self.D + 1)] for _ in range(n + 1)]
        self.mixw = [[1.0] * (self.D + 1) for _ in range(n + 1)]

    def feats(self, x):
        e = self.d / self.n
        gap = (self.t - self.last_seen[x]) * e
        return [1.0, self.c10[x] / (10 * e) - 1, self.c50[x] / (50 * e) - 1, self.c200[x] / (200 * e) - 1,
                min(gap, 5.0) - 1, 1.0 if x in self.prev else 0.0, 1.0 if (x - 1 in self.prev or x + 1 in self.prev) else 0.0]

    def kt(self, x):
        """Mixture-of-contexts probability that x is drawn next."""
        b, ws, ps = self.bits[x], self.mixw[x], []
        for depth in range(self.D + 1):
            key = tuple(b[-depth:]) if depth else ()
            a0, a1 = self.ctx[x][depth].get(key, (0, 0))
            ps.append((a1 + 0.5) / (a0 + a1 + 1))
        tot = sum(ws)
        return sum(w * p for w, p in zip(ws, ps)) / tot, ps

    def scores(self):
        n, rng = self.n, range(1, self.n + 1)
        out = {
            "Hot (last ~10 draws)": [self.w10[x] + 1 for x in rng],
            "Hot (last ~50 draws)": [self.w50[x] + 1 for x in rng],
            "All-time frequency": [self.cnt[x] + 1 for x in rng],
            "Cold (least frequent)": [1 / (self.w50[x] + 1) for x in rng],
            "Overdue (longest gap)": [self.t - self.last_seen[x] + 1 for x in rng],
            "Due by gap ratio": [max(0.1, (self.t - self.last_seen[x]) * self.d / n) for x in rng],
            "Repeat last draw": [3.0 if x in self.prev else 1.0 for x in rng],
            "Neighbours of last draw": [3.0 if (x - 1 in self.prev or x + 1 in self.prev) else 1.0 for x in rng],
            "Markov followers": [1 + sum(self.follow[y][x] for y in self.prev) for x in rng],
            "Pairs with last draw": [1 + sum(self.pair[y][x] for y in self.prev) for x in rng],
            "Machine learning (online logistic)": [1 / (1 + math.exp(-sum(w * f for w, f in zip(self.lr, self.feats(x))))) for x in rng],
            "Spectral cycles (Fourier)": [max(1e-3, 1 + self.spectral(x)) for x in rng],
            "Universal pattern learner (context mixture)": [self.kt(x)[0] for x in rng],
        }
        return {k: [0.0] + v for k, v in out.items()}

    def spectral(self, x):
        """Extrapolate the window one step with its 3 strongest cycles (periodic extension of the DFT)."""
        if self.t < self.W:
            return 0.0
        mags = sorted(range(len(self.K)), key=lambda i: -abs(self.X[x][i]))[:3]
        return sum(2 * self.X[x][i].real for i in mags) / self.W

    def update(self, nums):
        n, S = self.n, set(nums)
        # learning steps use the state *before* this draw
        e = self.d / n
        for x in range(1, n + 1):
            y = 1.0 if x in S else 0.0
            f = self.feats(x)
            p = 1 / (1 + math.exp(-sum(w * v for w, v in zip(self.lr, f))))
            self.lr = [w + 0.002 * (y - p) * v for w, v in zip(self.lr, f)]
            _, ps = self.kt(x)
            self.mixw[x] = [w * (p_ if y else 1 - p_) for w, p_ in zip(self.mixw[x], ps)]
            m = max(self.mixw[x])
            self.mixw[x] = [w / m for w in self.mixw[x]]
            b = self.bits[x]
            for depth in range(self.D + 1):
                key = tuple(b[-depth:]) if depth else ()
                a0, a1 = self.ctx[x][depth].get(key, (0, 0))
                self.ctx[x][depth][key] = (a0 + (y == 0), a1 + (y == 1))
            b.append(int(y))
            old = b[-self.W - 1] if len(b) > self.W else 0  # sliding DFT
            self.X[x] = [(Xk - old + y) * tw for Xk, tw in zip(self.X[x], self.tw)]
        for y_ in self.prev:
            for x in S:
                self.follow[y_][x] += 1
        for a in S:
            for b_ in S:
                if a != b_:
                    self.pair[a][b_] += 1
        self.hist.append(S)
        for c, w in ((self.c10, 10), (self.c50, 50), (self.c200, 200)):
            for x in S:
                c[x] += 1
            if len(self.hist) > w:
                for x in self.hist[-w - 1]:
                    c[x] -= 1
        self.w10 = [v * 0.5 ** (1 / 10) for v in self.w10]
        self.w50 = [v * 0.5 ** (1 / 50) for v in self.w50]
        for x in S:
            self.w10[x] += 1
            self.w50[x] += 1
            self.cnt[x] += 1
            self.last_seen[x] = self.t + 1
        self.prev = sorted(S)
        self.t += 1


# ---- Formula Lab: evolved formulas over date + previous-draw features ---------------------------------------
FEATURES = ["day", "month", "year", "day of year", "week", "weekday", "moon age", "day number", "date digit sum",
            "draw count"] + [f"last draw #{i + 1}" for i in range(6)] + ["last draw sum"] + [f"draw before #{i + 1}" for i in range(6)]


def formula_features(date, prev, prev2, t):
    """Everything known before a draw: its date and the previous draws (sorted). Same order as FEATURES."""
    day = dt.date.fromisoformat(date)
    digits = f"{day.day:02d}{day.month:02d}{day.year}"
    ds = sum(map(int, digits))
    dn = ds
    while dn > 9:
        dn = sum(map(int, str(dn)))
    age = ((dt.datetime(day.year, day.month, day.day, 18, 30) - dt.datetime(2000, 1, 6, 18, 14)).total_seconds() / 86400) % 29.530588853
    pad = lambda xs: (sorted(xs) + [0] * 6)[:6]
    return [day.day, day.month, day.year % 100, day.timetuple().tm_yday, day.isocalendar()[1], day.weekday(), int(age), dn, ds, t] \
        + pad(prev) + [sum(prev)] + pad(prev2)


def formula_value(f, feats, n):
    a, i, b, j, c, op = f
    return ((a * feats[i] * feats[j] if op else a * feats[i] + b * feats[j]) + c) % n + 1


def formula_ticket(formulas, feats, n, pick):
    out = []
    for f in formulas:
        v = formula_value(f, feats, n)
        if v not in out:
            out.append(v)
        if len(out) == pick:
            break
    while len(out) < pick:  # formulas agreed too often: fill with the next free numbers after the first answer
        v = out[-1] % n + 1 if out else 1
        while v in out:
            v = v % n + 1
        out.append(v)
    return sorted(out)


def formula_text(f, n):
    a, i, b, j, c, op = f
    if op:
        return f"({a} × {FEATURES[i]} × {FEATURES[j]} + {c}) mod {n} + 1"
    return f"({a} × {FEATURES[i]}{f' + {b} × {FEATURES[j]}' if b else ''} + {c}) mod {n} + 1"


def jacobi_max_eig(A, sweeps=12):
    """Largest eigenvalue of a symmetric matrix (cyclic Jacobi rotations; stdlib only)."""
    A = [r[:] for r in A]
    m = len(A)
    for _ in range(sweeps):
        off = sum(A[i][j] ** 2 for i in range(m) for j in range(m) if i != j)
        if off < 1e-12:
            break
        for p in range(m - 1):
            for q in range(p + 1, m):
                if abs(A[p][q]) < 1e-12:
                    continue
                th = (A[q][q] - A[p][p]) / (2 * A[p][q])
                t = (1 if th >= 0 else -1) / (abs(th) + math.sqrt(th * th + 1))
                c = 1 / math.sqrt(t * t + 1)
                s_ = t * c
                for k in range(m):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p], A[k][q] = c * akp - s_ * akq, s_ * akp + c * akq
                for k in range(m):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k], A[q][k] = c * apk - s_ * aqk, s_ * apk + c * aqk
    return max(A[i][i] for i in range(m))


def corr_max_eig(sets, n):
    T = len(sets)
    X = [[1.0 if x in S else 0.0 for x in range(1, n + 1)] for S in sets]
    mu = [sum(r[i] for r in X) / T for i in range(n)]
    sd = [math.sqrt(max(1e-12, sum((r[i] - mu[i]) ** 2 for r in X) / T)) for i in range(n)]
    C = [[sum((r[i] - mu[i]) * (r[j] - mu[j]) for r in X) / T / sd[i] / sd[j] for j in range(n)] for i in range(n)]
    return jacobi_max_eig(C)


def structure_tests(draws, n, d, sims=12, seed=3):
    """Two searches for hidden structure, each against simulated fair lotteries of the same length.
    1. Information carried draw-to-draw: overlap with the previous draw vs the hypergeometric law (chi-square).
    2. Random matrix theory: the largest eigenvalue of the number-number correlation matrix. Real structure (pairs
       or groups that travel together) would push it above what pure noise produces (Marchenko-Pastur bulk)."""
    sets = [set(dr["nums"]) for dr in draws][-1500:]
    T = len(sets)
    ov = [len(a & b) for a, b in zip(sets, sets[1:])]
    exp = [hypergeom(n, d, d, k) * len(ov) for k in range(d + 1)]
    obs = [ov.count(k) for k in range(d + 1)]
    bins_o, bins_e, acc_o, acc_e = [], [], 0, 0
    for o, e in zip(obs, exp):
        acc_o, acc_e = acc_o + o, acc_e + e
        if acc_e >= 5:
            bins_o.append(acc_o); bins_e.append(acc_e); acc_o = acc_e = 0
    if acc_e:
        bins_o[-1] += acc_o; bins_e[-1] += acc_e
    stat = sum((o - e) ** 2 / e for o, e in zip(bins_o, bins_e))
    p_overlap = chi2_sf(stat, len(bins_o) - 1) if len(bins_o) > 1 else 1.0
    lam = corr_max_eig(sets, n)
    rnd = random.Random(seed)
    null = sorted(corr_max_eig([set(rnd.sample(range(1, n + 1), d)) for _ in range(T)], n) for _ in range(sims))
    q = n / T
    return dict(draws=T, overlap_p=round(p_overlap, 4), eig=round(lam, 3), mp_edge=round((1 + math.sqrt(q)) ** 2, 3),
                null_lo=round(null[0], 3), null_hi=round(null[-1], 3), eig_rank=sum(v < lam for v in null), sims=sims)


def backtest(draws, n, d, pick, min_hits=3, start=100, seed=1):
    """Walk-forward over the whole history: before each draw every system picks its `pick` best numbers (one ticket)
    and gives a probability per ball (20% tilt off uniform, so no system is a strawman). Scored on hits, prize rate,
    and log-likelihood vs chance (Benter's yardstick). Plus a random ticket and a 'typical-looking' ticket."""
    rnd = random.Random(seed)
    P = Predictors(n, d, rnd)
    e_hits = pick * d / n
    var_hits = e_hits * (n - pick) / n * (n - d) / (n - 1)  # hypergeometric variance
    p_prize = sum(hypergeom(n, d, pick, j) for j in range(min_hits, min(pick, d) + 1))
    sums = []
    res = {}
    for t, dr in enumerate(draws):
        S = set(dr["nums"])
        if t >= start:
            for name, sc in P.scores().items():
                tot = sum(sc)
                q = [0.8 / n + 0.2 * v / tot for v in sc]
                top = sorted(range(1, n + 1), key=lambda x: (-q[x], x))[:pick]
                r = res.setdefault(name, dict(llr=[], hits=[]))
                r["llr"].append(sum(math.log(q[x] * n) for x in S))
                r["hits"].append(len(S & set(top)))
            res.setdefault("Random ticket", dict(llr=None, hits=[]))["hits"].append(len(S & set(rnd.sample(range(1, n + 1), pick))))
            lo, hi = sorted(sums)[len(sums) // 4], sorted(sums)[3 * len(sums) // 4]
            while True:  # 'typical-looking': sum in the middle half of past sums, balanced odd/even
                c = rnd.sample(range(1, n + 1), pick)
                if lo <= sum(c) * d / pick <= hi and abs(sum(x % 2 for x in c) - pick / 2) <= 1:
                    break
            res.setdefault("Typical-looking ticket (sum, odd/even)", dict(llr=None, hits=[]))["hits"].append(len(S & set(c)))
        sums.append(sum(dr["nums"]))
        P.update(dr["nums"])
    out, N = {}, len(draws) - start
    for name, r in res.items():
        h = r["hits"]
        m = sum(h) / len(h)
        z = (m - e_hits) / math.sqrt(var_hits / len(h))
        half = len(h) // 2
        row = dict(hits=round(m, 4), z=round(z, 2), p=round(math.erfc(abs(z) / math.sqrt(2)), 4),
                   prize=round(sum(x >= min_hits for x in h) / len(h), 4),
                   hits1=round(sum(h[:half]) / half, 4), hits2=round(sum(h[half:]) / (len(h) - half), 4))
        if r["llr"]:
            L = r["llr"]
            mu = sum(L) / len(L)
            se = math.sqrt(sum((v - mu) ** 2 for v in L) / (len(L) - 1) / len(L))
            row.update(llr=round(mu * 1000, 2), se=round(se * 1000, 2),
                       llr1=round(sum(L[:half]) / half * 1000, 2), llr2=round(sum(L[half:]) / (len(L) - half) * 1000, 2))
        out[name] = row
    best = max((k for k in out if "llr1" in out[k]), key=lambda k: out[k]["llr1"])  # chosen on 1st half only
    return dict(draws=N, expected_hits=round(e_hits, 4), p_prize=round(p_prize, 4), models=out,
                oos=dict(model=best, first=out[best]["llr1"], second=out[best]["llr2"],
                         hits_first=out[best]["hits1"], hits_second=out[best]["hits2"]))


# ---- player popularity, fitted Benter-style to official winner counts ------------------------------------
# Each draw, loto.ro publishes how many tickets won each category. If players favour some numbers, draws made
# of those numbers produce more winners than sales x odds predicts. Fitting that across ~1,400 draws per game
# gives each number's popularity, which is what decides how many people you'd share a prize with.

# category -> (numbers matched, only the first 5 balls count, Joker ball required)
CATS = {
    "649": {"I": (6, 0, 0), "II": (5, 0, 0), "III": (4, 0, 0), "IV": (3, 0, 0)},
    "540": {"I": (5, 1, 0), "II": (5, 0, 0), "III": (4, 0, 0)},
    "joker": {"I": (5, 0, 1), "II": (5, 0, 0), "III": (4, 0, 1), "IV": (4, 0, 0), "V": (3, 0, 1),
              "VI": (3, 0, 0), "VII": (2, 0, 1), "VIII": (1, 0, 1)},
}
PAYOUT = 0.40  # share of sales paid into prize funds: fund/sales = 0.400 for all three games (checked Oct 2026)
MIN_WIN = {"649": (3, None), "540": (4, None), "joker": (3, 1)}  # fewest hits that pay (without, with Joker)
TICKETS = 100  # ranked single tickets generated (and saved) per draw
HOLDOUT = 150


def cat_probs(key, n, d, pick, xn):
    out = {}
    for c, (k, first5, j) in CATS[key].items():
        p = 1 / math.comb(n, pick) if first5 else hypergeom(n, d, pick, k)
        if key == "540" and c == "II":
            p -= 1 / math.comb(n, pick)  # 5 of 6, minus the jackpot case
        out[c] = p * ((1 / xn if j else 1 - 1 / xn) if xn else 1)
    return out


def sales_series(prizes):
    """Lei sold per draw = (fund - previous rollovers) / payout. Supplemented or reseeded jackpots distort
    single draws, so values more than 2x off the local median are dropped."""
    raw, prev = [], None
    for p in prizes:
        if prev is not None and p["fund"] > prev:
            raw.append(((p["date"], p["seq"]), (p["fund"] - prev) / PAYOUT))
        prev = sum(v[2] for v in p["cats"].values())
    out = {}
    for i, (k, v) in enumerate(raw):
        med = statistics.median(x for _, x in raw[max(0, i - 8):i + 9])
        if 0.5 < v / med < 2:
            out[k] = v
    return out


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for c in range(n - 1, -1, -1):
        x[c] = (M[c][n] - sum(M[c][k] * x[k] for k in range(c + 1, n))) / M[c][c]
    return x


def poisson_fit(rows, ridge, iters=40):
    """rows: (y, offset, [(param, weight)]). Ridge-penalised Poisson regression by damped Newton steps."""
    P = len(ridge)
    th = [0.0] * P
    for _ in range(iters):
        g = [-ridge[j] * th[j] for j in range(P)]
        H = [[ridge[j] if j == k else 0.0 for k in range(P)] for j in range(P)]
        for y, off, feats in rows:
            mu = math.exp(min(50, off + sum(th[j] * w for j, w in feats)))
            for j, w in feats:
                g[j] += w * (y - mu)
                Hj = H[j]
                for k, v in feats:
                    Hj[k] += w * v * mu
        step = [max(-1.0, min(1.0, s)) for s in solve(H, g)]
        th = [a + s for a, s in zip(th, step)]
        if max(map(abs, step)) < 1e-6:
            break
    return th


def deviance(rows, th):
    dev = 0.0
    for y, off, feats in rows:
        mu = math.exp(off + sum(th[j] * w for j, w in feats))
        dev += 2 * ((y * math.log(y / mu) if y else 0) - (y - mu))
    return dev


def fit_popularity(key, n, d, pick, xn, prizes):
    """Winners_c ~ Poisson(sales x P(c) x exp(cat + year + k/m x sum of drawn numbers' popularity [+ Joker's])).
    Why k/m: a category-k winner holds k of the m drawn numbers, so on log scale they carry k/m of the set's
    popularity. That same scale makes a full ticket's popularity the plain sum, which the jackpot split needs."""
    probs, sales = cat_probs(key, n, d, pick, xn), sales_series(prizes)
    years = sorted({p["date"][:4] for p in prizes})
    cats = list(CATS[key])
    ci = {c: n + (xn or 0) + i for i, c in enumerate(cats)}
    yi = {y: n + (xn or 0) + len(cats) + i for i, y in enumerate(years)}
    draws = []
    for p in prizes:
        s = sales.get((p["date"], p["seq"]))
        if not s or not p["nums"]:
            continue
        rows = []
        for c, (w, v, r) in p["cats"].items():
            if c not in CATS[key]:
                continue
            k, first5, j = CATS[key][c]
            S = p["nums"][:5] if first5 else p["nums"]
            feats = [(x - 1, k / len(S)) for x in S] + [(ci[c], 1.0), (yi[p["date"][:4]], 1.0)]
            if j and p["extra"]:
                feats.append((n + p["extra"] - 1, 1.0))
            rows.append((w, math.log(s / p.get("price", 1) * probs[c]), feats))
        draws.append(rows)
    if len(draws) < 300:
        return None
    P = n + (xn or 0) + len(cats) + len(years)
    ridge = [10.0] * (n + (xn or 0)) + [1e-6] * len(cats) + [1.0] * len(years)
    flat = lambda ds: [r for rows in ds for r in rows]
    train, test = flat(draws[:-HOLDOUT]), flat(draws[-HOLDOUT:])
    null_ridge = ridge[:]
    for j in range(n + (xn or 0)):
        null_ridge[j] = 1e12  # null model: pins every number's popularity at 0
    full, null = poisson_fit(train, ridge), poisson_fit(train, null_ridge)
    improve = 1 - deviance(test, full) / deviance(test, null)
    th = poisson_fit(flat(draws), ridge)  # final model on everything
    if improve < 0.01:  # doesn't predict held-out draws: don't steer picks with noise (happens on 5/40)
        th = [0.0] * len(th)
    b = th[:n]
    mb = sum(b) / n
    bj = th[n:n + (xn or 0)]
    mj = sum(bj) / xn if xn else 0
    return dict(b=[0] + [round(v - mb, 4) for v in b], bj=[0] + [round(v - mj, 4) for v in bj] if xn else None,
                draws=len(draws), holdout=HOLDOUT, improve=round(improve, 4),
                reliable=improve >= 0.01, since=prizes[0]["date"][:4], sales=statistics.median(list(sales.values())[-10:]))


EVIDENCE_CAT = {"649": "III", "540": "III", "joker": "IV"}  # 4-number prize: many winners, always shared


def evidence(key, prizes, pop):
    """Median official payout of the 4-number prize in draws whose numbers were least vs most popular."""
    c = EVIDENCE_CAT[key]
    rows = sorted((sum(pop["b"][x] for x in p["nums"]), p["cats"][c][1]) for p in prizes[-500:]
                  if p["nums"] and c in p["cats"] and p["cats"][c][0])
    if len(rows) < 100:
        return None
    q = len(rows) // 4
    return dict(cat=c, draws=len(rows), low=round(statistics.median(v for _, v in rows[:q])),
                high=round(statistics.median(v for _, v in rows[-q:])))


def ticket_pop(nums, extra, pop, last):
    """Log popularity of a ticket: fitted per-number terms, plus two pattern terms the per-number fit can't see.
    ponytail: the pattern weights (runs like 1-2-3-4-5-6 or 5-10-15, replaying last draw) are hand-set."""
    s = sorted(nums)
    lp = sum(pop["b"][x] for x in s) if pop else 0.0
    if pop and extra and pop["bj"]:
        lp += pop["bj"][extra]
    diffs = [y - x for x, y in zip(s, s[1:])]
    lp += 1.0 if len(set(diffs)) == 1 else 0
    return lp + 0.3 * len(set(s) & set(last))


def split_factor(lam):
    """E[1/(1+K)], K~Poisson(lam) other winners: the share of the jackpot you keep on average."""
    return 1.0 if lam < 1e-9 else (1 - math.exp(-lam)) / lam


def prize_model(key, g, prizes, jackpot, pop):
    """What one line pays on average, by category, from the last 100 official prize tables."""
    if not prizes:
        return None
    recent = prizes[-100:]
    vals, contrib = {}, []
    for p in recent:
        for c, (w, v, r) in p["cats"].items():
            if c == "I":
                if not w:
                    contrib.append(v)
            elif w:
                vals.setdefault(c, []).append(v)
    typical = {c: (statistics.median(v), max(v) - min(v) < 0.01) for c, v in vals.items()}
    J = (jackpot or recent[-1]["cats"]["I"][2]) + (statistics.median(contrib) if contrib else 0)
    lines = (pop["sales"] if pop else 0) / g["price"]
    return dict(J=J, typical=typical, lines=lines)


def line_ev(key, probs, pm, mult, pick, drawn):
    """Expected lei back for one line whose popularity is `mult` x the average ticket's (before tax).
    A category-k win shares the pot with holders of those k numbers: by the fit's k/m rule their crowd scales
    by mult^((k/m)(k/pick)). ponytail: treats the k hit numbers as average for the ticket."""
    if not pm:
        return None, None
    keep = split_factor(pm["lines"] * probs["I"] * mult)
    ev = probs["I"] * pm["J"] * keep
    for c, (val, fixed) in pm["typical"].items():
        k, first5, _ = CATS[key][c]
        ev += probs[c] * (val if fixed else val * mult ** (-(k / (5 if first5 else drawn)) * (k / pick)))
    return ev, keep


def trend_weights(draws, n, h=50):
    """Experimental 'trend' score input: the Hot (~50 draws) model from the backtest, as a probability per ball."""
    decay, w = 0.5 ** (1 / h), [0.0] * (n + 1)
    for dr in draws:
        w = [v * decay for v in w]
        for x in dr["nums"]:
            w[x] += 1
    s = sum(w[1:]) + n
    return [0] + [0.8 / n + 0.2 * (w[x] + 1) / s for x in range(1, n + 1)]


def trend_score(nums, q, n):
    """+ = these numbers came up more than usual lately. The backtest says this has no predictive value;
    it's saved with every pick so the track record can show whether that holds."""
    return round(100 * sum(math.log(q[x] * n) for x in nums), 1)


def p_any(key, n, d, k, xn):
    lo, lo_j = MIN_WIN[key]
    h = [hypergeom(n, d, k, j) if j <= min(d, k) else 0 for j in range(max(d, k) + 1)]
    p = sum(h[lo:])
    if lo_j is not None:
        p += sum(h[lo_j:lo]) / xn
    return p


def make_picks(key, g, D, pop, probs, pm, seed):
    n, pick, xn = g["n"], g["pick"], g["extra"]
    rnd, last, q = random.Random(seed), D[-1]["nums"], trend_weights(D, n)
    pool = []
    for _ in range(60000):
        c = sorted(rnd.sample(range(1, n + 1), pick))
        e = rnd.randint(1, xn) if xn else None
        pool.append((ticket_pop(c, e, pop, last), c, e))
    norm = sum(math.exp(lp) for lp, _, _ in pool) / len(pool)  # average ticket = 1x
    pool.sort(key=lambda t: t[0])

    def card(lp, c, e):
        mult = math.exp(lp) / norm
        ev, keep = line_ev(key, probs, pm, mult, pick, g["drawn"])
        return dict(nums=c, extra=e, mult=round(mult, 3), ev=ev and round(ev, 3), keep=keep and round(keep, 4),
                    trend=trend_score(c, q, n))

    chosen = []
    for overlap in (1, 2, 3, pick):  # least popular first; spread the board, loosening only when needed
        for lp, c, e in pool[:6000]:
            if len(chosen) == TICKETS:
                break
            if all(c != o["nums"] and len(set(c) & set(o["nums"])) <= overlap for o in chosen):
                chosen.append(card(lp, c, e))
    avg_ev, avg_keep = line_ev(key, probs, pm, 1.0, pick, g["drawn"])

    systems = {}
    for k in range(pick + 1, pick + 5):  # system tickets: k numbers = every pick-sized line among them
        cand = []
        for _ in range(15000):
            c = sorted(rnd.sample(range(1, n + 1), k))
            cand.append((ticket_pop(c, None, pop, last), rnd.random(), c))  # random tie-break: no bias to low numbers
        cand.sort()
        e = min(range(1, xn + 1), key=lambda x: pop["bj"][x]) if xn and pop else (1 if xn else None)
        out = []
        for _, _, c in cand[:2000]:
            if len(out) == 3:
                break
            if any(len(set(c) & set(o["nums"])) > k // 2 for o in out):
                continue
            lines = list(combinations(c, pick))
            ev = sum(line_ev(key, probs, pm, math.exp(ticket_pop(ln, e, pop, last)) / norm, pick, g["drawn"])[0] or 0 for ln in lines)
            out.append(dict(nums=c, extra=e, lines=len(lines), cost=len(lines) * g["price"],
                            p_jackpot=len(lines) * probs["I"], p_any=p_any(key, g["n"], g["drawn"], k, xn),
                            ev=round(ev, 2) if pm else None, trend=trend_score(c, q, n)))
        systems[k] = sorted(out, key=lambda t: -(t["ev"] or 0) / t["cost"])  # strongest by payback per leu
    return chosen, systems, dict(ev=avg_ev and round(avg_ev, 3), keep=avg_keep and round(avg_keep, 4))


def payout(key, prize_row, cat):
    """What a winning line actually received that draw (sole winner of a rolled-over category gets the pot)."""
    if not prize_row or cat not in prize_row["cats"]:
        return 0.0
    w, v, r = prize_row["cats"][cat]
    return v if w else v + r


def track_record(key, g, ledger, D, prizes):
    """Score every saved pick against the draw(s) on its date, with the official payouts of that draw."""
    by_date = {}
    for dr in D:
        by_date.setdefault(dr["date"], []).append(dr)
    prow = {}
    for p in prizes:
        if p["nums"]:
            prow[(p["date"], tuple(sorted(p["nums"])))] = p
    pick, xn = g["pick"], g["extra"]

    def score(nums, extra, dr):
        hits = len(set(nums) & set(dr["nums"]))
        bonus = extra == dr["extra"] if xn else set(nums) <= set(dr["nums"][:5])
        cat = prize_category(key, hits, bonus)
        return hits, cat, payout(key, prow.get((dr["date"], tuple(sorted(dr["nums"])))), cat) if cat else 0.0

    dates, rows_for_trend = [], []
    tiers = {1: [0, 0.0, {}], 5: [0, 0.0, {}], 10: [0, 0.0, {}], TICKETS: [0, 0.0, {}]}
    for date in sorted({r["date"] for r in ledger}):
        if date not in by_date:
            continue
        saved = [r for r in ledger if r["date"] == date]
        singles = sorted((r for r in saved if r["kind"] == "single"), key=lambda r: r["rank"])
        entry = dict(date=date, draws=[[dr["nums"], dr["extra"]] for dr in by_date[date]], tickets=[], systems=[])
        for r in singles:
            for dr in by_date[date]:
                hits, cat, won = score(r["nums"], r["extra"], dr)
                if r["rank"] <= 10:
                    entry["tickets"].append(dict(rank=r["rank"], nums=r["nums"], extra=r["extra"], hits=hits, cat=cat, won=won))
                for t, acc in tiers.items():
                    if r["rank"] <= t:
                        acc[0] += 1
                        acc[1] += won
                        if cat:
                            acc[2][cat] = acc[2].get(cat, 0) + 1
                if r["meta"].get("trend") is not None:
                    rows_for_trend.append((r["meta"]["trend"], hits))
        for r in (r for r in saved if r["kind"] == "system"):
            for dr in by_date[date]:
                won, cats = 0.0, {}
                for ln in combinations(r["nums"], pick):
                    _, cat, w = score(ln, r["extra"], dr)
                    won += w
                    if cat:
                        cats[cat] = cats.get(cat, 0) + 1
                entry["systems"].append(dict(size=len(r["nums"]), nums=r["nums"], extra=r["extra"],
                                             cost=r["meta"].get("cost"), won=won, cats=cats))
        dates.append(entry)
    e = pick * g["drawn"] / g["n"]
    var = e * (g["n"] - pick) / g["n"] * (g["n"] - g["drawn"]) / (g["n"] - 1)

    def paper(kind):  # tickets tracked on paper: hits vs chance on future draws only
        h = [len(set(r["nums"]) & set(dr["nums"])) for r in ledger if r["kind"] == kind for dr in by_date.get(r["date"], [])]
        return h and dict(draws=len(h), hits=round(sum(h) / len(h), 3), chance=round(e, 3),
                          z=round((sum(h) / len(h) - e) / math.sqrt(var / len(h)), 2))
    candidate, lab = paper("candidate"), paper("formula")
    summary = {t: dict(lines=a[0], cost=a[0] * g["price"], won=round(a[1], 2), wins=a[2]) for t, a in tiers.items() if a[0]}
    trend = None
    if len(rows_for_trend) >= 30:
        rows_for_trend.sort()
        third = len(rows_for_trend) // 3
        avg = lambda xs: round(sum(h for _, h in xs) / len(xs), 3)
        trend = dict(lines=len(rows_for_trend), low=avg(rows_for_trend[:third]), high=avg(rows_for_trend[-third:]),
                     chance=round(pick * g["drawn"] / g["n"], 3))
    return dict(dates=dates[::-1], summary=summary, trend=trend, candidate=candidate or None, lab=lab or None)


def next_draws(now, k=4):
    out, day = [], now.date()
    while len(out) < k:
        if day.weekday() in (3, 6):
            t = dt.datetime(day.year, day.month, day.day, *DRAW_HOUR, tzinfo=TZ)
            if t > now:
                out.append(t.isoformat())
        day += dt.timedelta(days=1)
    return out


def prev_draw_date(now):
    """Most recent regular draw date whose draw time has passed."""
    t = now - dt.timedelta(hours=DRAW_HOUR[0], minutes=DRAW_HOUR[1])
    day = t.date()
    while day.weekday() not in (3, 6):
        day -= dt.timedelta(days=1)
    return day.isoformat()


def lab_ticket(key, D, date, n, pick):
    """Formula Lab ticket for the draw on `date`, from formulas.json (written by lab.py)."""
    try:
        lab = json.load(open(Path(__file__).with_name("formulas.json")))
    except (OSError, ValueError):
        return None
    L = lab.get(key)
    if not L:
        return None
    feats = formula_features(date, D[-1]["nums"], D[-2]["nums"] if len(D) > 1 else [], len(D))
    return dict(nums=formula_ticket(L["formulas"], feats, n, pick), evolved=lab.get("evolved"),
                formulas=[formula_text(tuple(f), n) for f in L["formulas"][:5]],
                **{k: L[k] for k in ("train", "val", "test", "test_hits", "chance_hits", "test_draws")})


def analyze(games, draws, jackpots, announcements, prizes):
    now = dt.datetime.now(TZ)
    out = dict(generated=now.isoformat(), next_draws=next_draws(now), prev_draw=prev_draw_date(now),
               announcements=announcements, games={})
    for key, g in games.items():
        D = draws[key]
        if not D:
            continue
        n, d, pick, xn = g["n"], g["drawn"], g["pick"], g["extra"]
        probs = cat_probs(key, n, d, pick, xn)
        nums = number_stats(D, n, d)
        pc = {}
        for dr in D:
            for a, b in combinations(dr["nums"], 2):
                pc[(a, b)] = pc.get((a, b), 0) + 1
        exp_pair = len(D) * d * (d - 1) / (n * (n - 1))
        top_pairs = [dict(a=a, b=b, count=c) for (a, b), c in sorted(pc.items(), key=lambda kv: -kv[1])[:10]]
        odd = [sum(x % 2 for x in dr["nums"]) for dr in D]
        P = prizes.get(key, [])
        for p in P:
            p["price"] = g["price"]  # ponytail: today's price for all years; year effects absorb old prices
        pop = fit_popularity(key, n, d, pick, xn, P)
        jp = jackpots.get(key, [])
        pm = prize_model(key, g, P, jp[-1]["lei"] if jp else None, pop)
        picks, systems, avg = make_picks(key, g, D, pop, probs, pm, seed=len(D))
        lower = sum(probs[c] * v for c, (v, _) in (pm["typical"].items() if pm else []))
        keep_avg = avg["keep"] or 1
        out["games"][key] = dict(
            name=g["name"], n=n, drawn=d, pick=pick, extra=xn, price=g["price"],
            total=len(D), first=D[0]["date"], last=D[-1]["date"],
            history=[[x["date"], x["seq"], x["nums"], x["extra"]] for x in D],
            numbers=nums, p_ball=round(d / n, 4),
            extra_numbers=number_stats(D, xn, 1, key="extra") if xn else None,
            odds={c: round(1 / p) for c, p in sorted(probs.items(), key=lambda kv: kv[1])},
            p_any=p_any(key, n, d, pick, xn),
            fairness=dict(all=chi2_uniform([x["count"] for x in nums]),
                          recent=chi2_uniform([sum(x in dr["nums"] for dr in D[-500:]) for x in range(1, n + 1)]),
                          extra=chi2_uniform([x["count"] for x in number_stats(D, xn, 1, key="extra")]) if xn else None),
            odd_split=[dict(k=k, seen=round(odd.count(k) / len(D), 4), theory=round(hypergeom(n, sum(x % 2 for x in range(1, n + 1)), d, k), 4))
                       for k in range(d + 1)],
            pairs=dict(expected=round(exp_pair, 1), top=top_pairs),
            backtest=backtest(D, n, d, pick, MIN_WIN[key][0]),
            structure=structure_tests(D, n, d),
            popularity=pop and dict(mult=[round(math.exp(v), 3) for v in pop["b"]],
                                    joker_mult=[round(math.exp(v), 3) for v in pop["bj"]] if pop["bj"] else None,
                                    draws=pop["draws"], since=pop["since"], holdout=pop["holdout"], improve=pop["improve"],
                                    reliable=pop["reliable"],
                                    evidence=evidence(key, P, pop) if pop["reliable"] else None),
            picks=picks, systems=systems,
            lab=lab_ticket(key, D, next_draws(now)[0][:10], n, pick),
            candidate=dict(nums=sorted(sorted(range(1, n + 1), key=lambda x: (-trend_weights(D, n)[x], x))[:pick]), extra=None,
                           rule="Hot numbers, last ~50 draws", registered="2026-10-08"),
            ev=pm and dict(jackpot=round(pm["J"]), as_of=jp[-1]["date"] if jp else P[-1]["date"], price=g["price"],
                           lines=round(pm["lines"]), ev_avg=avg["ev"], keep_avg=avg["keep"],
                           prizes={"I": round(pm["J"]), **{c: round(v, 2) for c, (v, _) in pm["typical"].items()}},
                           breakeven=round((g["price"] - lower) / probs["I"] / keep_avg), history=jp[-60:]),
            p_jackpot=probs["I"])
    return out


if __name__ == "__main__":  # self-check: math + a fair simulated history must look fair and unbeatable
    assert abs(sum(hypergeom(49, 6, 6, k) for k in range(7)) - 1) < 1e-12
    assert round(1 / hypergeom(49, 6, 6, 6)) == 13983816
    assert abs(chi2_sf(48, 48) - 0.47) < 0.02 and chi2_sf(100, 48) < 0.001
    assert prize_category("joker", 5, True) == "I" and prize_category("joker", 2, False) is None
    assert prize_category("540", 5, True) == "I" and prize_category("540", 5, False) == "II"
    assert abs(split_factor(1.0) - 0.632) < 0.001 and split_factor(0) == 1
    rnd = random.Random(1)
    fake = [dict(date="x", seq=1, nums=sorted(rnd.sample(range(1, 50), 6)), extra=None) for _ in range(3000)]
    bt = backtest(fake, 49, 6, 6)
    for name, m in bt["models"].items():  # on truly random data no system may "win" by > 3 SE
        assert m.get("llr", 0) < 3 * m.get("se", 0) + 0.5 and m["z"] < 3.5, (name, m)
    assert ticket_pop([1, 2, 3, 4, 5, 6], None, None, []) > ticket_pop([33, 38, 41, 44, 46, 49], None, None, [])
    assert abs(sum(cat_probs("joker", 45, 5, 5, 20).values()) - p_any("joker", 45, 5, 5, 20)) < 1e-12
    assert abs(1 / cat_probs("540", 40, 6, 5, None)["II"] - 131602) < 1
    # popularity fit must recover planted popularity from simulated winner counts
    rnd2, true_b = random.Random(7), [0] + [0.25 if x <= 12 else (-0.15 if x > 31 else 0) for x in range(1, 50)]
    sim, probs = [], cat_probs("649", 49, 6, 6, None)
    for t in range(700):
        S = rnd2.sample(range(1, 50), 6)
        cats = {}
        for c, (k, _, _) in CATS["649"].items():
            lam = 2e6 * probs[c] * math.exp(k / 6 * sum(true_b[x] for x in S))
            L, y, pr = math.exp(-lam), 0, 1.0
            if lam > 50:
                y = max(0, round(rnd2.gauss(lam, math.sqrt(lam))))
            else:
                while True:
                    pr *= rnd2.random()
                    if pr < L:
                        break
                    y += 1
            cats[c] = [y, 0, 0]
        sim.append(dict(date=f"{2012 + t // 100}-01-01", seq=1, nums=S, extra=None, cats=cats, fund=0, price=1))
    sales_series_orig = globals()["sales_series"]
    globals()["sales_series"] = lambda P: {(p["date"], p["seq"]): 2e6 for p in P}
    fit = fit_popularity("649", 49, 6, 6, None, sim)
    globals()["sales_series"] = sales_series_orig
    centred = [v - sum(true_b[1:]) / 49 for v in true_b]
    err = max(abs(fit["b"][x] - centred[x]) for x in range(1, 50))
    assert err < 0.06 and fit["improve"] > 0.2, (err, fit["improve"])
    print("model ok; popularity fit err", round(err, 3), "holdout improvement", fit["improve"])
