#!/usr/bin/env python3
"""Formula Lab: evolve formulas f(date, previous draws) -> number, 3 rounds of learning, then a one-time final test.
Splits history chronologically: train 60% / validation 20% / test 20%. Writes formulas.json for the app.
  python3 lab.py"""
import json, random, sys, time
from pathlib import Path
import loto, model

ROOT = Path(__file__).resolve().parent
KEEP = 30  # ranked formulas per game; the ticket takes the first `pick` distinct outputs


def rate(f, rows, n):
    return sum(formula_hit(f, r, n) for r in rows) / len(rows)


def formula_hit(f, r, n):
    feats, S = r
    return model.formula_value(f, feats, n) in S


def canon(f):  # equivalent formulas collapse to one (0 x feature, unused second coefficient in products)
    a, i, b, j, c, op = f
    if op:
        return (a, min(i, j), 0, max(i, j), c, 1)
    return (a, i, 0, 0, c, 0) if b == 0 else (a, i, b, j, c, 0)


def evolve(rows, n, rnd, seeds=(), pop=800, gens=25, ops=(0, 1)):
    F = len(model.FEATURES)
    rand = lambda: canon((rnd.randint(1, 12), rnd.randrange(F), rnd.randint(0, 12), rnd.randrange(F), rnd.randrange(n), rnd.choice(ops)))
    def mutate(f):
        f = list(f)
        k = rnd.randrange(6)
        f[k] = {0: rnd.randint(1, 12), 1: rnd.randrange(F), 2: rnd.randint(0, 12), 3: rnd.randrange(F), 4: rnd.randrange(n), 5: rnd.choice(ops)}[k]
        return canon(tuple(f))
    popl = list(seeds) + [rand() for _ in range(pop - len(seeds))]
    for _ in range(gens):
        scored = sorted(set(popl), key=lambda f: -rate(f, rows, n))
        elite = scored[:50]
        popl = elite + [mutate(rnd.choice(elite)) for _ in range(pop - 50 - 50)] + [rand() for _ in range(50)]
    return sorted(set(popl), key=lambda f: -rate(f, rows, n))


def run(key, g, D, rnd, log):
    n, pick, p = g["n"], g["pick"], g["drawn"] / g["n"]
    rows = [(model.formula_features(x["date"], D[t - 1]["nums"] if t else [], D[t - 2]["nums"] if t > 1 else [], t), set(x["nums"]))
            for t, x in enumerate(D)][2:]
    a, b = int(len(rows) * .6), int(len(rows) * .8)
    train, val, test = rows[:a], rows[a:b], rows[b:]
    seeds = []
    for rnd_i in range(3):  # learn: evolve on train, keep what holds on validation, breed from those next round
        cands = evolve(train, n, rnd, seeds)[:120]
        ranked = sorted(cands, key=lambda f: -rate(f, val, n))
        seeds = ranked[:20]
        best = ranked[:KEEP]
        tr = sum(rate(f, train, n) for f in best) / KEEP / p
        va = sum(rate(f, val, n) for f in best) / KEEP / p
        log(f"   round {rnd_i + 1}: top {KEEP} formulas hit {tr:.2f}x chance on training, {va:.2f}x on validation")
    te = sum(rate(f, test, n) for f in best) / KEEP / p
    hits = [len(set(model.formula_ticket(best, feats, n, pick)) & S) for feats, S in test]
    e = pick * p
    log(f"   final test (touched once, {len(test)} draws): formulas {te:.2f}x chance; formula ticket {sum(hits)/len(hits):.3f} hits vs {e:.3f} chance")
    return dict(formulas=best, train=round(tr, 3), val=round(va, 3), test=round(te, 3),
                test_hits=round(sum(hits) / len(hits), 4), chance_hits=round(e, 4), test_draws=len(test))


if __name__ == "__main__":
    draws, *_ = loto.load(loto.db())
    out, t0 = {}, time.time()
    for key, g in loto.GAMES.items():
        print(f"== {g['name']}")
        out[key] = run(key, g, draws[key], random.Random(2026), print)
        for f in out[key]["formulas"][:3]:
            print("   ", model.formula_text(f, g["n"]))
    out["evolved"] = loto.dt.date.today().isoformat()
    (ROOT / "formulas.json").write_text(json.dumps(out))
    print(f"wrote formulas.json ({time.time() - t0:.0f}s)")
