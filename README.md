# Loto odds: 6/49, 5/40, Joker

Python stdlib only (3.9+). There's nothing to install.

```sh
python3 loto.py backfill        # once: full history since 1993 -> loto.db, data.json
python3 loto.py update          # latest results, jackpots, special-draw announcements, ticket check, notify
python3 model.py                # self-check of the maths
python3 lab.py                  # Formula Lab: evolve a new generation of formulas -> formulas.json (~30s)
python3 -m http.server 8649     # open http://localhost:8649
```

- `tickets.json`: the lines you actually play (Joker: 6th number is the Joker). Checked after every new draw.
- Notifications (optional env vars): `NTFY_TOPIC` (push via ntfy.sh app) and/or
  `SMTP_HOST SMTP_PORT SMTP_USER SMTP_PASS MAIL_TO` (email).
- Prices are in `GAMES` at the top of `loto.py`.
- Before each draw the 100 ranked tickets and the system tickets are saved to the `ledger` table (once the
  previous results are in), shown unchanged until the draw, then scored with that draw's official payouts.

## VPS cron (`crontab -e`)

```cron
CRON_TZ=Europe/Bucharest
# results nights: every 15 min after the 18:30 draw until they're published
*/15 19-23 * * 0,4  cd /srv/loto && NTFY_TOPIC=your-topic python3 loto.py update >> update.log 2>&1
# every day: jackpots, special-draw announcements, extra holiday draws
0 9 * * *           cd /srv/loto && NTFY_TOPIC=your-topic python3 loto.py update >> update.log 2>&1
```

Serve the folder with nginx (static: `index.html` + `data.json`). Don't serve `loto.db` and `tickets.json` publicly:

```nginx
location ~ \.(db|py|log)$|tickets\.json { deny all; }
```

## Data

- Draw history: loto49.ro archives (6/49 since 1993, 5/40 since 1995, Joker since 2000).
- Official prize tables per draw since 2012 (winners and payout per category, prize fund, drawn numbers):
  loto.ro results pages. They also fill draws the archive is missing. Sales per draw = fund change / 40%.

## What the "algorithm" is

Draws are independent and uniform, so no model can predict them. The page shows that honestly with a
walk-forward backtest scored by log-likelihood vs chance (Benter's yardstick). It also monitors fairness
with chi-square tests. The one real lever is picking combinations other players avoid, which raises
your share of the jackpot if you win. Which numbers players favour is fitted Benter-style: a Poisson
regression of official winner counts on the drawn numbers (offset by sales x odds), validated on held-out draws.
Suggested tickets are ranked by that, with full expected payback across all prize categories, and spread across
the board. Each pick also carries an experimental "trend score" so the track record can test it.

Prediction study (reruns on every build): 15 systems, from hot/cold/overdue to Fourier cycles, an online
logistic model and a context-mixture universal predictor, each predicting every past draw from earlier draws
only; Bonferroni-corrected, with a first-half/second-half out-of-sample check. Structure tests: draw-to-draw
overlap vs hypergeometric, and the largest eigenvalue of the number correlation matrix vs simulated fair lotteries.
None finds an edge. What does change outcomes is how lines are arranged: the bundle calculator compares system
tickets with a covering design of the same lines (same cost, jackpot odds and average; very different spread).
