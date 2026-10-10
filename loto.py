#!/usr/bin/env python3
"""Romanian lottery (6/49, 5/40, Joker): scrape history -> SQLite -> stats -> data.json.

  python3 loto.py backfill   # first run: every archive page since 1993
  python3 loto.py update     # cron: latest pages, jackpots, announcements, check tickets, notify
  python3 loto.py build      # just rebuild data.json from the DB
"""
import datetime as dt
import html
import json
import os
import re
import smtplib
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from email.message import EmailMessage
from pathlib import Path

import model

ROOT = Path(__file__).resolve().parent
DB = ROOT / "loto.db"
OUT = ROOT / "data.json"
TICKETS = ROOT / "tickets.json"
UA = "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/130 Safari/537.36"
ARCH = "https://www.loto49.ro/"

# ponytail: prices from July 2026 press; edit when Loteria Română changes them.
GAMES = {
    "649":   dict(name="Loto 6/49", n=49, drawn=6, pick=6, extra=None, price=9,
                  prizes=("649_si_noroc/rezultate_extragere.html", "6/49"),
                  current=["arhiva-loto49.php"],
                  old=["arhiva-loto49-1993-2000.php", "arhiva-loto49-2001-2005.php", "arhiva-loto49-2006-2010.php",
                       "arhiva-loto49-2011-2015.php", "arhiva-loto-6-49-din-perioada-2016-2019.php",
                       "arhiva-loto-6-49-din-perioada-2020-2023.php"]),
    "540":   dict(name="Loto 5/40", n=40, drawn=6, pick=5, extra=None, price=6,
                  prizes=("540_si_super_noroc/rezultate_extrageri.html", "5/40"),
                  current=["arhiva-superloto.php"], old=[]),
    "joker": dict(name="Joker", n=45, drawn=5, pick=5, extra=20, price=8,
                  prizes=("joker_si_noroc_plus/rezultate_extrageri.html", "joker"),
                  current=["arhiva-joker.php"],
                  old=["arhiva-joker-2000-2005.php", "arhiva-joker-2006-2010.php", "arhiva-joker-2011-2015.php"]),
}


PRIZES = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/"
PRIZES_FROM = 2012  # winner counts per category; enough draws to fit player popularity


def get(url, data=None):
    req = urllib.request.Request(url, data=data and urllib.parse.urlencode(data).encode(), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "ignore")


def text(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def parse_archive(page, g):
    """Yield (date, seq, nums, extra). Same-date rows = double draws (holidays), numbered in page order."""
    seen, rows = {}, []
    for c in (text(c) for c in re.findall(r"(?s)<td[^>]*>(.*?)</td>", page)):
        if m := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", c):  # split on date cells: <tr> tags often missing
            rows.append([f"{int(m[1]):04}-{int(m[2]):02}-{int(m[3]):02}"])
        elif rows:
            rows[-1].append(c)
    for cells in rows:
        vals = [int(x) for c in cells[1:] for x in re.findall(r"\d+", c)]
        nums, extra = (vals[:5], vals[5] if len(vals) > 5 else None) if g["extra"] else (vals[:g["drawn"]], None)
        # validate: wrong-format rows (typos, old game rules) are skipped, not guessed at
        if len(nums) != g["drawn"] or len(set(nums)) != g["drawn"] or not all(1 <= x <= g["n"] for x in nums):
            print(f"  skip {g['name']} {cells[0]}: {cells[1:]}", file=sys.stderr)
            continue
        if g["extra"] and not (extra and 1 <= extra <= g["extra"]):
            print(f"  skip {g['name']} {cells[0]}: bad joker {extra}", file=sys.stderr)
            continue
        seen[cells[0]] = seen.get(cells[0], 0) + 1
        yield cells[0], seen[cells[0]], nums, extra  # draw order kept: 5/40 jackpot needs the first 5


def db():
    con = sqlite3.connect(DB)
    con.executescript("""
      CREATE TABLE IF NOT EXISTS draws(game TEXT, date TEXT, seq INT, nums TEXT, extra INT,
                                       PRIMARY KEY(game, date, seq));
      CREATE TABLE IF NOT EXISTS prizes(game TEXT, date TEXT, seq INT, nums TEXT, extra INT, cats TEXT, fund REAL,
                                        PRIMARY KEY(game, date, seq));
      CREATE TABLE IF NOT EXISTS ledger(game TEXT, draw_date TEXT, kind TEXT, rank INT, nums TEXT, extra INT,
                                        meta TEXT, saved_at TEXT, PRIMARY KEY(game, draw_date, kind, rank));
      CREATE TABLE IF NOT EXISTS jackpots(game TEXT, seen_on TEXT, lei REAL, PRIMARY KEY(game, seen_on));
      CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);""")
    return con


def scrape(con, full):
    new = []
    for key, g in GAMES.items():
        for p in (g["old"] if full else []) + g["current"]:
            print(f"fetch {p}", file=sys.stderr)
            for d, seq, nums, extra in parse_archive(get(ARCH + p), g):
                cur = con.execute("INSERT OR IGNORE INTO draws VALUES(?,?,?,?,?)", (key, d, seq, json.dumps(nums), extra))
                if cur.rowcount:
                    new.append((key, d, seq, nums, extra))
    con.commit()
    return new


def num(s):
    if re.fullmatch(r"\d{1,3}(,\d{3})+\.\d\d", s):  # a few cells use English format: 1,804,236.20
        return float(s.replace(",", ""))
    return float(s.replace(".", "").replace(",", "."))


def parse_prizes(page, label):
    """Official results page -> (date, seq, nums, extra, {cat: [winners, value_lei, report_lei]}, total_fund).
    winners = 0 when the category rolled over (REPORT); value is then this draw's contribution.
    Each table follows its own ball images; holiday "EXTRAGERI SPECIALE" tables get seq 2."""
    raw, out, balls = re.sub(r"\s+", " ", page), [], []
    for m in re.finditer(r'bile/([^"]+?)\.png|(?P<reset>numere-extrase)|Detalii castiguri (EXTRAGERI SPECIALE )?la (\S+) din <span>(\d\d)\.(\d\d)\.(\d{4})'
                         r"</span>(.*?)Fond total de castiguri: (?:<[^>]+> ?)*([\d.,]+)", raw):
        if m["reset"]:
            balls = []
            continue
        if m[1]:
            balls.append(m[1])
            continue
        main = [int(x) for x in balls if x.isdigit()]
        jok = [int(x.split("/")[1]) for x in balls if x.startswith("joker/")]
        balls = []
        if m[4] != label:
            continue
        ok = len(main) in (5, 6) and len(jok) <= 1  # anything else is ambiguous: keep prizes, drop numbers
        cats = {}
        for c, w, v, r in re.findall(r"\b(I|II|III|IV|V|VI|VII|VIII) \([^)]*\) (REPORT|[\d.]+) ([\d.,]+|-)(?: ([\d.,]+|-))?", text(m[8])):
            try:
                cats[c] = [0 if w == "REPORT" else int(num(w)), 0 if v == "-" else num(v), num(r) if r and r != "-" else 0]
            except ValueError:  # the site has a few garbled cells (e.g. "1.804.23620"); skip that row only
                print(f"  skip garbled prize row {m[7]}-{m[6]}-{m[5]} {c}", file=sys.stderr)
        if cats:
            out.append((f"{m[7]}-{m[6]}-{m[5]}", 2 if m[3] else 1, main if ok else None,
                        jok[0] if ok and jok else None, cats, num(m[9])))
    return out[::-1]  # page is newest first


def scrape_prizes(con, full):
    today = dt.date.today()
    months = [(y, m) for y in range(PRIZES_FROM, today.year + 1) for m in range(1, 13) if (y, m) <= (today.year, today.month)]
    if not full:  # update: this month + last month (covers month boundaries and late postings)
        months = months[-2:]
    for key, g in GAMES.items():
        path, label = g["prizes"]
        for y, m in months:
            try:
                rows = parse_prizes(get(PRIZES + path, {"select-year": y, "select-month": m}), label)
            except Exception as e:
                print(f"  prizes {key} {y}-{m:02} failed: {e}", file=sys.stderr)
                continue
            con.executemany("INSERT OR REPLACE INTO prizes VALUES(?,?,?,?,?,?,?)",
                            [(key, d, s, n and json.dumps(n), x, json.dumps(c), f) for d, s, n, x, c, f in rows])
            # official numbers fill dates the loto49.ro archive is missing (it skips many 5/40 draws)
            con.executemany("INSERT INTO draws SELECT ?,?,?,?,? WHERE NOT EXISTS "
                            "(SELECT 1 FROM draws WHERE game=? AND date=?)",
                            [(key, d, s, json.dumps(n), x, key, d) for d, s, n, x, c, f in rows if n])
            if full:
                print(f"  prizes {key} {y}-{m:02}: {len(rows)}", file=sys.stderr)
                time.sleep(0.3)  # be polite to loto.ro
        con.commit()


def scrape_jackpots(con):
    t = text(get("https://www.loto.ro/"))
    today = dt.date.today().isoformat()
    for key, label in [("649", "LOTO 6/49"), ("540", "LOTO 5/40"), ("joker", "JOKER")]:
        m = re.search(r"REPORT " + re.escape(label) + r" REPORTURI ([\d.,]+)", t)
        if m:
            lei = float(m.group(1).replace(".", "").replace(",", "."))
            con.execute("INSERT OR REPLACE INTO jackpots VALUES(?,?,?)", (key, today, lei))
        else:
            print(f"  jackpot for {label} not found on loto.ro (layout changed?)", file=sys.stderr)
    con.commit()


def scrape_announcements(con):
    """No public API for special/double draws; loto.ro announces them as posts. Keep the recent ones."""
    items = {}
    for q in ["trageri speciale", "tragere suplimentara", "trageri duble"]:
        page = get("https://www.loto.ro/?s=" + q.replace(" ", "+"))
        for href, title, when in re.findall(
                r'(?s)<h3 class="entry-title"><a href="([^"]+)"[^>]*>(.*?)</a>.*?datetime="([^"]+)"', page):
            items[href] = dict(url=href, title=text(title), date=when[:10])
    cutoff = (dt.date.today() - dt.timedelta(days=90)).isoformat()
    recent = sorted((i for i in items.values() if i["date"] >= cutoff), key=lambda i: i["date"], reverse=True)
    con.execute("INSERT OR REPLACE INTO meta VALUES('announcements', ?)", (json.dumps(recent),))
    con.commit()


def load(con):
    draws = {k: [] for k in GAMES}
    for key, d, seq, nums, extra in con.execute("SELECT * FROM draws ORDER BY date, seq"):
        draws[key].append(dict(date=d, seq=seq, nums=json.loads(nums), extra=extra))
    jp = {k: [] for k in GAMES}
    for key, d, lei in con.execute("SELECT * FROM jackpots ORDER BY seen_on"):
        jp[key].append(dict(date=d, lei=lei))
    ann = con.execute("SELECT v FROM meta WHERE k='announcements'").fetchone()
    prizes = {k: [] for k in GAMES}
    for key, d, seq, nums, extra, cats, fund in con.execute("SELECT * FROM prizes ORDER BY date, seq"):
        prizes[key].append(dict(date=d, seq=seq, nums=nums and json.loads(nums), extra=extra, cats=json.loads(cats), fund=fund))
    return draws, jp, json.loads(ann[0]) if ann else [], prizes


def ledger(con, key, date=None):
    q = "SELECT draw_date, kind, rank, nums, extra, meta, saved_at FROM ledger WHERE game=?" + (" AND draw_date=?" if date else "")
    return [dict(date=d, kind=k, rank=r, nums=json.loads(n), extra=x, meta=json.loads(m), saved_at=t)
            for d, k, r, n, x, m, t in con.execute(q + " ORDER BY draw_date, kind, rank", (key, date) if date else (key,))]


def freeze(con, key, g, date):
    """Save this draw's picks once, before the draw, so they can be scored honestly afterwards."""
    now = dt.datetime.now().isoformat(timespec="seconds")
    rows = [(key, date, "single", i, json.dumps(t["nums"]), t["extra"],
             json.dumps({k: v for k, v in t.items() if k not in ("nums", "extra")}), now) for i, t in enumerate(g["picks"], 1)]
    rows += [(key, date, "system", int(size) * 10 + i, json.dumps(t["nums"]), t["extra"],
              json.dumps({k: v for k, v in t.items() if k not in ("nums", "extra")}), now)
             for size, ts in g["systems"].items() for i, t in enumerate(ts)]
    c = g["candidate"]  # pre-registered hypothesis ticket, scored on future draws only
    rows.append((key, date, "candidate", 1, json.dumps(c["nums"]), c["extra"], json.dumps({"rule": c["rule"]}), now))
    if g.get("lab"):
        rows.append((key, date, "formula", 1, json.dumps(g["lab"]["nums"]), None, json.dumps({"evolved": g["lab"]["evolved"]}), now))
    con.executemany("INSERT OR IGNORE INTO ledger VALUES(?,?,?,?,?,?,?,?)", rows)
    con.commit()


def check_tickets(new):
    """tickets.json: {"649": [[1,2,3,4,5,6]], "joker": [[1,2,3,4,5,9]] (last = joker)}"""
    if not TICKETS.exists() or not new:
        return []
    mine, lines = json.loads(TICKETS.read_text()), []
    for key, d, seq, nums, extra in new:
        g = GAMES[key]
        for t in mine.get(key, []):
            main, j = (t[:5], t[5]) if g["extra"] else (t, None)
            hits = len(set(main) & set(nums))
            prize = model.prize_category(key, hits, j == extra if g["extra"] else set(main) <= set(nums[:5]))
            lines.append(f"{g['name']} {d}#{seq} {nums}{' +' + str(extra) if extra else ''} | "
                         f"ticket {t}: {hits} hit{'s' * (hits != 1)}{' + Joker' if j and j == extra else ''}"
                         f"{' -> WIN category ' + prize if prize else ''}")
    return lines


def notify(subject, body):
    """Push via ntfy.sh (NTFY_TOPIC) and/or email via SMTP (SMTP_HOST, SMTP_USER, SMTP_PASS, MAIL_TO)."""
    print(f"{subject}\n{body}")
    if topic := os.environ.get("NTFY_TOPIC"):
        urllib.request.urlopen(urllib.request.Request(
            f"https://ntfy.sh/{topic}", data=body.encode(), headers={"Title": subject}), timeout=30)
    if host := os.environ.get("SMTP_HOST"):
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, os.environ["SMTP_USER"], os.environ["MAIL_TO"]
        msg.set_content(body)
        with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", 587))) as s:
            s.starttls()
            s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
            s.send_message(msg)


def build(con):
    draws, jp, ann, prizes = load(con)
    data = model.analyze(GAMES, draws, jp, ann, prizes)
    nxt = data["next_draws"][0][:10]
    for key, g in data["games"].items():
        saved = ledger(con, key, nxt)
        # ponytail: freezes only once the latest regular draw is in the DB; a cancelled regular draw (holiday)
        # delays freezing until the next results arrive, so that one draw goes untracked
        if not saved and g["last"] >= data["prev_draw"]:
            freeze(con, key, g, nxt)
            saved = ledger(con, key, nxt)
        if saved:  # show exactly what was saved for this draw, never a regenerated set
            g["picks"] = [dict(nums=r["nums"], extra=r["extra"], **r["meta"]) for r in saved if r["kind"] == "single"]
            systems = {}
            for r in saved:
                if r["kind"] == "system":
                    systems.setdefault(len(r["nums"]), []).append(dict(nums=r["nums"], extra=r["extra"], **r["meta"]))
            g["systems"] = systems or g["systems"]
            if g.get("lab"):
                g["lab"]["nums"] = next((r["nums"] for r in saved if r["kind"] == "formula"), g["lab"]["nums"])
            g["candidate"] = next((dict(g["candidate"], nums=r["nums"]) for r in saved if r["kind"] == "candidate"), g["candidate"])
        g["saved_at"] = saved[0]["saved_at"] if saved else None
        g["track"] = model.track_record(key, g, ledger(con, key), draws[key], prizes[key])
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    tmp.replace(OUT)  # atomic: the page never reads a half-written file
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)", file=sys.stderr)


def main(cmd):
    con = db()
    if cmd in ("backfill", "update"):
        new = scrape(con, full=cmd == "backfill")
        for step in (scrape_jackpots, scrape_announcements, lambda c: scrape_prizes(c, cmd == "backfill")):
            try:
                step(con)
            except Exception as e:  # loto.ro being down must not block the results update
                print(f"  {step.__name__} failed: {e}", file=sys.stderr)
        if cmd == "update" and new:
            body = "\n".join(f"{GAMES[k]['name']} {d}#{s}: {n}{' +' + str(x) if x else ''}" for k, d, s, n, x in new)
            if hits := check_tickets(new):
                body += "\n\nYour tickets:\n" + "\n".join(hits)
            for k, d, _, nums, _ in new:  # how the saved top-10 picks did
                top = [r for r in ledger(con, k, d) if r["kind"] == "single" and r["rank"] <= 10]
                if top:
                    best = max(len(set(r["nums"]) & set(nums)) for r in top)
                    body += f"\nSaved {GAMES[k]['name']} picks (top 10): best line matched {best}"
            notify(f"New lottery results ({len(new)} draws)", body)
    build(con)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "update")
