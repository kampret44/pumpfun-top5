#!/usr/bin/env python3
# pumpfun-top5-forecast-01 entry: a Monte Carlo top-5 forecaster.
#
# Reads /input/task.json (places, size-bin edges, the seven answer reads, cadence) and
# /input/history.ndjson (10-minute marks) and writes /output/forecast.json:
#   probabilities: per coin, per place, the probability the coin holds that place,
#                  calibrated so each place's named mass leaves ~40% unnamed (for the
#                  coins born after the cut and the coins the candidate set leaves out);
#   sizes:         per place, the bins (log10 cap in SOL on task.json's edges) the
#                  place's market cap falls in, p rounded down to 6 decimals;
#   top5:          the 5 mints with the highest P(place 1), tie-break by expected place.
#
# Each candidate coin is a log10-cap random walk (drift and volatility shrunk toward
# the global priors) plus a per-step collapse hazard; per simulated read the surviving
# candidate caps are ranked for per-place order and size probabilities. All random draws
# come from one fixed-seed random.Random, so two runs are byte-identical. It never reads
# the network, an API key or the clock (the seed is a constant).
import json
import math
import os
import random

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261112

# Calibration knobs. Fixed defaults, chosen by the reasoning in the header (the dev
# reference is sealed on the server, so there is no local grade to tune against).
N_SIMS = 2000             # Monte Carlo replications per read
DRIFT_PSEUDO = 3.0        # pseudo-observations pulling a coin's drift toward the prior
VOL_PSEUDO = 8.0          # pseudo-observations pulling a coin's volatility toward the prior
COLLAPSE_FLOOR_LOG = 1.4  # log10(cap SOL) below which a coin is out of contention
COLLAPSE_PER_STEP = 0.004  # base per-step collapse hazard (shrunk by a pseudo-count)
COLLAPSE_PSEUDO = 0.5     # pseudo-count for the collapse-hazard shrinkage
CANDIDATE_TOP = 60        # top coins by last-mark cap drawn into the candidate set
RISER_COUNT = 12          # fastest recent risers, also drawn in
RISER_MARKS = 3           # window (marks) over which "recent riser" is measured
CALIB = 0.58              # share of a place's candidate probability bet on named coins,
                          # i.e. ~42% of each place is left unnamed for born-after coins
MIN_NAME_P = 0.02         # a lower named probability is dropped (its mass returns unnamed)
SIZE_SMOOTH = 40.0        # pseudo-samples spread over the size bins (robust to few sims)
TREND_A = 0.13            # place-cap trend (log form): the top places' caps grow as new
                          # coins pump and take the top, so a simulated place cap is
                          # shifted up by TREND_A * ln(1 + H) for the SIZE distribution
                          # only (the ranking is unchanged). A max over a growing
                          # population grows logarithmically, so the shift saturates
                          # over long horizons instead of running away linearly.


def log_cap_sol(lamports):
    return math.log10(int(lamports)) - 9.0


def read_task(input_dir):
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf8") as f:
        return json.load(f)


def read_marks(input_dir):
    text = open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf8").read()
    marks = []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        if m.get("ok") and isinstance(m.get("rows"), list):
            marks.append(m)
    return marks


def build_series(marks):
    # mint -> list of (mark_index, log_cap) in ascending mark order
    series = {}
    for mi, m in enumerate(marks):
        for r in m["rows"]:
            series.setdefault(r["mint"], []).append((mi, log_cap_sol(r["market_cap_lamports"])))
    return series


def parse_iso(iso):
    from datetime import datetime, timezone
    s = iso[:-1] + "+00:00" if iso.endswith("Z") else iso
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


def steps_to_reads(task, last_mark_at):
    cad = task.get("cadence_minutes", 10) * 60.0
    last_t = parse_iso(last_mark_at)
    out = []
    for r in task["reads"]:
        d = parse_iso(r) - last_t
        steps = int(round(d / cad))
        out.append(max(1, steps))
    return out


def estimate_params(series):
    """Global drift/vol priors, then per-coin shrunk drift, volatility and collapse hazard."""
    rets = [b - a for v in series.values() for (_, a), (_, b) in zip(v, v[1:])]
    if rets:
        g_mean = sum(rets) / len(rets)
        var = sum((x - g_mean) ** 2 for x in rets) / (len(rets) - 1) if len(rets) > 1 else 0.0144
        g_sd = max(math.sqrt(var) if var > 0 else 0.12, 0.05)
    else:
        g_mean, g_sd = 0.0, 0.12

    params = {}
    for mint, v in series.items():
        last = v[-1][1]
        if len(v) >= 2:
            own = [b - a for (_, a), (_, b) in zip(v, v[1:])]
            n = len(own)
            own_mean = sum(own) / n
            drift = (n * own_mean + DRIFT_PSEUDO * g_mean) / (n + DRIFT_PSEUDO)
            var = sum((x - own_mean) ** 2 for x in own) / (n - 1) if n > 1 else g_sd ** 2
            sd = math.sqrt((var * (n - 1) + VOL_PSEUDO * g_sd ** 2) / (n - 1 + VOL_PSEUDO))
            collapsed = sum(1 for _, lc in v if lc <= COLLAPSE_FLOOR_LOG)
            hazard = (n * (collapsed / n) + COLLAPSE_PSEUDO * COLLAPSE_PER_STEP) / (n + COLLAPSE_PSEUDO)
        else:
            drift = g_mean
            sd = g_sd
            hazard = COLLAPSE_PER_STEP
        params[mint] = {"drift": drift, "sd": sd, "hazard": hazard, "last": last}
    return params, g_mean, g_sd


def pick_candidates(series, params):
    # top coins by last-mark cap, plus the fastest recent risers (largest per-step
    # log-cap gain over the last span of the known window, capped at RISER_MARKS marks).
    by_cap = sorted(series, key=lambda m: params[m]["last"], reverse=True)[:CANDIDATE_TOP]
    cand_set = set(by_cap)
    scores = []
    for mint, v in series.items():
        if mint in cand_set or len(v) < 2:
            continue
        start = max(0, v[-1][0] - RISER_MARKS)
        # the first observation no earlier than the start of the recent window
        recent = [l for i, l in v if i >= start] or [v[-1][1]]
        n_steps = max(1, v[-1][0] - start)
        gain = (v[-1][1] - recent[0]) / n_steps
        scores.append((gain, mint))
    scores.sort(reverse=True, key=lambda t: t[0])
    for _, mint in scores:
        if len(by_cap) < CANDIDATE_TOP + RISER_COUNT and len(by_cap) < len(series):
            if mint not in cand_set:
                by_cap.append(mint)
                cand_set.add(mint)
    return by_cap


def bin_of(lc, edges, nbins):
    b = nbins - 1
    for i, e in enumerate(edges):
        if lc < e:
            b = i
            break
    return b


def simulate(task, params, candidates, steps_list, rng):
    places = task["places"]
    edges = task["size"]["edges"]
    nbins = len(edges) + 1
    nreads = len(steps_list)
    denom = N_SIMS * nreads
    order_counts = {}
    size_counts = {}
    for _sim in range(N_SIMS):
        for H in steps_list:
            caps = []
            for cand in candidates:
                p = params[cand]
                surv = (1.0 - p["hazard"]) ** H
                if rng.random() >= surv:
                    caps.append((COLLAPSE_FLOOR_LOG, cand))
                else:
                    shift = rng.gauss(H * p["drift"], p["sd"] * math.sqrt(H))
                    caps.append((p["last"] + shift, cand))
            caps.sort(reverse=True, key=lambda t: t[0])
            for k in range(1, places + 1):
                if len(caps) >= k and caps[k - 1][0] > COLLAPSE_FLOOR_LOG:
                    cand = caps[k - 1][1]
                    order_counts[(k, cand)] = order_counts.get((k, cand), 0) + 1
            for k in range(1, places + 1):
                if len(caps) >= k and caps[k - 1][0] > COLLAPSE_FLOOR_LOG:
                    lc = caps[k - 1][0] + TREND_A * math.log1p(H)
                    key = (k, bin_of(lc, edges, nbins))
                    size_counts[key] = size_counts.get(key, 0) + 1
    return order_counts, size_counts, denom


def build_forecast(task, params, candidates, order_counts, size_counts, denom):
    places = task["places"]
    edges = task["size"]["edges"]
    nbins = len(edges) + 1

    probs = {}
    for k in range(1, places + 1):
        for cand in candidates:
            c = order_counts.get((k, cand), 0)
            if c:
                probs[(k, cand)] = c / denom

    probabilities = []
    for k in range(1, places + 1):
        scaled = sorted(((cand, probs.get((k, cand), 0.0)) for cand in candidates),
                        key=lambda t: t[1], reverse=True)
        for cand, p in scaled:
            if p < MIN_NAME_P:
                break
            p_named = round(p * CALIB, 6)
            if p_named > 0:
                probabilities.append({"mint": cand, "place": k, "p": p_named})

    sizes = []
    for k in range(1, places + 1):
        tot = sum(size_counts.get((k, b), 0) for b in range(nbins))
        if tot == 0:
            continue
        prior = SIZE_SMOOTH / nbins
        for b in range(nbins):
            raw = size_counts.get((k, b), 0) / tot
            p = math.floor((raw * tot + prior) / (tot + SIZE_SMOOTH) * 1e6) / 1e6
            if p > 0:
                sizes.append({"place": k, "bin": b, "p": p})

    scored = sorted(candidates, key=lambda c: tuple(probs.get((k, c), 0.0) for k in range(1, places + 1)), reverse=True)
    top5 = scored[:places]
    by_cap = sorted(candidates, key=lambda c: params[c]["last"], reverse=True)
    for m in by_cap:
        if len(top5) == places:
            break
        if m not in top5:
            top5.append(m)

    return {"probabilities": probabilities, "top5": top5, "sizes": sizes}


def main():
    task = read_task(INPUT_DIR)
    marks = read_marks(INPUT_DIR)
    if not marks:
        raise RuntimeError("the history holds no ok mark")
    series = build_series(marks)
    if not series:
        raise RuntimeError("the history ranks no coins")
    steps_list = steps_to_reads(task, marks[-1]["at"])
    params, g_mean, g_sd = estimate_params(series)
    candidates = pick_candidates(series, params)
    rng = random.Random(SEED)
    order_counts, size_counts, denom = simulate(task, params, candidates, steps_list, rng)
    forecast = build_forecast(task, params, candidates, order_counts, size_counts, denom)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task["output"]), "w", encoding="utf8") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
