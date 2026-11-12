#!/usr/bin/env python3
"""Local scorer mirroring grader/forecast/ranked.ts (orderLogScore, sizeLogScore, agreementOf).

The dev kit's reference (the seven answer reads' charts and cap bins) is sealed on the
server (the kit CLI refuses grade-local for program/v1), so this needs reference data the
kit does not publish. It is included so a forecast can be scored locally the moment a
reference (or an honest proxy) is available:
  python3 grade_locally.py --forecast <forecast.json> --reference <reference.json>
Reference format, one JSON object:
  {"places": 5, "edges": [...task edges...], "history_mints": [...],
   "reads": [ {"chart": [ {"mint": ..., "log10_cap_sol": ...}, ... up to 100 ], ... }, x7 ]}
where each read's chart is its 100 ranked coins (rank order), and each row's log10_cap_sol
is the cap the answer priced. A row naming a mint the history never named is dropped and its
place's p falls back to the unnamed share, exactly as forecast.ts knownOnly does.
Uses the exact floor 1e-4 and the mean-over-reads order + size log scores from the harness.
"""
import argparse
import json
import math

P_FLOOR = 1e-4


def bin_of(r, edges):
    lo = 0
    hi = len(edges)
    while lo < hi:
        mid = (lo + hi) >> 1
        if edges[mid] <= r:
            lo = mid + 1
        else:
            hi = mid
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", required=True)
    ap.add_argument("--reference", required=True)
    args = ap.parse_args()

    doc = json.load(open(args.forecast))
    ref = json.load(open(args.reference))
    places, edges = ref["places"], ref["edges"]
    nbins = len(edges) + 1
    history = set(ref["history_mints"])

    # order rows, split per place
    order = {}
    order_sum = [0.0] * (places + 1)
    for r in doc["probabilities"]:
        order[(r["place"], r["mint"])] = r["p"]
        order_sum[r["place"]] += r["p"]

    # size rows, split per place
    size = {}
    size_sum = [0.0] * (places + 1)
    size_named = [set() for _ in range(places + 1)]
    for r in doc["sizes"]:
        size[(r["place"], r["bin"])] = r["p"]
        size_sum[r["place"]] += r["p"]
        size_named[r["place"]].add(r["bin"])

    order_total = 0.0
    size_total = 0.0
    agree_total = 0
    nreads = 0

    for read in ref["reads"]:
        chart = []      # (rank, mint) drop rows not in history
        caps = [None] * (places + 1)
        for i, row in enumerate(read["chart"], start=1):
            if row["mint"] in history:
                chart.append((i, row["mint"]))
            if i <= places:
                caps[i] = row
        if not chart:
            continue
        nreads += 1

        # order score
        named_sets = {}
        for k in range(1, places + 1):
            named_sets[k] = {m for (pl, m) in order if pl == k}
        for k in range(1, min(places, len(chart)) + 1):
            _, actual = chart[k - 1]
            p = order.get((k, actual))
            if p is None:
                unnamed = max(0.0, 1.0 - order_sum[k])
                n_unnamed = sum(1 for _, m in chart if m not in named_sets[k])
                p = unnamed / n_unnamed if n_unnamed else 0.0
            order_total += math.log(max(p, P_FLOOR))

        # size score: only history mints; place k's cap is caps[k] if present
        for k in range(1, min(places, len(chart)) + 1):
            row = caps[k]
            if row is None or row["mint"] not in history:
                continue
            b = bin_of(row["log10_cap_sol"], edges)
            p = size.get((k, b))
            if p is None:
                n_unnamed = nbins - len(size_named[k])
                p = max(0.0, 1.0 - size_sum[k]) / n_unnamed if n_unnamed else 0.0
            size_total += math.log(max(p, P_FLOOR))

        # agreement (top5 vs this read's top 5, only history mints)
        head = [m for _, m in chart[:places]]
        top = doc["top5"][:places]
        for i, m in enumerate(top):
            if m in head:
                j = head.index(m)
                agree_total += places - abs(i - j)

    ln = max(nreads, 1)
    order_mean = order_total / ln
    size_mean = size_total / ln
    print(f"reads={nreads}")
    print(f"order_mean={order_mean:.6f}  size_mean={size_mean:.6f}  log_score={order_mean + size_mean:.6f}")
    print(f"agreement_sum={agree_total}")


if __name__ == "__main__":
    main()
