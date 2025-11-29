#!/usr/bin/env python3
"""
analyze_boruvka.py

Analyze results.csv produced by boruvka_kmachines_fixed output with logging.

CSV columns expected:
 run_id,n,m,K,total_time,phases_count,edges_in_mst,per_phase_max_time_json,edges_added_per_phase_json,notes

Outputs (PNG):
 - strong_scaling.png
 - weak_scaling.png
 - heatmap.png
 - per_phase_time.png
 - edges_added_per_phase.png
"""

import csv
import json
import sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
from statistics import mean, stdev

CSV_PATH = "results.csv"

def read_results(path):
    rows = []
    try:
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    r_parsed = {
                        "run_id": r.get("run_id",""),
                        "n": int(r.get("n", "0")),
                        "m": int(r.get("m", "0")),
                        "K": int(r.get("K", "0")),
                        "total_time": float(r.get("total_time", "nan")),
                        "phases_count": int(r.get("phases_count", "0")),
                        "edges_in_mst": int(r.get("edges_in_mst", "0")),
                        "per_phase_max_time": [],
                        "edges_added_per_phase": [],
                        "notes": r.get("notes","")
                    }
                except Exception as e:
                    # skip malformed row
                    print("Skipping malformed CSV row:", e)
                    continue

                # parse JSON arrays (they are quoted in the CSV)
                def try_load_json_field(fieldname):
                    raw = r.get(fieldname, "")
                    if raw is None: return []
                    raw = raw.strip()
                    if raw == "": return []
                    # If the CSV quoting preserved quotes, raw will start with [ or with "[
                    # Remove wrapping quotes if present
                    if raw.startswith('"') and raw.endswith('"'):
                        raw = raw[1:-1]
                    if raw.startswith("'") and raw.endswith("'"):
                        raw = raw[1:-1]
                    try:
                        parsed = json.loads(raw)
                        return parsed
                    except Exception:
                        # fallback: try replacing single-quotes with double-quotes
                        try:
                            parsed = json.loads(raw.replace("'", "\""))
                            return parsed
                        except Exception:
                            # as last resort, attempt to evaluate simple list of numbers
                            try:
                                cleaned = raw.strip()
                                cleaned = cleaned.strip("[] ")
                                if cleaned == "": return []
                                parts = [x.strip() for x in cleaned.split(",")]
                                nums = []
                                for p in parts:
                                    if p == "": continue
                                    nums.append(float(p))
                                return nums
                            except Exception:
                                return []

                r_parsed["per_phase_max_time"] = try_load_json_field("per_phase_max_time_json")
                r_parsed["edges_added_per_phase"] = try_load_json_field("edges_added_per_phase_json")

                rows.append(r_parsed)
    except FileNotFoundError:
        print("Error: CSV not found at", path)
        return []
    return rows

def group_by(rows, keyfn):
    g = defaultdict(list)
    for r in rows:
        g[keyfn(r)].append(r)
    return g

def plot_strong_scaling(rows, n_fixed, m_fixed, out):
    filtered = [r for r in rows if r['n']==n_fixed and r['m']==m_fixed]
    if not filtered:
        print("No rows for strong-scaling target n=", n_fixed, "m=", m_fixed)
        return
    byK = defaultdict(list)
    for r in filtered:
        byK[r['K']].append(r['total_time'])
    Ks = sorted(byK.keys())
    means = [mean(byK[k]) for k in Ks]
    errs = [stdev(byK[k]) if len(byK[k])>1 else 0.0 for k in Ks]

    plt.figure()
    plt.errorbar(Ks, means, yerr=errs, marker='o')
    plt.xlabel('MPI ranks (K)')
    plt.ylabel('Total time (s)')
    plt.title(f'Strong scaling (n={n_fixed}, m={m_fixed})')
    plt.grid(True)
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print("Wrote", out)

def plot_weak_scaling(rows, per_rank_n, m_ratio, out):
    # pick rows where n == per_rank_n * K (approx) and m approx equals n * m_ratio
    byK = defaultdict(list)
    for r in rows:
        if r['K'] <= 0: continue
        if r['n'] == per_rank_n * r['K']:
            # allow rounding of m ratio
            if math.isclose(r['m'], int(r['n'] * m_ratio), rel_tol=1e-3, abs_tol=1):
                byK[r['K']].append(r['total_time'])
    if not byK:
        print("No rows for weak-scaling with n_per_rank=", per_rank_n, "m_ratio=", m_ratio)
        return
    Ks = sorted(byK.keys())
    means = [mean(byK[k]) for k in Ks]
    errs = [stdev(byK[k]) if len(byK[k])>1 else 0.0 for k in Ks]

    plt.figure()
    plt.errorbar(Ks, means, yerr=errs, marker='o')
    plt.xlabel('MPI ranks (K)')
    plt.ylabel('Total time (s)')
    plt.title(f'Weak scaling (n_per_rank={per_rank_n}, m_ratio≈{m_ratio:.2f})')
    plt.grid(True)
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print("Wrote", out)

def plot_heatmap(rows, ns, Ks, out):
    mat = np.full((len(ns), len(Ks)), np.nan)
    for i,n in enumerate(ns):
        for j,K in enumerate(Ks):
            vals = [r['total_time'] for r in rows if r['n']==n and r['K']==K]
            if vals:
                mat[i,j] = mean(vals)
    plt.figure()
    plt.imshow(mat, aspect='auto', origin='lower')
    plt.xlabel('K (ranks)')
    plt.ylabel('n (vertices)')
    plt.title('Mean total_time heatmap')
    plt.xticks(range(len(Ks)), Ks, rotation=45)
    plt.yticks(range(len(ns)), ns)
    plt.colorbar()
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print("Wrote", out)

def plot_per_phase(rows_filtered, out):
    if not rows_filtered:
        print("No rows for per-phase plot")
        return
    # Use per_phase_max_time averaged across runs, pad shorter runs with NaN and compute mean ignoring NaN
    lists = [r['per_phase_max_time'] for r in rows_filtered]
    maxlen = max((len(l) for l in lists), default=0)
    arr = np.full((len(lists), maxlen), np.nan)
    for i,l in enumerate(lists):
        for j,v in enumerate(l):
            arr[i,j] = v
    mean_per_phase = np.nanmean(arr, axis=0)
    phases = list(range(1, len(mean_per_phase)+1))
    plt.figure()
    plt.plot(phases, mean_per_phase, marker='o')
    plt.xlabel('phase')
    plt.ylabel('time (s) [per-phase max across ranks]')
    first = rows_filtered[0]
    plt.title(f'Avg per-phase max time (n={first["n"]}, K={first["K"]})')
    plt.grid(True)
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print("Wrote", out)

def plot_edges_added_per_phase(rows_filtered, out):
    if not rows_filtered:
        print("No rows for edges-added-per-phase plot")
        return
    lists = [r['edges_added_per_phase'] for r in rows_filtered]
    maxlen = max((len(l) for l in lists), default=0)
    arr = np.full((len(lists), maxlen), np.nan)
    for i,l in enumerate(lists):
        for j,v in enumerate(l):
            arr[i,j] = v
    mean_per_phase = np.nanmean(arr, axis=0)
    phases = list(range(1, len(mean_per_phase)+1))
    plt.figure()
    plt.plot(phases, mean_per_phase, marker='o')
    plt.xlabel('phase')
    plt.ylabel('edges added (avg)')
    first = rows_filtered[0]
    plt.title(f'Avg edges added per phase (n={first["n"]}, K={first["K"]})')
    plt.grid(True)
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print("Wrote", out)

def main():
    parser = argparse.ArgumentParser(description="Analyze Boruvka results CSV")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to results.csv")
    parser.add_argument("--n", type=int, default=None, help="n for strong-scaling plot")
    parser.add_argument("--m", type=int, default=None, help="m for strong-scaling plot")
    parser.add_argument("--per_rank_n", type=int, default=None, help="n per rank for weak scaling plot")
    parser.add_argument("--m_ratio", type=float, default=None, help="m/n ratio expected for weak scaling")
    parser.add_argument("--pick_config", default=None, help="pick a config for per-phase plots in form n:K (e.g. 16000:8)")
    args = parser.parse_args()

    rows = read_results(args.csv)
    if not rows:
        print("No data found in", args.csv)
        return

    # Attach edges_added_per_phase and per_phase_max_time fields into rows (they are present but we normalized names)
    # The CSV parser saved them into keys already as lists

    # Determine defaults if not provided
    if args.n is None or args.m is None:
        # pick the most common (n,m) pair
        pair_counts = defaultdict(int)
        for r in rows:
            pair_counts[(r['n'], r['m'])] += 1
        (n_def, m_def), _ = max(pair_counts.items(), key=lambda kv: kv[1])
    else:
        n_def, m_def = args.n, args.m

    if args.per_rank_n is None:
        # try derive from the first row
        first = rows[0]
        per_rank_n_def = first['n'] // max(1, first['K'])
    else:
        per_rank_n_def = args.per_rank_n

    if args.m_ratio is None:
        first = rows[0]
        m_ratio_def = first['m'] / float(max(1, first['n']))
    else:
        m_ratio_def = args.m_ratio

    # strong scaling
    plot_strong_scaling(rows, n_def, m_def, 'strong_scaling.png')

    # weak scaling
    plot_weak_scaling(rows, per_rank_n_def, m_ratio_def, 'weak_scaling.png')

    # heatmap over ns and Ks present
    ns = sorted(list(set(r['n'] for r in rows)))
    Ks = sorted(list(set(r['K'] for r in rows)))
    plot_heatmap(rows, ns, Ks, 'heatmap.png')

    # pick config for per-phase plots
    if args.pick_config:
        parts = args.pick_config.split(':')
        if len(parts)==2:
            pick_n = int(parts[0]); pick_K = int(parts[1])
        else:
            print("Invalid pick_config; expecting n:K")
            pick_n = n_def; pick_K = rows[0]['K']
    else:
        pick_n = n_def
        # choose most common K for that n
        Ks_for_n = sorted([r['K'] for r in rows if r['n']==pick_n])
        pick_K = Ks_for_n[0] if Ks_for_n else rows[0]['K']

    rows_filtered = [r for r in rows if r['n']==pick_n and r['K']==pick_K]
    if not rows_filtered:
        print("No rows for chosen (n,K) = (", pick_n, ",", pick_K, ") -- picking first available config")
        rows_filtered = [rows[0]]

    plot_per_phase(rows_filtered, 'per_phase_time.png')
    plot_edges_added_per_phase(rows_filtered, 'edges_added_per_phase.png')

    print("All plots generated.")

if __name__ == "__main__":
    main()
