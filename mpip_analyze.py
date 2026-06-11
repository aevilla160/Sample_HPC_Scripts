#!/usr/bin/env python3
"""
mpip_analyze.py -- parse mpiP report files into clean CSVs and plots.

Handles a single .mpiP file or a directory/glob of them. For each file it
writes the six mpiP tables as tidy CSVs and a set of PNG figures. When more
than one file is given (e.g. a 2/4/8/16-rank scaling sweep), it also emits a
cross-file summary CSV and scaling plots.

Usage:
    python mpip_analyze.py FILE_OR_DIR [FILE_OR_DIR ...] [-o OUTDIR] [--no-plots]

Example:
    python mpip_analyze.py <filenmae.mpiP>
    python mpip_analyze.py <filename.mpiP>  -o mpip_analysis
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

# Section title keyword -> short key. Order does not matter.
SECTION_KEYS = {
    "MPI Time": "mpi_time",
    "Aggregate Time": "agg_time",
    "Aggregate Sent Message Size": "agg_msg",
    "Callsite Time statistics": "callsite_time",
    "Callsite Message Sent statistics": "callsite_msg",
}

# Column schema per section (header token -> output column name, in order).
SCHEMAS = {
    "mpi_time":      ["Task", "AppTime", "MPITime", "MPIpct"],
    "agg_time":      ["Call", "Site", "Time_ms", "Apppct", "MPIpct", "Count"],
    "agg_msg":       ["Call", "Site", "Count", "Total_bytes", "Avg_bytes", "Sentpct"],
    "callsite_time": ["Name", "Site", "Rank", "Count", "Max_ms", "Mean_ms",
                      "Min_ms", "Apppct", "MPIpct"],
    "callsite_msg":  ["Name", "Site", "Rank", "Count", "Max_bytes", "Mean_bytes",
                      "Min_bytes", "Sum_bytes"],
}

DASH_RE = re.compile(r"^-+$")
BANNER_RE = re.compile(r"^@---\s*(.*?)\s*-*$")


def _to_num(tok):
    """Coerce a token to float; keep '*' as the literal string."""
    if tok == "*":
        return "*"
    try:
        return float(tok)
    except ValueError:
        return tok


def parse_mpip(path):
    """Return (meta: dict, tables: dict[str, DataFrame])."""
    with open(path) as fh:
        lines = fh.readlines()

    meta = {"file": os.path.basename(path), "task_hosts": {}}
    tables = {}

    # --- header metadata (@ key : value), before the first @--- banner ---
    for ln in lines:
        if ln.startswith("@---"):
            break
        m = re.match(r"^@\s+([^:]+?)\s*:\s*(.*)$", ln.rstrip())
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip()
        if key == "MPI Task Assignment":
            parts = val.split()
            if len(parts) >= 2:
                meta["task_hosts"][parts[0]] = parts[1]
        elif key == "MPIP env var":
            meta["case"] = val.replace("-k", "").strip()
        elif key == "Command":
            meta["command"] = val
        elif key in ("Start time", "Stop time", "Version"):
            meta[key.lower().replace(" ", "_")] = val

    # --- split into sections by @--- banners ---
    banner_idx = [i for i, ln in enumerate(lines) if ln.startswith("@---")]
    banner_idx.append(len(lines))
    for b in range(len(banner_idx) - 1):
        start = banner_idx[b]
        end = banner_idx[b + 1]
        title_m = BANNER_RE.match(lines[start].rstrip())
        title = title_m.group(1) if title_m else ""
        key = next((k for kw, k in SECTION_KEYS.items() if kw in title), None)
        if key is None:
            continue

        # data rows: drop banner, dash separators, blanks; first real line is
        # the column header, rest are data.
        body = []
        for ln in lines[start + 1:end]:
            s = ln.strip()
            if not s or DASH_RE.match(s) or s.startswith("@"):
                continue
            body.append(s)
        if not body:
            continue

        cols = SCHEMAS[key]
        rows = []
        for s in body[1:]:                      # body[0] is the column header
            toks = s.split()
            if len(toks) != len(cols):
                # tolerate ragged lines rather than crashing
                continue
            rows.append([_to_num(t) for t in toks])
        if rows:
            tables[key] = pd.DataFrame(rows, columns=cols)

    # --- derive nprocs / label ---
    nprocs = None
    if "mpi_time" in tables:
        nprocs = tables["mpi_time"]["Task"].apply(
            lambda x: x != "*").sum()
    if not nprocs:
        m = re.search(r"\.(\d+)\.\d+\.\d+\.mpiP$", meta["file"])
        nprocs = int(m.group(1)) if m else len(meta["task_hosts"]) or None
    meta["nprocs"] = int(nprocs) if nprocs else None
    case = meta.get("case")
    if case in (None, "", "[null]"):
        case = None
    meta["label"] = case or os.path.splitext(meta["file"])[0]
    return meta, tables


# --------------------------------------------------------------------------
# Output: CSVs
# --------------------------------------------------------------------------

def write_csvs(meta, tables, outdir):
    case_dir = os.path.join(outdir, meta["label"])
    os.makedirs(case_dir, exist_ok=True)
    for key, df in tables.items():
        df.to_csv(os.path.join(case_dir, f"{key}.csv"), index=False)
    return case_dir


# --------------------------------------------------------------------------
# Output: per-file plots
# --------------------------------------------------------------------------

TOP_N = 20          # cap rows shown in any single-file plot
MAX_IN = 22.0       # cap any computed figure dimension (inches)


def _dim(per_item, n, lo):
    return max(lo, min(MAX_IN, per_item * n))


def _savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_file(meta, tables, case_dir):
    if not HAVE_MPL:
        return
    plot_dir = os.path.join(case_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    label = meta["label"]

    # 1. Aggregate MPI time by call (ms), descending.
    if "agg_time" in tables:
        df = tables["agg_time"].nlargest(TOP_N, "Time_ms").sort_values("Time_ms")
        fig, ax = plt.subplots(figsize=(8, _dim(0.4, len(df), 3)))
        ax.barh(df["Call"], df["Time_ms"], color="#3b6ea5")
        for y, (t, p) in enumerate(zip(df["Time_ms"], df["MPIpct"])):
            ax.text(t, y, f" {p:.0f}%", va="center", fontsize=8)
        ax.set_xlabel("Aggregate time (ms)")
        ax.set_title(f"MPI time by call — {label}")
        _savefig(fig, os.path.join(plot_dir, "agg_time_by_call.png"))

    # 2. Per-rank MPI% (load imbalance).
    if "mpi_time" in tables:
        df = tables["mpi_time"]
        df = df[df["Task"] != "*"].copy()
        df["Task"] = df["Task"].astype(float).astype(int)
        df = df.sort_values("Task")
        fig, ax = plt.subplots(figsize=(_dim(0.18, len(df), 8), 4))
        ax.bar(df["Task"], df["MPIpct"], color="#c0504d")
        ax.set_xlabel("MPI rank")
        ax.set_ylabel("MPI time (% of app time)")
        ax.set_title(f"Per-rank MPI fraction (imbalance) — {label}")
        _savefig(fig, os.path.join(plot_dir, "per_rank_mpi_pct.png"))

    # 3. Sent message volume by call (bytes, log).
    if "agg_msg" in tables:
        df = tables["agg_msg"].nlargest(TOP_N, "Total_bytes").sort_values("Total_bytes")
        fig, ax = plt.subplots(figsize=(8, _dim(0.4, len(df), 3)))
        ax.barh(df["Call"], df["Total_bytes"], color="#4f8a5b")
        ax.set_xscale("log")
        ax.set_xlabel("Total bytes sent (log)")
        ax.set_title(f"Sent message volume by call — {label}")
        _savefig(fig, os.path.join(plot_dir, "msg_volume_by_call.png"))

    # 4. Per-call time imbalance: Max / Mean / Min from aggregate (*) rows.
    if "callsite_time" in tables:
        df = tables["callsite_time"]
        agg = df[df["Rank"] == "*"].copy()
        if not agg.empty:
            agg = agg.nlargest(TOP_N, "Max_ms")
            x = range(len(agg))
            fig, ax = plt.subplots(figsize=(_dim(0.7, len(agg), 6), 4))
            w = 0.27
            ax.bar([i - w for i in x], agg["Max_ms"], w, label="Max", color="#c0504d")
            ax.bar(list(x), agg["Mean_ms"], w, label="Mean", color="#3b6ea5")
            ax.bar([i + w for i in x], agg["Min_ms"], w, label="Min", color="#9bbb59")
            ax.set_yscale("log")
            ax.set_xticks(list(x))
            ax.set_xticklabels(agg["Name"], rotation=45, ha="right")
            ax.set_ylabel("Per-call time (ms, log)")
            ax.set_title(f"Max/Mean/Min call time across ranks — {label}")
            ax.legend()
            _savefig(fig, os.path.join(plot_dir, "call_time_imbalance.png"))


# --------------------------------------------------------------------------
# Output: cross-file summary + scaling plots
# --------------------------------------------------------------------------

def build_summary(parsed):
    rows = []
    for meta, tables in parsed:
        row = {"label": meta["label"], "nprocs": meta["nprocs"],
               "file": meta["file"]}
        if "mpi_time" in tables:
            agg = tables["mpi_time"][tables["mpi_time"]["Task"] == "*"]
            if not agg.empty:
                row["AppTime_s"] = float(agg["AppTime"].iloc[0])
                row["MPITime_s"] = float(agg["MPITime"].iloc[0])
                row["MPIpct"] = float(agg["MPIpct"].iloc[0])
        if "agg_time" in tables:
            at = tables["agg_time"]
            for call in ("Barrier", "Alltoall", "Alltoallv", "Allreduce"):
                hit = at[at["Call"] == call]
                row[f"{call}_ms"] = float(hit["Time_ms"].iloc[0]) if not hit.empty else 0.0
            top = at.sort_values("Time_ms", ascending=False).iloc[0]
            row["top_call"] = top["Call"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("nprocs", na_position="last")
    return df


def plot_scaling(summary, outdir):
    if not HAVE_MPL or summary["nprocs"].notna().sum() < 2:
        return
    sdir = os.path.join(outdir, "scaling")
    os.makedirs(sdir, exist_ok=True)
    s = summary.dropna(subset=["nprocs"]).sort_values("nprocs")
    x = s["nprocs"].astype(int).astype(str)

    # MPI% vs nprocs
    if "MPIpct" in s:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, s["MPIpct"], "o-", color="#c0504d")
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("MPI time (% of app)")
        ax.set_title("Communication fraction vs scale")
        _savefig(fig, os.path.join(sdir, "mpi_pct_vs_scale.png"))

    # Collective time vs nprocs (grouped)
    calls = [c for c in ("Barrier", "Alltoall", "Alltoallv", "Allreduce")
             if f"{c}_ms" in s and (s[f"{c}_ms"] > 0).any()]
    if calls:
        fig, ax = plt.subplots(figsize=(7, 4))
        for c in calls:
            ax.plot(x, s[f"{c}_ms"], "o-", label=c)
        ax.set_yscale("log")
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("Aggregate time (ms, log)")
        ax.set_title("Per-call MPI time vs scale")
        ax.legend()
        _savefig(fig, os.path.join(sdir, "call_time_vs_scale.png"))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def collect_files(inputs):
    files = []
    for inp in inputs:
        if os.path.isdir(inp):
            files += glob.glob(os.path.join(inp, "**", "*.mpiP"), recursive=True)
        elif any(ch in inp for ch in "*?["):
            files += glob.glob(inp, recursive=True)
        elif os.path.isfile(inp):
            files.append(inp)
        else:
            print(f"[warn] no match: {inp}", file=sys.stderr)
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser(description="Parse mpiP reports into CSVs and plots.")
    ap.add_argument("inputs", nargs="+", help="mpiP file(s), directory, or glob")
    ap.add_argument("-o", "--outdir", default="mpip_analysis", help="output directory")
    ap.add_argument("--no-plots", action="store_true", help="CSV only, skip figures")
    args = ap.parse_args()

    files = collect_files(args.inputs)
    if not files:
        print("No .mpiP files found.", file=sys.stderr)
        sys.exit(1)
    if args.no_plots:
        global HAVE_MPL
        HAVE_MPL = False
    if not HAVE_MPL and not args.no_plots:
        print("[warn] matplotlib not available -- writing CSVs only.", file=sys.stderr)

    os.makedirs(args.outdir, exist_ok=True)
    parsed = []
    for f in files:
        meta, tables = parse_mpip(f)
        case_dir = write_csvs(meta, tables, args.outdir)
        plot_file(meta, tables, case_dir)
        parsed.append((meta, tables))
        print(f"[ok] {meta['label']:40s} nprocs={meta['nprocs']}  "
              f"tables={len(tables)}  -> {case_dir}")

    summary = build_summary(parsed)
    summary.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
    plot_scaling(summary, args.outdir)
    print(f"\nSummary ({len(parsed)} file(s)) -> {os.path.join(args.outdir, 'summary.csv')}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
