#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser(description="Extract post-training metrics from CAT logs.")
    ap.add_argument("--dataset", "-d", required=True, type=str, help="Dataset name (e.g., Portrait)")
    ap.add_argument("--dir", default="../preds/", help="Directory containing log files (default: current folder)")
    return ap.parse_args()

def main():
    args = parse_args()
    ds = args.dataset
    log_dir = args.dir

    metrics = ["rmse", "mae", "doa", "pc-er", "rm"]

    # regex for metrics after training
    metric_re = re.compile(r".*?\b(rmse|mae|doa|pc-er|rm)\b\s*:\s*([0-9eE\.\-]+)")

    # extract testid from filename: launch_IMPACT_Portrait_1256230_0.out
    file_re = re.compile(rf"launch_IMPACT_{ds}_(\d+)_\d+\.out")

    # group logs by testid
    groups = {}  # testid → list of files
    for fname in os.listdir(log_dir):
        m = file_re.match(fname)
        if m:
            tid = m.group(1)
            groups.setdefault(tid, []).append(fname)

    if not groups:
        print(f"[ERROR] No log files launch_IMPACT_{ds}_<testid>_*.out found in {log_dir}")
        return

    print(f"[INFO] Found {len(groups)} tests for dataset '{ds}'")

    # process each testid separately
    for testid, files in groups.items():
        print(f"\n=== Test ID: {testid} ({len(files)} logs) ===")
        data = {m: [] for m in metrics}

        for fname in files:
            fpath = os.path.join(log_dir, fname)

            with open(fpath, "r") as f:
                lines = f.readlines()

            # find end training index
            end_idx = next((i for i, l in enumerate(lines) if "-- END Training --" in l), None)
            if end_idx is None:
                print(f"[WARN] Missing END marker in {fname}, skip")
                continue

            for line in lines[end_idx+1:]:
                mm = metric_re.search(line)
                if mm:
                    key, val = mm.group(1), float(mm.group(2))
                    data[key].append(val)

        # write result file
        out_file = os.path.join(log_dir, f"--results_IMPACT_{ds}_{testid}.txt")

        with open(out_file, "w") as fout:
            fout.write(f"Dataset: {ds}\nTest ID: {testid}\nLogs: {len(files)}\n\n")
            fout.write(f"{'Metric':6s}  Mean        Std       N\n")
            fout.write("-" * 35 + "\n")

            for key, vals in data.items():
                if vals:
                    fout.write(f"{key:6s}  {np.mean(vals):.6f}  {np.std(vals):.6f}  {len(vals)}\n")
                    print(f"{key:6s} | mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, n={len(vals)}")
                else:
                    fout.write(f"{key:6s}  NO DATA\n")
                    print(f"{key:6s} | no data")

        print(f"→ Results saved to {out_file}")

if __name__ == "__main__":
    main()
