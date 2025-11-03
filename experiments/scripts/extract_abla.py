#!/usr/bin/env python3
import os, re, argparse, math, sys
from collections import defaultdict, OrderedDict
import numpy as np

# ------------------------- CONFIG -------------------------

# Table 1 metrics (as before)
METRICS_1 = ["rmse", "mae", "doa", "pc-er", "rm"]
MINIMIZE = {"rmse", "mae"}  # rest are maximized

# Table 2 metrics (new)
METRICS_2 = ["mi_acc", "mi_acc_w1", "mi_acc_w2"]

# Ablation code order
ABLATION_ORDER = ["BASE", "WI", "WR", "WL", "WIR", "WIL", "WLR", "WILR"]

ABLATION_LABEL = {
    "BASE":  r"\method",
    "WI":    r"Without init",
    "WR":    r"Without $\mathcal{R}_{I}$",
    "WL":    r"Without $\lossintmul$",
    "WIR":   r"Without init and $\mathcal{R}_{I}$",
    "WIL":   r"Without init and $\lossintmul$",
    "WLR":   r"Without $\mathcal{R}_{I}$ and $\lossintmul$",
    "WILR":  r"Without init, $\mathcal{R}_{I}$ and $\lossintmul$",
}

DATASET_LABEL = {
    "postcovid": r"\postcovid",
    "promis":    r"\promis",
    "portrait":  r"\portrait",
    "movielens": r"\movielens",
}
DATASET_ORDER = ["postcovid", "promis", "portrait", "movielens"]

# Filename pattern
FNAME_RE = re.compile(
    r"^launch_IMPACT_"
    r"(?P<ds>[A-Za-z0-9\-]+)_"
    r"(?P<tid>[A-Za-z0-9\-]+?)"
    r"(?:-(?P<abl>WI|WR|WL|WIR|WIL|WLR|WILR))?"
    r"_(?P<fold>\d+)\.(?:out|txt|log)$"
)

END_MARK = "-- END Training --"

METRIC_RE = re.compile(
    r"\b(rmse|mae|doa|pc-er|rm|mi_acc(?:_w[12])?)\b\s*:\s*([0-9eE\.\-]+)"
)

# ------------------------- HELPERS -------------------------

def safe_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except:
        return None

def collect_values(log_dir):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seen = defaultdict(lambda: defaultdict(int))

    for fname in sorted(os.listdir(log_dir)):
        m = FNAME_RE.match(fname)
        if not m:
            continue

        ds = m["ds"].lower()
        abl = m["abl"] if m["abl"] else "BASE"
        fpath = os.path.join(log_dir, fname)

        try:
            with open(fpath, "r") as f:
                lines = f.readlines()
        except:
            continue

        # find END marker
        end = next((i for i, l in enumerate(lines) if END_MARK in l), None)
        tail = lines[end+1:] if end is not None else lines[-200:]

        found = False
        for line in tail:
            mm = METRIC_RE.search(line)
            if mm:
                metric, val = mm.group(1), safe_float(mm.group(2))
                if val is not None:
                    data[ds][abl][metric].append(val)
                    found = True
        if found:
            seen[ds][abl] += 1

    return data, seen

def mean_std(arr):
    a = np.asarray(arr, float)
    return float(np.mean(a)), float(np.std(a))

def bolding(stats, metric):
    rows = [(abl, mu, sd) for abl, (mu, sd) in stats.items() if mu is not None]
    if not rows:
        return set()

    if metric in MINIMIZE:
        best = min(rows, key=lambda x: x[1])
        thr = best[1] + best[2]
        return {abl for abl, mu, sd in rows if mu <= thr + 1e-12}
    else:
        best = max(rows, key=lambda x: x[1])
        thr = best[1] - best[2]
        return {abl for abl, mu, sd in rows if mu >= thr - 1e-12}

def fmt(mu, sd, do_bold):
    if mu is None:
        return r"\textemdash"
    s = f"{mu:.4f} $\\pm$ {sd:.4f}"
    return r"\textbf{" + s + "}" if do_bold else s

# ------------------------- TABLE GENERATOR -------------------------

def build_table(data, metrics, caption, label):
    out = []
    out.append(r"\begin{table*}[!htb]")
    out.append(r"\centering")
    out.append(r"\resizebox{1\textwidth}{!}{%")
    out.append(r"\begin{tabular}{l|" + "c" * len(metrics) + "}")
    out.append(r"\hline")
    out.append("Datasets & " + " & ".join(m.upper() for m in metrics) + r" \\")
    out.append(r"\hline")

    for ds in DATASET_ORDER:
        if ds not in data:
            continue

        out.append(f" & \\multicolumn{{{len(metrics)}}}{{c}}{{{DATASET_LABEL[ds]}}} \\\\ \\hline")

        bold_for = {
            m: bolding({abl: mean_std(data[ds][abl].get(m, []))
                        for abl in ABLATION_ORDER if abl in data[ds]}, m)
            for m in metrics
        }

        for abl in ABLATION_ORDER:
            if abl not in data[ds]:
                continue
            cells = []
            for m in metrics:
                vals = data[ds][abl].get(m, [])
                mu, sd = mean_std(vals) if vals else (None, None)
                cells.append(fmt(mu, sd, abl in bold_for[m]))
            out.append(f"{ABLATION_LABEL[abl]}\n  & " + " & ".join(cells) + r"\\")
        out.append(r"\hline")

    out.append(r"\end{tabular}}")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append(r"\end{table*}")
    return "\n".join(out)

# ------------------------- MAIN -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../preds_abla/")
    ap.add_argument("--out-base", default="../data/",
                    help="Base filename to save both tables, e.g. results/abla")
    args = ap.parse_args()

    data, seen = collect_values(args.dir)

    print("[INFO] Found datasets/ablations:")
    for ds, abls in seen.items():
        print(f"  {ds}: ", ", ".join(f"{a}:{c}" for a, c in abls.items()))

    tex1 = build_table(
        data,
        METRICS_1,
        "Ablation results: prediction (RMSE, MAE) and embedding (DOA, PC-ER, RM).",
        "tab:ablation_main"
    )

    tex2 = build_table(
        data,
        METRICS_2,
        "Ablation results: mastery metrics (Accuracy, Accuracy±1, Accuracy±2).",
        "tab:ablation_accuracy"
    )

    print("\n===== TABLE 1 (RMSE/MAE/DOA/PCE/RM) =====\n")
    print(tex1)
    print("\n===== TABLE 2 (Acc / Acc±1 / Acc±2) =====\n")
    print(tex2)

    if args.out_base:
        with open(args.out_base + "_main.tex", "w") as f: f.write(tex1)
        with open(args.out_base + "_acc.tex", "w")  as f: f.write(tex2)
        print(f"\n[OK] Written:\n  {args.out_base}_main.tex\n  {args.out_base}_acc.tex")

if __name__ == "__main__":
    main()
