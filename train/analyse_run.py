# analyse_run.py (sağlamlaştırılmış)
import os, sys, csv, json, math
from pathlib import Path
import matplotlib.pyplot as plt

def to_float(x):
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "nan"}: return None
    try:
        return float(s)
    except:
        # virgüllü ondalık vs. gelirse
        try:
            return float(s.replace(",", "."))
        except:
            return None

def load_csv(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            clean = {}
            for k, v in row.items():
                # mümkünse float'a çevir, değilse string bırak
                fv = to_float(v)
                clean[k] = fv if fv is not None else v
            rows.append(clean)
    return rows

def series_numeric(rows, key):
    out = []
    for r in rows:
        v = r.get(key, None)
        v = to_float(v)
        if v is not None:
            out.append(v)
    return out

def first_reach_threshold(series, threshold):
    for i, v in enumerate(series):
        if v is not None and v >= threshold:
            return i
    return None

def volatility(xs):
    xs = [to_float(x) for x in xs]
    xs = [x for x in xs if x is not None]
    if len(xs) < 2: return 0.0
    m = sum(xs) / len(xs)
    if len(xs) < 2: return 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    denom = abs(m) + 1e-8
    return math.sqrt(var) / denom

def analyse(run_dir):
    steps_path = os.path.join(run_dir, "steps.csv")
    epochs_path = os.path.join(run_dir, "epochs.csv")
    params_path = os.path.join(run_dir, "params.json")

    if not os.path.exists(steps_path):
        raise FileNotFoundError(f"steps.csv yok: {steps_path}")
    if not os.path.exists(epochs_path):
        # epoch olmazsa boş dizi ile ilerle
        epochs = []
    else:
        epochs = load_csv(epochs_path)

    steps = load_csv(steps_path)
    params = {}
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as f:
            try:
                params = json.load(f)
            except:
                params = {}

    # seriler
    lr     = series_numeric(steps, "lr")
    loss   = series_numeric(steps, "loss")
    ips    = series_numeric(steps, "imgs_per_sec")
    scale  = series_numeric(steps, "scale")
    e_map50 = series_numeric(epochs, "map50") if epochs else []

    # KPI'lar
    res = {}
    res["run"] = Path(run_dir).name
    res["final_loss"] = loss[-1] if loss else None
    res["best_map50"] = max(e_map50) if e_map50 else 0.0
    res["time_to_map50_0.80_epoch_idx"] = first_reach_threshold(e_map50, 0.80)
    res["mean_ips"] = (sum(ips) / len(ips)) if ips else 0.0
    res["lr_volatility"] = volatility(lr)
    # son 200 adımda oynaklık (yoksa tamamı)
    last_loss_window = loss[-min(200, len(loss)):] if loss else []
    res["loss_volatility"] = volatility(last_loss_window) if last_loss_window else 0.0
    # event sayıları CSV'de tutuluyorsa al, yoksa 0
    try:
        res["scale_up_events"] = int(to_float(steps[-1].get("events_up"))) if steps else 0
        res["scale_down_events"] = int(to_float(steps[-1].get("events_down"))) if steps else 0
    except:
        res["scale_up_events"] = 0
    return res, steps, epochs, params

def plot_series(ax, xs, ys, label, xlabel, ylabel):
    ax.plot(xs, ys, label=label)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True); ax.legend()

def main():
    if len(sys.argv) < 2:
        print("Kullanım: python train/analyse_run.py <run_dir_1> [<run_dir_2> ...]")
        sys.exit(1)

    out_dir = "artifacts/analysis"
    os.makedirs(out_dir, exist_ok=True)

    analyses = []
    for rd in sys.argv[1:]:
        res, steps, epochs, params = analyse(rd)
        analyses.append((res, steps, epochs, params))

    # Grafikler
    # LR
    fig1 = plt.figure(figsize=(10,6)); ax1 = fig1.add_subplot(111)
    for res, steps, _, _ in analyses:
        xs = series_numeric(steps, "step")
        ys = series_numeric(steps, "lr")
        if xs and ys:
            plot_series(ax1, xs, ys, res["run"], "step", "lr")
    fig1.tight_layout(); fig1.savefig(os.path.join(out_dir, "lr_curves.png"))

    # Loss
    fig2 = plt.figure(figsize=(10,6)); ax2 = fig2.add_subplot(111)
    for res, steps, _, _ in analyses:
        xs = series_numeric(steps, "step")
        ys = series_numeric(steps, "loss")
        if xs and ys:
            plot_series(ax2, xs, ys, res["run"], "step", "train loss")
    fig2.tight_layout(); fig2.savefig(os.path.join(out_dir, "loss_curves.png"))

    # mAP50
    fig3 = plt.figure(figsize=(10,6)); ax3 = fig3.add_subplot(111)
    for res, _, epochs, _ in analyses:
        xs = series_numeric(epochs, "epoch")
        ys = series_numeric(epochs, "map50")
        if xs and ys:
            plot_series(ax3, xs, ys, res["run"], "epoch", "mAP50 (val)")
    fig3.tight_layout(); fig3.savefig(os.path.join(out_dir, "map50_curves.png"))

    # Markdown rapor
    md = ["# Karşılaştırmalı Eğitim Raporu\n"]
    md.append("| Run | Final Loss | Best mAP50 | Epoch idx @ mAP50≥0.80 | Mean IPS | LR Volatility | Loss Volatility | ↑Events | ↓Events |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for res, _, _, _ in analyses:
        md.append(f"| {res['run']} | {res['final_loss'] if res['final_loss'] is not None else 'NA'} | "
                  f"{res['best_map50']:.3f} | {str(res['time_to_map50_0.80_epoch_idx'])} | "
                  f"{res['mean_ips']:.1f} | {res['lr_volatility']:.3f} | {res['loss_volatility']:.3f} | "
                  f"{int(res.get('scale_up_events',0))} | {int(res.get('scale_down_events',0))} |")
    md.append("\n## Grafikler\n")
    md.append("![LR Eğrileri](artifacts/analysis/lr_curves.png)\n")
    md.append("![Loss Eğrileri](artifacts/analysis/loss_curves.png)\n")
    md.append("![mAP50 Eğrileri](artifacts/analysis/map50_curves.png)\n")

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

if __name__ == "__main__":
    main()
