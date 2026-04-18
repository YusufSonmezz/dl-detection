"""
Grid Search — Fuzzy LR Scheduler Hyperparameter Optimization

Kullanim:
    python train/grid_search.py --dry-run                    # Kombinasyonlari listele
    python train/grid_search.py                              # Varsayilan grid ile calistir
    python train/grid_search.py --grid train/grid_config.yaml  # Ozel grid dosyasi
    python train/grid_search.py --resume                     # Yarida kalan grid'den devam et
"""

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


# ============================================================================
#  PATHS
# ============================================================================
RESULTS_DIR_DEFAULT = os.path.join("artifacts", "grid_search")
RUNS_DIR = os.path.join("artifacts", "runs")

# ============================================================================
#  DEFAULTS
# ============================================================================
DEFAULT_GRID = {
    "base_lr":    [0.0001, 0.00005, 0.00001],
    "scale_min":  [0.85, 0.90],
    "scale_max":  [1.10, 1.20],
    "accel_boost": [1.0, 1.5],
    "brake_penalty": [0.5, 0.7],
}

DEFAULT_TRAINING = {
    "epochs": 50,
    "batch": 16,
    "imgsz": 640,
    "data": "train/data.yaml",
    "seed": 42,
}

# ============================================================================
#  GRID GENERATION
# ============================================================================
def load_grid_config(yaml_path):
    """YAML grid config dosyasini yukle."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    grid = cfg.get("grid", DEFAULT_GRID)
    training = {**DEFAULT_TRAINING, **cfg.get("training", {})}
    output = cfg.get("output", {})
    return grid, training, output


def expand_grid(grid: dict) -> list[dict]:
    """
    Grid sozlugunden tum kombinasyonlari uret.
    Her kombinasyon bir dict: {"base_lr": 0.0001, "scale_min": 0.85, ...}
    """
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        combo = dict(zip(keys, vals))
        # scale_min >= scale_max olan gecersiz kombinasyonlari atla
        if "scale_min" in combo and "scale_max" in combo:
            if combo["scale_min"] >= combo["scale_max"]:
                continue
        combos.append(combo)
    return combos


def combo_to_run_name(prefix: str, idx: int, combo: dict) -> str:
    """Kombinasyondan okunabilir run ismi olustur."""
    parts = [prefix]
    for k in sorted(combo.keys()):
        v = combo[k]
        short_key = k.replace("_", "")
        if isinstance(v, float):
            if v < 0.001:
                parts.append(f"{short_key}{v:.0e}".replace("+", ""))
            else:
                parts.append(f"{short_key}{v}")
        else:
            parts.append(f"{short_key}{v}")
    return "_".join(parts)


# ============================================================================
#  TRAINING EXECUTION
# ============================================================================
def build_train_command(combo: dict, training: dict, run_name: str) -> list[str]:
    """train.py icin komut satirini olustur."""
    cmd = [
        sys.executable, "train/train.py",
        "--run_name", run_name,
        "--epochs", str(training["epochs"]),
        "--batch", str(training["batch"]),
        "--imgsz", str(training["imgsz"]),
        "--data", training["data"],
        "--seed", str(training["seed"]),
    ]

    # Grid search parametreleri
    param_map = {
        "base_lr":              "--base_lr",
        "scale_min":            "--scale_min",
        "scale_max":            "--scale_max",
        "accel_boost":          "--accel_boost",
        "brake_penalty":        "--brake_penalty",
        "overfitting_threshold": "--overfitting_threshold",
        "warmup_epochs":        "--warmup_epochs",
        "lrf":                  "--lrf",
        "hysteresis_min":       "--hysteresis_min",
        "hysteresis_max":       "--hysteresis_max",
    }

    for param_key, cli_flag in param_map.items():
        if param_key in combo:
            cmd.extend([cli_flag, str(combo[param_key])])

    # Ablation flags
    if combo.get("no_fuzzy"):
        cmd.append("--no_fuzzy")
    if combo.get("no_three_phase"):
        cmd.append("--no_three_phase")
    if combo.get("no_warmup"):
        cmd.append("--no_warmup")

    return cmd


def run_training(cmd: list[str], run_name: str, timeout_minutes: int = 120) -> dict:
    """Tek bir egitim calistir ve sonuclari topla."""
    t0 = time.time()
    result = {"run_name": run_name, "status": "running", "started_at": datetime.now().isoformat()}

    try:
        proc = subprocess.run(
            cmd,
            timeout=timeout_minutes * 60,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        result["returncode"] = proc.returncode

        if proc.returncode == 0:
            result["status"] = "complete"
        else:
            result["status"] = "failed"
            # Son 500 karakter hata ciktisi
            result["stderr_tail"] = (proc.stderr or "")[-500:]

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["elapsed_s"] = timeout_minutes * 60

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


# ============================================================================
#  RESULT COLLECTION
# ============================================================================
def to_float(x):
    if x is None:
        return None
    try:
        v = float(x)
        return v if not (math.isnan(v) or math.isinf(v)) else None
    except (ValueError, TypeError):
        return None


def collect_run_metrics(run_name: str) -> dict:
    """Tamamlanan bir run'in metriklerini topla."""
    run_dir = os.path.join(RUNS_DIR, run_name)
    metrics = {"run_name": run_name}

    # epochs.csv'den mAP50
    epochs_path = os.path.join(run_dir, "epochs.csv")
    if os.path.exists(epochs_path):
        try:
            with open(epochs_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                map50_vals = [to_float(r.get("map50")) for r in rows]
                map50_vals = [v for v in map50_vals if v is not None]
                if map50_vals:
                    metrics["best_map50"] = max(map50_vals)
                    metrics["final_map50"] = map50_vals[-1]
                    metrics["best_epoch"] = map50_vals.index(max(map50_vals)) + 1
                    metrics["total_epochs"] = len(map50_vals)

                map5095_vals = [to_float(r.get("map5095")) for r in rows]
                map5095_vals = [v for v in map5095_vals if v is not None]
                if map5095_vals:
                    metrics["best_map5095"] = max(map5095_vals)

                # Val/train gap (son epoch)
                vtg = to_float(rows[-1].get("val_train_gap"))
                if vtg is not None:
                    metrics["final_val_train_gap"] = vtg

                # Train loss
                train_loss_vals = [to_float(r.get("train_loss")) for r in rows]
                train_loss_vals = [v for v in train_loss_vals if v is not None]
                if train_loss_vals:
                    metrics["final_train_loss"] = train_loss_vals[-1]
        except Exception:
            pass

    # run_summary.json
    summary_path = os.path.join(run_dir, "run_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            for key in ["best_map50", "total_epochs", "total_steps"]:
                if key not in metrics and key in summary:
                    metrics[key] = summary[key]
        except Exception:
            pass

    return metrics


# ============================================================================
#  REPORTING
# ============================================================================
def generate_results_csv(results: list[dict], output_path: str):
    """Sonuclari CSV'ye yaz."""
    if not results:
        return

    # Tum anahtarlari topla
    all_keys = []
    for r in results:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    # Siralama: onemli kolonlar basta
    priority = ["run_name", "status", "best_map50", "final_map50", "best_map5095",
                "best_epoch", "total_epochs", "final_train_loss", "final_val_train_gap",
                "elapsed_s"]
    ordered_keys = [k for k in priority if k in all_keys]
    ordered_keys += [k for k in all_keys if k not in ordered_keys]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"[SAVED] {output_path}")


def generate_report(results: list[dict], grid: dict, training: dict, output_path: str):
    """Markdown rapor olustur."""
    md = []
    md.append("# Grid Search Sonuclari\n")
    md.append(f"**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    md.append(f"**Toplam Kombinasyon:** {len(results)}\n")

    # Grid parametreleri
    md.append("## Grid Parametreleri\n")
    md.append("| Parametre | Degerler |")
    md.append("|---|---|")
    for k, v in sorted(grid.items()):
        md.append(f"| `{k}` | {v} |")

    md.append(f"\n**Egitim:** {training['epochs']} epoch, batch={training['batch']}, "
              f"imgsz={training['imgsz']}, data={training['data']}\n")

    # Sonuc tablosu
    completed = [r for r in results if r.get("status") == "complete" and r.get("best_map50")]
    failed = [r for r in results if r.get("status") != "complete"]

    if completed:
        # mAP50'ye gore sirala
        completed.sort(key=lambda x: x.get("best_map50", 0), reverse=True)

        md.append("## Sonuclar (mAP50 siralama)\n")
        md.append("| # | Run | mAP50 | mAP50-95 | Best Ep | Val/Train Gap | Sure |")
        md.append("|--:|-----|------:|--------:|--------:|--------------:|-----:|")

        for i, r in enumerate(completed, 1):
            name = r.get("run_name", "?")
            map50 = f"{r.get('best_map50', 0):.4f}"
            map5095 = f"{r.get('best_map5095', 0):.4f}" if r.get("best_map5095") else "N/A"
            best_ep = str(r.get("best_epoch", "N/A"))
            vtg = f"{r.get('final_val_train_gap', 0):.3f}" if r.get("final_val_train_gap") is not None else "N/A"
            elapsed = f"{r.get('elapsed_s', 0) / 60:.1f}m" if r.get("elapsed_s") else "N/A"
            md.append(f"| {i} | `{name}` | {map50} | {map5095} | {best_ep} | {vtg} | {elapsed} |")

        # En iyi konfigurasyon
        best = completed[0]
        md.append(f"\n## En Iyi Konfigurasyon\n")
        md.append(f"**Run:** `{best['run_name']}`\n")
        md.append(f"- **mAP50:** {best.get('best_map50', 0):.4f}")
        md.append(f"- **Best Epoch:** {best.get('best_epoch', 'N/A')}")

        # Parametreleri cikar
        md.append("\n**Parametreler:**\n")
        for k in sorted(grid.keys()):
            if k in best:
                md.append(f"- `{k}`: {best[k]}")

    if failed:
        md.append(f"\n## Basarisiz Calistirimalar ({len(failed)})\n")
        for r in failed:
            md.append(f"- `{r.get('run_name', '?')}`: {r.get('status', '?')} "
                      f"({r.get('error', r.get('stderr_tail', '')[:100])})")

    report_text = "\n".join(md)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[SAVED] {output_path}")


def generate_heatmap(results: list[dict], grid: dict, output_dir: str):
    """Parametre cifti icin mAP50 heatmap olustur."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib/numpy bulunamadi, heatmap atlanıyor.")
        return

    completed = [r for r in results if r.get("status") == "complete" and r.get("best_map50")]
    if len(completed) < 4:
        return

    # En fazla varyasyona sahip 2 parametreyi sec
    param_vars = {}
    for k, v in grid.items():
        if isinstance(v, list) and len(v) > 1:
            param_vars[k] = v

    if len(param_vars) < 2:
        return

    # En cok deger iceren 2 parametre
    sorted_params = sorted(param_vars.items(), key=lambda x: len(x[1]), reverse=True)
    p1_name, p1_vals = sorted_params[0]
    p2_name, p2_vals = sorted_params[1]

    # Heatmap matrisi
    heatmap = np.full((len(p1_vals), len(p2_vals)), np.nan)
    for r in completed:
        v1 = r.get(p1_name)
        v2 = r.get(p2_name)
        if v1 in p1_vals and v2 in p2_vals:
            i = p1_vals.index(v1)
            j = p2_vals.index(v2)
            val = r.get("best_map50", 0)
            # Ayni hucreye birden fazla deger gelirse en iyisini al
            if np.isnan(heatmap[i, j]) or val > heatmap[i, j]:
                heatmap[i, j] = val

    fig, ax = plt.subplots(figsize=(max(8, len(p2_vals) * 2), max(6, len(p1_vals) * 1.5)))
    im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(p2_vals)))
    ax.set_xticklabels([str(v) for v in p2_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(p1_vals)))
    ax.set_yticklabels([str(v) for v in p1_vals])
    ax.set_xlabel(p2_name, fontsize=12)
    ax.set_ylabel(p1_name, fontsize=12)
    ax.set_title("Grid Search mAP50 Heatmap", fontsize=14, fontweight="bold")

    # Deger etiketleri
    for i in range(len(p1_vals)):
        for j in range(len(p2_vals)):
            val = heatmap[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        color="white" if val > np.nanmean(heatmap) else "black", fontsize=9)

    fig.colorbar(im, ax=ax, label="mAP@0.5")
    plt.tight_layout()
    path = os.path.join(output_dir, "grid_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")


# ============================================================================
#  MAIN
# ============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Grid Search for Fuzzy LR Scheduler")
    ap.add_argument("--grid", type=str, default=None,
                    help="YAML grid config dosyasi (varsayilan: built-in grid)")
    ap.add_argument("--dataset", type=str, default=None,
                    help="Dataset config (overrides grid config)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Epoch sayisi (overrides grid config)")
    ap.add_argument("--prefix", type=str, default="grid",
                    help="Run isim on-eki (default: grid)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Sadece kombinasyonlari listele, calistirma")
    ap.add_argument("--resume", action="store_true",
                    help="Tamamlanan run'lari atla, kalan kombinasyonlardan devam et")
    ap.add_argument("--timeout", type=int, default=120,
                    help="Run basi max sure (dakika, default: 120)")
    ap.add_argument("--results-dir", type=str, default=None,
                    help="Sonuc dizini (default: artifacts/grid_search)")
    return ap.parse_args()


def main():
    args = parse_args()

    # Grid yukle
    if args.grid:
        grid, training, output_cfg = load_grid_config(args.grid)
    else:
        grid = DEFAULT_GRID.copy()
        training = DEFAULT_TRAINING.copy()
        output_cfg = {}

    # CLI override'lar
    if args.dataset:
        training["data"] = args.dataset
    if args.epochs:
        training["epochs"] = args.epochs

    results_dir = args.results_dir or output_cfg.get("results_dir", RESULTS_DIR_DEFAULT)
    prefix = args.prefix or output_cfg.get("prefix", "grid")

    # Kombinasyonlari uret
    combos = expand_grid(grid)

    print("=" * 80)
    print("FUZZY LR SCHEDULER — GRID SEARCH")
    print("=" * 80)
    print(f"Grid parametreleri:")
    for k, v in sorted(grid.items()):
        print(f"  {k}: {v}")
    print(f"\nToplam kombinasyon: {len(combos)}")
    print(f"Egitim: {training['epochs']} epoch, batch={training['batch']}, data={training['data']}")
    print(f"Sonuc dizini: {results_dir}")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN] Kombinasyonlar:\n")
        for i, combo in enumerate(combos, 1):
            name = combo_to_run_name(prefix, i, combo)
            params = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
            print(f"  {i:3d}. {name}")
            print(f"       {params}")
        print(f"\nToplam: {len(combos)} calistirma")
        est_time = len(combos) * training["epochs"] * 0.35  # ~0.35 dk/epoch tahmini
        print(f"Tahmini sure: ~{est_time:.0f} dakika ({est_time / 60:.1f} saat)")
        return

    # Sonuc dizini
    os.makedirs(results_dir, exist_ok=True)

    # Mevcut sonuclari yukle (resume icin)
    state_path = os.path.join(results_dir, "grid_state.json")
    all_results = []
    completed_names = set()

    if args.resume and os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        all_results = state.get("results", [])
        completed_names = {r["run_name"] for r in all_results if r.get("status") == "complete"}
        print(f"[RESUME] {len(completed_names)} tamamlanmis run bulundu, kalanlar calistirilacak.\n")

    # Calistir
    total = len(combos)
    for i, combo in enumerate(combos, 1):
        run_name = combo_to_run_name(prefix, i, combo)

        # Resume: zaten tamamlanmis mi?
        if run_name in completed_names:
            print(f"[{i}/{total}] {run_name} — ATLANDI (zaten tamamlanmis)")
            continue

        # Zaten artifacts/runs/ altinda var mi?
        run_dir = os.path.join(RUNS_DIR, run_name)
        if os.path.exists(os.path.join(run_dir, "epochs.csv")) and args.resume:
            metrics = collect_run_metrics(run_name)
            if metrics.get("best_map50"):
                result = {**combo, **metrics, "status": "complete", "elapsed_s": 0}
                all_results.append(result)
                completed_names.add(run_name)
                print(f"[{i}/{total}] {run_name} — MEVCUT (mAP50={metrics['best_map50']:.4f})")
                continue

        params_str = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
        print(f"\n[{i}/{total}] {run_name}")
        print(f"  Parametreler: {params_str}")

        cmd = build_train_command(combo, training, run_name)
        print(f"  Komut: {' '.join(cmd[:6])} ...")

        # Calistir
        train_result = run_training(cmd, run_name, timeout_minutes=args.timeout)

        # Metrikleri topla
        if train_result["status"] == "complete":
            metrics = collect_run_metrics(run_name)
            result = {**combo, **metrics, **train_result}
            map50 = metrics.get("best_map50", 0)
            print(f"  TAMAMLANDI — mAP50={map50:.4f} ({train_result.get('elapsed_s', 0) / 60:.1f} dk)")
        else:
            result = {**combo, **train_result}
            print(f"  BASARISIZ — {train_result['status']}")

        all_results.append(result)
        completed_names.add(run_name)

        # Her run sonrasi state kaydet (crash koruması)
        state = {"results": all_results, "grid": grid, "training": training, "updated_at": datetime.now().isoformat()}
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    # Raporlama
    print("\n" + "=" * 80)
    print("GRID SEARCH TAMAMLANDI — RAPOR OLUSTURULUYOR")
    print("=" * 80)

    csv_path = os.path.join(results_dir, "grid_results.csv")
    generate_results_csv(all_results, csv_path)

    report_path = os.path.join(results_dir, "grid_report.md")
    generate_report(all_results, grid, training, report_path)

    generate_heatmap(all_results, grid, results_dir)

    # Ozet
    completed_runs = [r for r in all_results if r.get("best_map50")]
    if completed_runs:
        best = max(completed_runs, key=lambda x: x.get("best_map50", 0))
        print(f"\nEN IYI SONUC: {best['run_name']}")
        print(f"  mAP50 = {best['best_map50']:.4f}")
        print(f"  Parametreler:")
        for k in sorted(grid.keys()):
            if k in best:
                print(f"    {k} = {best[k]}")

    print(f"\nSonuclar: {results_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
