"""
Learning Rate Deneylerini Kapsamlı Karşılaştırma Scripti

Bu script, farklı learning rate'lerle yapılan deneyleri detaylı şekilde analiz eder:
- Convergence hızı
- Stability (loss volatility)
- Final performance
- Training dynamics
- Overfitting analizi
"""
import os
import sys
import csv
import json
import math
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Sınıf isimleri
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor", "background"
]

def to_float(x):
    """Güvenli float dönüşümü"""
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "nan", "inf", "-inf"}: return None
    try:
        return float(s)
    except:
        try:
            return float(s.replace(",", "."))
        except:
            return None

def load_csv(path):
    """CSV dosyasını yükle ve float'lara dönüştür"""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {}
            for k, v in row.items():
                k = k.strip()
                fv = to_float(v)
                clean[k] = fv if fv is not None else v
            rows.append(clean)
    return rows

def series_numeric(rows, key):
    """Belirli bir anahtarın numerik serisini çıkar"""
    out = []
    for r in rows:
        v = r.get(key, None)
        v = to_float(v)
        if v is not None and not math.isnan(v) and not math.isinf(v):
            out.append(v)
        else:
            out.append(None)
    return out

def calculate_statistics(series):
    """Bir seri için istatistikler hesapla"""
    valid = [x for x in series if x is not None]
    if not valid:
        return {
            'mean': None, 'std': None, 'min': None, 'max': None,
            'median': None, 'q1': None, 'q3': None
        }

    valid_sorted = sorted(valid)
    n = len(valid_sorted)

    return {
        'mean': np.mean(valid),
        'std': np.std(valid),
        'min': min(valid),
        'max': max(valid),
        'median': valid_sorted[n // 2],
        'q1': valid_sorted[n // 4],
        'q3': valid_sorted[3 * n // 4]
    }

def calculate_volatility(series, window=None):
    """
    Volatility hesapla (coefficient of variation)
    window: Son N değeri kullan (None ise tümü)
    """
    if window and len(series) > window:
        series = series[-window:]

    valid = [x for x in series if x is not None]
    if len(valid) < 2:
        return 0.0

    mean = sum(valid) / len(valid)
    variance = sum((x - mean) ** 2 for x in valid) / (len(valid) - 1)
    std = math.sqrt(variance)

    # Coefficient of variation (normalize by mean)
    if abs(mean) > 1e-8:
        return std / abs(mean)
    return 0.0

def first_reach_threshold(series, threshold):
    """İlk kez threshold'u geçtiği indeksi bul"""
    for i, v in enumerate(series):
        if v is not None and v >= threshold:
            return i
    return None

def convergence_speed(series, final_fraction=0.95):
    """
    Convergence hızı: Final değerin %95'ine kaç adımda ulaştı?
    """
    valid = [x for x in series if x is not None]
    if len(valid) < 10:
        return None

    # Son %10'luk kısmın ortalamasını "final" kabul et
    final_window = max(1, len(valid) // 10)
    final_value = sum(valid[-final_window:]) / final_window

    target = final_value * final_fraction

    return first_reach_threshold(series, target)

def detect_overfitting(train_series, val_series):
    """
    Overfitting tespiti: Train-val gap analizi
    Returns: overfitting_score (yüksek değer = daha fazla overfitting)
    """
    if not train_series or not val_series:
        return None

    # Son %20'lik kısımda ortalama gap
    n = min(len(train_series), len(val_series))
    if n < 5:
        return None

    window = max(1, n // 5)

    train_final = [x for x in train_series[-window:] if x is not None]
    val_final = [x for x in val_series[-window:] if x is not None]

    if not train_final or not val_final:
        return None

    train_mean = sum(train_final) / len(train_final)
    val_mean = sum(val_final) / len(val_final)

    # Gap (val daha yüksek loss = overfitting)
    return val_mean - train_mean

def analyze_run(run_dir):
    """Tek bir run'ı detaylı analiz et"""
    run_name = Path(run_dir).name

    # Load data
    results = load_csv(os.path.join(run_dir, "results.csv"))
    epochs = load_csv(os.path.join(run_dir, "epochs.csv"))
    steps = load_csv(os.path.join(run_dir, "steps.csv"))

    params_path = os.path.join(run_dir, "params.json")
    params = {}
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as f:
            try:
                params = json.load(f)
            except:
                pass

    # Extract learning rate from run name
    lr_match = re.search(r'lr([\d.]+)', run_name)
    learning_rate = float(lr_match.group(1)) if lr_match else None

    # Series extraction
    analysis = {
        'run_name': run_name,
        'learning_rate': learning_rate,
        'params': params,
    }

    # === EPOCHS DATA ===
    if epochs:
        map50_series = series_numeric(epochs, 'map50')
        map5095_series = series_numeric(epochs, 'map5095')
        precision_series = series_numeric(epochs, 'precision')
        recall_series = series_numeric(epochs, 'recall')

        analysis['epochs_data'] = {
            'map50': map50_series,
            'map5095': map5095_series,
            'precision': precision_series,
            'recall': recall_series,
        }

        # Best metrics
        valid_map50 = [x for x in map50_series if x is not None]
        analysis['best_map50'] = max(valid_map50) if valid_map50 else 0.0
        analysis['final_map50'] = valid_map50[-1] if valid_map50 else 0.0

        valid_map5095 = [x for x in map5095_series if x is not None]
        analysis['best_map5095'] = max(valid_map5095) if valid_map5095 else 0.0
        analysis['final_map5095'] = valid_map5095[-1] if valid_map5095 else 0.0

        # Convergence
        analysis['epochs_to_best'] = valid_map50.index(analysis['best_map50']) + 1 if valid_map50 else None
        analysis['convergence_epoch_90pct'] = convergence_speed(map50_series, 0.90)

        # Stability
        analysis['map50_volatility'] = calculate_volatility(map50_series)
        analysis['map50_std'] = calculate_statistics(map50_series)['std']

    # === RESULTS DATA (train/val losses) ===
    if results:
        train_box_loss = series_numeric(results, 'train/box_loss')
        train_cls_loss = series_numeric(results, 'train/cls_loss')
        train_dfl_loss = series_numeric(results, 'train/dfl_loss')
        val_box_loss = series_numeric(results, 'val/box_loss')
        val_cls_loss = series_numeric(results, 'val/cls_loss')

        analysis['results_data'] = {
            'train_box_loss': train_box_loss,
            'train_cls_loss': train_cls_loss,
            'train_dfl_loss': train_dfl_loss,
            'val_box_loss': val_box_loss,
            'val_cls_loss': val_cls_loss,
        }

        # Final losses
        analysis['final_train_box_loss'] = train_box_loss[-1] if train_box_loss and train_box_loss[-1] is not None else None
        analysis['final_val_box_loss'] = val_box_loss[-1] if val_box_loss and val_box_loss[-1] is not None else None

        # Loss volatility (son 10 epoch)
        analysis['train_box_loss_volatility'] = calculate_volatility(train_box_loss, window=10)

        # Overfitting detection
        analysis['overfitting_score'] = detect_overfitting(train_box_loss, val_box_loss)

    # === STEPS DATA ===
    if steps:
        lr_series = series_numeric(steps, 'lr')
        loss_series = series_numeric(steps, 'loss')

        analysis['steps_data'] = {
            'lr': lr_series,
            'loss': loss_series,
        }

        analysis['lr_volatility'] = calculate_volatility(lr_series)
        analysis['step_loss_volatility'] = calculate_volatility(loss_series, window=200)

    # === TRAINING HEALTH ===
    # Check for NaN/Inf issues
    all_values = []
    if results:
        for key in ['train/box_loss', 'train/cls_loss', 'val/box_loss']:
            all_values.extend([v for v in series_numeric(results, key) if v is not None])

    analysis['has_nan_issues'] = any(v is None or math.isnan(v) or math.isinf(v)
                                     for v in all_values) if all_values else True
    analysis['training_collapsed'] = analysis['best_map50'] < 0.01 if 'best_map50' in analysis else True

    return analysis

def create_comparison_plots(analyses, output_dir):
    """Karşılaştırma grafikleri oluştur"""
    os.makedirs(output_dir, exist_ok=True)

    # Sort by learning rate
    analyses = sorted(analyses, key=lambda x: x['learning_rate'] if x['learning_rate'] else 0)

    # === FIGURE 1: mAP50 Progression ===
    fig1 = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig1)

    ax1 = fig1.add_subplot(gs[0, :])  # mAP50 curves
    ax2 = fig1.add_subplot(gs[1, 0])  # Final mAP50 bar chart
    ax3 = fig1.add_subplot(gs[1, 1])  # Convergence speed

    # mAP50 curves
    for analysis in analyses:
        if 'epochs_data' in analysis:
            map50 = analysis['epochs_data']['map50']
            epochs = list(range(len(map50)))
            lr = analysis['learning_rate']
            label = f"LR={lr}" if lr else analysis['run_name']
            ax1.plot(epochs, map50, marker='o', label=label, linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('mAP@0.5', fontsize=12)
    ax1.set_title('Validation mAP@0.5 Progression', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    # Final mAP50 bar chart
    names = [f"LR={a['learning_rate']}" if a['learning_rate'] else a['run_name'] for a in analyses]
    final_map50s = [a.get('final_map50', 0) for a in analyses]
    colors = ['green' if not a.get('training_collapsed', True) else 'red' for a in analyses]

    bars = ax2.bar(range(len(names)), final_map50s, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Final mAP@0.5', fontsize=12)
    ax2.set_title('Final Performance Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Convergence speed
    conv_epochs = [a.get('epochs_to_best', 0) if a.get('epochs_to_best') else 0 for a in analyses]
    ax3.bar(range(len(names)), conv_epochs, color='steelblue', alpha=0.7)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.set_ylabel('Epochs to Best mAP@0.5', fontsize=12)
    ax3.set_title('Convergence Speed', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'map50_comparison.png'), dpi=150)
    print(f"[SAVED] {output_dir}/map50_comparison.png")

    # === FIGURE 2: Loss Curves ===
    fig2 = plt.figure(figsize=(14, 6))
    ax_train = fig2.add_subplot(121)
    ax_val = fig2.add_subplot(122)

    for analysis in analyses:
        if 'results_data' in analysis:
            train_loss = analysis['results_data']['train_box_loss']
            val_loss = analysis['results_data']['val_box_loss']
            epochs = list(range(len(train_loss)))
            lr = analysis['learning_rate']
            label = f"LR={lr}" if lr else analysis['run_name']

            ax_train.plot(epochs, train_loss, marker='o', label=label, linewidth=2, alpha=0.8)
            ax_val.plot(epochs, val_loss, marker='s', label=label, linewidth=2, alpha=0.8)

    ax_train.set_xlabel('Epoch', fontsize=12)
    ax_train.set_ylabel('Box Loss', fontsize=12)
    ax_train.set_title('Training Box Loss', fontsize=14, fontweight='bold')
    ax_train.grid(True, alpha=0.3)
    ax_train.legend(loc='best')

    ax_val.set_xlabel('Epoch', fontsize=12)
    ax_val.set_ylabel('Box Loss', fontsize=12)
    ax_val.set_title('Validation Box Loss', fontsize=14, fontweight='bold')
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(loc='best')

    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'loss_comparison.png'), dpi=150)
    print(f"[SAVED] {output_dir}/loss_comparison.png")

    # === FIGURE 3: Stability Analysis ===
    fig3 = plt.figure(figsize=(14, 6))
    ax_vol = fig3.add_subplot(121)
    ax_overfit = fig3.add_subplot(122)

    # Volatility
    volatilities = [a.get('map50_volatility', 0) for a in analyses]
    ax_vol.bar(range(len(names)), volatilities, color='coral', alpha=0.7)
    ax_vol.set_xticks(range(len(names)))
    ax_vol.set_xticklabels(names, rotation=45, ha='right')
    ax_vol.set_ylabel('Coefficient of Variation', fontsize=12)
    ax_vol.set_title('mAP@0.5 Stability (Lower = More Stable)', fontsize=12, fontweight='bold')
    ax_vol.grid(True, alpha=0.3, axis='y')

    # Overfitting score
    overfit_scores = [a.get('overfitting_score', 0) if a.get('overfitting_score') is not None else 0
                      for a in analyses]
    colors_overfit = ['green' if score < 0.5 else 'orange' if score < 1.0 else 'red'
                      for score in overfit_scores]
    ax_overfit.bar(range(len(names)), overfit_scores, color=colors_overfit, alpha=0.7)
    ax_overfit.set_xticks(range(len(names)))
    ax_overfit.set_xticklabels(names, rotation=45, ha='right')
    ax_overfit.set_ylabel('Val Loss - Train Loss', fontsize=12)
    ax_overfit.set_title('Overfitting Score (Lower = Better)', fontsize=12, fontweight='bold')
    ax_overfit.grid(True, alpha=0.3, axis='y')
    ax_overfit.axhline(y=0, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'stability_analysis.png'), dpi=150)
    print(f"[SAVED] {output_dir}/stability_analysis.png")

    plt.close('all')

def generate_markdown_report(analyses, output_dir):
    """Detaylı Markdown rapor oluştur"""
    analyses = sorted(analyses, key=lambda x: x['learning_rate'] if x['learning_rate'] else 0)

    md = []
    md.append("# Learning Rate Deneyleri - Kapsamlı Karşılaştırma Raporu\n")
    md.append(f"**Tarih:** {Path(output_dir).name}\n")
    md.append(f"**Toplam Deney Sayısı:** {len(analyses)}\n")
    md.append("---\n")

    # === Summary Table ===
    md.append("## Ozet Tablo\n")
    md.append("| LR | Best mAP@0.5 | Final mAP@0.5 | Epochs to Best | Final Train Loss | Final Val Loss | Overfitting | Status |")
    md.append("|---:|-------------:|--------------:|---------------:|-----------------:|---------------:|------------:|:------:|")

    for a in analyses:
        lr = f"{a['learning_rate']:.5f}" if a['learning_rate'] else "N/A"
        best_map = f"{a.get('best_map50', 0):.4f}"
        final_map = f"{a.get('final_map50', 0):.4f}"
        epochs_to_best = str(a.get('epochs_to_best', 'N/A'))
        train_loss = f"{a.get('final_train_box_loss', 0):.4f}" if a.get('final_train_box_loss') else "N/A"
        val_loss = f"{a.get('final_val_box_loss', 0):.4f}" if a.get('final_val_box_loss') else "N/A"
        overfit_score = f"{a.get('overfitting_score', 0):.4f}" if a.get('overfitting_score') is not None else "N/A"

        status = "OK" if not a.get('training_collapsed', True) else "FAIL"

        md.append(f"| {lr} | {best_map} | {final_map} | {epochs_to_best} | {train_loss} | {val_loss} | {overfit_score} | {status} |")

    md.append("\n")

    # === Detailed Analysis ===
    md.append("## Detayli Analiz\n")

    for a in analyses:
        lr = a['learning_rate'] if a['learning_rate'] else "Unknown"
        md.append(f"### Learning Rate: {lr}\n")
        md.append(f"**Run Name:** `{a['run_name']}`\n")

        # Performance
        md.append("#### Performance Metrikleri\n")
        md.append(f"- **Best mAP@0.5:** {a.get('best_map50', 0):.4f}")
        md.append(f"- **Final mAP@0.5:** {a.get('final_map50', 0):.4f}")
        md.append(f"- **Best mAP@0.5:0.95:** {a.get('best_map5095', 0):.4f}")
        md.append(f"- **Epochs to Best:** {a.get('epochs_to_best', 'N/A')}\n")

        # Stability
        md.append("#### Stability ve Convergence\n")
        overfit_val = a.get('overfitting_score')
        overfit_str = f"{overfit_val:.4f}" if overfit_val is not None else 'N/A'

        md.append(f"- **mAP@0.5 Volatility:** {a.get('map50_volatility', 0):.4f} (dusuk = stabil)")
        md.append(f"- **Train Box Loss Volatility:** {a.get('train_box_loss_volatility', 0):.4f}")
        md.append(f"- **Overfitting Score:** {overfit_str} (dusuk = daha az overfitting)\n")

        # Training Health
        md.append("#### Training Health\n")
        md.append(f"- **NaN/Inf Issues:** {'WARNING: YES' if a.get('has_nan_issues', False) else 'OK: NO'}")
        md.append(f"- **Training Collapsed:** {'FAIL: YES' if a.get('training_collapsed', True) else 'OK: NO'}\n")

        md.append("---\n")

    # === Recommendations ===
    md.append("## Oneriler\n")

    # En iyi performing run
    best_run = max(analyses, key=lambda x: x.get('best_map50', 0))
    md.append(f"### En İyi Performans\n")
    md.append(f"**LR={best_run.get('learning_rate', 'N/A')}** (mAP@0.5: {best_run.get('best_map50', 0):.4f})\n")

    # En stabil run
    stable_runs = [a for a in analyses if not a.get('training_collapsed', True)]
    if stable_runs:
        most_stable = min(stable_runs, key=lambda x: x.get('map50_volatility', float('inf')))
        md.append(f"### En Stabil Eğitim\n")
        md.append(f"**LR={most_stable.get('learning_rate', 'N/A')}** (Volatility: {most_stable.get('map50_volatility', 0):.4f})\n")

    # Hızlı convergence
    fast_converge = [a for a in analyses if a.get('epochs_to_best')]
    if fast_converge:
        fastest = min(fast_converge, key=lambda x: x.get('epochs_to_best', float('inf')))
        md.append(f"### En Hızlı Convergence\n")
        md.append(f"**LR={fastest.get('learning_rate', 'N/A')}** ({fastest.get('epochs_to_best', 'N/A')} epoch)\n")

    # === Visualizations ===
    md.append("\n## Grafikler\n")
    md.append("### mAP@0.5 Comparison\n")
    md.append("![mAP50 Comparison](map50_comparison.png)\n")
    md.append("### Loss Comparison\n")
    md.append("![Loss Comparison](loss_comparison.png)\n")
    md.append("### Stability Analysis\n")
    md.append("![Stability Analysis](stability_analysis.png)\n")

    # Save report
    report_path = os.path.join(output_dir, "comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"[SAVED] {report_path}")

def main():
    # Otomatik olarak baseline_lr* klasörlerini bul
    runs_dir = Path("artifacts/runs")

    # Pattern: baseline_lr{value}_e50
    lr_runs = sorted(runs_dir.glob("baseline_lr*_e50"))

    if not lr_runs:
        print("[ERROR] baseline_lr*_e50 pattern'ine uyan klasör bulunamadı!")
        print("Kullanım: python train/compare_lr_experiments.py")
        print("  veya")
        print("Kullanım: python train/compare_lr_experiments.py <run_dir_1> <run_dir_2> ...")
        sys.exit(1)

    # Manual override varsa
    if len(sys.argv) > 1:
        lr_runs = [Path(arg) for arg in sys.argv[1:]]

    print("=" * 80)
    print("LEARNING RATE DENEYLERİ - KAPSAMLI KARŞILAŞTIRMA")
    print("=" * 80)
    print(f"Toplam {len(lr_runs)} deney bulundu:\n")
    for run in lr_runs:
        print(f"  - {run.name}")
    print("=" * 80)

    # Analyze each run
    analyses = []
    for run_dir in lr_runs:
        print(f"\n[ANALYZING] {run_dir.name}...")
        try:
            analysis = analyze_run(str(run_dir))
            analyses.append(analysis)
            print(f"  [OK] Best mAP@0.5: {analysis.get('best_map50', 0):.4f}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

    if not analyses:
        print("\n[ERROR] Hiçbir deney analiz edilemedi!")
        sys.exit(1)

    # Output directory
    output_dir = "artifacts/lr_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GRAFIK OLUŞTURULUYOR...")
    print("=" * 80)
    create_comparison_plots(analyses, output_dir)

    # Generate report
    print("\n" + "=" * 80)
    print("RAPOR OLUŞTURULUYOR...")
    print("=" * 80)
    generate_markdown_report(analyses, output_dir)

    print("\n" + "=" * 80)
    print("[SUCCESS] TAMAMLANDI!")
    print("=" * 80)
    print(f"Çıktı klasörü: {output_dir}/")
    print(f"Rapor: {output_dir}/comparison_report.md")
    print("=" * 80)

if __name__ == "__main__":
    main()
