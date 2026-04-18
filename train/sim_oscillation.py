"""
Osilasyon duzeltmesi simulasyonu.
EMA beta=0.9 vs beta=0.95 karsilastirmasi.
Her iki durumda da YENi infer_scale (stabilizator + frenleme duzeltmeli) kullanilir.
"""
import csv
import sys
import os
import statistics
sys.path.insert(0, os.path.dirname(__file__))

from scheduler.fuzzy_lr import FuzzyLRController, FuzzyLRConfig, EMA


def run_simulation(steps_csv_path, run_name):
    cfg = FuzzyLRConfig()

    ctrl_09 = FuzzyLRController(base_lr=0.0001, cfg=cfg)
    ctrl_095 = FuzzyLRController(base_lr=0.0001, cfg=cfg)

    # Raw loss degerlerini oku ve EMA'yi iki farkli beta ile yeniden hesapla
    rows = []
    with open(steps_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # EMA'lari raw loss'tan yeniden hesapla
    ema_09 = EMA(beta=0.9)
    ema_095 = EMA(beta=0.95)

    scales_09 = []
    scales_095 = []
    ev_up_09 = 0
    ev_dn_09 = 0
    ev_up_095 = 0
    ev_dn_095 = 0

    for row in rows:
        loss = float(row['loss'])
        gn = float(row['grad_norm'])
        ps = int(float(row['plateau_steps']))
        vtg = float(row['val_train_gap'])
        er = float(row['epoch_ratio'])

        # Delta loss: EMA farki
        dl_09 = ema_09.update(loss)
        dl_095 = ema_095.update(loss)

        s_09 = ctrl_09.infer_scale(dl_09, gn, ps, vtg, er)
        s_095 = ctrl_095.infer_scale(dl_095, gn, ps, vtg, er)

        scales_09.append(s_09)
        scales_095.append(s_095)

        if s_09 > 1.001: ev_up_09 += 1
        elif s_09 < 0.999: ev_dn_09 += 1
        if s_095 > 1.001: ev_up_095 += 1
        elif s_095 < 0.999: ev_dn_095 += 1

    print(f"\n{'='*70}")
    print(f"  OSILASYON SIMULASYONU: {run_name}")
    print(f"  Toplam adim: {len(rows)}")
    print(f"  Her iki durumda da YENI infer_scale kullaniliyor")
    print(f"{'='*70}")

    def calc_stats(scales):
        return {
            'mean': statistics.mean(scales),
            'std': statistics.stdev(scales),
            'min': min(scales),
            'max': max(scales),
            'p5': sorted(scales)[int(len(scales)*0.05)],
            'p50': sorted(scales)[int(len(scales)*0.50)],
            'p95': sorted(scales)[int(len(scales)*0.95)],
            'neutral': sum(1 for s in scales if 0.999 <= s <= 1.001) / len(scales) * 100,
        }

    s09 = calc_stats(scales_09)
    s095 = calc_stats(scales_095)

    print(f"\n{'Metrik':<25} {'beta=0.90':>12} {'beta=0.95':>12} {'FARK':>12}")
    print("-" * 65)
    for key, label in [('mean','Scale ortalama'), ('std','Scale std'),
                       ('min','Scale min'), ('max','Scale max'),
                       ('p5','Scale P5'), ('p50','Scale P50'),
                       ('p95','Scale P95')]:
        print(f"{label:<25} {s09[key]:>12.4f} {s095[key]:>12.4f} {s095[key]-s09[key]:>+12.4f}")
    print(f"{'Notr adim (%)':<25} {s09['neutral']:>11.1f}% {s095['neutral']:>11.1f}% {s095['neutral']-s09['neutral']:>+11.1f}%")
    print(f"{'Events UP':<25} {ev_up_09:>12d} {ev_up_095:>12d} {ev_up_095-ev_up_09:>+12d}")
    print(f"{'Events DN':<25} {ev_dn_09:>12d} {ev_dn_095:>12d} {ev_dn_095-ev_dn_09:>+12d}")
    r09 = ev_up_09 / max(ev_dn_09, 1)
    r095 = ev_up_095 / max(ev_dn_095, 1)
    print(f"{'UP/DN orani':<25} {r09:>12.1f} {r095:>12.1f} {r095-r09:>+12.1f}")

    # Osilasyon analizi
    osc_09 = sum(1 for i in range(1, len(scales_09))
                 if (scales_09[i]-1.0)*(scales_09[i-1]-1.0) < 0)
    osc_095 = sum(1 for i in range(1, len(scales_095))
                  if (scales_095[i]-1.0)*(scales_095[i-1]-1.0) < 0)
    pct_09 = osc_09 / max(len(scales_09)-1, 1) * 100
    pct_095 = osc_095 / max(len(scales_095)-1, 1) * 100
    print(f"\n  Osilasyon orani: beta=0.90: {pct_09:.1f}%  beta=0.95: {pct_095:.1f}%  (fark: {pct_095-pct_09:+.1f}%)")

    # Ardisik ayni yon sayisi (streak analizi)
    def streak_stats(scales):
        streaks = []
        current = 1
        for i in range(1, len(scales)):
            same_dir = (scales[i]-1.0) * (scales[i-1]-1.0) > 0
            if same_dir:
                current += 1
            else:
                streaks.append(current)
                current = 1
        streaks.append(current)
        return statistics.mean(streaks), max(streaks), statistics.median(streaks)

    mean_s09, max_s09, med_s09 = streak_stats(scales_09)
    mean_s095, max_s095, med_s095 = streak_stats(scales_095)
    print(f"\n  Ayni yonde ardisik adim (streak):")
    print(f"  {'Metrik':<20} {'beta=0.90':>12} {'beta=0.95':>12}")
    print(f"  {'-'*46}")
    print(f"  {'Ortalama streak':<20} {mean_s09:>12.1f} {mean_s095:>12.1f}")
    print(f"  {'Median streak':<20} {med_s09:>12.1f} {med_s095:>12.1f}")
    print(f"  {'Max streak':<20} {max_s09:>12d} {max_s095:>12d}")

    # Delta loss dagilimi karsilastirmasi
    ema_09_reset = EMA(beta=0.9)
    ema_095_reset = EMA(beta=0.95)
    dls_09 = []
    dls_095 = []
    for row in rows:
        loss = float(row['loss'])
        dls_09.append(ema_09_reset.update(loss))
        dls_095.append(ema_095_reset.update(loss))

    print(f"\n  Delta loss dagilimi:")
    print(f"  {'Metrik':<20} {'beta=0.90':>12} {'beta=0.95':>12}")
    print(f"  {'-'*46}")
    print(f"  {'Ortalama':<20} {statistics.mean(dls_09):>12.5f} {statistics.mean(dls_095):>12.5f}")
    print(f"  {'Std':<20} {statistics.stdev(dls_09):>12.5f} {statistics.stdev(dls_095):>12.5f}")
    abs09 = [abs(d) for d in dls_09]
    abs095 = [abs(d) for d in dls_095]
    print(f"  {'Ort. |delta|':<20} {statistics.mean(abs09):>12.5f} {statistics.mean(abs095):>12.5f}")
    print(f"  {'P95 |delta|':<20} {sorted(abs09)[int(len(abs09)*0.95)]:>12.5f} {sorted(abs095)[int(len(abs095)*0.95)]:>12.5f}")

    # Epoch bazinda osilasyon
    print(f"\n  Epoch bazinda osilasyon orani:")
    print(f"  {'Epoch':<8} {'b=0.90':>8} {'b=0.95':>8} {'FARK':>8}")
    print(f"  {'-'*36}")

    epoch_idx = {}
    for i, row in enumerate(rows):
        ep = int(float(row['epoch']))
        epoch_idx.setdefault(ep, []).append(i)

    for ep in sorted(epoch_idx.keys()):
        idxs = epoch_idx[ep]
        if len(idxs) < 2:
            continue
        osc_e09 = sum(1 for j in range(1, len(idxs))
                      if (scales_09[idxs[j]]-1.0)*(scales_09[idxs[j-1]]-1.0) < 0)
        osc_e095 = sum(1 for j in range(1, len(idxs))
                       if (scales_095[idxs[j]]-1.0)*(scales_095[idxs[j-1]]-1.0) < 0)
        pct_e09 = osc_e09 / (len(idxs)-1) * 100
        pct_e095 = osc_e095 / (len(idxs)-1) * 100
        marker = " <<<" if abs(pct_e095 - pct_e09) > 10 else ""
        print(f"  {ep:<8} {pct_e09:>7.1f}% {pct_e095:>7.1f}% {pct_e095-pct_e09:>+7.1f}%{marker}")


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    runs_dir = os.path.join(base, "..", "artifacts", "runs")

    neudet_path = os.path.join(runs_dir, "neudet_fuzzy_hybrid", "steps.csv")
    if os.path.exists(neudet_path):
        run_simulation(neudet_path, "neudet_fuzzy_hybrid")

    gc10_path = os.path.join(runs_dir, "gc10_hybrid_v2", "steps.csv")
    if os.path.exists(gc10_path):
        run_simulation(gc10_path, "gc10_hybrid_v2")
