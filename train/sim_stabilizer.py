"""
Stabilizator duzeltmesi simulasyonu.
Eski vs yeni infer_scale() karsilastirmasi — gerçek eğitim verileri uzerinde.
"""
import csv
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from scheduler.fuzzy_lr import FuzzyLRController, FuzzyLRConfig

def run_simulation(steps_csv_path, run_name):
    cfg = FuzzyLRConfig()

    ctrl_old = FuzzyLRController(base_lr=0.0001, cfg=cfg)
    ctrl_new = FuzzyLRController(base_lr=0.0001, cfg=cfg)

    # Monkey-patch: eski infer_scale (stabilizator orijinal)
    original_infer = FuzzyLRController.infer_scale

    def infer_old(self, dl, gn, psteps, vtg=0.0, epoch_ratio=0.0):
        """Orijinal stabilizator: VTG["healthy"] * 1.0 (ağirlik=1.0, kosulsuz)"""
        DL = self._mf_delta_loss(dl)
        GN = self._mf_grad_norm(gn)
        PL = self._mf_plateau(psteps)
        VTG = self._mf_val_train_gap(vtg)

        rules = []
        rules += [(DL["hizla_azalan"], 1.12)]
        rules += [(DL["az_azalan"], 1.05)]
        rules += [(min(DL["az_azalan"], GN["kucuk"]), 1.07)]
        rules += [(min(DL["az_azalan"], GN["orta"]), 1.04)]
        escape = min(PL["uzun"], max(DL["az_azalan"], DL["hizla_azalan"]))
        rules += [(escape * 2.0, 1.12)]
        rules += [(min(DL["artan"], GN["buyuk"]), 0.93)]
        rules += [(VTG["warning"], 0.96)]
        rules += [(min(VTG["overfitting"], PL["uzun"]), 0.90)]
        overfitting_emergency = min(PL["cok_uzun"], max(VTG["warning"], VTG["overfitting"]))
        rules += [(overfitting_emergency * 3.0, 0.70)]
        explore_emergency = min(PL["cok_uzun"], VTG["healthy"])
        rules += [(explore_emergency * 2.5, 1.20)]
        escape_emergency = min(PL["cok_uzun"], max(DL["az_azalan"], DL["hizla_azalan"]))
        rules += [(escape_emergency * 2.0, 1.25)]
        # ESKI STABILIZATÖR: kosulsuz, tam ağirlik
        rules += [(VTG["healthy"], 1.0)]

        from scheduler.fuzzy_lr import clamp
        import math
        num = 0.0
        den = 0.0
        for weight, consequent in rules:
            if consequent > 1.0:
                w = weight * self.cfg.accel_boost
            elif consequent < 1.0:
                w = weight * self.cfg.brake_penalty
            else:
                w = weight
            num += w * consequent
            den += w
        scale = (num / (den + 1e-12)) if den > 0 else 1.0

        max_step = self.cfg.hysteresis_frac_max - (0.15 * epoch_ratio)
        max_step = clamp(max_step, self.cfg.hysteresis_frac_min, self.cfg.hysteresis_frac_max)
        delta = scale - self._last_scale
        if abs(delta) > max_step:
            scale = self._last_scale + math.copysign(max_step, delta)
        scale = clamp(scale, self.cfg.scale_min, self.cfg.scale_max)
        self._last_scale = scale
        return scale

    # Read steps
    rows = []
    with open(steps_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"\n{'='*70}")
    print(f"  SIMÜLASYON: {run_name}")
    print(f"  Toplam adim: {len(rows)}")
    print(f"{'='*70}")

    old_scales = []
    new_scales = []
    old_events_up = 0
    old_events_down = 0
    new_events_up = 0
    new_events_down = 0

    for row in rows:
        dl = float(row['delta_loss'])
        gn = float(row['grad_norm'])
        ps = int(float(row['plateau_steps']))
        vtg = float(row['val_train_gap'])
        er = float(row['epoch_ratio'])

        s_old = infer_old(ctrl_old, dl, gn, ps, vtg, er)
        s_new = ctrl_new.infer_scale(dl, gn, ps, vtg, er)

        old_scales.append(s_old)
        new_scales.append(s_new)

        if s_old > 1.001:
            old_events_up += 1
        elif s_old < 0.999:
            old_events_down += 1

        if s_new > 1.001:
            new_events_up += 1
        elif s_new < 0.999:
            new_events_down += 1

    # Istatistikler
    import statistics

    def stats(scales, label):
        mean_s = statistics.mean(scales)
        std_s = statistics.stdev(scales) if len(scales) > 1 else 0
        min_s = min(scales)
        max_s = max(scales)
        p5 = sorted(scales)[int(len(scales) * 0.05)]
        p50 = sorted(scales)[int(len(scales) * 0.50)]
        p95 = sorted(scales)[int(len(scales) * 0.95)]
        neutral = sum(1 for s in scales if 0.999 <= s <= 1.001)
        neutral_pct = neutral / len(scales) * 100
        return mean_s, std_s, min_s, max_s, p5, p50, p95, neutral_pct

    old_mean, old_std, old_min, old_max, old_p5, old_p50, old_p95, old_neutral = stats(old_scales, "ESKI")
    new_mean, new_std, new_min, new_max, new_p5, new_p50, new_p95, new_neutral = stats(new_scales, "YENI")

    print(f"\n{'Metrik':<25} {'ESKI':>12} {'YENI':>12} {'FARK':>12}")
    print("-" * 65)
    print(f"{'Scale ortalama':<25} {old_mean:>12.4f} {new_mean:>12.4f} {new_mean-old_mean:>+12.4f}")
    print(f"{'Scale std':<25} {old_std:>12.4f} {new_std:>12.4f} {new_std-old_std:>+12.4f}")
    print(f"{'Scale min':<25} {old_min:>12.4f} {new_min:>12.4f} {new_min-old_min:>+12.4f}")
    print(f"{'Scale max':<25} {old_max:>12.4f} {new_max:>12.4f} {new_max-old_max:>+12.4f}")
    print(f"{'Scale P5':<25} {old_p5:>12.4f} {new_p5:>12.4f} {new_p5-old_p5:>+12.4f}")
    print(f"{'Scale P50 (median)':<25} {old_p50:>12.4f} {new_p50:>12.4f} {new_p50-old_p50:>+12.4f}")
    print(f"{'Scale P95':<25} {old_p95:>12.4f} {new_p95:>12.4f} {new_p95-old_p95:>+12.4f}")
    print(f"{'Notr adim (%)':<25} {old_neutral:>11.1f}% {new_neutral:>11.1f}% {new_neutral-old_neutral:>+11.1f}%")
    print(f"{'Events UP (hizlanma)':<25} {old_events_up:>12d} {new_events_up:>12d} {new_events_up-old_events_up:>+12d}")
    print(f"{'Events DN (frenleme)':<25} {old_events_down:>12d} {new_events_down:>12d} {new_events_down-old_events_down:>+12d}")

    old_ratio = old_events_up / max(old_events_down, 1)
    new_ratio = new_events_up / max(new_events_down, 1)
    print(f"{'UP/DN orani':<25} {old_ratio:>12.1f} {new_ratio:>12.1f} {new_ratio-old_ratio:>+12.1f}")

    # Epoch bazinda ortalama scale
    print(f"\n  Epoch bazinda ortalama scale:")
    print(f"  {'Epoch':<8} {'ESKI':>8} {'YENI':>8} {'FARK':>8}")
    print(f"  {'-'*36}")

    epoch_old = {}
    epoch_new = {}
    for i, row in enumerate(rows):
        ep = int(float(row['epoch']))
        epoch_old.setdefault(ep, []).append(old_scales[i])
        epoch_new.setdefault(ep, []).append(new_scales[i])

    for ep in sorted(epoch_old.keys()):
        m_old = statistics.mean(epoch_old[ep])
        m_new = statistics.mean(epoch_new[ep])
        marker = " <<<" if abs(m_new - m_old) > 0.02 else ""
        print(f"  {ep:<8} {m_old:>8.4f} {m_new:>8.4f} {m_new-m_old:>+8.4f}{marker}")

    # Osilasyon analizi
    old_dir_changes = sum(1 for i in range(1, len(old_scales))
                         if (old_scales[i] - 1.0) * (old_scales[i-1] - 1.0) < 0)
    new_dir_changes = sum(1 for i in range(1, len(new_scales))
                         if (new_scales[i] - 1.0) * (new_scales[i-1] - 1.0) < 0)
    old_osc = old_dir_changes / max(len(old_scales)-1, 1) * 100
    new_osc = new_dir_changes / max(len(new_scales)-1, 1) * 100
    print(f"\n  Osilasyon orani: ESKI={old_osc:.1f}%, YENI={new_osc:.1f}% (fark: {new_osc-old_osc:+.1f}%)")


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    runs_dir = os.path.join(base, "..", "artifacts", "runs")

    # NEU-DET fuzzy hybrid (ana basari run'i)
    neudet_path = os.path.join(runs_dir, "neudet_fuzzy_hybrid", "steps.csv")
    if os.path.exists(neudet_path):
        run_simulation(neudet_path, "neudet_fuzzy_hybrid")

    # GC10 hybrid v2 (akilli emergency run'i)
    gc10_path = os.path.join(runs_dir, "gc10_hybrid_v2", "steps.csv")
    if os.path.exists(gc10_path):
        run_simulation(gc10_path, "gc10_hybrid_v2")
