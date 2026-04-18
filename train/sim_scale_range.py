# train/sim_scale_range.py
# Scale aralik simülasyonu: dar vs genis scale'in MIMO'daki etkisi
#
# Amac: [0.90, 1.10] vs [0.80, 1.20] vs [0.75, 1.30] karsilastirmasi
# Soru: Scale araligi genisletildiginde dengeler nasil degisir?

import math
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# ============================================================================
#  YARDIMCI FONKSIYONLAR (fuzzy_lr.py'den)
# ============================================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def trimf(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    if x < b:  return (x - a) / (b - a + 1e-12)
    return (c - x) / (c - b + 1e-12)

def trapmf(x, a, b, c, d):
    if x < a or x > d: return 0.0
    elif b <= x <= c: return 1.0
    elif a <= x < b: return (x - a) / (b - a + 1e-12)
    else: return (d - x) / (d - c + 1e-12)


# ============================================================================
#  SCALE INFERENCE (fuzzy_lr.py infer_scale ile ayni mantik)
# ============================================================================
def infer_scale(dl, gn, psteps, vtg, epoch_ratio, last_scale,
                scale_min, scale_max, consequents,
                accel_boost=1.5, brake_penalty=0.7,
                stab_weight=0.5):
    """
    consequents: dict with keys for each rule's crisp output
    """
    # MF'ler (degismiyor)
    DL = {
        "hizla_azalan": trapmf(dl, -999, -999, -0.10, -0.04),
        "az_azalan":    trimf(dl, -0.10, -0.025, 0.0),
        "durdu":        trimf(dl, -0.015, 0.0, 0.015),
        "artan":        trapmf(dl, 0.01, 0.05, 999, 999),
    }
    GN = {
        "kucuk": trapmf(gn, 0.0, 0.0, 0.5, 3.0),
        "orta":  trimf(gn, 2.0, 6.0, 9.0),
        "buyuk": trapmf(gn, 8.0, 9.5, 999, 999),
    }
    VTG = {
        "healthy":     trapmf(vtg, -999, -999, 0.15, 0.30),
        "warning":     trimf(vtg, 0.25, 0.45, 0.65),
        "overfitting": trapmf(vtg, 0.55, 0.8, 999, 999),
    }
    PL = {
        "kisa":      trapmf(psteps, 0, 0, 4, 8),
        "orta":      trimf(psteps, 4, 10, 16),
        "uzun":      trimf(psteps, 12, 18, 24),
        "cok_uzun":  trapmf(psteps, 16, 24, 999, 999),
    }

    c = consequents  # kisa isim
    rules = [
        # ACCEL
        (DL["hizla_azalan"],                      c["accel_strong"]),
        (DL["az_azalan"],                         c["accel_mild"]),
        (min(DL["az_azalan"], GN["kucuk"]),       c["accel_fallback"]),
        # BRAKE
        (DL["artan"],                             c["brake_mild"]),
        (min(DL["artan"], GN["buyuk"]),           c["brake_strong"]),
        (VTG["warning"],                          c["brake_warning"]),
        (min(VTG["overfitting"], PL["uzun"]),     c["brake_overfit"]),
        # STABILIZER
        (VTG["healthy"] * PL["kisa"] * stab_weight, 1.0),
    ]

    num, den = 0.0, 0.0
    for weight, cons in rules:
        if cons > 1.0:
            w = weight * accel_boost
        elif cons < 1.0:
            w = weight * brake_penalty
        else:
            w = weight
        num += w * cons
        den += w

    scale = (num / (den + 1e-12)) if den > 0 else 1.0

    # Hysteresis
    max_step = clamp(0.10 - 0.15 * epoch_ratio, 0.08, 0.10)
    delta = scale - last_scale
    if abs(delta) > max_step:
        scale = last_scale + math.copysign(max_step, delta)

    # Global clamp
    scale = clamp(scale, scale_min, scale_max)
    return scale


# ============================================================================
#  SENARYOLAR
# ============================================================================
scenarios = {
    "A: Mevcut [0.90, 1.10]": {
        "scale_min": 0.90, "scale_max": 1.10,
        "consequents": {
            "accel_strong": 1.08, "accel_mild": 1.04, "accel_fallback": 1.05,
            "brake_mild": 0.97, "brake_strong": 0.94, "brake_warning": 0.96,
            "brake_overfit": 0.92,
        },
        "stab_weight": 0.5,
    },
    "B: Genis [0.80, 1.20]": {
        "scale_min": 0.80, "scale_max": 1.20,
        "consequents": {
            "accel_strong": 1.15, "accel_mild": 1.07, "accel_fallback": 1.08,
            "brake_mild": 0.95, "brake_strong": 0.88, "brake_warning": 0.93,
            "brake_overfit": 0.85,
        },
        "stab_weight": 0.5,
    },
    "C: Eski [0.75, 1.30]": {
        "scale_min": 0.75, "scale_max": 1.30,
        "consequents": {
            "accel_strong": 1.20, "accel_mild": 1.08, "accel_fallback": 1.10,
            "brake_mild": 0.93, "brake_strong": 0.85, "brake_warning": 0.90,
            "brake_overfit": 0.80,
        },
        "stab_weight": 0.5,
    },
    "D: Mevcut + stab=0.3": {
        "scale_min": 0.90, "scale_max": 1.10,
        "consequents": {
            "accel_strong": 1.08, "accel_mild": 1.04, "accel_fallback": 1.05,
            "brake_mild": 0.97, "brake_strong": 0.94, "brake_warning": 0.96,
            "brake_overfit": 0.92,
        },
        "stab_weight": 0.3,
    },
    "E: Orta [0.85, 1.15]": {
        "scale_min": 0.85, "scale_max": 1.15,
        "consequents": {
            "accel_strong": 1.12, "accel_mild": 1.06, "accel_fallback": 1.07,
            "brake_mild": 0.96, "brake_strong": 0.90, "brake_warning": 0.94,
            "brake_overfit": 0.88,
        },
        "stab_weight": 0.5,
    },
}


# ============================================================================
#  EGITIM SINYALI URETECI
# ============================================================================
def generate_signals(n_epochs=100, steps_per_epoch=90):
    """Gercekci egitim sinyalleri uret."""
    signals = []
    plateau_counter = 0
    best_loss = 999.0
    loss = 3.5

    for ep in range(n_epochs):
        epoch_ratio = ep / n_epochs
        # VTG: yavas artis
        vtg = 0.02 + 0.15 * epoch_ratio + random.gauss(0, 0.03)
        vtg = max(-0.05, vtg)

        for step in range(steps_per_epoch):
            # Delta loss: genelde negatif (ogrenme), bazen pozitif
            if epoch_ratio < 0.3:
                dl = random.gauss(-0.03, 0.04)  # erken: guclu ogrenme
            elif epoch_ratio < 0.7:
                dl = random.gauss(-0.005, 0.035)  # orta: yavaslayan
            else:
                dl = random.gauss(0.002, 0.03)  # gec: neredeyse durgun

            # Grad norm: bimodal
            if random.random() < 0.25:
                gn = random.gauss(0.5, 0.2)  # fallback
            else:
                gn = random.gauss(9.0, 1.5)  # normal

            gn = max(0.0, gn)

            signals.append({
                "dl": dl,
                "gn": gn,
                "psteps": plateau_counter,
                "vtg": vtg,
                "epoch_ratio": epoch_ratio,
            })

        # Epoch sonu: plateau guncelle
        loss *= (0.98 - 0.005 * epoch_ratio)  # yavas azalan loss
        noise = random.gauss(0, 0.02)
        current = loss + noise

        if current < best_loss * 0.998:
            best_loss = current
            plateau_counter = 0
        else:
            plateau_counter += 1

    return signals


# ============================================================================
#  SIMULASYON
# ============================================================================
def run_simulation():
    signals = generate_signals()
    print(f"Toplam adim: {len(signals)}\n")

    results = {}
    for name, cfg in scenarios.items():
        scales = []
        last_scale = 1.0

        for sig in signals:
            s = infer_scale(
                dl=sig["dl"], gn=sig["gn"], psteps=sig["psteps"],
                vtg=sig["vtg"], epoch_ratio=sig["epoch_ratio"],
                last_scale=last_scale,
                scale_min=cfg["scale_min"], scale_max=cfg["scale_max"],
                consequents=cfg["consequents"],
                stab_weight=cfg["stab_weight"],
            )
            scales.append(s)
            last_scale = s

        scales = np.array(scales)

        results[name] = {
            "mean": np.mean(scales),
            "std": np.std(scales),
            "min": np.min(scales),
            "max": np.max(scales),
            "near_identity": np.mean((scales >= 0.98) & (scales <= 1.02)) * 100,
            "strong_accel": np.mean(scales > 1.05) * 100,
            "strong_brake": np.mean(scales < 0.95) * 100,
            "any_accel": np.mean(scales > 1.01) * 100,
            "any_brake": np.mean(scales < 0.99) * 100,
        }

    # ========================================================================
    #  SONUC TABLOSU
    # ========================================================================
    print("=" * 100)
    print("SCALE ARALIK SIMULASYONU SONUCLARI")
    print("=" * 100)
    print(f"{'Senaryo':<30} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6} "
          f"{'~1.0%':>6} {'>1.05%':>7} {'<0.95%':>7} {'>1.01%':>7} {'<0.99%':>7}")
    print("-" * 100)

    for name, r in results.items():
        print(f"{name:<30} {r['mean']:>6.4f} {r['std']:>6.4f} {r['min']:>6.3f} {r['max']:>6.3f} "
              f"{r['near_identity']:>5.1f}% {r['strong_accel']:>6.1f}% {r['strong_brake']:>6.1f}% "
              f"{r['any_accel']:>6.1f}% {r['any_brake']:>6.1f}%")

    # ========================================================================
    #  LR ETKISI ANALIZI
    # ========================================================================
    print("\n" + "=" * 100)
    print("LR ETKISI: Cosine vs Scale karsilastirmasi (lr0=0.0001, lrf=0.10)")
    print("=" * 100)

    lr0 = 0.0001
    lrf = 0.10

    # Cosine LR range
    cosine_max = lr0  # progress=0
    cosine_min = lr0 * lrf  # progress=1
    cosine_range = cosine_max - cosine_min
    cosine_pct = (cosine_range / cosine_max) * 100

    print(f"\nCosine LR araligi: {cosine_min:.7f} - {cosine_max:.7f} (degisim: {cosine_pct:.1f}%)")

    print(f"\n{'Senaryo':<30} {'Scale LR aralik':>16} {'Scale %':>8} {'Cosine/Scale':>13} {'Toplam LR aralik':>18}")
    print("-" * 100)

    for name, r in results.items():
        # Scale'in tipik LR etkisi (orta cosine noktasinda)
        mid_cosine = lr0 * 0.55  # ~orta nokta
        scale_min_lr = mid_cosine * r['min']
        scale_max_lr = mid_cosine * r['max']
        scale_range = scale_max_lr - scale_min_lr
        scale_pct = (scale_range / mid_cosine) * 100

        # Toplam LR araligi
        total_min = cosine_min * r['min']
        total_max = cosine_max * r['max']

        ratio = cosine_pct / scale_pct if scale_pct > 0 else float('inf')

        print(f"{name:<30} {scale_range:.8f} ({scale_pct:>5.1f}%) "
              f"  {ratio:>5.1f}x cosine  "
              f"  {total_min:.7f} - {total_max:.7f}")

    # ========================================================================
    #  EPOCH BAZLI DAGILIM
    # ========================================================================
    print("\n" + "=" * 100)
    print("EPOCH BAZLI ORTALAMA SCALE (her 10 epoch)")
    print("=" * 100)

    steps_per_epoch = 90
    header = f"{'Epoch':<10}"
    for name in scenarios:
        short = name.split(":")[0]
        header += f" {short:>10}"
    print(header)
    print("-" * 70)

    for name, cfg in scenarios.items():
        # Yeniden hesapla (epoch bazli)
        last_scale = 1.0
        epoch_means = []
        epoch_scales = []

        for i, sig in enumerate(signals):
            s = infer_scale(
                dl=sig["dl"], gn=sig["gn"], psteps=sig["psteps"],
                vtg=sig["vtg"], epoch_ratio=sig["epoch_ratio"],
                last_scale=last_scale,
                scale_min=cfg["scale_min"], scale_max=cfg["scale_max"],
                consequents=cfg["consequents"],
                stab_weight=cfg["stab_weight"],
            )
            epoch_scales.append(s)
            last_scale = s

            if (i + 1) % steps_per_epoch == 0:
                epoch_means.append(np.mean(epoch_scales))
                epoch_scales = []

        scenarios[name]["epoch_means"] = epoch_means

    for ep_idx in range(0, 100, 10):
        row = f"Ep {ep_idx:>3}-{ep_idx+9:<3}"
        for name in scenarios:
            means = scenarios[name]["epoch_means"]
            block = means[ep_idx:ep_idx+10]
            avg = np.mean(block)
            row += f" {avg:>10.4f}"
        print(row)


if __name__ == "__main__":
    run_simulation()
