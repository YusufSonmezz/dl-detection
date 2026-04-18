# train/sim_plateau_break.py
# Scale araliginin plato kirma kapasitesine etkisi
#
# Soru: Genis scale, platoyu daha hizli kirar mi?
# Mekanizma: Plato uzadikca PL["uzun/cok_uzun"] aktif → scale yukari
#            Genis aralikta bu kick daha guclu → LR daha cok degisir
#            → Model farkli bolgeleri kesfeder → platodan cikar

import math
import numpy as np

def clamp(x, lo, hi): return max(lo, min(hi, x))
def trimf(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    return (x - a) / (b - a + 1e-12) if x < b else (c - x) / (c - b + 1e-12)
def trapmf(x, a, b, c, d):
    if x < a or x > d: return 0.0
    elif b <= x <= c: return 1.0
    elif a <= x < b: return (x - a) / (b - a + 1e-12)
    else: return (d - x) / (d - c + 1e-12)


def simulate_plateau_response(scale_min, scale_max, consequents, label,
                               accel_boost=1.5, brake_penalty=0.7, stab_weight=0.5):
    """
    Plato senaryosunda scale tepkisini simule et.

    Senaryo: Model ep20'de platoya girer, 30 epoch boyunca plato devam eder.
    Her epoch 90 step. Delta_loss ~0 (durgun), grad_norm bimodal, VTG saglikli.
    """
    epoch_scale = 2.0  # 100ep → s=2
    results = []
    last_scale = 1.0

    for plateau_ep in range(0, 31):  # 0-30 epoch plato
        psteps = plateau_ep
        epoch_scales = []

        for step in range(90):
            # Plato sirasindaki tipik sinyaller
            dl = np.random.normal(-0.003, 0.015)   # neredeyse durgun
            gn = 9.0 + np.random.normal(0, 1.0) if np.random.random() > 0.25 else 0.5
            gn = max(0.0, gn)
            vtg = 0.08  # saglikli
            epoch_ratio = 0.4  # orta faz

            # MF'ler
            DL = {
                "hizla_azalan": trapmf(dl, -999, -999, -0.10, -0.04),
                "az_azalan":    trimf(dl, -0.10, -0.025, 0.0),
                "durdu":        trimf(dl, -0.015, 0.0, 0.015),
                "artan":        trapmf(dl, 0.01, 0.05, 999, 999),
            }
            GN = {
                "kucuk": trapmf(gn, 0.0, 0.0, 0.5, 3.0),
                "buyuk": trapmf(gn, 8.0, 9.5, 999, 999),
            }
            VTG = {
                "healthy": trapmf(vtg, -999, -999, 0.15, 0.30),
                "warning": trimf(vtg, 0.25, 0.45, 0.65),
                "overfitting": trapmf(vtg, 0.55, 0.8, 999, 999),
            }
            s = epoch_scale
            PL = {
                "kisa":     trapmf(psteps, 0, 0, 2*s, 4*s),
                "orta":     trimf(psteps, 2*s, 5*s, 8*s),
                "uzun":     trimf(psteps, 6*s, 9*s, 12*s),
                "cok_uzun": trapmf(psteps, 8*s, 12*s, 999, 999),
            }

            c = consequents
            rules = [
                (DL["hizla_azalan"],                      c["accel_strong"]),
                (DL["az_azalan"],                         c["accel_mild"]),
                (min(DL["az_azalan"], GN["kucuk"]),       c["accel_fallback"]),
                (DL["artan"],                             c["brake_mild"]),
                (min(DL["artan"], GN["buyuk"]),           c["brake_strong"]),
                (VTG["warning"],                          c["brake_warning"]),
                (min(VTG["overfitting"], PL["uzun"]),     c["brake_overfit"]),
                (VTG["healthy"] * PL["kisa"] * stab_weight, 1.0),
            ]

            num, den = 0.0, 0.0
            for weight, cons in rules:
                if cons > 1.0: w = weight * accel_boost
                elif cons < 1.0: w = weight * brake_penalty
                else: w = weight
                num += w * cons
                den += w

            scale = (num / (den + 1e-12)) if den > 0 else 1.0

            # Hysteresis
            max_step = clamp(0.10 - 0.15 * epoch_ratio, 0.08, 0.10)
            delta = scale - last_scale
            if abs(delta) > max_step:
                scale = last_scale + math.copysign(max_step, delta)
            scale = clamp(scale, scale_min, scale_max)
            last_scale = scale
            epoch_scales.append(scale)

        mean_s = np.mean(epoch_scales)
        # LR etkisi (gate=0.3 varsayimi, tipik plato)
        lr0 = 0.0001
        base_lr = lr0 * 0.7 + lr0 * 0.85 * 0.3  # gate=0.3, cosine ~0.85
        effective_lr = base_lr * mean_s
        lr_change_pct = (mean_s - 1.0) * 100

        results.append({
            "psteps": plateau_ep,
            "mean_scale": mean_s,
            "min_scale": min(epoch_scales),
            "max_scale": max(epoch_scales),
            "lr_change_pct": lr_change_pct,
            "effective_lr": effective_lr,
        })

    return results


# ============================================================================
#  SENARYOLAR
# ============================================================================
scenarios = {
    "A: [0.90, 1.10]": {
        "scale_min": 0.90, "scale_max": 1.10,
        "consequents": {
            "accel_strong": 1.08, "accel_mild": 1.04, "accel_fallback": 1.05,
            "brake_mild": 0.97, "brake_strong": 0.94, "brake_warning": 0.96,
            "brake_overfit": 0.92,
        },
    },
    "B: [0.85, 1.15]": {
        "scale_min": 0.85, "scale_max": 1.15,
        "consequents": {
            "accel_strong": 1.12, "accel_mild": 1.06, "accel_fallback": 1.07,
            "brake_mild": 0.96, "brake_strong": 0.90, "brake_warning": 0.94,
            "brake_overfit": 0.88,
        },
    },
    "C: [0.80, 1.20]": {
        "scale_min": 0.80, "scale_max": 1.20,
        "consequents": {
            "accel_strong": 1.15, "accel_mild": 1.07, "accel_fallback": 1.08,
            "brake_mild": 0.95, "brake_strong": 0.88, "brake_warning": 0.93,
            "brake_overfit": 0.85,
        },
    },
}

np.random.seed(42)

print("=" * 110)
print("PLATO KIRMA KAPASITESI: Scale araliginin plato tepkisine etkisi")
print("=" * 110)
print("\nSenaryo: Model platoya girmis, delta_loss ~0, VTG saglikli, epoch_ratio=0.4")
print("PL MF epoch-olcekli (100ep, s=2): kisa<8, orta=4-16, uzun=12-24, cok_uzun>16\n")

# Her senaryo icin plato tepkisi
all_results = {}
for name, cfg in scenarios.items():
    np.random.seed(42)  # Ayni sinyaller
    all_results[name] = simulate_plateau_response(
        cfg["scale_min"], cfg["scale_max"], cfg["consequents"], name
    )

# Tablo 1: Plato suresi vs ortalama scale
print(f"{'Plato(ep)':<12}", end="")
for name in scenarios:
    short = name.split(":")[0].strip()
    print(f" {short+' scale':>12} {short+' LR%':>10}", end="")
print()
print("-" * 110)

for pstep in [0, 2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30]:
    print(f"  {pstep:>3} ep    ", end="")
    for name in scenarios:
        r = all_results[name][pstep]
        print(f" {r['mean_scale']:>12.4f} {r['lr_change_pct']:>+9.2f}%", end="")
    print()

# Tablo 2: Plato kirma metrikleri
print("\n" + "=" * 110)
print("PLATO KIRMA METRIKLERI")
print("=" * 110)

print(f"\n{'Metrik':<40}", end="")
for name in scenarios:
    short = name.split(":")[0].strip()
    print(f" {short:>18}", end="")
print()
print("-" * 110)

# LR'nin %5'ten fazla degistigi ilk plato epoch'u
for name in scenarios:
    results = all_results[name]
    first_5pct = next((r["psteps"] for r in results if abs(r["lr_change_pct"]) > 5), ">30")
    scenarios[name]["first_5pct"] = first_5pct

print(f"{'LR > %5 degisim (ilk epoch)':<40}", end="")
for name in scenarios:
    print(f" {str(scenarios[name]['first_5pct']):>18}", end="")
print()

# Plato 10. epoch'taki scale
print(f"{'Plato 10ep: ort. scale':<40}", end="")
for name in scenarios:
    print(f" {all_results[name][10]['mean_scale']:>18.4f}", end="")
print()

# Plato 20. epoch'taki scale
print(f"{'Plato 20ep: ort. scale':<40}", end="")
for name in scenarios:
    print(f" {all_results[name][20]['mean_scale']:>18.4f}", end="")
print()

# Max scale ulasilabilir
print(f"{'Max scale (30ep plato)':<40}", end="")
for name in scenarios:
    print(f" {all_results[name][30]['max_scale']:>18.4f}", end="")
print()

# LR pertürbasyon gücü (plato 15ep'te)
print(f"{'LR pertürbasyon gucu (15ep plato)':<40}", end="")
for name in scenarios:
    r = all_results[name][15]
    lr0 = 0.0001
    lr_perturb = lr0 * abs(r["mean_scale"] - 1.0)
    print(f" {lr_perturb:.8f}", end="")
print()

# Stabilizator baskinligi (plato 0'da scale ne kadar 1.0'a yakin)
print(f"{'Stab. baskinligi (plato=0, |s-1|)':<40}", end="")
for name in scenarios:
    r = all_results[name][0]
    print(f" {abs(r['mean_scale'] - 1.0):>18.4f}", end="")
print()

# ============================================================================
#  GATE + SCALE ETKILESIMI
# ============================================================================
print("\n" + "=" * 110)
print("GATE + SCALE ETKILESIMI: Plato sirasinda efektif LR")
print("=" * 110)
print("\nGate degerleri: 0.0 (saf fuzzy), 0.3 (gecis), 0.6 (cogunlukla cosine), 1.0 (tam cosine)")
print("Cosine progress=0.4 varsayimi (lr0=0.0001)\n")

lr0 = 0.0001
lrf = 0.10
cosine_factor = (1.0 + math.cos(math.pi * 0.4)) / 2.0
cosine_lr = lr0 * (cosine_factor * (1.0 - lrf) + lrf)

print(f"Cosine LR @ progress=0.4: {cosine_lr:.7f}")
print()

gates = [0.0, 0.3, 0.6, 1.0]
pstep_check = 15  # 15 epoch plato

print(f"{'Gate':<8}", end="")
for name in scenarios:
    short = name.split(":")[0].strip()
    print(f" {short+' LR':>14} {short+' vs noScale':>16}", end="")
print()
print("-" * 110)

for gate in gates:
    base_lr = lr0 * (1 - gate) + cosine_lr * gate
    no_scale_lr = base_lr  # scale=1.0

    print(f"  {gate:<6.1f}", end="")
    for name in scenarios:
        r = all_results[name][pstep_check]
        eff_lr = base_lr * r["mean_scale"]
        diff_pct = (eff_lr / no_scale_lr - 1.0) * 100
        print(f" {eff_lr:>14.8f} {diff_pct:>+15.2f}%", end="")
    print()

print("\n" + "=" * 110)
print("SONUC")
print("=" * 110)
print("""
PLATO KIRMA MEKANIZMASI:
  1. Plato uzar → PL["uzun/cok_uzun"] aktif → accel consequent'ler ateşlenir
  2. Scale yukari → LR artar → model farkli bölgeleri kesfeder
  3. Eger kesif basarili → plato kirilir (mAP50 artar)
  4. Eger basarisiz → gate artar → cosine LR'yi dusurur → fine-tuning

SCALE ARALIGININ ETKISI:
  - Dar [0.90, 1.10]: Scale kick cok zayif, plato kiramaz
  - Orta [0.85, 1.15]: Anlamli kick, cosine hala dominant
  - Genis [0.80, 1.20]: Guclu kick, ama cosine ile catisma riski
""")
