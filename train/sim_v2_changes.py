# train/sim_v2_changes.py
# MIMO v2 degisikliklerinin birlesik simulasyonu
#
# 3 degisiklik test ediliyor:
#   1. Plateau MF esikleri dusuruldu (tepki 2 epoch erken)
#   2. Scale plato kacis kurallari eklendi
#   3. Scale araligi [0.85, 1.15] + orantili consequent'ler
#
# Karsilastirma:
#   - v1 (mevcut MIMO)
#   - v2 (3 degisiklik birlikte)
#   - neudet_org_fuzzy referans (eski genis scale [0.75, 1.30])

import math
import numpy as np

np.random.seed(42)

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


# ============================================================================
#  MIMO SIMULATORU
# ============================================================================
class MIMOSimulator:
    def __init__(self, config):
        self.cfg = config
        self.last_scale = 1.0
        self.last_gate = 0.0
        self.progress = 0.0
        self.lr0 = 0.0001
        self.lrf = config.get("lrf", 0.10)
        self.epoch_scale = 100.0 / 50.0  # 100ep, s=2

    def _mf_delta_loss(self, dl):
        return {
            "hizla_azalan": trapmf(dl, -999, -999, -0.10, -0.04),
            "az_azalan":    trimf(dl, -0.10, -0.025, 0.0),
            "durdu":        trimf(dl, -0.015, 0.0, 0.015),
            "artan":        trapmf(dl, 0.01, 0.05, 999, 999),
        }

    def _mf_grad_norm(self, gn):
        return {
            "kucuk": trapmf(gn, 0.0, 0.0, 0.5, 3.0),
            "orta":  trimf(gn, 2.0, 6.0, 9.0),
            "buyuk": trapmf(gn, 8.0, 9.5, 999, 999),
        }

    def _mf_plateau(self, p):
        s = self.epoch_scale
        pl_cfg = self.cfg["plateau_mf"]
        return {
            "kisa":     trapmf(p, 0, 0, pl_cfg["kisa_ust"]*s, pl_cfg["kisa_end"]*s),
            "orta":     trimf(p, pl_cfg["orta_start"]*s, pl_cfg["orta_peak"]*s, pl_cfg["orta_end"]*s),
            "uzun":     trimf(p, pl_cfg["uzun_start"]*s, pl_cfg["uzun_peak"]*s, pl_cfg["uzun_end"]*s),
            "cok_uzun": trapmf(p, pl_cfg["cok_uzun_start"]*s, pl_cfg["cok_uzun_full"]*s, 999, 999),
        }

    def _mf_vtg(self, gap):
        return {
            "healthy":     trapmf(gap, -999, -999, 0.15, 0.30),
            "warning":     trimf(gap, 0.25, 0.45, 0.65),
            "overfitting": trapmf(gap, 0.55, 0.8, 999, 999),
        }

    def _mf_cosine_progress(self, p):
        return {
            "erken":  trapmf(p, 0, 0, 0.3, 0.5),
            "orta":   trimf(p, 0.3, 0.5, 0.8),
            "gec":    trapmf(p, 0.7, 0.9, 1.0, 1.0),
        }

    def infer_scale(self, dl, gn, psteps, vtg, epoch_ratio):
        DL = self._mf_delta_loss(dl)
        GN = self._mf_grad_norm(gn)
        VTG = self._mf_vtg(vtg)
        PL = self._mf_plateau(psteps)

        c = self.cfg["consequents"]
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
            (VTG["healthy"] * PL["kisa"] * 0.5,       1.0),
        ]

        # PLATO KACIS KURALLARI (sadece v2'de aktif)
        if self.cfg.get("plato_escape", False):
            rules.append((PL["uzun"],                              c.get("escape_uzun", 1.12)))
            rules.append((min(PL["cok_uzun"], VTG["healthy"]),     c.get("escape_cok_uzun", 1.15)))

        num, den = 0.0, 0.0
        ab, bp = 1.5, 0.7
        for weight, cons in rules:
            if cons > 1.0: w = weight * ab
            elif cons < 1.0: w = weight * bp
            else: w = weight
            num += w * cons
            den += w

        scale = (num / (den + 1e-12)) if den > 0 else 1.0

        # Hysteresis
        max_step = clamp(0.10 - 0.15 * epoch_ratio, 0.08, 0.10)
        delta = scale - self.last_scale
        if abs(delta) > max_step:
            scale = self.last_scale + math.copysign(max_step, delta)

        smin, smax = self.cfg["scale_min"], self.cfg["scale_max"]
        scale = clamp(scale, smin, smax)
        self.last_scale = scale
        return scale

    def infer_gate(self, dl, gn, psteps, vtg, epoch_ratio):
        DL = self._mf_delta_loss(dl)
        PL = self._mf_plateau(psteps)
        VTG = self._mf_vtg(vtg)

        rules = [
            (min(DL["hizla_azalan"], PL["kisa"]),   0.0),
            (min(DL["az_azalan"], PL["kisa"]),       0.0),
            (min(PL["kisa"], VTG["healthy"]),         0.0),
            (min(PL["orta"], VTG["healthy"]),         0.3),
            (PL["uzun"],                              0.7),
            (PL["cok_uzun"],                          1.0),
            (VTG["warning"],                          0.5),
            (VTG["overfitting"],                      0.8),
        ]

        num, den = 0.0, 0.0
        for w, c in rules:
            num += w * c
            den += w
        gate = (num / (den + 1e-12)) if den > 0 else 0.0
        gate = clamp(gate, 0.0, 1.0)

        # Gate hysteresis (0.15/epoch)
        delta = gate - self.last_gate
        if abs(delta) > 0.15:
            gate = self.last_gate + math.copysign(0.15, delta)
        gate = clamp(gate, 0.0, 1.0)
        self.last_gate = gate
        return gate

    def infer_speed(self, dl, gn, psteps, vtg, epoch_ratio):
        DL = self._mf_delta_loss(dl)
        PL = self._mf_plateau(psteps)
        VTG = self._mf_vtg(vtg)
        CP = self._mf_cosine_progress(self.progress)

        rules = [
            (DL["hizla_azalan"],                         0.5),
            (min(DL["az_azalan"], VTG["healthy"]),       0.7),
            (DL["durdu"],                                1.0),
            (PL["orta"],                                 1.5),
            (PL["uzun"],                                 2.0),
            (VTG["warning"],                             2.0),
            (VTG["overfitting"],                         3.0),
            (min(PL["cok_uzun"], VTG["healthy"]),        1.5),
            (min(CP["gec"], PL["uzun"]),                 0.3),
            (min(CP["gec"], VTG["healthy"]),             0.5),
            (min(CP["gec"], DL["hizla_azalan"]),         0.3),
        ]

        num, den = 0.0, 0.0
        for w, c in rules:
            num += w * c
            den += w
        speed = (num / (den + 1e-12)) if den > 0 else 1.0
        return clamp(speed, 0.3, 3.0)

    def step_epoch(self, dl, gn, psteps, vtg, epoch_ratio):
        """Bir epoch'luk MIMO inference (ortalama 90 step)."""
        scale = self.infer_scale(dl, gn, psteps, vtg, epoch_ratio)
        gate = self.infer_gate(dl, gn, psteps, vtg, epoch_ratio)
        speed = self.infer_speed(dl, gn, psteps, vtg, epoch_ratio)

        # Cosine progress ilerlet
        if epoch_ratio >= 3/100:  # warmup sonrasi
            effective_total = max(100 - 3, 1)
            self.progress += (1.0 / effective_total) * speed
            self.progress = min(self.progress, 1.0)

        # LR hesapla
        cosine_factor = (1.0 + math.cos(math.pi * self.progress)) / 2.0
        cosine_lr = self.lr0 * (cosine_factor * (1.0 - self.lrf) + self.lrf)

        if epoch_ratio < 3/100:  # warmup
            base_lr = 0.5 * self.lr0 + 0.5 * self.lr0 * (epoch_ratio * 100 + 1) / 3
        else:
            base_lr = self.lr0 * (1 - gate) + cosine_lr * gate

        effective_lr = base_lr * scale
        effective_lr = clamp(effective_lr, 1e-6, 0.001)

        return {
            "lr": effective_lr, "scale": scale, "gate": gate,
            "speed": speed, "progress": self.progress,
            "cosine_lr": cosine_lr, "base_lr": base_lr,
        }


# ============================================================================
#  KONFIGURASYONLAR
# ============================================================================
V1_CONFIG = {
    "name": "MIMO v1 (mevcut)",
    "scale_min": 0.90, "scale_max": 1.10,
    "lrf": 0.10,
    "plato_escape": False,
    "plateau_mf": {
        "kisa_ust": 2, "kisa_end": 4,
        "orta_start": 2, "orta_peak": 5, "orta_end": 8,
        "uzun_start": 6, "uzun_peak": 9, "uzun_end": 12,
        "cok_uzun_start": 8, "cok_uzun_full": 12,
    },
    "consequents": {
        "accel_strong": 1.08, "accel_mild": 1.04, "accel_fallback": 1.05,
        "brake_mild": 0.97, "brake_strong": 0.94, "brake_warning": 0.96,
        "brake_overfit": 0.92,
    },
}

V2_CONFIG = {
    "name": "MIMO v2 (3 degisiklik)",
    "scale_min": 0.85, "scale_max": 1.15,
    "lrf": 0.10,
    "plato_escape": True,
    "plateau_mf": {
        "kisa_ust": 1, "kisa_end": 3,        # 2,4 -> 1,3
        "orta_start": 1, "orta_peak": 3, "orta_end": 6,  # 2,5,8 -> 1,3,6
        "uzun_start": 4, "uzun_peak": 7, "uzun_end": 10,  # 6,9,12 -> 4,7,10
        "cok_uzun_start": 6, "cok_uzun_full": 10,  # 8,12 -> 6,10
    },
    "consequents": {
        "accel_strong": 1.12, "accel_mild": 1.06, "accel_fallback": 1.07,
        "brake_mild": 0.96, "brake_strong": 0.90, "brake_warning": 0.94,
        "brake_overfit": 0.88,
        "escape_uzun": 1.12,
        "escape_cok_uzun": 1.15,
    },
}


# ============================================================================
#  EGITIM SIMULASYONU (100 epoch, gercekci sinyal)
# ============================================================================
def simulate_training(config, seed=42):
    np.random.seed(seed)
    sim = MIMOSimulator(config)

    # Gercekci mAP50 trajectory (neudet_mimo_v1'den esinlenilmis)
    # Hizli yukselis -> plato -> hafif dusus -> recovery -> final plato
    epochs = []
    psteps = 0
    best_metric = 0.0
    loss = 3.5

    for ep in range(100):
        epoch_ratio = ep / 100.0

        # Gercekci mAP50 uretimi
        if ep < 3:
            base_map = 0.15 + ep * 0.17  # warmup: 0.15 -> 0.66
        elif ep < 10:
            base_map = 0.65 + (ep - 3) * 0.008  # hizli: 0.65 -> 0.71
        elif ep < 25:
            base_map = 0.71 + (ep - 10) * 0.0015  # yavas: 0.71 -> 0.73
        elif ep < 50:
            base_map = 0.73 - (ep - 25) * 0.0008  # hafif dusus: 0.73 -> 0.71
        elif ep < 70:
            base_map = 0.71 + (ep - 50) * 0.0005  # recovery: 0.71 -> 0.72
        else:
            base_map = 0.72 - (ep - 70) * 0.0003  # final: 0.72 -> 0.71

        noise = np.random.normal(0, 0.015)
        map50 = clamp(base_map + noise, 0.1, 0.85)

        # Hybrid metric (0.7*m50 + 0.3*m95, m95 ~ m50*0.5)
        map5095 = map50 * 0.5 + np.random.normal(0, 0.01)
        hybrid = map50 * 0.7 + map5095 * 0.3

        # Plateau tracker
        if best_metric == 0 or map50 > best_metric * 1.002:
            best_metric = map50
            psteps = 0
        else:
            psteps += 1

        # Delta loss
        loss *= (0.985 - 0.003 * epoch_ratio)
        dl_noise = np.random.normal(0, 0.02)
        if ep < 10:
            dl = -0.03 + dl_noise
        elif ep < 30:
            dl = -0.01 + dl_noise
        elif ep < 60:
            dl = -0.003 + dl_noise
        else:
            dl = 0.001 + dl_noise

        # Grad norm (bimodal)
        gn = np.random.normal(9.0, 1.5) if np.random.random() > 0.25 else 0.5
        gn = max(0.0, gn)

        # VTG
        vtg = 0.05 + 0.12 * epoch_ratio + np.random.normal(0, 0.02)
        vtg = max(-0.05, vtg)

        # MIMO step
        result = sim.step_epoch(dl, gn, psteps, vtg, epoch_ratio)

        epochs.append({
            "epoch": ep,
            "map50": map50,
            "psteps": psteps,
            **result,
        })

    return epochs


# ============================================================================
#  CALISTIR VE KARSILASTIR
# ============================================================================
configs = [V1_CONFIG, V2_CONFIG]
all_results = {}

for cfg in configs:
    name = cfg["name"]
    all_results[name] = simulate_training(cfg)

print("=" * 120)
print("MIMO v1 vs v2: 100 Epoch Egitim Simulasyonu")
print("=" * 120)

# Tablo 1: Her 5 epoch'ta karsilastirma
print(f"\n{'Ep':<5}", end="")
for name in all_results:
    short = name.split("(")[1].rstrip(")")
    print(f" | {short:>6} {'LR':>10} {'scale':>6} {'gate':>5} {'spd':>5} {'cos%':>5} {'plat':>5}", end="")
print()
print("-" * 120)

for ep in range(0, 100, 5):
    print(f"{ep:>3}  ", end="")
    for name in all_results:
        r = all_results[name][ep]
        print(f" | {r['map50']:>6.4f} {r['lr']:.8f} {r['scale']:>6.3f} {r['gate']:>5.2f} "
              f"{r['speed']:>5.2f} {r['progress']*100:>4.0f}% {r['psteps']:>5}", end="")
    print()

# Tablo 2: Plato tepki karsilastirmasi
print("\n" + "=" * 120)
print("PLATO TEPKI ANALIZI")
print("=" * 120)

for name, epochs in all_results.items():
    print(f"\n--- {name} ---")

    # Plato baslangic ve bitis noktalari
    plato_phases = []
    in_plato = False
    plato_start = 0

    for r in epochs:
        if r["psteps"] >= 3 and not in_plato:
            in_plato = True
            plato_start = r["epoch"]
        elif r["psteps"] == 0 and in_plato:
            in_plato = False
            plato_phases.append((plato_start, r["epoch"], r["epoch"] - plato_start))

    if in_plato:
        plato_phases.append((plato_start, 99, 99 - plato_start))

    print(f"  Plato sayisi: {len(plato_phases)}")
    for start, end, dur in plato_phases:
        print(f"    Ep {start:>2} - {end:>2} ({dur:>2} epoch)")

    # Gate ilk tepki (gate > 0.1)
    first_gate = next((r["epoch"] for r in epochs if r["gate"] > 0.1), ">100")
    print(f"  Gate ilk tepki (>0.1): epoch {first_gate}")

    # Scale ilk anlamli tepki (|scale-1| > 0.05)
    first_scale = next((r["epoch"] for r in epochs if abs(r["scale"] - 1.0) > 0.05), ">100")
    print(f"  Scale ilk anlamli tepki (>5%): epoch {first_scale}")

    # Cosine %50'ye ulasma
    cos50 = next((r["epoch"] for r in epochs if r["progress"] >= 0.5), ">100")
    print(f"  Cosine %50: epoch {cos50}")

    # Cosine %90'a ulasma
    cos90 = next((r["epoch"] for r in epochs if r["progress"] >= 0.9), ">100")
    print(f"  Cosine %90: epoch {cos90}")

# Tablo 3: LR trajectory karsilastirma
print("\n" + "=" * 120)
print("LR TRAJECTORY (her 10 epoch)")
print("=" * 120)

print(f"\n{'Ep':<5}", end="")
for name in all_results:
    short = name.split("(")[1].rstrip(")")
    print(f" | {short:>12} {'base_lr':>12} {'cosine_lr':>12} {'scale':>7} {'gate':>5}", end="")
print()
print("-" * 120)

for ep in range(0, 100, 10):
    print(f"{ep:>3}  ", end="")
    for name in all_results:
        r = all_results[name][ep]
        print(f" | {r['lr']:>12.8f} {r['base_lr']:>12.8f} {r['cosine_lr']:>12.8f} "
              f"{r['scale']:>7.4f} {r['gate']:>5.2f}", end="")
    print()

# Tablo 4: Ozet metrikleri
print("\n" + "=" * 120)
print("OZET METRIKLERI")
print("=" * 120)

for name, epochs in all_results.items():
    scales = [r["scale"] for r in epochs]
    gates = [r["gate"] for r in epochs]
    lrs = [r["lr"] for r in epochs]

    print(f"\n--- {name} ---")
    print(f"  Scale: mean={np.mean(scales):.4f}, std={np.std(scales):.4f}, "
          f"min={np.min(scales):.3f}, max={np.max(scales):.3f}")
    print(f"  Gate:  mean={np.mean(gates):.4f}, final={gates[-1]:.4f}")
    print(f"  LR:    start={lrs[3]:.8f}, final={lrs[-1]:.8f}, "
          f"min={min(lrs[3:]):.8f}, max={max(lrs[3:]):.8f}")
    print(f"  Cosine: final progress={epochs[-1]['progress']*100:.1f}%")

    # Scale > 1.05 ve < 0.95 oranlari
    accel = sum(1 for s in scales if s > 1.05) / len(scales) * 100
    brake = sum(1 for s in scales if s < 0.95) / len(scales) * 100
    neutral = sum(1 for s in scales if 0.98 <= s <= 1.02) / len(scales) * 100
    print(f"  Scale dagilimi: accel(>1.05)={accel:.1f}%, brake(<0.95)={brake:.1f}%, "
          f"notr(0.98-1.02)={neutral:.1f}%")

    # Toplam plato suresi
    total_plato = sum(1 for r in epochs if r["psteps"] >= 3)
    print(f"  Plato suresi (psteps>=3): {total_plato} epoch / 100")

# Epoch onerisi
print("\n" + "=" * 120)
print("EPOCH ONERISI")
print("=" * 120)

for name, epochs in all_results.items():
    print(f"\n--- {name} ---")

    # Cosine %90'a ulasma
    cos90 = next((r["epoch"] for r in epochs if r["progress"] >= 0.9), 100)
    cos95 = next((r["epoch"] for r in epochs if r["progress"] >= 0.95), 100)
    cos99 = next((r["epoch"] for r in epochs if r["progress"] >= 0.99), 100)

    print(f"  Cosine %90: epoch {cos90}")
    print(f"  Cosine %95: epoch {cos95}")
    print(f"  Cosine %99: epoch {cos99}")

    # LR minimum noktasi
    min_lr_ep = min(range(3, len(epochs)), key=lambda i: epochs[i]["lr"])
    print(f"  Min LR: epoch {min_lr_ep} (LR={epochs[min_lr_ep]['lr']:.8f})")

    # Onerilen epoch
    # Cosine %95 + 5 epoch fine-tuning marji
    suggested = min(cos95 + 5, 100)
    print(f"  Onerilen toplam epoch: {suggested}")
