"""
MIMO Fuzzy-Supervised Cosine Decay Simulasyonu (v2)

Gercek egitim verilerini (neudet_org_fuzzy_100ep) kullanarak
mevcut sistem vs MIMO mimarisinin LR trajektorisini karsilastirir.

v2 degisiklikleri:
- cosine_progress MF eklendi (speed kural tabanina 6. giris)
- 3 yeni speed kurali: CP["gec"] ile erken tamamlanma onlenir
- Guard kodu YOK — tum kararlar fuzzy kurallarla

Cikti: Epoch bazinda gate, speed, progress, base_lr ve LR karsilastirmasi.
"""

import csv
import math
import os
import sys

# Mevcut fuzzy MF'leri ve yardimlari import et
sys.path.insert(0, os.path.dirname(__file__))
from scheduler.fuzzy_lr import (
    FuzzyLRController, FuzzyLRConfig, PlateauTracker,
    trimf, trapmf, clamp
)


# ============================================================================
#  MIMO CONTROLLER (Simulasyon versiyonu v2)
# ============================================================================
class MIMOFuzzyController:
    """
    3-cikisli MIMO Fuzzy Controller:
      1. scale  -- mikro LR perturbasyonu [0.90, 1.10]
      2. gate   -- cosine aktivasyon seviyesi [0, 1]
      3. speed  -- cosine ilerleme hizi carpani [0.3, 3.0]

    Paylasilan uyelik fonksiyonlari, bagimsiz kural tabanlari.

    v2: cosine_progress MF eklendi — speed kural tabaninda kullanilir.
    Progress 0.7'den itibaren yumusak gecisle speed yavaslar.
    Sert kesim yok, %100 fuzzy.
    """

    def __init__(self, cfg: FuzzyLRConfig, total_epochs: int = 100):
        self.cfg = cfg
        self._epoch_scale = total_epochs / 50.0
        self._last_scale = 1.0
        self._last_gate = 0.0

    # --- PAYLASILAN MF'LER (mevcut sistemden aynen) ---

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
        s = self._epoch_scale
        return {
            "kisa":     trapmf(p, 0, 0, 2*s, 4*s),
            "orta":     trimf(p, 2*s, 5*s, 8*s),
            "uzun":     trimf(p, 6*s, 9*s, 12*s),
            "cok_uzun": trapmf(p, 8*s, 12*s, 999, 999),
        }

    def _mf_val_train_gap(self, gap):
        return {
            "healthy":    trapmf(gap, -999, -999, 0.15, 0.30),
            "warning":    trimf(gap, 0.25, 0.45, 0.65),
            "overfitting": trapmf(gap, 0.55, self.cfg.overfitting_threshold, 999, 999),
        }

    def _mf_cosine_progress(self, p):
        """
        Cosine ilerleme durumu MF'si.
        Sadece speed kural tabaninda kullanilir.
        Progress 0.7'den itibaren "gec" uyeligi baslar → speed yavaslar.
        Guard kodu yerine fuzzy kural ile cozum.
        """
        return {
            "erken":  trapmf(p, 0, 0, 0.3, 0.5),
            "orta":   trimf(p, 0.3, 0.5, 0.8),
            "gec":    trapmf(p, 0.7, 0.9, 1.0, 1.0),
        }

    # --- KURAL TABANI 1: SCALE (mikro perturbasyon, sadelesmis) ---

    def infer_scale(self, dl, gn, psteps, vtg, epoch_ratio):
        DL = self._mf_delta_loss(dl)
        GN = self._mf_grad_norm(gn)
        VTG = self._mf_val_train_gap(vtg)
        PL = self._mf_plateau(psteps)

        rules = []

        # A. HIZLANMA (mikro boost)
        rules.append((DL["hizla_azalan"],                            1.08))
        rules.append((DL["az_azalan"],                               1.04))
        rules.append((min(DL["az_azalan"], GN["kucuk"]),             1.05))

        # B. FRENLEME (mikro brake)
        rules.append((DL["artan"],                                   0.97))
        rules.append((min(DL["artan"], GN["buyuk"]),                 0.94))
        rules.append((VTG["warning"],                                0.96))
        rules.append((min(VTG["overfitting"], PL["uzun"]),           0.92))

        # C. STABILIZATOR
        rules.append((VTG["healthy"] * PL["kisa"] * 0.5,            1.0))

        # Asymmetric weighted average
        num, den = 0.0, 0.0
        for w, c in rules:
            if c > 1.0:
                w *= self.cfg.accel_boost
            elif c < 1.0:
                w *= self.cfg.brake_penalty
            num += w * c
            den += w

        scale = (num / (den + 1e-12)) if den > 0 else 1.0

        # Hysteresis
        max_step = self.cfg.hysteresis_frac_max - (0.15 * epoch_ratio)
        max_step = clamp(max_step, self.cfg.hysteresis_frac_min, self.cfg.hysteresis_frac_max)
        delta = scale - self._last_scale
        if abs(delta) > max_step:
            scale = self._last_scale + math.copysign(max_step, delta)

        # Daraltilmis mikro aralik
        scale = clamp(scale, 0.90, 1.10)
        self._last_scale = scale
        return scale

    # --- KURAL TABANI 2: GATE (cosine aktivasyonu) ---

    def infer_gate(self, dl, gn, psteps, vtg, epoch_ratio):
        DL = self._mf_delta_loss(dl)
        PL = self._mf_plateau(psteps)
        VTG = self._mf_val_train_gap(vtg)

        rules = []

        # Model aktif ogreniyor + plato yok -> gate kapali tut
        rules.append((min(DL["hizla_azalan"], PL["kisa"]),   0.0))
        rules.append((min(DL["az_azalan"], PL["kisa"]),      0.0))
        rules.append((min(PL["kisa"], VTG["healthy"]),        0.0))

        # Erken plato, hala saglikli -> yavas ac
        rules.append((min(PL["orta"], VTG["healthy"]),        0.3))

        # Uzun plato -> daha fazla ac
        rules.append((PL["uzun"],                             0.7))

        # Cok uzun plato -> tam ac
        rules.append((PL["cok_uzun"],                         1.0))

        # Overfitting sinyalleri -> cosine'i devreye sok
        rules.append((VTG["warning"],                         0.5))
        rules.append((VTG["overfitting"],                     0.8))

        # Basit weighted average (asymmetric yok -- gate'te accel/brake mantigi yok)
        num, den = 0.0, 0.0
        for w, c in rules:
            num += w * c
            den += w
        gate = (num / (den + 1e-12)) if den > 0 else 0.0
        gate = clamp(gate, 0.0, 1.0)

        # Gate hysteresis: max 0.15/epoch degisim
        gate_hyst = 0.15
        delta = gate - self._last_gate
        if abs(delta) > gate_hyst:
            gate = self._last_gate + math.copysign(gate_hyst, delta)
        gate = clamp(gate, 0.0, 1.0)

        self._last_gate = gate
        return gate

    # --- KURAL TABANI 3: SPEED (cosine ilerleme hizi) ---

    def infer_speed(self, dl, gn, psteps, vtg, epoch_ratio, cosine_progress):
        """
        Speed kural tabani — cosine_progress MF dahil.

        v2: 3 yeni kural eklendi (kural 9-11):
        - CP["gec"] AND PL["uzun"]          -> 0.3  (cosine sona yakin + plato -> koru)
        - CP["gec"] AND VTG["healthy"]       -> 0.5  (cosine sona yakin + saglikli -> yavasla)
        - CP["gec"] AND DL["hizla_azalan"]   -> 0.3  (cosine sona yakin + ogrenme -> koru)

        Bu kurallar progress 0.7'den itibaren yumusak gecisle speed'i yavaslatir.
        Guard kodu (if progress > 0.9: clamp) YERINE fuzzy cozum.
        """
        DL = self._mf_delta_loss(dl)
        PL = self._mf_plateau(psteps)
        VTG = self._mf_val_train_gap(vtg)
        CP = self._mf_cosine_progress(cosine_progress)

        rules = []

        # --- MEVCUT KURALLAR (1-8) ---
        # Model iyi ogreniyor -> cosine yavaslasin (LR yuksek kalsin)
        rules.append((DL["hizla_azalan"],                         0.5))   # K1
        rules.append((min(DL["az_azalan"], VTG["healthy"]),       0.7))   # K2

        # Notr -> normal hiz
        rules.append((DL["durdu"],                                1.0))   # K3

        # Plato -> hizlandir
        rules.append((PL["orta"],                                 1.5))   # K4
        rules.append((PL["uzun"],                                 2.0))   # K5

        # Overfitting -> cok hizli indir
        rules.append((VTG["warning"],                             2.0))   # K6
        rules.append((VTG["overfitting"],                         3.0))   # K7

        # Cok uzun plato + saglikli -> orta hizda (kesfet icin scale devrede)
        rules.append((min(PL["cok_uzun"], VTG["healthy"]),        1.5))   # K8

        # --- YENI KURALLAR: COSINE PROGRESS KORUMA (9-11) ---
        # Progress 0.7'den itibaren "gec" uyeligi baslar.
        # Bu kurallar speed'i yavaslatarak cosine butcesini korur.
        # Sert if/else yerine yumusak fuzzy gecis.

        # K9: Cosine sona yakin + plato -> kalan butceyi koru
        rules.append((min(CP["gec"], PL["uzun"]),                 0.3))   # K9

        # K10: Cosine sona yakin + saglikli -> hafif yavasla
        rules.append((min(CP["gec"], VTG["healthy"]),             0.5))   # K10

        # K11: Cosine sona yakin + aktif ogrenme -> kalan butceyi koru
        rules.append((min(CP["gec"], DL["hizla_azalan"]),         0.3))   # K11

        num, den = 0.0, 0.0
        for w, c in rules:
            num += w * c
            den += w
        speed = (num / (den + 1e-12)) if den > 0 else 1.0
        return clamp(speed, 0.3, 3.0)

    # --- MIMO INTERFACE ---

    def infer_mimo(self, dl, gn, psteps, vtg, epoch_ratio, cosine_progress):
        scale = self.infer_scale(dl, gn, psteps, vtg, epoch_ratio)
        gate  = self.infer_gate(dl, gn, psteps, vtg, epoch_ratio)
        speed = self.infer_speed(dl, gn, psteps, vtg, epoch_ratio, cosine_progress)
        return scale, gate, speed


# ============================================================================
#  SIMULASYON
# ============================================================================
def load_epoch_data(epochs_csv):
    """epochs.csv'den epoch-level verileri yukle."""
    with open(epochs_csv, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_step_aggregates(steps_csv):
    """steps.csv'den epoch bazli ortalama delta_loss, grad_norm hesapla."""
    with open(steps_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    epoch_agg = {}
    for r in rows:
        ep = int(r["epoch"])
        if ep not in epoch_agg:
            epoch_agg[ep] = {"dls": [], "gns": [], "scales": [], "lrs": [], "base_lrs": []}
        epoch_agg[ep]["dls"].append(float(r["delta_loss"]))
        epoch_agg[ep]["gns"].append(float(r["grad_norm"]))
        epoch_agg[ep]["scales"].append(float(r["scale"]))
        epoch_agg[ep]["lrs"].append(float(r["lr"]))
        epoch_agg[ep]["base_lrs"].append(float(r["phase_base_lr"]))

    result = {}
    for ep, data in epoch_agg.items():
        result[ep] = {
            "avg_dl": sum(data["dls"]) / len(data["dls"]),
            "avg_gn": sum(data["gns"]) / len(data["gns"]),
            "avg_scale": sum(data["scales"]) / len(data["scales"]),
            "avg_lr": sum(data["lrs"]) / len(data["lrs"]),
            "avg_base_lr": sum(data["base_lrs"]) / len(data["base_lrs"]),
        }
    return result


def simulate_mimo(epoch_data, step_agg, lr0=0.0001, lrf=0.10, total_epochs=100, warmup_epochs=3):
    """
    MIMO sistemi ile LR trajektorisini simule et.
    Gercek egitim sinyallerini kullanarak gate, speed, progress, base_lr hesapla.
    """
    cfg = FuzzyLRConfig(lr0=lr0, lrf=lrf)
    mimo = MIMOFuzzyController(cfg, total_epochs=total_epochs)

    # Plateau tracker (callback ile ayni)
    plateau = PlateauTracker(rel_tol=0.002, grace_epochs=0)

    progress = 0.0
    effective_total = total_epochs - warmup_epochs
    results = []

    for ep_row in epoch_data:
        ep = int(ep_row["epoch"])
        map50 = float(ep_row["map50"])
        map5095 = float(ep_row.get("map5095", 0))
        vtg = float(ep_row.get("val_train_gap", 0))

        # Plateau update (hybrid metric)
        hybrid = map50 * 0.7 + map5095 * 0.3
        if hybrid > 0:
            plateau.update(hybrid, epoch=ep)
        psteps = plateau.steps

        # Step aggregates
        agg = step_agg.get(ep, {"avg_dl": 0, "avg_gn": 1.0, "avg_scale": 1.0,
                                 "avg_lr": lr0, "avg_base_lr": lr0})
        avg_dl = agg["avg_dl"]
        avg_gn = agg["avg_gn"]
        epoch_ratio = ep / total_epochs

        # --- WARMUP ---
        if ep < warmup_epochs:
            warmup_factor = (ep + 1) / warmup_epochs
            warmup_lr = 0.5 * lr0 + 0.5 * lr0 * warmup_factor
            # Warmup'ta: gate=0, speed=1, scale uygulanir
            scale = mimo.infer_scale(avg_dl, avg_gn, psteps, vtg, epoch_ratio)
            gate = 0.0
            speed = 1.0
            mimo_base_lr = warmup_lr
            mimo_lr = warmup_lr * scale
        else:
            # --- MIMO INFERENCE ---
            # cosine_progress: speed kural tabanina girdi olarak verilir
            scale, gate, speed = mimo.infer_mimo(avg_dl, avg_gn, psteps, vtg, epoch_ratio, progress)

            # Cosine at current progress
            cosine_factor = (1.0 + math.cos(math.pi * progress)) / 2.0
            cosine_lr = lr0 * (cosine_factor * (1.0 - lrf) + lrf)

            # Gate blending: Phase 1 <-> Phase 2
            mimo_base_lr = lr0 * (1.0 - gate) + cosine_lr * gate
            mimo_lr = clamp(mimo_base_lr * scale, cfg.lr_min, cfg.lr_max)

            # Advance progress (epoch sonunda)
            base_increment = 1.0 / max(effective_total, 1)
            progress += base_increment * speed
            progress = min(progress, 1.0)

        results.append({
            "epoch": ep,
            "map50": map50,
            "psteps": psteps,
            "vtg": vtg,
            "avg_dl": avg_dl,
            # MIMO outputs
            "gate": gate,
            "speed": speed,
            "progress": progress,
            "scale": scale,
            "mimo_base_lr": mimo_base_lr,
            "mimo_lr": mimo_lr,
            # Old system (actual)
            "old_base_lr": agg["avg_base_lr"],
            "old_lr": agg["avg_lr"],
            "old_scale": agg["avg_scale"],
        })

    return results


def print_results(results, title="MIMO FUZZY-SUPERVISED COSINE DECAY"):
    """Sonuclari tablo olarak yazdir."""
    print()
    print("=" * 165)
    print(f"{title} -- SIMULASYON SONUCLARI")
    print("=" * 165)
    print(f"{'Ep':>3} | {'mAP50':>6} | {'Plato':>5} | {'VTG':>6} | "
          f"{'Gate':>5} | {'Speed':>5} | {'Prog%':>5} | "
          f"{'MIMO_Base':>11} | {'MIMO_LR':>11} | {'Scale':>5} | "
          f"{'OLD_Base':>11} | {'OLD_LR':>11} | {'OldScl':>6} | "
          f"{'Fark':>7}")
    print("-" * 165)

    for r in results:
        lr_diff_pct = (r["mimo_lr"] / r["old_lr"] - 1) * 100 if r["old_lr"] > 0 else 0

        print(f"{r['epoch']:>3} | "
              f"{r['map50']:.4f} | "
              f"{r['psteps']:>5} | "
              f"{r['vtg']:>+.3f} | "
              f"{r['gate']:>5.2f} | "
              f"{r['speed']:>5.2f} | "
              f"{r['progress']*100:>5.1f} | "
              f"{r['mimo_base_lr']:.9f} | "
              f"{r['mimo_lr']:.9f} | "
              f"{r['scale']:>5.3f} | "
              f"{r['old_base_lr']:.9f} | "
              f"{r['old_lr']:.9f} | "
              f"{r['old_scale']:>6.3f} | "
              f"{lr_diff_pct:>+6.1f}%")


def print_analysis(results, label=""):
    """Ozet istatistikler."""
    print()
    print("=" * 80)
    print(f"OZET ANALIZ {label}")
    print("=" * 80)

    # Faz analizi
    phase1 = [r for r in results if r["gate"] < 0.05]
    transition = [r for r in results if 0.05 <= r["gate"] < 0.95]
    phase2 = [r for r in results if r["gate"] >= 0.95]

    print(f"\nFaz 1 (gate < 0.05, pure fuzzy):    {len(phase1)} epoch")
    print(f"Gecis (0.05 <= gate < 0.95):         {len(transition)} epoch")
    print(f"Faz 2 (gate >= 0.95, full cosine):   {len(phase2)} epoch")

    # LR istatistikleri
    if results:
        mimo_lrs = [r["mimo_lr"] for r in results]
        old_lrs = [r["old_lr"] for r in results]
        print(f"\nMIMO LR araligi: [{min(mimo_lrs):.8f}, {max(mimo_lrs):.8f}]")
        print(f"Eski LR araligi: [{min(old_lrs):.8f}, {max(old_lrs):.8f}]")

        # Speed istatistikleri
        speeds = [r["speed"] for r in results if r["epoch"] >= 3]
        if speeds:
            print(f"\nSpeed araligi: [{min(speeds):.2f}, {max(speeds):.2f}]")
            print(f"Speed ortalama: {sum(speeds)/len(speeds):.2f}")

        # Ep17 (best mAP50) civarinda karsilastirma
        ep17 = next((r for r in results if r["epoch"] == 17), None)
        if ep17:
            print(f"\nEp17 (best mAP50 = {ep17['map50']:.4f}):")
            print(f"  MIMO: gate={ep17['gate']:.2f}, speed={ep17['speed']:.2f}, "
                  f"progress={ep17['progress']*100:.1f}%, LR={ep17['mimo_lr']:.8f}")
            print(f"  Eski: LR={ep17['old_lr']:.8f}")

        # Ep34 (eski sistemde emergency baslangici) karsilastirma
        ep34 = next((r for r in results if r["epoch"] == 34), None)
        if ep34:
            print(f"\nEp34 (eski sistemde emergency baslangici):")
            print(f"  MIMO: gate={ep34['gate']:.2f}, speed={ep34['speed']:.2f}, "
                  f"progress={ep34['progress']*100:.1f}%, LR={ep34['mimo_lr']:.8f}")
            print(f"  Eski: LR={ep34['old_lr']:.8f} (scale={ep34['old_scale']:.3f})")

        # Progress'in 1.0'a ulastigi epoch
        full_prog = next((r for r in results if r["progress"] >= 0.99), None)
        if full_prog:
            print(f"\nCosine tamamlandi: Ep{full_prog['epoch']} "
                  f"(progress={full_prog['progress']*100:.1f}%)")
        else:
            last = results[-1]
            print(f"\nCosine son durum: Ep{last['epoch']}, "
                  f"progress={last['progress']*100:.1f}%")

        # Gate ilk acilma noktasi
        first_gate = next((r for r in results if r["gate"] > 0.05), None)
        if first_gate:
            print(f"\nGate ilk acildi: Ep{first_gate['epoch']} "
                  f"(gate={first_gate['gate']:.2f}, psteps={first_gate['psteps']})")

        # Ep34-99 arasi scale UP/DN analizi
        late_mimo = [r for r in results if r["epoch"] >= 34]
        if late_mimo:
            scales_up = sum(1 for r in late_mimo if r["scale"] > 1.005)
            scales_dn = sum(1 for r in late_mimo if r["scale"] < 0.995)
            scales_ne = sum(1 for r in late_mimo if 0.995 <= r["scale"] <= 1.005)
            print(f"\nEp34-99 MIMO Scale dagilimi: UP={scales_up}, DN={scales_dn}, NEU={scales_ne}")
            print(f"  (Eski sistemde: UP=5846, DN=4, NEU=0 -- fuzzy kilitlenmisti)")

        # cosine_progress MF etkisi: speed'in gec epochlardaki davranisi
        late_speed = [r for r in results if r["progress"] >= 0.7]
        if late_speed:
            avg_late_speed = sum(r["speed"] for r in late_speed) / len(late_speed)
            early_speed = [r for r in results if r["progress"] < 0.3 and r["epoch"] >= 3]
            avg_early_speed = sum(r["speed"] for r in early_speed) / len(early_speed) if early_speed else 0
            print(f"\nCosine Progress MF Etkisi:")
            print(f"  Erken (progress<0.3): ort speed={avg_early_speed:.2f} ({len(early_speed)} epoch)")
            print(f"  Gec (progress>=0.7):  ort speed={avg_late_speed:.2f} ({len(late_speed)} epoch)")
            if avg_early_speed > 0:
                print(f"  Yavaslatma orani: {(1 - avg_late_speed/avg_early_speed)*100:.1f}%")


def main():
    run_dir = os.path.join("artifacts", "runs", "neudet_org_fuzzy_100ep")
    epochs_csv = os.path.join(run_dir, "epochs.csv")
    steps_csv = os.path.join(run_dir, "steps.csv")

    if not os.path.exists(epochs_csv):
        print(f"[ERROR] {epochs_csv} bulunamadi!")
        return

    print("Veriler yukleniyor...")
    epoch_data = load_epoch_data(epochs_csv)
    step_agg = load_step_aggregates(steps_csv)
    print(f"  {len(epoch_data)} epoch, {len(step_agg)} epoch aggregate yuklendi.")

    # =========================================================================
    #  SENARYO 1: Standart lr0=0.0001 (v1 ile karsilastirma)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SENARYO 1: lr0=0.0001 (standart) — cosine_progress MF AKTIF")
    print("=" * 80)

    results = simulate_mimo(
        epoch_data, step_agg,
        lr0=0.0001, lrf=0.10,
        total_epochs=100, warmup_epochs=3
    )

    print_results(results, "MIMO v2 (cosine_progress MF)")
    print_analysis(results, "(lr0=0.0001)")

    # =========================================================================
    #  SENARYO 2: Yuksek lr0=0.001 (auto-discovery testi)
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("SENARYO 2: lr0=0.001 (yuksek LR) — cosine_progress MF AKTIF")
    print("=" * 80)

    results_high = simulate_mimo(
        epoch_data, step_agg,
        lr0=0.001, lrf=0.10,
        total_epochs=100, warmup_epochs=3
    )

    # Tam tablo
    print_results(results_high, "MIMO v2 Yuksek LR")
    print_analysis(results_high, "(lr0=0.001)")

    # =========================================================================
    #  KARSILASTIRMA: v1 (cosine_progress yok) vs v2 (cosine_progress var)
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("KARSILASTIRMA: v2 vs v1 (cosine_progress MF etkisi)")
    print("=" * 80)

    # v1 simulasyonu: cosine_progress olmadan (speed'e her zaman progress=0 ver)
    cfg_v1 = FuzzyLRConfig(lr0=0.0001, lrf=0.10)
    mimo_v1 = MIMOFuzzyController(cfg_v1, total_epochs=100)
    plateau_v1 = PlateauTracker(rel_tol=0.002, grace_epochs=0)
    progress_v1 = 0.0
    effective_total = 97  # 100 - 3 warmup

    v1_progress_history = []
    v2_progress_history = []

    # v1 simulate (cosine_progress=0 always -> MF effect disabled)
    for ep_row in epoch_data:
        ep = int(ep_row["epoch"])
        map50 = float(ep_row["map50"])
        map5095 = float(ep_row.get("map5095", 0))
        vtg = float(ep_row.get("val_train_gap", 0))

        hybrid = map50 * 0.7 + map5095 * 0.3
        if hybrid > 0:
            plateau_v1.update(hybrid, epoch=ep)
        psteps = plateau_v1.steps
        agg = step_agg.get(ep, {"avg_dl": 0, "avg_gn": 1.0})
        epoch_ratio = ep / 100

        if ep >= 3:
            # v1: cosine_progress = 0 -> CP["gec"] = 0 always -> yeni kurallar etkisiz
            _, _, speed_v1 = mimo_v1.infer_mimo(agg["avg_dl"], agg["avg_gn"],
                                                 psteps, vtg, epoch_ratio, 0.0)
            progress_v1 += (1.0 / effective_total) * speed_v1
            progress_v1 = min(progress_v1, 1.0)

        v1_progress_history.append({"epoch": ep, "progress": progress_v1,
                                     "speed": speed_v1 if ep >= 3 else 1.0})

    print(f"\n{'Ep':>3} | {'v1_Prog%':>8} | {'v1_Speed':>8} | {'v2_Prog%':>8} | {'v2_Speed':>8} | {'Fark':>6}")
    print("-" * 55)
    for i, r in enumerate(results):
        v1 = v1_progress_history[i]
        if r["epoch"] % 5 == 0 or r["epoch"] < 5 or r["progress"] >= 0.95:
            print(f"{r['epoch']:>3} | "
                  f"{v1['progress']*100:>7.1f}% | "
                  f"{v1['speed']:>8.2f} | "
                  f"{r['progress']*100:>7.1f}% | "
                  f"{r['speed']:>8.2f} | "
                  f"{(r['progress'] - v1['progress'])*100:>+5.1f}%")

    # v1 vs v2: progress 1.0'a ulasma epoch'u
    v1_full = next((v for v in v1_progress_history if v["progress"] >= 0.99), None)
    v2_full = next((r for r in results if r["progress"] >= 0.99), None)
    print(f"\nv1 cosine tamamlanma: Ep{v1_full['epoch'] if v1_full else 'N/A'}")
    print(f"v2 cosine tamamlanma: Ep{v2_full['epoch'] if v2_full else 'N/A'}")
    if v1_full and v2_full:
        diff = v2_full["epoch"] - v1_full["epoch"]
        print(f"v2, cosine'i {abs(diff)} epoch {'daha gec' if diff > 0 else 'daha erken'} tamamliyor")
        print(f"  -> cosine_progress MF, erken tamamlanmayi {abs(diff)} epoch geciktirdi")


if __name__ == "__main__":
    main()
