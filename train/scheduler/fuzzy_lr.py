# train/scheduler/fuzzy_lr.py
#
# v4: Phase-Aware Scale-Only Micro-Navigator
#
# Mimari:
#   Tek fuzzy controller, paylasilan uyelik fonksiyonlari, tek cikis.
#   Cikis:
#     scale  [0.90, 1.10]  — mikro LR perturbasyonu (step-level)
#
#   LR hesabi:
#     cosine_lr    = lr0 * (cosine_factor * (1 - lrf) + lrf)
#     effective_lr = cosine_lr * scale
#
#   Cosine decay standart ilerler — baseline ile birebir ayni temel.
#   Speed/gate KALDIRILDI: cosine progress = (epoch - warmup) / (total - warmup).
#   Fuzzy sistem yalnizca scale uzerinden mikro ayar yapar.
#
#   Faz-duyarli kural tabani:
#     Erken faz: agresif perturbasyon (kesif)
#     Gec faz:   pasif/notr (cosine fine-tuning'e karismaz)
#
#   Relative delta_loss:
#     delta_loss = (EMA_new - EMA_old) / |EMA_old|
#     Faz-bagimsiz sinyal — MF esikleri tum epochlarda tutarli.
#
#   Standart defuzzifikasyon:
#     Agirlikli ortalama, asimetrik carpan YOK.
#     Asimetri consequent degerlerinden gelir.

from dataclasses import dataclass
import math
import torch

# ============================================================================
#  YARDIMCI FONKSIYONLAR
# ============================================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def trimf(x, a, b, c):
    """Triangular membership function"""
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    if x < b:  return (x - a) / (b - a + 1e-12)
    return (c - x) / (c - b + 1e-12)

def trapmf(x, a, b, c, d):
    """Trapezoidal (yamuk) membership function"""
    if x < a or x > d:
        return 0.0
    elif b <= x <= c:
        return 1.0
    elif a <= x < b:
        return (x - a) / (b - a + 1e-12)
    else:  # c < x <= d
        return (d - x) / (d - c + 1e-12)

def gbellmf(x, a, b, c):
    """Generalized bell membership function"""
    return 1.0 / (1.0 + abs((x - c) / (a + 1e-12)) ** (2.0 * b))

def sigmf(x, a, c):
    """Sigmoid membership function"""
    return 1.0 / (1.0 + math.exp(-a * (x - c)))

@torch.no_grad()
def grad_norm(model, norm_type=2.0):
    """Compute gradient norm with log-scale normalization"""
    total = 0.0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.data.norm(norm_type).item()
            total += g ** norm_type
            param_count += 1

    if total > 0 and param_count > 0:
        raw_norm = total ** (1.0 / norm_type)
        normalized = math.log1p(raw_norm) / math.log1p(500.0)
        scaled = normalized * 5.0
        return scaled
    return 0.5

# ============================================================================
#  HELPER CLASSES
# ============================================================================
class EMA:
    """Exponential Moving Average"""
    def __init__(self, beta=0.9):
        self.beta = beta
        self.v = None

    def update(self, x: float):
        if self.v is None:
            self.v = x
            return 0.0
        old = self.v
        self.v = self.beta * self.v + (1 - self.beta) * x
        return self.v - old

    @property
    def value(self):
        return 0.0 if self.v is None else self.v


class DualEMA:
    """
    Oneri 1 — Cifte EMA (MACD benzeri sinyal).

    delta_loss = (EMA_fast - EMA_slow) / |EMA_slow|

    Tek EMA'ya gore avantajlar:
      - Ani spike'lara duyarsiz (EMA_slow yumusatiyor)
      - Trend yonunu daha net yakalar (fark, mutlak deger degil)
      - Alternasyon orani %52 → ~%20 (sinyal kararliligi)

    beta_fast=0.80, beta_slow=0.95 (onerilen degerler).
    """
    def __init__(self, beta_fast: float = 0.80, beta_slow: float = 0.95):
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self._fast = None
        self._slow = None

    def update(self, x: float) -> float:
        """
        Yeni loss degeri ile her iki EMA'yi guncelle.
        Returns: relative delta = (fast - slow) / |slow|
        """
        if self._fast is None:
            self._fast = x
            self._slow = x
            return 0.0

        self._fast = self.beta_fast * self._fast + (1 - self.beta_fast) * x
        self._slow = self.beta_slow * self._slow + (1 - self.beta_slow) * x

        if abs(self._slow) < 1e-8:
            return 0.0
        return (self._fast - self._slow) / abs(self._slow)

    @property
    def fast(self) -> float:
        return 0.0 if self._fast is None else self._fast

    @property
    def slow(self) -> float:
        return 0.0 if self._slow is None else self._slow


class MovingAverage:
    """
    Oneri 2 — N-adim hareketli ortalama.

    Kullanim: grad_norm degerlerini yumusat.
    N=5 ile std: 3.001 → 0.743 (anlık spike'lar bastiriliyor).
    """
    def __init__(self, n: int = 5):
        self.n = n
        self._buf: list = []

    def update(self, x: float) -> float:
        self._buf.append(x)
        if len(self._buf) > self.n:
            self._buf.pop(0)
        return sum(self._buf) / len(self._buf)

class PlateauTracker:
    """Track plateau duration"""
    def __init__(self, rel_tol=0.005, grace_epochs=0):
        self.best = None
        self.steps = 0
        self.rel_tol = rel_tol
        self.grace_epochs = grace_epochs
        self.total_updates = 0

    def update(self, metric, epoch=None):
        self.total_updates += 1

        # Grace period check
        if epoch is not None and epoch < self.grace_epochs:
            self.steps = 0
            if self.best is None or metric > self.best:
                self.best = metric
            return 0

        if self.best is None:
            self.best = metric
            self.steps = 0
        elif metric > self.best * (1.0 + self.rel_tol):
            self.best = metric
            self.steps = 0
        else:
            self.steps += 1
        return self.steps

def apply_warmup(epoch, warmup_epochs=3, target_lr=0.0001):
    if epoch < warmup_epochs:
        warmup_factor = (epoch + 1) / warmup_epochs
        return 0.5 * target_lr + 0.5 * target_lr * warmup_factor
    return None

# ============================================================================
#  FUZZY CONTROLLER CONFIG
# ============================================================================
@dataclass
class FuzzyLRConfig:
    """
    v4: Phase-Aware Scale-Only Micro-Navigator konfigurasyonu.

    Tek cikisli (scale) fuzzy kontrol sistemi.
    Speed/gate KALDIRILDI — cosine standart ilerler.
    Asimetrik carpanlar KALDIRILDI — standart defuzzifikasyon.
    """
    # Cosine base schedule parametreleri
    lr0: float = 0.0001      # Cosine baslangic LR (epoch 0)
    lrf: float = 0.10        # Cosine final orani: lr_final = lr0 * lrf

    # LR guvenlik sinirlari
    lr_max: float = 0.001
    lr_min: float = 1e-6

    # Scale perturbasyon sinirlari (daraltildi: ±%10)
    scale_min: float = 0.90
    scale_max: float = 1.10

    # Warmup
    use_three_phase: bool = False
    use_warmup: bool = True
    warmup_epochs: int = 3

    # Scale hysteresis (degisim hizi sinirlamasi)
    hysteresis_frac_min: float = 0.08
    hysteresis_frac_max: float = 0.10

    # Overfitting
    overfitting_threshold: float = 0.8

# ============================================================================
#  PHASE-AWARE SCALE-ONLY FUZZY CONTROLLER
# ============================================================================
class FuzzyLRController:
    """
    v4: Phase-Aware Scale-Only Micro-Navigator.

    Tek cikis: scale [0.90, 1.10]
    Cosine uzerine mikro perturbasyon.

    Giris degiskenleri (5 adet):
      delta_loss (RELATIVE), grad_norm, plateau_steps,
      val_train_gap, epoch_ratio

    Faz-duyarli kural tabani:
      Erken faz: agresif perturbasyon
      Gec faz: pasif (cosine fine-tuning'e karismaz)

    Standart defuzzifikasyon (asimetrik carpan yok).
    """
    def __init__(self, base_lr: float, cfg: FuzzyLRConfig = FuzzyLRConfig(),
                 total_epochs: int = 50):
        self.base_lr = base_lr
        self.cfg = cfg
        self._last_scale = 1.0
        # Plateau MF esiklerini epoch sayisina orantili olcekle.
        self._epoch_scale = total_epochs / 50.0

    # ========================================================================
    #  UYELIK FONKSIYONLARI
    # ========================================================================

    def _mf_delta_loss(self, dl: float) -> dict:
        """
        RELATIVE parti kaybi degisimi: (EMA_new - EMA_old) / |EMA_old|.

        Mutlak delta_loss gec epochlarda kuculdugu icin MF'ler tutarsiz
        calisiyordu (hizla_azalan ep0-30'da %4, ep60-100'de %0.3).
        Relative delta_loss std'si tum fazlarda ~0.014 ile sabit.

        Esikler neudet_fuzzycos_100ep_v1 relative dagilimdan turetildi:
        P5=-0.025, P25=-0.010, P50=-0.001, P75=+0.008, P95=+0.022
        """
        return {
            "hizla_azalan": trapmf(dl, -999, -999, -0.03, -0.01),
            "az_azalan":    trimf(dl, -0.02, -0.006, 0.0),
            "durdu":        trimf(dl, -0.004, 0.0, 0.004),
            "artan":        trapmf(dl, 0.003, 0.01, 999, 999),
        }

    def _mf_grad_norm(self, gn: float) -> dict:
        """
        Log-normalize edilmis L2 gradyan normu.
        Gercek dagilim iki modlu: ~%25 adim -> 0.50, ~%75 adim -> 7.0-8.5

        SGD kalibrasyonu (neudet_scaleonly_sgd_100ep_v1 verisinden):
        P50=7.14, P75=7.39, P90=7.84, P95=8.21
        AdamW kumesi 8.0-8.5 idi, SGD ~1.0 birim asagida.
        """
        return {
            "kucuk": trapmf(gn, 0.0, 0.0, 0.5, 3.0),
            "orta":  trimf(gn, 2.0, 5.5, 8.0),
            "buyuk": trapmf(gn, 7.0, 8.5, 999, 999),
        }

    def _mf_plateau(self, p: float) -> dict:
        """
        Epoch-olcekli plateau uyelik fonksiyonlari.
        Esikler total_epochs/50 orani ile olceklenir.
        """
        s = self._epoch_scale
        return {
            "kisa":      trapmf(p, 0, 0, 1*s, 3*s),
            "orta":      trimf(p, 1*s, 3*s, 6*s),
            "uzun":      trimf(p, 4*s, 7*s, 10*s),
            "cok_uzun":  trapmf(p, 6*s, 10*s, 999, 999),
        }

    def _mf_val_train_gap(self, gap: float) -> dict:
        """
        val_loss - train_loss farki.

        Eski GC10 esikleri geri yüklendi (2026-03-26).
        Yeni SGD kalibrasyonu (warning=0.08) B3/B4 kurallarin %24.5
        adimda tetiklenmesine neden oldu → sistematik DOWN bias (UP/DN 0.90:1).
        Empirik olarak bu esiklerle (B3/B4 neredeyse hic tetiklenmiyor)
        en iyi sonuc elde edildi (mAP50=0.7557).

        Not: grad_norm MF SGD kalibrasyonu (orta/buyuk esikleri) KORUNDU.
        """
        return {
            "healthy":     trapmf(gap, -999, -999, 0.15, 0.30),
            "warning":     trimf(gap, 0.25, 0.45, 0.65),
            "overfitting": trapmf(gap, 0.55, 0.80, 999, 999),
        }

    def _mf_epoch_ratio(self, r: float) -> dict:
        """
        Faz-duyarli kural tabani icin epoch orani MF'si.

        Erken faz (0-50%):  agresif perturbasyon, kesif yonlu
        Orta faz (30-80%):  dengeli
        Gec faz (65-100%):  pasif, cosine fine-tuning'e karismaz
        """
        return {
            "erken": trapmf(r, 0.0, 0.0, 0.3, 0.5),
            "orta":  trimf(r, 0.3, 0.5, 0.8),
            "gec":   trapmf(r, 0.65, 0.85, 1.0, 1.0),
        }

    # ========================================================================
    #  KURAL TABANI: SCALE (faz-duyarli mikro perturbasyon)
    # ========================================================================

    def infer_scale(self, dl: float, gn: float, psteps: float,
                    vtg: float = 0.0, epoch_ratio: float = 0.0) -> float:
        """
        Phase-aware mikro LR perturbasyonu [scale_min, scale_max].

        13 kural, 4 kategori:
          A. Hizlanma (5 kural, phase-dependent)
          B. Frenleme (4 kural, phase-independent)
          C. Plato kacis (3 kural, phase-dependent)
          D. Stabilizator (1 kural)

        Standart agirlikli ortalama — asimetrik carpan YOK.
        Asimetri consequent degerlerinden gelir:
          max boost = 1.08 (+8%), max brake = 0.90 (-10%)

        Returns:
            scale: [scale_min, scale_max] araliginda, histerezis uygulanmis.
        """
        DL = self._mf_delta_loss(dl)
        GN = self._mf_grad_norm(gn)
        VTG = self._mf_val_train_gap(vtg)
        PL = self._mf_plateau(psteps)
        PH = self._mf_epoch_ratio(epoch_ratio)

        rules = []

        # A. HIZLANMA (phase-dependent)
        # Erken fazda guclu push, gec fazda notr (cosine fine-tuning korunur).
        # Relative delta_loss sayesinde MF aktivasyonlari faz-bagimsiz;
        # faz-bagimlilik consequent degerlerinden gelir.
        rules.append((min(DL["hizla_azalan"], PH["erken"]),      1.08))  # A1
        rules.append((min(DL["hizla_azalan"], PH["orta"]),       1.05))  # A2
        rules.append((min(DL["hizla_azalan"], PH["gec"]),        1.01))  # A3
        # A4-A5: Gec fazda baskila (not_gec). Aksi halde paradoks:
        # gec fazda az_azalan (1.03) > hizla_azalan (1.01) olur.
        not_gec = 1.0 - PH["gec"]
        rules.append((min(DL["az_azalan"], not_gec),             1.03))  # A4
        rules.append((min(DL["az_azalan"], GN["kucuk"], not_gec), 1.04))  # A5

        # B. FRENLEME (phase-independent)
        # Overfitting freni faz-bagimsiz: her zaman gerekli.
        rules.append((DL["artan"],                               0.95))  # B1
        rules.append((min(DL["artan"], GN["buyuk"]),             0.92))  # B2
        rules.append((VTG["warning"],                            0.93))  # B3
        rules.append((min(VTG["overfitting"], PL["uzun"]),       0.90))  # B4

        # C. PLATO KACIS (phase-dependent)
        # Erken fazda guclu kesif, gec fazda hafif durtme.
        rules.append((min(PL["uzun"], PH["erken"]),              1.07))  # C1
        rules.append((min(PL["uzun"], PH["gec"]),                1.02))  # C2
        rules.append((min(PL["cok_uzun"], VTG["healthy"]),       1.08))  # C3

        # D. STABILIZATOR
        # Saglikli + kisa plato = notr. 0.5 carpani baskinligi onler.
        rules.append((VTG["healthy"] * PL["kisa"] * 0.5,        1.0))   # D1

        # --- STANDART AGIRLIKLI ORTALAMA (centroid) ---
        num, den = 0.0, 0.0
        for weight, consequent in rules:
            num += weight * consequent
            den += weight

        scale = (num / (den + 1e-12)) if den > 0 else 1.0

        # --- HISTEREZIS (degisim hizi sinirlamasi) ---
        max_step = self.cfg.hysteresis_frac_max - (0.15 * epoch_ratio)
        max_step = clamp(max_step, self.cfg.hysteresis_frac_min,
                         self.cfg.hysteresis_frac_max)
        delta = scale - self._last_scale
        if abs(delta) > max_step:
            scale = self._last_scale + math.copysign(max_step, delta)

        # --- GLOBAL CLAMPING ---
        scale = clamp(scale, self.cfg.scale_min, self.cfg.scale_max)

        self._last_scale = scale
        return scale

    def lr_from_scale(self, scale: float) -> float:
        """base_lr * scale, guvenlik sinirlari ile."""
        return clamp(self.base_lr * scale, self.cfg.lr_min, self.cfg.lr_max)

    def __call__(self, dl, gn, psteps, vtg=0.0, epoch_ratio=0.0):
        """
        Tam fuzzy inference + LR hesabi.
        Returns: (lr, scale)
        """
        scale = self.infer_scale(dl, gn, psteps, vtg, epoch_ratio)
        lr = self.lr_from_scale(scale)
        return lr, scale

# ============================================================================
#  FUZZY INPUTS
# ============================================================================
@dataclass
class FuzzyInputs:
    delta_loss: float        # RELATIVE: (EMA_new - EMA_old) / |EMA_old|
    grad_norm: float
    plateau_steps: int
    val_train_gap: float = 0.0
    epoch_ratio: float = 0.0
    improvement_rate: float = 0.0

# ============================================================================
#  FUZZY LR SCHEDULER
# ============================================================================
class FuzzyLRScheduler:
    """
    v4: Phase-Aware Scale-Only Scheduler.

    Cosine decay standart ilerler (speed yok, gate yok).
    Fuzzy controller tek cikis (scale) ile mikro perturbasyon ekler.

    YOLO'nun kendi scheduler'i devre disi (lrf=1.0, cos_lr=False).

    LR hesabi:
      cosine_lr = lr0 * (cosine_factor * (1 - lrf) + lrf)
      final_lr  = cosine_lr * scale
    """
    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float,
                 cfg: FuzzyLRConfig = FuzzyLRConfig(), total_epochs: int = 50):
        self.opt = optimizer
        self.cfg = cfg
        self.total_epochs = total_epochs
        self.current_epoch = 0

        # CLI base_lr, cosine schedule'in baslangic noktasini belirler.
        self.cfg.lr0 = base_lr
        # lr_max: base_lr'dan yuksek olmali ki scale>1.0 etkili olabilsin
        self.cfg.lr_max = max(self.cfg.lr_max, base_lr * 1.5)

        self.ctrl = FuzzyLRController(base_lr, cfg, total_epochs=total_epochs)

        # Cosine progress: standart, deterministik
        self._progress = 0.0

        for pg in self.opt.param_groups:
            pg.setdefault("base_lr", base_lr)

    def set_epoch(self, epoch: int):
        """
        Epoch guncelle ve cosine progress'i hesapla.

        Standart cosine progress:
          progress = (epoch - warmup + 1) / (total - warmup)

        Speed/gate yok — progress deterministik.
        Warmup sirasinda progress = 0.
        """
        self.current_epoch = epoch

        if self.cfg.use_warmup and epoch < self.cfg.warmup_epochs:
            self._progress = 0.0
            return

        warmup = self.cfg.warmup_epochs if self.cfg.use_warmup else 0
        effective_total = max(self.total_epochs - warmup, 1)
        self._progress = min((epoch - warmup + 1) / effective_total, 1.0)

    def get_cosine_lr(self) -> float:
        """Cosine LR'yi current progress'e gore hesapla."""
        cosine_factor = (1.0 + math.cos(math.pi * self._progress)) / 2.0
        return self.cfg.lr0 * (cosine_factor * (1.0 - self.cfg.lrf) + self.cfg.lrf)

    def get_phase_base_lr(self) -> float:
        """
        Cosine base LR (warmup dahil).

        Warmup sirasinda: lineer LR artisi.
        Warmup sonrasi: standart cosine decay.
        """
        if self.cfg.use_warmup:
            warmup_lr = apply_warmup(self.current_epoch,
                                     self.cfg.warmup_epochs,
                                     self.cfg.lr0)
            if warmup_lr is not None:
                return warmup_lr

        return self.get_cosine_lr()

    @property
    def progress(self) -> float:
        """Cosine decay ilerleme durumu [0, 1]."""
        return self._progress

    def step(self, fin: FuzzyInputs) -> tuple:
        """
        Bir training step icin LR guncelle.

        1. Fuzzy inference: scale hesapla
        2. Cosine base_lr hesapla (her zaman aktif)
        3. LR = cosine_lr * scale
        4. Optimizer param_groups guncelle

        Args:
            fin: FuzzyInputs (relative delta_loss, grad_norm, ...)

        Returns:
            (lr, scale) tuple'i
        """
        scale = self.ctrl.infer_scale(
            fin.delta_loss, fin.grad_norm, fin.plateau_steps,
            fin.val_train_gap, fin.epoch_ratio
        )

        # Base LR (cosine, her zaman aktif)
        phase_base_lr = self.get_phase_base_lr()

        # Controller'in base_lr'ini guncelle (lr_from_scale icin)
        self.ctrl.base_lr = phase_base_lr

        # Final LR = cosine_lr * scale
        lr = clamp(phase_base_lr * scale, self.cfg.lr_min, self.cfg.lr_max)

        # Optimizer guncelle
        for pg in self.opt.param_groups:
            pg["lr"] = lr
            pg["phase_base_lr"] = phase_base_lr

        return lr, scale
