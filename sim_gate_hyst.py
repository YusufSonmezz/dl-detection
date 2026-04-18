"""
Gate Hysteresis Stratejisi Simülasyonu
İki stratejiyi v2 eğitim verisi üzerinde karşılaştırır.
  A: Step-level hyst=0.03/step
  B: Epoch-level hyst=0.30/epoch

Çalıştırma: python sim_gate_hyst.py
"""
import csv
import math
import statistics
from collections import defaultdict

# ============================================================
# YARDIMCI FONKSIYONLAR
# ============================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def cosine_lr(progress, lr0=1e-4, lrf=0.10):
    factor = (1.0 + math.cos(math.pi * min(progress, 1.0))) / 2.0
    return lr0 * (factor * (1.0 - lrf) + lrf)

# ============================================================
# VERİ YÜKLEME
# ============================================================
steps = []
with open("artifacts/runs/neudet_mimo_100ep_v2/steps.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append({
            "step": int(row["step"]),
            "epoch": int(row["epoch"]),
            "gate": float(row["gate"]),
            "speed": float(row["speed"]),
            "scale": float(row["scale"]),
            "lr": float(row["lr"]),
            "phase_base_lr": float(row["phase_base_lr"]),
            "cosine_progress": float(row["cosine_progress"]),
            "plateau_steps": float(row["plateau_steps"]),
        })

epoch_steps = defaultdict(list)
for s in steps:
    epoch_steps[s["epoch"]].append(s)
epochs_list = sorted(epoch_steps.keys())

# Epoch bazli ozetler
epoch_gate_real = {}
epoch_nsteps = {}

for ep in epochs_list:
    s_list = epoch_steps[ep]
    epoch_gate_real[ep] = s_list[0]["gate"]
    epoch_nsteps[ep] = len(s_list)

# ============================================================
# HAM GATE HEDEFLERİ
# Hist-sınırlı geçişler tespit edilir:
#   |delta| ~ 0.15 ve delta < 0 → sonraki epoch'ta ani 0 → ham=0.0
#   |delta| ~ 0.15 ve delta > 0 → ham = real + extra (en az 0.15 daha yüksek)
# ============================================================
def get_ham_gate(ep):
    real = epoch_gate_real[ep]
    if ep > 0:
        prev = epoch_gate_real[ep - 1]
        delta = real - prev
        if abs(abs(delta) - 0.15) < 0.005:
            if delta < 0:
                # Ani dusuş: hedef 0.0
                return 0.0
            else:
                # Ani artış: hedef en az 0.15 daha yüksek (güvenli üst sınır 1.0)
                return min(real + 0.15, 1.0)
    return real

ham_gates = {ep: get_ham_gate(ep) for ep in epochs_list}

# ============================================================
# LR HESABI
# ============================================================
def calc_lr(gate, cp, scale, lr0=1e-4):
    cos_lr = cosine_lr(cp, lr0=lr0)
    base = lr0 * (1.0 - gate) + cos_lr * gate
    return base * scale

# ============================================================
# STRATEJİ A: step-level hyst=0.03/step
# Her epoch için ham gate hedefine adım adım yaklaşılır.
# Epoch sonu gate değeri sonraki epoch'a aktarılır.
# ============================================================
gate_A_epoch_mean = {}
gate_A_epoch_start = {}   # Epoch başındaki gate (karşılaştırma için)
gate_A_step_all = []      # (epoch, step_idx, gate) - tüm step'ler

last_gate_A = 0.0
for ep in epochs_list:
    target = ham_gates[ep]
    n_steps = epoch_nsteps[ep]
    s_list = epoch_steps[ep]
    step_gates = []
    gate_A_epoch_start[ep] = last_gate_A
    for i in range(n_steps):
        delta = target - last_gate_A
        if abs(delta) > 0.03:
            last_gate_A += 0.03 * (1 if delta > 0 else -1)
        else:
            last_gate_A = target
        last_gate_A = clamp(last_gate_A, 0.0, 1.0)
        step_gates.append(last_gate_A)
        gate_A_step_all.append((ep, i, last_gate_A))
    gate_A_epoch_mean[ep] = statistics.mean(step_gates)

# ============================================================
# STRATEJİ B: epoch-level hyst=0.30/epoch
# Gate epoch başında hesaplanır, epoch boyunca sabit kalır.
# ============================================================
gate_B_epoch = {}
last_gate_B = 0.0
for ep in epochs_list:
    target = ham_gates[ep]
    delta = target - last_gate_B
    if abs(delta) > 0.30:
        last_gate_B += 0.30 * (1 if delta > 0 else -1)
    else:
        last_gate_B = target
    last_gate_B = clamp(last_gate_B, 0.0, 1.0)
    gate_B_epoch[ep] = last_gate_B

# ============================================================
# LR HESAPLAMA - Her strateji için epoch ortalaması
# ============================================================
lr_real_epoch = {}
lr_A_epoch = {}
lr_B_epoch = {}
lr_real_std_epoch = {}
lr_A_std_epoch = {}
lr_B_std_epoch = {}

# Strateji A için step-level gate'i yeniden hesapla (epoch mean güncellemek için)
last_gate_A2 = 0.0
gate_A_step_map = {}  # (ep, step_idx) -> gate

for ep in epochs_list:
    target = ham_gates[ep]
    n_steps = epoch_nsteps[ep]
    temp_g = last_gate_A2
    for i in range(n_steps):
        delta = target - temp_g
        if abs(delta) > 0.03:
            temp_g += 0.03 * (1 if delta > 0 else -1)
        else:
            temp_g = target
        temp_g = clamp(temp_g, 0.0, 1.0)
        gate_A_step_map[(ep, i)] = temp_g
    last_gate_A2 = temp_g

for ep in epochs_list:
    s_list = epoch_steps[ep]
    gate_b = gate_B_epoch[ep]

    lrs_real = []
    lrs_A = []
    lrs_B = []

    for i, s in enumerate(s_list):
        cp = s["cosine_progress"]
        scale = s["scale"]
        lr_r = s["lr"]
        gate_a = gate_A_step_map.get((ep, i), 0.0)

        lrs_real.append(lr_r)
        lrs_A.append(calc_lr(gate_a, cp, scale))
        lrs_B.append(calc_lr(gate_b, cp, scale))

    lr_real_epoch[ep] = statistics.mean(lrs_real)
    lr_A_epoch[ep] = statistics.mean(lrs_A)
    lr_B_epoch[ep] = statistics.mean(lrs_B)

    if len(lrs_real) > 1:
        lr_real_std_epoch[ep] = statistics.stdev(lrs_real)
        lr_A_std_epoch[ep] = statistics.stdev(lrs_A)
        lr_B_std_epoch[ep] = statistics.stdev(lrs_B)
    else:
        lr_real_std_epoch[ep] = 0.0
        lr_A_std_epoch[ep] = 0.0
        lr_B_std_epoch[ep] = 0.0

# ============================================================
# BÖLÜM 1: GATE TRAJECTORY TABLOSU (her 5 epoch dilimi)
# ============================================================
print("=" * 75)
print("BÖLÜM 1: GATE TRAJECTORY (ortalama gate, her 5 epoch)")
print("=" * 75)
print(f"{'Ep':>4}  {'Mevcut(0.15)':>14}  {'A(0.03/step)':>14}  {'B(0.30/epoch)':>15}  {'Ham Hedef':>10}")
print("-" * 75)
for ep in range(0, 99, 5):
    if ep not in epochs_list:
        continue
    real = epoch_gate_real[ep]
    a = gate_A_epoch_mean[ep]
    b = gate_B_epoch[ep]
    ham = ham_gates[ep]
    print(f"{ep:>4}  {real:>14.4f}  {a:>14.4f}  {b:>15.4f}  {ham:>10.4f}")

# ============================================================
# BÖLÜM 2: LR TRAJECTORY TABLOSU (her 5 epoch)
# ============================================================
print()
print("=" * 75)
print("BÖLÜM 2: ORTALAMA LR TRAJECTORY (her 5 epoch)")
print("=" * 75)
print(f"{'Ep':>4}  {'Mevcut':>12}  {'A(0.03)':>12}  {'B(0.30)':>12}")
print("-" * 75)
for ep in range(0, 99, 5):
    if ep not in epochs_list:
        continue
    print(f"{ep:>4}  {lr_real_epoch[ep]:>12.7f}  {lr_A_epoch[ep]:>12.7f}  {lr_B_epoch[ep]:>12.7f}")

# ============================================================
# BÖLÜM 3: KRİTİK GEÇİŞ ANALİZİ - EP55-65 (HYST-aktif ep60 civarı)
# ============================================================
print()
print("=" * 85)
print("BÖLÜM 3: KRİTİK GEÇİŞ ANALİZİ EP55-65 (epoch bazli)")
print("=" * 85)
print(f"{'Ep':>4}  {'MevGate':>9}  {'A_Gate':>9}  {'B_Gate':>9}  {'MevLR':>11}  {'A_LR':>11}  {'B_LR':>11}")
print("-" * 85)
for ep in range(55, 65):
    if ep not in epochs_list:
        continue
    real_g = epoch_gate_real[ep]
    a_g = gate_A_epoch_mean[ep]
    b_g = gate_B_epoch[ep]
    real_lr = lr_real_epoch[ep]
    a_lr = lr_A_epoch[ep]
    b_lr = lr_B_epoch[ep]
    print(f"{ep:>4}  {real_g:>9.4f}  {a_g:>9.4f}  {b_g:>9.4f}  {real_lr:>11.7f}  {a_lr:>11.7f}  {b_lr:>11.7f}")

# ============================================================
# BÖLÜM 4: TATLI NOKTA KALIŞ SÜRESİ
# Gate [0.55, 0.85] → LR orta bölge
# ============================================================
print()
print("=" * 70)
print("BÖLÜM 4: TATLI NOKTA [gate 0.55-0.85] KALIS SÜRESİ")
print("=" * 70)

total_steps_all = sum(epoch_nsteps[ep] for ep in epochs_list)

# Mevcut: step-level gate (steps.csv)
sweet_real = sum(1 for s in steps if 0.55 <= s["gate"] <= 0.85)

# Strateji A: step-level (gate_A_step_map)
sweet_A = sum(1 for (ep, i), g in gate_A_step_map.items() if 0.55 <= g <= 0.85)

# Strateji B: epoch-level (gate_B_epoch, her step epoch gatesi)
sweet_B = 0
for ep in epochs_list:
    b_g = gate_B_epoch[ep]
    if 0.55 <= b_g <= 0.85:
        sweet_B += epoch_nsteps[ep]

print(f"Toplam step sayisi: {total_steps_all}")
print(f"Strateji            Tatli noktada step   Yuzde")
print(f"Mevcut (0.15/epoch) {sweet_real:>20}   {100*sweet_real/total_steps_all:.1f}%")
print(f"A (0.03/step)       {sweet_A:>20}   {100*sweet_A/total_steps_all:.1f}%")
print(f"B (0.30/epoch)      {sweet_B:>20}   {100*sweet_B/total_steps_all:.1f}%")

# ============================================================
# BÖLÜM 5: LR STABİLİTE METRİKLERİ
# ============================================================
print()
print("=" * 70)
print("BÖLÜM 5: LR STABİLİTE METRİKLERİ")
print("=" * 70)

# Epoch-içi LR std ortalaması
avg_real_std = statistics.mean(lr_real_std_epoch[ep] for ep in epochs_list)
avg_A_std = statistics.mean(lr_A_std_epoch[ep] for ep in epochs_list)
avg_B_std = statistics.mean(lr_B_std_epoch[ep] for ep in epochs_list)

# Step-to-step LR değişimi
def step_to_step_diff(lr_list):
    if len(lr_list) < 2:
        return []
    return [abs(lr_list[i+1] - lr_list[i]) for i in range(len(lr_list)-1)]

# Mevcut LR listesi
lrs_real_all = [s["lr"] for s in steps]

# Strateji A LR listesi
lrs_A_all = []
for ep in epochs_list:
    s_list = epoch_steps[ep]
    for i, s in enumerate(s_list):
        g = gate_A_step_map.get((ep, i), 0.0)
        lrs_A_all.append(calc_lr(g, s["cosine_progress"], s["scale"]))

# Strateji B LR listesi
lrs_B_all = []
for ep in epochs_list:
    gate_b = gate_B_epoch[ep]
    s_list = epoch_steps[ep]
    for s in s_list:
        lrs_B_all.append(calc_lr(gate_b, s["cosine_progress"], s["scale"]))

diffs_real = step_to_step_diff(lrs_real_all)
diffs_A = step_to_step_diff(lrs_A_all)
diffs_B = step_to_step_diff(lrs_B_all)

print(f"Metrik                         Mevcut(0.15)   A(0.03/step)   B(0.30/epoch)")
print("-" * 70)
print(f"Epoch-ici LR std (ort)         {avg_real_std:.6f}     {avg_A_std:.6f}     {avg_B_std:.6f}")
print(f"Step-to-step degisim (ort)     {statistics.mean(diffs_real):.6f}     {statistics.mean(diffs_A):.6f}     {statistics.mean(diffs_B):.6f}")
print(f"Maks tek-step LR siçramasi     {max(diffs_real):.6f}     {max(diffs_A):.6f}     {max(diffs_B):.6f}")

# ============================================================
# BÖLÜM 6: COSİNE BÜTÇE ETKİSİ
# Strateji B'de speed epoch başında sabit olurdu.
# Cosine progress: progress += (1/eff_total) * speed
# Mevcut: her step sonunda speed güncellenir, set_epoch'ta progress ilerler.
# B simülasyonu: Her epoch için o epoch'taki ilk step'in speed değerini kullan.
# ============================================================
print()
print("=" * 70)
print("BÖLÜM 6: COSİNE BÜTÇE ETKİSİ")
print("=" * 70)

EFF_TOTAL = N_EPOCHS = 99 - 3  # warmup=3 sonrası effective
progress_real_final = epoch_steps[epochs_list[-1]][-1]["cosine_progress"]

# Strateji B için cosine progress simülasyonu
# Her epoch: speed = o epoch'taki ilk step speed (B'de sabit)
# progress başlangicı: warmup biter (ep=3 başında)
progress_B = 0.0
for ep in epochs_list:
    if ep < 3:  # warmup
        continue
    s_list = epoch_steps[ep]
    # B'de speed epoch başında hesaplanır
    # Mevcut veriden ilk step speed'i al (fuzzy çıktısı)
    speed_b = s_list[0]["speed"]
    base_inc = 1.0 / max(EFF_TOTAL, 1)
    progress_B += base_inc * speed_b
    progress_B = min(progress_B, 1.0)

print(f"Mevcut cosine_progress (final):  {progress_real_final:.4f}")
print(f"Strateji B simüle (final):       {progress_B:.4f}")
print(f"Fark (B - Mevcut):               {progress_B - progress_real_final:+.4f}")
print()
print("Not: Strateji A cosine progress'i etkilemez (gate degismez, speed degismez).")
print("     Strateji B'de speed epoch-icinde sabit olur - ancak speed zaten")
print("     epoch-level hesaplandigi icin pratikte fark minimal.")

# ============================================================
# BÖLÜM 7: SONUÇ TABLOSU
# ============================================================
print()
print("=" * 85)
print("BÖLÜM 7: ÖZET SONUÇ TABLOSU")
print("=" * 85)

# 1.0 -> 0.0 gecis suresi
# Mevcut: ep28: 0.7931 -> 0.6431 (-0.15, hyst), ep29: 0.0000 (ani)
# Toplam: ep27=0.7931, ep28=0.6431 (1 epoch sınırlı), ep29=0.0000
# Effectif: 1 epoch (tek sıçrama)
# Strateji A: gate_A hız: 0.03/step, 90 step/epoch → 2.7/epoch
# 0.79 -> 0.0: 0.79/2.7 = ~0.3 epoch (çok hızlı)
# Strateji B: 0.30/epoch, 0.79/0.30 = ~2.7 epoch

# Plato tepki gecikmesi:
# Mevcut: gate 0 -> 0.3 geçişi: ep15-ep23 (~8 epoch)
# A: 0.03*90=2.7/epoch max → 0.3/2.7 = ~0.11 epoch (neredeyse aninda)
# B: 0.30/epoch, 0->0.3 = 1 epoch

# Hesapla
# Mevcut: ilk >0 gate ep15, 0.3 ulaşma ep18 → 3 epoch
# A: 0'dan 0.3'e: 0.3/2.7_per_epoch = ~0.11 epoch
# B: 0.30/epoch, 0->0.30 = 1 epoch, 0->0.60 = 2 epoch

# Cosine budget: progress_real vs progress_B
cosine_budget_diff = abs(progress_B - progress_real_final)

print(f"{'Metrik':<35}  {'Mevcut(0.15/ep)':>16}  {'A(0.03/step)':>13}  {'B(0.30/epoch)':>14}")
print("-" * 85)
print(f"{'1.0->0.0 gecis suresi':<35}  {'~1 epoch':>16}  {'~0.3 epoch':>13}  {'~2.7 epoch':>14}")
print(f"{'0->plato_tepki (0.3 gate)':<35}  {'~3 epoch':>16}  {'~0.1 epoch':>13}  {'~1 epoch':>14}")
print(f"{'Tatli noktada step':<35}  {sweet_real:>16}  {sweet_A:>13}  {sweet_B:>14}")
print(f"{'Tatli noktada yuzde':<35}  {100*sweet_real/total_steps_all:>15.1f}%  {100*sweet_A/total_steps_all:>12.1f}%  {100*sweet_B/total_steps_all:>13.1f}%")
print(f"{'Epoch-ici LR std (ort)':<35}  {avg_real_std:>16.7f}  {avg_A_std:>13.7f}  {avg_B_std:>14.7f}")
print(f"{'Step-to-step LR degisim ort':<35}  {statistics.mean(diffs_real):>16.7f}  {statistics.mean(diffs_A):>13.7f}  {statistics.mean(diffs_B):>14.7f}")
print(f"{'Maks tek-step LR sicramasi':<35}  {max(diffs_real):>16.7f}  {max(diffs_A):>13.7f}  {max(diffs_B):>14.7f}")
print(f"{'Cosine progress farki (final)':<35}  {'ref':>16}  {'~0.0':>13}  {cosine_budget_diff:>14.4f}")

# ============================================================
# BÖLÜM 8: NİHAİ ÖNERİ destekleyen ek metrikler
# ============================================================
print()
print("=" * 70)
print("BÖLÜM 8: DESTEKLEYEN EK ANALİZLER")
print("=" * 70)

# Hangi strateji gate [0, 0.1] (saf fuzzy) bölgesinde daha az zaman geçiriyor?
# Düşük gate → cosine etkisiz, LR sabit lr0
low_gate_real = sum(1 for s in steps if s["gate"] < 0.1)
low_gate_A = sum(1 for (ep, i), g in gate_A_step_map.items() if g < 0.1)
low_gate_B = 0
for ep in epochs_list:
    if gate_B_epoch[ep] < 0.1:
        low_gate_B += epoch_nsteps[ep]

print(f"Dusuk gate (<0.1, saf fuzzy) step sayisi:")
print(f"  Mevcut: {low_gate_real} ({100*low_gate_real/total_steps_all:.1f}%)")
print(f"  A:      {low_gate_A} ({100*low_gate_A/total_steps_all:.1f}%)")
print(f"  B:      {low_gate_B} ({100*low_gate_B/total_steps_all:.1f}%)")

# Yüksek gate (>0.9, tam cosine)
high_gate_real = sum(1 for s in steps if s["gate"] > 0.9)
high_gate_A = sum(1 for (ep, i), g in gate_A_step_map.items() if g > 0.9)
high_gate_B = 0
for ep in epochs_list:
    if gate_B_epoch[ep] > 0.9:
        high_gate_B += epoch_nsteps[ep]

print(f"Yuksek gate (>0.9, tam cosine) step sayisi:")
print(f"  Mevcut: {high_gate_real} ({100*high_gate_real/total_steps_all:.1f}%)")
print(f"  A:      {high_gate_A} ({100*high_gate_A/total_steps_all:.1f}%)")
print(f"  B:      {high_gate_B} ({100*high_gate_B/total_steps_all:.1f}%)")

# LR range per strateji
print(f"\nLR aralik (min-max):")
print(f"  Mevcut: {min(lrs_real_all):.7f} - {max(lrs_real_all):.7f}")
print(f"  A:      {min(lrs_A_all):.7f} - {max(lrs_A_all):.7f}")
print(f"  B:      {min(lrs_B_all):.7f} - {max(lrs_B_all):.7f}")

# Epoch 28-35 detail (reset dongüsü)
print()
print("--- Ep28-37 (1. reset dongusu) ---")
print(f"{'Ep':>3}  {'Ham':>6}  {'MevGate':>8}  {'A_Gate':>8}  {'B_Gate':>8}  {'MevLR':>10}  {'A_LR':>10}  {'B_LR':>10}")
for ep in range(27, 38):
    if ep not in epochs_list:
        continue
    print(f"{ep:>3}  {ham_gates[ep]:>6.3f}  {epoch_gate_real[ep]:>8.4f}  {gate_A_epoch_mean[ep]:>8.4f}  {gate_B_epoch[ep]:>8.4f}  {lr_real_epoch[ep]:>10.7f}  {lr_A_epoch[ep]:>10.7f}  {lr_B_epoch[ep]:>10.7f}")
