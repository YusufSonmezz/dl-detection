# Fuzzy Scheduler Monitoring Guide

Bu kılavuz, fuzzy scheduler training sırasında terminal çıktısını nasıl yorumlayacağınızı ve neye dikkat etmeniz gerektiğini açıklar.

---

## Terminal Çıktısı Formatı

Her epoch sonunda şu formatta çıktı görürsünüz:

```
[FUZZY] Epoch X/Y | Status: {durum} | Base LR: {lr} | Best mAP50: {map}
```

### Parametreler:

#### 1. **Status (Durum)**
Fuzzy scheduler'ın şu anki durumu:

| Status | Anlamı | Ne Beklemeli? |
|--------|--------|---------------|
| `Warm-up` | İlk 3 epoch, LR yavaşça artıyor | LR: 0.00005 → 0.0001 |
| `Improving` | mAP iyileşiyor (plateau_steps=0) | Scale: 1.05-1.20 (artıyor) |
| `Plateau(N)` | N epoch mAP durdu (N<5) | Scale: ~1.0 (bekliyor) |
| `Long Plateau(N)` | N epoch durdu (5≤N<15) | Scale: 1.05-1.15 (keşif) |
| `🚨 EMERGENCY(N)` | **15+ epoch durdu!** | **Scale: 1.25-1.30 (ACİL KAÇIŞ!)** |

#### 2. **Base LR**
Fuzzy scaling'den **ÖNCE** base learning rate:
- Warm-up sırasında: Artıyor (0.00005 → 0.0001)
- Warm-up sonrası: **SABİT kalmalı (0.0001)** ← Phase decay devre dışı!

⚠️ **UYARI**: Base LR 0.0001'den farklıysa, phase decay hala aktif demektir (BUG!)

#### 3. **Best mAP50**
Şimdiye kadar elde edilen en iyi mAP@0.5 değeri.
- **İyi**: Her epoch artıyor veya stabil
- **Kötü**: 10+ epoch hiç artmıyor (plateau)

---

## Başarılı vs Başarısız Training

### ✅ BAŞARILI Training Örneği

```
[FUZZY] Epoch 0/40 | Status: Warm-up | Base LR: 0.000067 | Best mAP50: 0.4771
[FUZZY] Epoch 2/40 | Status: Warm-up | Base LR: 0.000100 | Best mAP50: 0.7587
[FUZZY] Epoch 3/40 | Status: Improving | Base LR: 0.000100 | Best mAP50: 0.7693
[FUZZY] Epoch 5/40 | Status: Improving | Base LR: 0.000100 | Best mAP50: 0.7850  ✅ İyileşiyor
[FUZZY] Epoch 8/40 | Status: Plateau(2) | Base LR: 0.000100 | Best mAP50: 0.7850
[FUZZY] Epoch 12/40 | Status: Long Plateau(6) | Base LR: 0.000100 | Best mAP50: 0.7850
[FUZZY] Epoch 13/40 | Status: Improving | Base LR: 0.000100 | Best mAP50: 0.7920  ✅ Plateau'dan kaçtı!
[FUZZY] Epoch 20/40 | Status: Improving | Base LR: 0.000100 | Best mAP50: 0.7980
```

**Göstergeler:**
- Base LR 0.0001'de sabit ✅
- mAP düzenli artıyor ✅
- Plateau'lardan kaçıyor ✅

### ❌ BAŞARISIZ Training Örneği

```
[FUZZY] Epoch 3/40 | Status: Improving | Base LR: 0.000100 | Best mAP50: 0.7693  ← Peak
[FUZZY] Epoch 10/40 | Status: Long Plateau(7) | Base LR: 0.000100 | Best mAP50: 0.7693
[FUZZY] Epoch 15/40 | Status: Long Plateau(12) | Base LR: 0.000100 | Best mAP50: 0.7693
[FUZZY] Epoch 20/40 | Status: 🚨 EMERGENCY(17) | Base LR: 0.000100 | Best mAP50: 0.7693  ← Stuck!
[FUZZY] Epoch 25/40 | Status: 🚨 EMERGENCY(22) | Base LR: 0.000100 | Best mAP50: 0.7693  ← Still stuck!
```

**Sorun İşaretleri:**
- Epoch 3'ten beri mAP artmıyor ❌
- Emergency status 10+ epoch sürüyor ❌
- Scale boost çalışmıyor olabilir ❌

---

## steps.csv Analizi (Detaylı İnceleme)

Training bitince `artifacts/runs/{run_name}/steps.csv` dosyasını inceleyin:

### Önemli Kolonlar:

| Kolon | Açıklama | Ne Aranmalı? |
|-------|----------|--------------|
| `epoch` | Epoch numarası | - |
| `lr` | Gerçek LR (base_lr × scale) | **0.0001 ± %30** aralığında |
| `phase_base_lr` | Base LR (fuzzy öncesi) | **Her zaman 0.0001** (warm-up hariç) |
| `scale` | Fuzzy scale faktörü | **0.85-1.30** arası değişmeli |
| `plateau_steps` | Kaç epochtir plateau | Emergency durumda **15+** olmalı |
| `delta_loss` | Loss değişimi | **Negatif ise azalıyor (iyi!)** |

### Örnek Sorgu (CSV'de):

**Soru: Emergency escape çalışıyor mu?**

```python
# plateau_steps ≥ 15 olan satırları filtrele
emergency_rows = df[df['plateau_steps'] >= 15]

# Scale değerlerini kontrol et
print(emergency_rows[['epoch', 'plateau_steps', 'scale', 'lr']].tail(10))

# Beklenen:
# scale: 1.20-1.30 arası (emergency boost active!)
# lr: 0.00012-0.00013 (0.0001 × 1.25 = 0.000125)
```

**Soru: Base LR sabit mi kaldı?**

```python
# Warm-up sonrası (epoch > 3) base_lr'yi kontrol et
post_warmup = df[df['epoch'] > 3]
print(post_warmup['phase_base_lr'].unique())

# Beklenen: [0.0001] (tek değer!)
# Yanlış: [0.0001, 0.00007, 0.00003] (phase decay hala aktif!)
```

---

## Gerçek Zamanlı İzleme (Training Sırasında)

### 1. Scale Değerlerini İzle

```bash
# Son 20 step'in scale değerlerini göster
tail -20 artifacts/runs/fuzzy_fix_v2/steps.csv | cut -d',' -f9
```

**Beklenen Davranış:**
- Normal durum: 0.95-1.15 arası
- Plateau durumu: 1.05-1.20 arası
- **Emergency (plateau_steps≥15): 1.20-1.30 arası** ← BU ÇOK ÖNEMLİ!

### 2. mAP Trend Grafiği (Hızlı Kontrol)

```python
import pandas as pd
import matplotlib.pyplot as plt

epochs = pd.read_csv('artifacts/runs/fuzzy_fix_v2/epochs.csv')
plt.plot(epochs['epoch'], epochs['map50'])
plt.axhline(y=0.7997, color='r', linestyle='--', label='Baseline Best')
plt.xlabel('Epoch')
plt.ylabel('mAP@0.5')
plt.legend()
plt.show()
```

**İyi Trend:**
- Düzenli yükseliş
- Baseline çizgisini geçiyor
- Son 10 epoch'ta stabil/artıyor

**Kötü Trend:**
- Erken peak (epoch 3-5), sonra düz çizgi
- Baseline'ın altında kalıyor
- 20+ epoch hiç artış yok

---

## Troubleshooting

### Problem 1: "Emergency status var ama LR artmıyor"

**Belirtiler:**
```
[FUZZY] Epoch 20/40 | Status: 🚨 EMERGENCY(16) | Base LR: 0.000100 | Best mAP50: 0.7693
```
Ama `steps.csv`'de `scale` ≈ 1.05 (beklenen: 1.25-1.30)

**Neden:**
- Emergency rule weight boost yetersiz
- Diğer kurallar hala dominant

**Çözüm:**
```python
# fuzzy_lr.py'de weight multiplier'ı artır:
emergency_weight = min(1.0, PL["cok_uzun"] * 10.0)  # 5.0 → 10.0
```

### Problem 2: "Base LR değişiyor (phase decay hala aktif)"

**Belirtiler:**
```
[FUZZY] Epoch 15/40 | Status: Plateau | Base LR: 0.000072 | ...  ← 0.0001 değil!
```

**Neden:**
- `get_phase_base_lr()` hala phase decay kullanıyor

**Çözüm:**
```python
# fuzzy_lr.py, line 506 civarı - kontrol et:
# if self.cfg.use_three_phase:  ← Bu satır COMMENT olmalı!
```

### Problem 3: "Plateau çok geç tespit ediliyor"

**Belirtiler:**
- Epoch 3'te peak
- Plateau status epoch 20'de başlıyor

**Neden:**
- Grace period (10 epoch) çok uzun
- Relative tolerance (%0.5) çok geniş

**Çözüm:**
```python
# yolo_fuzzy_callback.py'de:
self.plateau = PlateauTracker(rel_tol=0.003, grace_epochs=5)  # Daha hassas
```

---

## Başarı Kriterleri (fuzzy_fix_v2)

Training sonunda şu hedeflere ulaşılmalı:

| Metrik | Hedef | Nasıl Kontrol? |
|--------|-------|----------------|
| **Best mAP@0.5** | **>0.77** (ideal: >0.79) | Terminal output |
| **Convergence Epoch** | **<30** | `epochs.csv`: mAP'in en yüksek olduğu epoch |
| **Emergency Escape** | **2-3 kere görünmeli** | `grep "EMERGENCY" training.log` |
| **LR Variance** | **>20%** | `steps.csv`: `scale.std()` |
| **Base LR Stability** | **0.0001 sabit** | `steps.csv`: `phase_base_lr.unique()` |
| **Final Plateau** | **<10 epoch** | Son epoch'ta `plateau_steps < 10` |

### Hızlı Başarı Kontrolü (Python)

```python
import pandas as pd

# Logs oku
epochs = pd.read_csv('artifacts/runs/fuzzy_fix_v2/epochs.csv')
steps = pd.read_csv('artifacts/runs/fuzzy_fix_v2/steps.csv')

# Metrikler
best_map = epochs['map50'].max()
converge_epoch = epochs['map50'].idxmax()
lr_var = steps['scale'].std()
base_lr_stable = (steps[steps['epoch'] > 3]['phase_base_lr'] == 0.0001).all()
emergency_count = (steps['plateau_steps'] >= 15).sum()

print(f"✅ Best mAP: {best_map:.4f} (Target: >0.77)")
print(f"✅ Converged at epoch: {converge_epoch} (Target: <30)")
print(f"✅ LR variance: {lr_var:.2f} (Target: >0.20)")
print(f"✅ Base LR stable: {base_lr_stable}")
print(f"✅ Emergency events: {emergency_count // 66} times")  # 66 steps/epoch

if best_map > 0.77 and converge_epoch < 30:
    print("\n🎉 SUCCESS! Fuzzy scheduler is working!")
else:
    print("\n⚠️ NEEDS TUNING - Check troubleshooting section")
```

---

## Özet: Neye Dikkat Etmeli?

1. **Terminal çıktısında:**
   - Base LR = 0.0001 sabit kalmalı ✅
   - Emergency status görülmeli (plateau ≥15) ✅
   - mAP düzenli artmalı ✅

2. **steps.csv'de:**
   - `scale`: 0.85-1.30 arası değişmeli ✅
   - `phase_base_lr`: 0.0001 sabit ✅
   - Emergency'de scale >1.20 olmalı ✅

3. **epochs.csv'de:**
   - `map50`: Epoch 30'a kadar 0.77+ ulaşmalı ✅
   - Plateau duration <10 epoch ✅

**En Önemli Gösterge:**
```
[FUZZY] Epoch 18/40 | Status: 🚨 EMERGENCY(16) | Base LR: 0.000100 | Best mAP50: 0.7850
```
Sonraki birkaç epoch'ta:
```
[FUZZY] Epoch 20/40 | Status: Improving | Base LR: 0.000100 | Best mAP50: 0.7920  ← KAÇTI!
```

Bu oluyorsa, **fuzzy adaptive intelligence çalışıyor demektir!** 🎯
