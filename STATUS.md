# Proje Durumu

> Son guncelleme: 2026-04-17

---

## 1. Su An Neredeyiz?

**SGD + Fuzzy v4 ile baseline ilk kez gecildi! MF kalibrasyonu tamamlandi.**

AdamW optimizer'inda YOLO'nun ic optimizasyonlari (bias LR, adaptif beta vb.) baseline'a haksiz avantaj sagliyordu.
SGD'ye gecince adil karsilastirma ortami olusturuldu ve fuzzy v4 baseline'i gecti.

SGD verisi analiz edildi: grad_norm ve VTG MF esikleri SGD/NEU-DET dagilimina gore kalibre edildi.
Kalibrasyon etkisini gormek icin `neudet_sgd_fuzzy_calibrated_v1` egitimi planlanmaktadir.

### En Son Deney Sonuclari (data_neudet_org.yaml, 100 epoch)

| Run | Optimizer | Mod | Best mAP50 | UP/DN | Not |
|---|---|---|---|---|---|
| neudet_baseline_org_v1 | AdamW | Baseline | 0.7582 | — | Eski en iyi (AdamW avantajli) |
| **neudet_scaleonly_sgd_100ep_v1** | **SGD** | **Fuzzy v4** | **0.7557** | **3916/3369 (1.16:1)** | **SGD'de en iyi, baseline'i gecti** |
| neudet_baseline_sgd_100ep_v1 | SGD | Baseline | 0.7544 | — | SGD baseline |
| neudet_v4_scaleonly_100ep | AdamW | Fuzzy v4 | 0.7456 | 6363/1364 (4.66:1) | AdamW ile fuzzy |
| neudet_scaleonly_sgd__baselr0.001 | SGD | Fuzzy v4 | 0.7454 | 5721/2114 | lr0=0.001 cok yuksek |
| neudet_ablation_scaleonly_v1 | AdamW | Ablation | 0.7320 | 0/0 | Cosine only, fuzzy yok |

### Kritik Bulgular

1. **SGD Fuzzy > SGD Baseline**: +0.13pp peak, **+1.1pp son 20ep ortalamasi**
2. **Stabilite**: Fuzzy son 20ep min=0.7436 vs baseline min=0.7106 (3.7x daha stabil)
3. **UP/DN dengesi**: SGD ile 1.16:1 (AdamW'de 4.66:1 idi) — kusursuz denge
4. **AdamW sorunlu**: YOLO ic optimizasyonlari fuzzy callback'i dezavantajli birakiyor

---

## 2. Bir Sonraki TEK Adim

**Tum sweep'ler (ablation, plateau, sgdr) bekleniyor — bitince analiz et.**

Sweep komutu:
```bash
python experiments/run_seed_sweep.py --modes ablation plateau sgdr --seeds 42 123 456
```

---

## 3. Bekleyen Isler (Sirayla)

### A. Deneyler (Oncelik sirasi)
1. [x] v4 mimari karari: scale-only + relative delta + phase-aware
2. [x] neudet_v4_scaleonly_100ep (AdamW) — mAP50=0.7456
3. [x] neudet_ablation_scaleonly_v1 (AdamW) — mAP50=0.7320
4. [x] AdamW optimizer avantaji kesfedildi — YOLO ic optimizasyonlari
5. [x] SGD'ye gecis karari
6. [x] neudet_baseline_sgd_100ep_v1 — mAP50=0.7544
7. [x] **neudet_scaleonly_sgd_100ep_v1 — mAP50=0.7557 (BASELINE'I GECTI!)**
8. [x] neudet_scaleonly_sgd__baselr0.001 — mAP50=0.7454 (lr yuksek, basarisiz)
9. [ ] **neudet_sgd_fuzzy_calibrated_v1 — MF kalibrasyonu etkisi (base_lr=0.01, lrf=0.01)**
10. [ ] Coklu seed dogrulamasi (3-5 seed, SGD fuzzy vs baseline)
11. [ ] Epoch uzatma (150-200 ep, SGD + fuzzy)
12. [ ] Augmented veri seti + SGD + fuzzy
13. [ ] Severstal veri seti (12.5k goruntu)

### B. Coklu Seed Sweep Durumu (2026-04-17)
- [x] fuzzy: seed42=0.7437, seed123=0.7401, seed456=0.7323
- [x] baseline: seed42=0.7387, seed123=0.7402, seed456=0.7396
- [ ] ablation: seed42, seed123, seed456 — DEVAM EDIYOR
- [ ] plateau: seed42, seed123, seed456 — DEVAM EDIYOR (bug fix: patience=5, threshold=0.02)
- [ ] sgdr: seed42, seed123, seed456 — DEVAM EDIYOR (bug fix: on_train_batch_start LR override)

**Genel istatistik (fuzzy vs baseline, 4 seed):**
fuzzy mean max=0.7430 +-0.0097  |  baseline mean max=0.7432 +-0.0075  |  p=0.97 (anlamli fark yok)

### C. Bekleyen Implementasyonlar (Sweep Bittikten Sonra)
1. [ ] **Sweep analizi**: ablation+plateau+sgdr sonuclari gelince tam karsilastirma yap, Faz 1 karari ver
2. [ ] **Yan destek mimarisi**: YOLO native cosine ACIK kalsin, fuzzy sadece `final_lr = yolo_lr x scale` uygulasın (simdi kapatiyoruz, bu avantaji kaldiriyor olabilir)
3. [ ] **Faz 1 implementasyonu** (analiz onaylarsa): erken epoch korumasi (ep<8 scale=1.0), accel_boost=1.3, EMA beta=0.95

### D. Sistem Iyilestirmeleri
- [x] v4: Speed+gate tamamen kaldirildi — scale-only micro-navigator
- [x] Relative delta_loss — faz-bagimsiz sinyal isleme
- [x] Phase-aware kural tabani — epoch_ratio MF ile faz-duyarli kurallar
- [x] Standart defuzzifikasyon — accel_boost/brake_penalty kaldirildi
- [x] AdamW→SGD gecisi
- [x] MF esiklerini SGD gradient dagilimina gore kalibre et (grad_norm + VTG)
- [x] Plateau bug fix: on_train_batch_start LR override, patience=5, threshold=0.02
- [ ] Loss EMA beta (0.8) → 0.95 (Faz 1 ile birlikte)
- [ ] 3-faz strateji testi (oncelik dusuk)

---

## 4. Tez Yazimi Nerede?

**Henuz baslanmadi.** Deney + analiz asamasindayiz.

Teze baslamak icin gereken minimum:
- [x] GC10 baseline + fuzzy sonuclari
- [x] NEU-DET baseline + fuzzy sonuclari (birden fazla veri seti)
- [x] Sistem davranis analizi (stabilizator, fren, metrik duzeltmeleri)
- [x] Ablation runlari (coklu baseline + ablation karsilastirmalari)
- [x] YOLO varsayilan karsilastirmasi
- [x] Literatur analizi (epoch, split, augmentation farklari)
- [x] 100 epoch egitim sonuclari
- [x] MIMO mimarisi ve v1/v2/v3 testleri
- [x] Gate kaldirma, speed tersine cevirme deneyleri
- [x] v4 mimari implementasyonu ve testleri
- [x] **AdamW→SGD optimizer kesfi ve SGD sonuclari**
- [x] **SGD fuzzy > SGD baseline dogrulamasi (ilk run)**
- [ ] **Coklu seed istatistiksel dogrulama**
- [ ] Augmented veri seti ile egitim sonuclari
- [ ] **Mimari finalizasyonu** (v4 + SGD kesinlesti mi?)
- [ ] Sonuclarin yorumlanmasi ve karsilastirma tablolari
- [ ] Tez yazmaya baslama
