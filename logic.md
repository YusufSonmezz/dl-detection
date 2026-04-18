## Mantık

1-  Image Extraction sinifi oncelikle veriyi frame frame almalı. Cunku tum goruntu uzerindeki metal levhaları bulacak ve onlari ayristiracak.
2-  Ayristirilan metal levha butun goruntuden koparilacak ve extract_metal_sheet fonksiyonu koparilan parcayi geri dondurecek.
3-  Dondurulen resim parcasi modele verilerek analiz edilecek ve sonuc tespit edilecek.
4-  Modelden gelen sonuc ile resmin kendisi birlestirilerek son cikti elde edilecek.

Bash command

# Baseline (fuzzy kapalı)
python train/train.py --no_fuzzy --run_name baseline_y8s_e10_pascal --epochs 10 --batch 16 --imgsz 640 --base_lr 1e-3

python train/train.py --data train/config_small.yaml --no_fuzzy --run_name baseline_y8s_e10_pascal --epochs 10 --batch 16 --imgsz 640 --base_lr 1e-3

# Fuzzy açık
python train/train.py --run_name fuzzy_y8s_e10_pascal --epochs 10 --batch 16 --imgsz 640 --base_lr 1e-3

## Fuzzy + 3-Phase
python train/train.py --data train/config_small.yaml --epochs 50 --batch 16 --base_lr 0.0001 --run_name fuzzy_3phase_e40

# Analyse
python train/analyse_run.py artifacts/runs/fuzzy_y8s_e10_pascal artifacts/runs/baseline_y8s_e10_pascal

artifacts/runs/baseline_lr0.001_e50