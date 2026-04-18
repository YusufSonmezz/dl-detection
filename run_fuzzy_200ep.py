"""
Fuzzy calibrated — 200 epoch, 3 seed ardarda.
Local'de calistir:
    python run_fuzzy_200ep.py
"""
import subprocess, sys, os

SEEDS = [42, 123, 456]
EPOCHS = 200
BASE_LR = 0.01
LRF = 0.01
DATA = "train/data_neudet_org.yaml"

for seed in SEEDS:
    run_name = f"neudet_v4_cal200_seed{seed}"
    print(f"\n{'='*60}")
    print(f"  FUZZY 200ep | seed={seed} | run={run_name}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "train/train.py",
        "--data",     DATA,
        "--epochs",   str(EPOCHS),
        "--base_lr",  str(BASE_LR),
        "--lrf",      str(LRF),
        "--seed",     str(seed),
        "--run_name", run_name,
    ]
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"[ERROR] seed={seed} basarisiz oldu (returncode={result.returncode}). Sonraki seed'e geciliyor.")

print("\nTum fuzzy runlar tamamlandi.")
