"""
Baseline cosine — 200 epoch, 3 seed ardarda.
Colab'da calistir:
    !python run_baseline_200ep.py
"""
import subprocess, sys, os

SEEDS = [42, 123, 456]
EPOCHS = 200
DATA = "train/data_neudet_org.yaml"

for seed in SEEDS:
    run_name = f"neudet_baseline200_seed{seed}"
    print(f"\n{'='*60}")
    print(f"  BASELINE 200ep | seed={seed} | run={run_name}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "train/train.py",
        "--data",     DATA,
        "--epochs",   str(EPOCHS),
        "--no_fuzzy",
        "--seed",     str(seed),
        "--run_name", run_name,
    ]
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"[ERROR] seed={seed} basarisiz oldu (returncode={result.returncode}). Sonraki seed'e geciliyor.")

print("\nTum baseline runlar tamamlandi.")
