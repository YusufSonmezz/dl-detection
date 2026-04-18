# train/train.py
import argparse, os, random, numpy as np, torch
from models.yolov8 import Yolov8
from models.yolo_fuzzy_callback import FuzzyYOLOCallback
from models.yolo_baseline_callback import BaselineLoggerCallback
from models.yolo_plateau_callback import PlateauLRCallback
from models.yolo_sgdr_callback import SGDRCallback
from scheduler.fuzzy_lr import FuzzyLRConfig
from scheduler.exp_logger import StepLogger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    ap = argparse.ArgumentParser(description="YOLOv8 training with Enhanced Fuzzy LR Scheduler")
    ap.add_argument("--epochs", type=int, default=40,
                    help="Total training epochs (default: 40)")
    ap.add_argument("--batch", type=int, default=16,
                    help="Batch size (default: 16)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Image size (default: 640)")
    ap.add_argument("--data", type=str, default="train/config.yaml",
                    help="Dataset config file")
    ap.add_argument("--run_name", type=str, default="fuzzy_3phase",
                    help="Run name for artifacts folder")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--base_lr", type=float, default=None,
                    help="Base learning rate (fuzzy default: 0.0001, baseline default: 0.001)")
    ap.add_argument("--no_fuzzy", action="store_true",
                    help="Disable fuzzy LR (baseline mode with fixed LR)")
    ap.add_argument("--ablation", action="store_true",
                    help="Ablation mode: same cosine schedule as fuzzy but scale=1.0 always. "
                         "Uses FuzzyYOLOCallback with use_fuzzy=False for fair comparison.")
    ap.add_argument("--no_three_phase", action="store_true",
                    help="Disable 3-phase strategy (use only fuzzy scaling)")
    ap.add_argument("--no_warmup", action="store_true",
                    help="Disable warm-up")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="Custom output directory (default: artifacts/runs/<run_name>)")
    ap.add_argument("--plateau", action="store_true",
                    help="ReduceLROnPlateau scheduler mode")
    ap.add_argument("--sgdr", action="store_true",
                    help="CosineAnnealingWarmRestarts (SGDR) scheduler mode")

    # FuzzyLRConfig hyperparameters (for grid search)
    ap.add_argument("--scale_min", type=float, default=None,
                    help="Fuzzy scale lower bound (default: 0.75)")
    ap.add_argument("--scale_max", type=float, default=None,
                    help="Fuzzy scale upper bound (default: 1.30)")
    ap.add_argument("--overfitting_threshold", type=float, default=None,
                    help="Overfitting detection threshold (default: 0.8)")
    ap.add_argument("--warmup_epochs", type=int, default=None,
                    help="Warmup epoch count (default: 3)")
    ap.add_argument("--lrf", type=float, default=None,
                    help="Cosine final LR ratio (default: 0.10)")
    ap.add_argument("--hysteresis_min", type=float, default=None,
                    help="Hysteresis min fraction (default: 0.08)")
    ap.add_argument("--hysteresis_max", type=float, default=None,
                    help="Hysteresis max fraction (default: 0.10)")
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # Validate mutually exclusive flags
    exclusive_modes = sum([args.no_fuzzy, args.ablation, args.plateau, args.sgdr])
    if exclusive_modes > 1:
        print("[ERROR] --no_fuzzy, --ablation, --plateau, --sgdr birbirini dislar. Sadece birini kullan.")
        return

    # Resolve base_lr: None means use mode-specific default
    args.base_lr_explicit = args.base_lr is not None
    if args.base_lr is None:
        # SGD YOLO default: 0.01 (hem fuzzy hem baseline için)
        args.base_lr = 0.01

    print("=" * 80)
    print("YOLOV8 TRAINING WITH ENHANCED FUZZY LR SCHEDULER")
    print("=" * 80)
    if args.ablation:
        mode_str = "Ablation (Cosine-only, no fuzzy perturbation)"
    elif args.no_fuzzy:
        mode_str = "Baseline (YOLO native scheduler)"
    elif args.plateau:
        mode_str = "ReduceLROnPlateau"
    elif args.sgdr:
        mode_str = "SGDR (CosineAnnealingWarmRestarts)"
    else:
        mode_str = "Fuzzy LR with Cosine Macro Schedule"
    print(f"Mode: {mode_str}")
    print(f"Base LR: {args.base_lr}")
    print(f"Total Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch}")
    print(f"Run Name: {args.run_name}")
    print("=" * 80)

    # Model
    yolov8Class = Yolov8()
    model_yolov8 = yolov8Class.return_model()

    # Artifact directory (mutlak yol — YOLO relative path'i runs/detect/ ile prefiksliyor)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.output_dir is not None:
        run_dir = os.path.join(project_root, args.output_dir)
    else:
        run_dir = os.path.join(project_root, "artifacts", "runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Logger
    step_logger = StepLogger(run_dir)
    step_logger.save_params({"cli_args": vars(args), "note": "Enhanced Fuzzy LR with 3-Phase Strategy"})

    # ========================================================================
    #  FUZZY LR CONFIG (Based on Baseline Experiments)
    # ========================================================================
    #
    # Baseline Experiment Results:
    # - LR=0.00001: Best mAP (0.7997) but slow (36 epochs)
    # - LR=0.0001:  Fast convergence (0.7662 @ 5 epochs)
    # - LR=0.001:   Poor performance (0.5021)
    # - LR=0.01+:   Training collapse
    #
    # Strategy: 3-Phase
    # - Phase 1 (0-25%):     Start at 0.0001 for rapid exploration
    # - Phase 2 (25-62.5%):  Decay to 0.00003 for consolidation
    # - Phase 3 (62.5-100%): Fine-tune at 0.00001 for best performance
    #
    # Goal: Achieve 0.79+ mAP in 35-40 epochs (vs 36 in baseline)
    # ========================================================================

    # Build config from CLI overrides (None values use dataclass defaults)
    cfg_overrides = {}
    if args.scale_min is not None:       cfg_overrides["scale_min"] = args.scale_min
    if args.scale_max is not None:       cfg_overrides["scale_max"] = args.scale_max
    if args.overfitting_threshold is not None: cfg_overrides["overfitting_threshold"] = args.overfitting_threshold
    if args.warmup_epochs is not None:   cfg_overrides["warmup_epochs"] = args.warmup_epochs
    if args.lrf is not None:             cfg_overrides["lrf"] = args.lrf
    if args.hysteresis_min is not None:  cfg_overrides["hysteresis_frac_min"] = args.hysteresis_min
    if args.hysteresis_max is not None:  cfg_overrides["hysteresis_frac_max"] = args.hysteresis_max
    fuzzy_cfg = FuzzyLRConfig(**cfg_overrides)

    # Register callbacks
    if args.plateau:
        print("[INFO] ReduceLROnPlateau mode")
        pl_cb = PlateauLRCallback(
            run_dir=run_dir,
            step_logger=step_logger,
            base_lr=args.base_lr,
            batch_size=args.batch,
            total_epochs=args.epochs,
        )
        model_yolov8.add_callback("on_train_start",        pl_cb.on_train_start)
        model_yolov8.add_callback("on_train_batch_start",  pl_cb.on_train_batch_start)
        model_yolov8.add_callback("on_train_batch_end",    pl_cb.on_train_batch_end)
        model_yolov8.add_callback("on_fit_epoch_end",      pl_cb.on_fit_epoch_end)
        model_yolov8.add_callback("on_train_end",          pl_cb.on_train_end)
        print("[INFO] Plateau callbacks registered")

    elif args.sgdr:
        print("[INFO] SGDR (CosineAnnealingWarmRestarts) mode")
        sgdr_cb = SGDRCallback(
            run_dir=run_dir,
            step_logger=step_logger,
            base_lr=args.base_lr,
            batch_size=args.batch,
            total_epochs=args.epochs,
        )
        model_yolov8.add_callback("on_train_start",        sgdr_cb.on_train_start)
        model_yolov8.add_callback("on_train_batch_start",  sgdr_cb.on_train_batch_start)
        model_yolov8.add_callback("on_train_batch_end",    sgdr_cb.on_train_batch_end)
        model_yolov8.add_callback("on_fit_epoch_end",      sgdr_cb.on_fit_epoch_end)
        model_yolov8.add_callback("on_train_end",          sgdr_cb.on_train_end)
        print("[INFO] SGDR callbacks registered")

    elif not args.no_fuzzy:
        # Fuzzy mode veya Ablation mode — ikisi de FuzzyYOLOCallback kullanır
        use_fuzzy = not args.ablation
        if use_fuzzy:
            print("[INFO] Fuzzy LR ENABLED")
        else:
            print("[INFO] Ablation mode — same cosine schedule, no fuzzy perturbation")

        cb = FuzzyYOLOCallback(
            run_dir=run_dir,
            step_logger=step_logger,
            base_lr=args.base_lr,
            cfg=fuzzy_cfg,
            batch_size=args.batch,
            total_epochs=args.epochs
        )
        cb.use_fuzzy = use_fuzzy

        print(f"\n[DEBUG] Registering {'fuzzy' if use_fuzzy else 'ablation'} callbacks...")
        model_yolov8.add_callback("on_train_start",       cb.on_train_start)
        model_yolov8.add_callback("on_train_batch_start", cb.on_train_batch_start)
        model_yolov8.add_callback("on_train_batch_end",   cb.on_train_batch_end)
        model_yolov8.add_callback("on_fit_epoch_end",     cb.on_fit_epoch_end)
        model_yolov8.add_callback("on_train_end",         cb.on_train_end)
        print(f"[DEBUG] {'Fuzzy' if use_fuzzy else 'Ablation'} callbacks registered successfully")

    else:
        print("[INFO] Baseline mode - YOLO handles LR scheduling")
        bl_cb = BaselineLoggerCallback(
            run_dir=run_dir,
            step_logger=step_logger,
            batch_size=args.batch,
            total_epochs=args.epochs
        )
        model_yolov8.add_callback("on_train_start",     bl_cb.on_train_start)
        model_yolov8.add_callback("on_train_batch_end", bl_cb.on_train_batch_end)
        model_yolov8.add_callback("on_fit_epoch_end",   bl_cb.on_fit_epoch_end)
        model_yolov8.add_callback("on_train_end",       bl_cb.on_train_end)
        print("[INFO] Baseline logger callback registered (LR untouched)")

    try:
        # Training
        print("\n" + "=" * 80)
        print("STARTING TRAINING...")
        print("=" * 80)

        """# Shared parameters for fair comparison between baseline and fuzzy
        common_train_kwargs = dict(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            optimizer='AdamW',
            momentum=0.937,
            weight_decay=0.0005,
            amp=True,
            close_mosaic=10,    # Disable mosaic last 10 epochs (YOLO default)
            workers=2,          # Windows bellek tasarrufu (varsayilan 8 cok fazla)
            project=os.path.join(project_root, "artifacts", "runs"),
            name=args.run_name,
            exist_ok=True,
            verbose=True,
            plots=True,
            save_period=-1,
        )"""

        common_train_kwargs = dict(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            optimizer='SGD',
            momentum=0.937,
            weight_decay=0.0005,
            amp=True,
            close_mosaic=10,
            workers=2,
            patience=100,               # Erken durdurma yok (best run ile aynı)
            seed=args.seed,             # YOLO'nun iç seed'i de sabitle
            deterministic=True,
            project=os.path.dirname(run_dir),
            name=os.path.basename(run_dir),
            exist_ok=True,
            verbose=True,
            plots=True,
            save_period=-1,
        )

        # Callback'in LR'yi yonetip yonetmedigine gore YOLO scheduler ayari
        callback_owns_lr = args.plateau or args.sgdr or (not args.no_fuzzy)

        if args.no_fuzzy:
            # Baseline: YOLO handles LR scheduling (warmup + cosine decay)
            baseline_kwargs = dict(
                **common_train_kwargs,
                lrf=0.01,
                cos_lr=True,
                warmup_epochs=3.0,
                warmup_momentum=0.8,
                warmup_bias_lr=0.0,
            )
            if args.base_lr_explicit:
                baseline_kwargs["lr0"] = args.base_lr
            model_yolov8.train(**baseline_kwargs)

        elif callback_owns_lr:
            # Fuzzy, Ablation, Plateau, SGDR:
            # YOLO scheduler tamamen devre disi — callback her adimda LR'yi yazar.
            # lr0: baslangic degeri (callback warmup bitmeden zaten override eder)
            # lrf=1.0, cos_lr=False, warmup_epochs=0: YOLO hicbir sey yapmasin
            model_yolov8.train(
                **common_train_kwargs,
                lr0=args.base_lr,
                lrf=1.0,
                cos_lr=False,
                warmup_epochs=0,
                warmup_momentum=0.8,
                warmup_bias_lr=0.0,
            )
    finally:
        step_logger.close()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {run_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
