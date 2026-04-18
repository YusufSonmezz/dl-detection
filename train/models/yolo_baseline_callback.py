# train/models/yolo_baseline_callback.py
import time
from scheduler.fuzzy_lr import EMA
from scheduler.exp_logger import StepLogger


class BaselineLoggerCallback:
    """
    Lightweight logging-only callback for baseline (no-fuzzy) training.

    - Does NOT touch the optimizer or learning rate in any way.
    - Reads metrics from YOLO's trainer object and writes to steps.csv / epochs.csv
      using the same StepLogger format as FuzzyYOLOCallback, enabling direct comparison.

    Fixed values for fuzzy-specific columns:
        scale        = 1.0   (YOLO controls LR, no fuzzy scaling)
        delta_loss   = 0.0   (EMA delta not meaningful without fuzzy context)
        grad_norm    = 1.0   (not calculated, placeholder)
        plateau_steps = 0    (fuzzy-specific concept)
        events_up/down = 0   (fuzzy-specific counters)
    """

    def __init__(self, run_dir: str, step_logger: StepLogger,
                 batch_size: int = 16, total_epochs: int = 50):
        self.run_dir = run_dir
        self.logger = step_logger
        self.batch_size = batch_size
        self.total_epochs = total_epochs

        self.global_step = 0
        self.current_epoch = 0
        self._epoch_t0 = None
        self._optimizer = None

        # Loss tracking
        self.train_loss_ema = EMA(beta=0.9)
        self.last_val_loss = None
        self.best_map50 = 0.0

    # =========================================================================
    #  LIFECYCLE CALLBACKS
    # =========================================================================

    def on_train_start(self, trainer):
        self._optimizer = trainer.optimizer
        self._epoch_t0 = time.perf_counter()

        params = {
            "mode": "baseline",
            "total_epochs": self.total_epochs,
            "batch_size": self.batch_size,
            "ultralytics_args": getattr(trainer.args, "__dict__", {}),
        }
        try:
            self.logger.save_params(params)
        except Exception as e:
            print(f"[BaselineCB] save_params error: {e}")

    def on_train_batch_end(self, trainer):
        # Extract batch loss
        try:
            loss = float(trainer.loss_items.sum()) if hasattr(trainer, "loss_items") \
                else float(trainer.loss)
        except Exception:
            loss = float(getattr(trainer, "loss", 0.0))

        self.global_step += 1
        self.train_loss_ema.update(loss)

        # Read current LR from optimizer (YOLO manages this — warmup, cosine, etc.)
        try:
            lr = float(self._optimizer.param_groups[0]["lr"])
        except Exception:
            lr = 0.0

        # Epoch info
        self.current_epoch = getattr(trainer, "epoch", self.current_epoch)
        epoch_ratio = self.current_epoch / self.total_epochs if self.total_epochs > 0 else 0.0

        # val/train gap (last known val_loss vs current train EMA)
        val_train_gap = 0.0
        if self.last_val_loss is not None:
            val_train_gap = self.last_val_loss - self.train_loss_ema.value

        # improvement_rate: distance from best mAP50 (normalized)
        improvement_rate = 0.0
        if self.best_map50 > 1e-6:
            current_best = self.best_map50
            improvement_rate = (current_best - current_best) / current_best  # = 0 until epoch end

        try:
            self.logger.log_step(
                step=self.global_step,
                epoch=self.current_epoch,
                loss=loss,
                delta_loss=0.0,         # not meaningful in baseline
                grad_norm=1.0,          # not calculated
                lr=lr,
                scale=1.0,              # YOLO controls LR, no scaling
                plateau_steps=0,        # fuzzy-specific
                batch_size=self.batch_size,
                accuracy=None,
                total_epochs=self.total_epochs,
                val_train_gap=val_train_gap,
                improvement_rate=improvement_rate,
                phase_base_lr=lr,
            )
        except Exception as e:
            print(f"[BaselineCB] log_step error: {e}")

    def on_fit_epoch_end(self, trainer):
        metrics = {}
        val_loss = None

        # Read validation metrics from YOLO's validator
        try:
            v = getattr(trainer.validator, "metrics", None)
            if v is not None:
                if hasattr(v, "results_dict"):
                    results = v.results_dict
                    metrics = {
                        "map50":     float(results.get("metrics/mAP50(B)", 0.0)),
                        "map5095":   float(results.get("metrics/mAP50-95(B)", 0.0)),
                        "precision": float(results.get("metrics/precision(B)", 0.0)),
                        "recall":    float(results.get("metrics/recall(B)", 0.0)),
                    }
                    # val loss: trainer.tloss is a tensor [box, cls, dfl]
                    tloss = getattr(trainer, "tloss", None)
                    if tloss is not None:
                        try:
                            import torch
                            t = tloss if isinstance(tloss, torch.Tensor) else torch.tensor(tloss)
                            val_loss = float(t.sum())
                        except Exception:
                            val_loss = None
                elif isinstance(v, dict):
                    metrics = {
                        "map50":   float(v.get("metrics/mAP50(B)", v.get("map50", 0.0))),
                        "map5095": float(v.get("metrics/mAP50-95(B)", v.get("map", 0.0))),
                    }
                    val_loss = v.get("val/box_loss", None)
                else:
                    metrics = {
                        "map50":   float(getattr(v, "map50", 0.0)),
                        "map5095": float(getattr(v, "map", 0.0)),
                    }
        except Exception as e:
            print(f"[BaselineCB] metric read error: {e}")

        self.current_epoch = getattr(trainer, "epoch", self.current_epoch + 1)

        if val_loss is not None:
            self.last_val_loss = float(val_loss)

        # Update best mAP50
        current_map50 = metrics.get("map50", 0.0)
        if current_map50 > self.best_map50:
            self.best_map50 = current_map50

        t1 = time.perf_counter()
        try:
            self.logger.log_epoch(
                epoch=self.current_epoch,
                metrics=metrics,
                t_epoch=t1 - self._epoch_t0,
                total_epochs=self.total_epochs,
                train_loss=self.train_loss_ema.value,
                val_loss=self.last_val_loss,
            )
        except Exception as e:
            print(f"[BaselineCB] log_epoch error: {e}")

        self._epoch_t0 = t1

    def on_train_end(self, trainer):
        try:
            self.logger.save_summary({
                "total_steps": self.global_step,
                "total_epochs": self.current_epoch,
                "best_map50": self.best_map50,
                "use_fuzzy": False,
            })
            self.logger.close()
        except Exception as e:
            print(f"[BaselineCB] save_summary error: {e}")
