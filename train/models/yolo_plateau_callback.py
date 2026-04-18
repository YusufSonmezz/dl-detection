# train/models/yolo_plateau_callback.py
#
# ReduceLROnPlateau callback — YOLO entegrasyonu
#
# YOLO'nun kendi scheduler'i devre disi birakilir (lrf=1.0, cos_lr=False).
# Bu callback optimizer.param_groups[0]["lr"]'yi dogrudan yazar.
# Logging formati BaselineLoggerCallback ile birebir ayni.

import time
import torch
from scheduler.fuzzy_lr import EMA
from scheduler.exp_logger import StepLogger


class PlateauLRCallback:
    """
    PyTorch ReduceLROnPlateau ile YOLO entegrasyonu.

    Strateji:
      - Warmup: ilk warmup_epochs epoch lineer LR artisi (baseline ile esit kosullar)
      - Warmup sonrasi: her epoch sonu mAP50 uzerinden plateau.step()
      - Plateau algilayinca: lr = lr * factor (default 0.5)
      - patience: kac epoch iyilesme olmadan beklesin
      - min_lr: minimum LR siniri

    Tez karsilastirmasi: "kural-tabanli adaptif" kategorisi.
    """

    def __init__(self, run_dir: str, step_logger: StepLogger,
                 base_lr: float = 0.01,
                 batch_size: int = 16,
                 total_epochs: int = 100,
                 warmup_epochs: int = 3,
                 factor: float = 0.5,
                 patience: int = 5,
                 min_lr: float = 1e-6,
                 threshold: float = 0.02):

        self.run_dir = run_dir
        self.logger = step_logger
        self.base_lr = base_lr
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold

        self.global_step = 0
        self.current_epoch = 0
        self._epoch_t0 = None
        self._optimizer = None
        self._scheduler = None
        self._current_lr = base_lr  # YOLO reset'ine karsi korunan LR durumu

        self.train_loss_ema = EMA(beta=0.9)
        self.last_val_loss = None
        self.best_map50 = 0.0

    def on_train_epoch_start(self, trainer):
        pass  # on_train_batch_start'ta override ediliyor (epoch_start YOLO scheduler.step()'den once calisar)

    def on_train_batch_start(self, trainer):
        """YOLO'nun scheduler.step() LR reset'ini epoch'un ilk batch'inde override et."""
        if self._optimizer is not None and getattr(trainer, 'epoch', -1) != getattr(self, '_last_override_epoch', -2):
            self._last_override_epoch = getattr(trainer, 'epoch', -1)
            for pg in self._optimizer.param_groups:
                pg["lr"] = self._current_lr

    def on_train_start(self, trainer):
        self._optimizer = trainer.optimizer

        # Warmup baslangici: LR'yi cok dusuk ayarla
        self._current_lr = self.base_lr * (1.0 / max(self.warmup_epochs, 1))
        for pg in self._optimizer.param_groups:
            pg["lr"] = self._current_lr

        # Warmup bittikten sonra _current_lr'yi base_lr'ye esitle
        self._current_lr = self.base_lr

        # ReduceLROnPlateau: mAP50 maksimize edildiginden mode="max"
        # threshold=0.02: kucuk spikelar "iyilesme" sayilmaz (%2 artis sarti)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            mode="max",
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
            threshold=self.threshold,
            threshold_mode="rel",
            verbose=False,
        )

        self.logger.save_params({
            "mode": "plateau",
            "base_lr": self.base_lr,
            "total_epochs": self.total_epochs,
            "warmup_epochs": self.warmup_epochs,
            "plateau_cfg": {
                "factor": self.factor,
                "patience": self.patience,
                "min_lr": self.min_lr,
                "threshold": self.threshold,
            },
            "ultralytics_args": getattr(trainer.args, "__dict__", {}),
        })
        self._epoch_t0 = time.perf_counter()

    def on_train_batch_end(self, trainer):
        try:
            loss = float(trainer.loss_items.sum()) if hasattr(trainer, "loss_items") \
                else float(trainer.loss)
        except Exception:
            loss = float(getattr(trainer, "loss", 0.0))

        self.global_step += 1
        self.train_loss_ema.update(loss)
        self.current_epoch = getattr(trainer, "epoch", self.current_epoch)

        # Warmup: lineer LR artisi
        if self.current_epoch < self.warmup_epochs:
            warmup_factor = (self.current_epoch + 1) / max(self.warmup_epochs, 1)
            self._current_lr = self.base_lr * warmup_factor
            for pg in self._optimizer.param_groups:
                pg["lr"] = self._current_lr

        try:
            lr = float(self._optimizer.param_groups[0]["lr"])
        except Exception:
            lr = self.base_lr

        val_train_gap = 0.0
        if self.last_val_loss is not None:
            val_train_gap = self.last_val_loss - self.train_loss_ema.value

        epoch_ratio = self.current_epoch / self.total_epochs if self.total_epochs > 0 else 0.0

        try:
            self.logger.log_step(
                step=self.global_step,
                epoch=self.current_epoch,
                loss=loss,
                delta_loss=0.0,
                grad_norm=1.0,
                lr=lr,
                scale=1.0,
                plateau_steps=0,
                batch_size=self.batch_size,
                accuracy=None,
                total_epochs=self.total_epochs,
                val_train_gap=val_train_gap,
                improvement_rate=0.0,
                phase_base_lr=lr,
            )
        except Exception as e:
            print(f"[PlateauCB] log_step error: {e}")

    def on_fit_epoch_end(self, trainer):
        metrics = {}
        val_loss = None

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
                    tloss = getattr(trainer, "tloss", None)
                    if tloss is not None:
                        try:
                            t = tloss if isinstance(tloss, torch.Tensor) else torch.tensor(tloss)
                            val_loss = float(t.sum())
                        except Exception:
                            val_loss = None
                elif isinstance(v, dict):
                    metrics = {
                        "map50":   float(v.get("metrics/mAP50(B)", v.get("map50", 0.0))),
                        "map5095": float(v.get("metrics/mAP50-95(B)", v.get("map", 0.0))),
                    }
                else:
                    metrics = {
                        "map50":   float(getattr(v, "map50", 0.0)),
                        "map5095": float(getattr(v, "map", 0.0)),
                    }
        except Exception as e:
            print(f"[PlateauCB] metric read error: {e}")

        self.current_epoch = getattr(trainer, "epoch", self.current_epoch + 1)

        if val_loss is not None:
            self.last_val_loss = float(val_loss)

        current_map50 = metrics.get("map50", 0.0)
        if current_map50 > self.best_map50:
            self.best_map50 = current_map50

        # Warmup bittikten sonra plateau scheduler'i guncelle
        if self.current_epoch >= self.warmup_epochs and current_map50 > 0:
            self._scheduler.step(current_map50)
            # Plateau scheduler LR'yi dusurmus olabilir — _current_lr'yi guncelle
            self._current_lr = float(self._optimizer.param_groups[0]["lr"])

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
            print(f"[PlateauCB] log_epoch error: {e}")

        self._epoch_t0 = t1

        try:
            lr = float(self._optimizer.param_groups[0]["lr"])
        except Exception:
            lr = 0.0
        print(f"[Plateau] Ep {self.current_epoch+1:>3}/{self.total_epochs} | "
              f"LR {lr:.6f} | mAP50 {current_map50:.4f} (best:{self.best_map50:.4f})")

    def on_train_end(self, trainer):
        try:
            self.logger.save_summary({
                "total_steps": self.global_step,
                "total_epochs": self.current_epoch,
                "best_map50": self.best_map50,
                "use_fuzzy": False,
                "mode": "plateau",
            })
            self.logger.close()
        except Exception as e:
            print(f"[PlateauCB] save_summary error: {e}")
