# train/models/yolo_sgdr_callback.py
#
# CosineAnnealingWarmRestarts (SGDR) callback — YOLO entegrasyonu
#
# YOLO'nun kendi scheduler'i devre disi birakilir (lrf=1.0, cos_lr=False).
# Bu callback optimizer.param_groups[0]["lr"]'yi dogrudan yazar.
# Logging formati BaselineLoggerCallback ile birebir ayni.
#
# Referans: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with
# Warm Restarts", ICLR 2017.

import math
import time
import torch
from scheduler.fuzzy_lr import EMA
from scheduler.exp_logger import StepLogger


class SGDRCallback:
    """
    CosineAnnealingWarmRestarts (SGDR) ile YOLO entegrasyonu.

    Strateji:
      - Warmup: ilk warmup_epochs epoch lineer LR artisi
      - Warmup sonrasi: SGDR cosine cycle, her T_0 epoch'ta restart
      - T_mult: her restart sonrasi periyot T_mult kati uzar
      - eta_min: minimum LR siniri (= base_lr * lrf)

    Parametreler (100 epoch, baseline ile karsilastirma):
      T_0 = 10        ilk cosine periyodu (epoch)
      T_mult = 2      restart sonrasi periyot 2x uzar: 10, 20, 40, 80...
      eta_min = 1e-4  cosine tabanı (= base_lr * 0.01)

    Tez karsilastirmasi: "periyodik adaptif" kategorisi (literatur yontemi).
    """

    def __init__(self, run_dir: str, step_logger: StepLogger,
                 base_lr: float = 0.01,
                 batch_size: int = 16,
                 total_epochs: int = 100,
                 warmup_epochs: int = 3,
                 T_0: int = 10,
                 T_mult: int = 2,
                 eta_min: float = 1e-4):

        self.run_dir = run_dir
        self.logger = step_logger
        self.base_lr = base_lr
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

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

        # Warmup baslangici
        self._current_lr = self.base_lr * (1.0 / max(self.warmup_epochs, 1))
        for pg in self._optimizer.param_groups:
            pg["lr"] = self._current_lr

        # SGDR: epoch-level scheduler (step_size=1 epoch)
        # last_epoch=-1 -> ilk step() cagrisinda epoch 0 baslar
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self._optimizer,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.eta_min,
            last_epoch=-1,
        )

        self.logger.save_params({
            "mode": "sgdr",
            "base_lr": self.base_lr,
            "total_epochs": self.total_epochs,
            "warmup_epochs": self.warmup_epochs,
            "sgdr_cfg": {
                "T_0": self.T_0,
                "T_mult": self.T_mult,
                "eta_min": self.eta_min,
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
            print(f"[SGDRCB] log_step error: {e}")

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
            print(f"[SGDRCB] metric read error: {e}")

        self.current_epoch = getattr(trainer, "epoch", self.current_epoch + 1)

        if val_loss is not None:
            self.last_val_loss = float(val_loss)

        current_map50 = metrics.get("map50", 0.0)
        if current_map50 > self.best_map50:
            self.best_map50 = current_map50

        # Warmup bittikten sonra SGDR scheduler'i guncelle
        if self.current_epoch >= self.warmup_epochs:
            sgdr_epoch = self.current_epoch - self.warmup_epochs
            self._scheduler.step(sgdr_epoch)
            # SGDR yeni LR'yi optimizer'a yazmis olacak — _current_lr'yi guncelle
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
            print(f"[SGDRCB] log_epoch error: {e}")

        self._epoch_t0 = t1

        try:
            lr = float(self._optimizer.param_groups[0]["lr"])
        except Exception:
            lr = 0.0

        # Hangi restart periyodundayiz?
        sgdr_epoch = max(self.current_epoch - self.warmup_epochs, 0)
        t_cur, t_i = self._get_cycle_info(sgdr_epoch)
        print(f"[SGDR]    Ep {self.current_epoch+1:>3}/{self.total_epochs} | "
              f"LR {lr:.6f} | mAP50 {current_map50:.4f} (best:{self.best_map50:.4f}) | "
              f"cycle {t_cur}/{t_i}")

    def _get_cycle_info(self, epoch: int):
        """Hangi SGDR dongu icinde oldugumuz (t_cur/T_i)."""
        T_0, T_mult = self.T_0, self.T_mult
        t_cur = epoch
        t_i = T_0
        while t_cur >= t_i:
            t_cur -= t_i
            t_i = int(t_i * T_mult)
        return t_cur, t_i

    def on_train_end(self, trainer):
        try:
            self.logger.save_summary({
                "total_steps": self.global_step,
                "total_epochs": self.current_epoch,
                "best_map50": self.best_map50,
                "use_fuzzy": False,
                "mode": "sgdr",
            })
            self.logger.close()
        except Exception as e:
            print(f"[SGDRCB] save_summary error: {e}")
