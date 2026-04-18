# train/models/yolo_fuzzy_callback.py
#
# v4: Phase-Aware Scale-Only Micro-Navigator — YOLO Entegrasyon Callback'i
#
# FuzzyLRScheduler.step() 2 deger donduruyor: (lr, scale)
# Cosine standart ilerler. Fuzzy sadece scale uzerine mikro ayar ekler.
# delta_loss RELATIVE olarak hesaplanir: (EMA_new - EMA_old) / |EMA_old|

import os, time
from scheduler.fuzzy_lr import (
    FuzzyLRScheduler, FuzzyLRConfig, FuzzyInputs,
    EMA, DualEMA, MovingAverage,
    PlateauTracker, grad_norm, clamp
)
from scheduler.exp_logger import StepLogger

# ============================================================================
#  PHASE-AWARE SCALE-ONLY CALLBACK
# ============================================================================
class FuzzyYOLOCallback:
    """
    v4: Phase-Aware Scale-Only Callback.

    Tek cikisli fuzzy controller (scale) ile
    cosine decay uzerine mikro perturbasyon ekler.

    Cosine decay standart ilerler (baseline ile birebir ayni).
    Fuzzy sistem:
      scale  [0.90, 1.10]  — mikro LR perturbasyonu (step-level)

    Speed/gate KALDIRILDI.
    Relative delta_loss: faz-bagimsiz sinyal.
    """

    def __init__(self, run_dir, step_logger, base_lr=1e-3,
                 cfg: FuzzyLRConfig = FuzzyLRConfig(),
                 batch_size=16, total_epochs=50):
        self.base_lr = base_lr
        self.cfg = cfg
        self.batch_size = batch_size
        self.run_dir = run_dir
        self.total_epochs = total_epochs

        # State tracking
        # Oneri 1: Cifte EMA (MACD benzeri) — delta_loss sinyali icin
        self.loss_dual_ema = DualEMA(beta_fast=0.80, beta_slow=0.95)
        # Eski tek EMA korunuyor (train_loss_ema icin kullanilir)
        self.loss_ema = EMA(0.8)
        # Oneri 2: grad_norm N=5 hareketli ortalama
        self._gnorm_ma = MovingAverage(n=5)
        # Secenek A: son gecerli grad_norm cache'i
        self._last_valid_gnorm = None
        self.plateau = PlateauTracker(rel_tol=0.002, grace_epochs=0)
        self.scheduler = None
        self.global_step = 0
        self.logger = step_logger
        self._epoch_t0 = None
        self._optimizer = None
        self.use_fuzzy = True

        # Loss tracking for overfitting detection
        self.train_loss_ema = EMA(0.9)
        self.last_val_loss = None
        self.last_train_loss = None

        # Improvement tracking
        self.best_val_metric = 0.0

        # Current epoch
        self.current_epoch = 0

        # Epoch-average accumulators (status satiri icin)
        self._epoch_scales = []

        # Transition detection (kosullu 2. satir icin)
        self._prev_vtg_tag = None

    # ========================================================================
    #  LIFECYCLE CALLBACKS
    # ========================================================================

    def on_fit_start(self, trainer):
        self.on_train_start(trainer)

    def on_train_start(self, trainer):
        """Initialize phase-aware scale-only scheduler at training start."""
        self._optimizer = trainer.optimizer

        # Create scheduler
        self.scheduler = FuzzyLRScheduler(
            self._optimizer,
            base_lr=self.base_lr,
            cfg=self.cfg,
            total_epochs=self.total_epochs
        )

        # Log parameters
        params = {
            "mode": "fuzzy_augmented" if self.use_fuzzy else "ablation",
            "base_lr": self.base_lr,
            "total_epochs": self.total_epochs,
            "fuzzy_cfg": vars(self.cfg),
            "use_warmup": self.cfg.use_warmup,
            "architecture": "Phase-Aware Scale-Only (v4)",
            "ultralytics_args": getattr(trainer.args, "__dict__", {}),
        }

        try:
            self.logger.save_params(params)
        except Exception as e:
            print(f"[FuzzyCB] save_params error: {e}")

        self._epoch_t0 = time.perf_counter()

    def on_train_batch_start(self, trainer):
        """Capture gradient norm before optimizer step."""
        try:
            raw_gnorm = grad_norm(trainer.model)
            # Secenek A: 0.5 fallback degerini reddet, cache'den al
            if raw_gnorm is None or raw_gnorm <= 0.5 or not (0 < raw_gnorm < 1000):
                # Gecerli grad_norm yok (accumulate adimi) — onceki cache'i kullan
                raw_gnorm = self._last_valid_gnorm
            else:
                # Gecerli grad_norm — cache'i guncelle
                self._last_valid_gnorm = raw_gnorm

            if raw_gnorm is None:
                # Hic cache yoksa (ilk adimlar) — MF'ler icin nötr nokta
                raw_gnorm = 5.5

            # Oneri 2: N=5 hareketli ortalama uygula
            self._last_gnorm = self._gnorm_ma.update(raw_gnorm)

        except Exception as e:
            if self.global_step < 10:
                print(f"[ERROR] Gradient norm calculation failed: {e}")
            # Exception'da da cache'i dene
            cached = self._last_valid_gnorm
            self._last_gnorm = self._gnorm_ma.update(cached if cached is not None else 5.5)

    def on_train_batch_end(self, trainer):
        """Update LR after each batch using phase-aware scale-only controller."""

        # Extract loss
        try:
            loss = float(trainer.loss_items.sum()) if hasattr(trainer, "loss_items") else float(trainer.loss)
        except Exception:
            loss = float(getattr(trainer, "loss", 0.0))

        self.global_step += 1

        # Oneri 1: Cifte EMA ile relative delta_loss hesapla
        # (fast=0.80, slow=0.95) → MACD benzeri trend sinyali
        # Alternasyon orani ~%52 → ~%20, spike'lara duyarsiz
        rel_delta = self.loss_dual_ema.update(loss)

        # Eski tek EMA train_loss takibi icin devam ediyor
        self.loss_ema.update(loss)
        self.train_loss_ema.update(loss)
        self.last_train_loss = self.train_loss_ema.value

        # Get gradient norm
        gnorm = getattr(self, "_last_gnorm", 1.0)

        # Calculate epoch ratio
        epoch_ratio = self.current_epoch / self.total_epochs if self.total_epochs > 0 else 0.0

        # Calculate val/train gap (use last known val_loss)
        val_train_gap = 0.0
        if self.last_val_loss is not None and self.last_train_loss is not None:
            val_train_gap = self.last_val_loss - self.last_train_loss

        # Improvement rate: normalized plateau steps [0, 1]
        improvement_rate = min(self.plateau.steps / 10.0, 1.0)

        # Plateau steps (epoch-based)
        psteps = self.plateau.steps

        # Update LR via phase-aware scale-only scheduler or ablation
        if self.use_fuzzy:
            # Prepare fuzzy inputs (delta_loss is RELATIVE)
            fuzzy_inputs = FuzzyInputs(
                delta_loss=rel_delta,
                grad_norm=gnorm,
                plateau_steps=psteps,
                val_train_gap=val_train_gap,
                epoch_ratio=epoch_ratio,
                improvement_rate=improvement_rate
            )

            # Fuzzy step: 2 deger doner (lr, scale)
            lr, scale = self.scheduler.step(fuzzy_inputs)

            # Epoch-average accumulation
            self._epoch_scales.append(scale)

            # Get phase_base_lr for logging
            phase_base_lr = self.scheduler.get_phase_base_lr()

        else:
            # Ablation mode: scale=1.0, same cosine schedule
            scale = 1.0
            phase_base_lr = self.scheduler.get_phase_base_lr()
            lr = clamp(phase_base_lr, self.cfg.lr_min, self.cfg.lr_max)
            for pg in self._optimizer.param_groups:
                pg["lr"] = lr

            # Epoch-average accumulation (ablation: sabit degerler)
            self._epoch_scales.append(scale)

        # Log step metrics
        try:
            self.logger.log_step(
                step=self.global_step,
                epoch=self.current_epoch,
                loss=loss,
                delta_loss=rel_delta,
                grad_norm=gnorm,
                lr=lr,
                scale=scale,
                plateau_steps=psteps,
                batch_size=self.batch_size,
                accuracy=None,
                total_epochs=self.total_epochs,
                val_train_gap=val_train_gap,
                improvement_rate=improvement_rate,
                phase_base_lr=phase_base_lr,
                cosine_progress=self.scheduler.progress
            )
        except Exception as e:
            print(f"[FuzzyCB] log_step error: {e}")

    def on_fit_epoch_end(self, trainer):
        """
        Update metrics, plateau tracker and print status.
        """
        metrics = {}
        val_loss = None

        try:
            v = getattr(trainer.validator, "metrics", None)
            if v is not None:
                if hasattr(v, 'results_dict'):
                    results = v.results_dict
                    metrics.update({
                        "map50": float(results.get("metrics/mAP50(B)", 0.0)),
                        "map5095": float(results.get("metrics/mAP50-95(B)", 0.0)),
                        "precision": float(results.get("metrics/precision(B)", 0.0)),
                        "recall": float(results.get("metrics/recall(B)", 0.0)),
                    })
                    tloss = getattr(trainer, "tloss", None)
                    if tloss is not None:
                        try:
                            import torch
                            t = tloss if isinstance(tloss, torch.Tensor) else torch.tensor(tloss)
                            val_loss = float(t.sum())
                        except Exception:
                            val_loss = None
                elif isinstance(v, dict):
                    metrics.update({
                        "map50": float(v.get("metrics/mAP50(B)", v.get("map50", 0.0))),
                        "map5095": float(v.get("metrics/mAP50-95(B)", v.get("map", 0.0))),
                    })
                    val_loss = v.get("val/box_loss", None)
                else:
                    metrics.update({
                        "map50": float(getattr(v, "map50", 0.0)),
                        "map5095": float(getattr(v, "map", 0.0)),
                    })
        except Exception as e:
            print(f"[FuzzyCB] metric read failed: {e}")

        # Update epoch — set_epoch computes standard cosine progress
        self.current_epoch = getattr(trainer, "epoch", self.current_epoch + 1)
        if self.scheduler is not None:
            self.scheduler.set_epoch(self.current_epoch)

        if val_loss is not None:
            self.last_val_loss = float(val_loss)

        # --- METRIC UPDATE ---
        current_map50 = metrics.get("map50", 0)
        current_map5095 = metrics.get("map5095", 0)

        if current_map50 > 0:
            # PlateauTracker + Best: ikisi de saf mAP50 kullanir
            self.plateau.update(current_map50, epoch=self.current_epoch)
            if current_map50 > self.best_val_metric:
                self.best_val_metric = current_map50

        # Log to CSV
        t1 = time.perf_counter()
        try:
            self.logger.log_epoch(
                epoch=self.current_epoch, metrics=metrics, t_epoch=t1 - self._epoch_t0,
                total_epochs=self.total_epochs, train_loss=self.last_train_loss, val_loss=self.last_val_loss
            )
        except Exception:
            pass
        self._epoch_t0 = t1

        # --- Epoch ortalamalari hesapla ---
        avg_scale = sum(self._epoch_scales) / len(self._epoch_scales) if self._epoch_scales else 1.0

        # Accumulatorlari sifirla
        self._epoch_scales.clear()

        # --- STATUS ---
        if self.scheduler is not None:
            self._print_status(current_map50, current_map5095, avg_scale)

    def _print_status(self, current_map50: float, current_map5095: float,
                      avg_scale: float):
        """
        Kompakt tek satirlik terminal status ciktisi.
        Kosullu 2. satir: yeni best veya VTG degisiminde.
        """
        # --- Metrikleri topla ---
        if self._optimizer and self._optimizer.param_groups:
            effective_lr = float(self._optimizer.param_groups[0]["lr"])
        else:
            effective_lr = self.scheduler.get_phase_base_lr()

        psteps = self.plateau.steps
        best_map50 = self.best_val_metric
        progress = self.scheduler.progress
        is_new_best = (psteps == 0 and current_map50 > 0)

        # --- VTG durumu (3 karakter) ---
        vtg = self.last_val_loss - self.last_train_loss if (self.last_val_loss and self.last_train_loss) else 0.0
        if vtg > 0.55:
            vtg_tag = "OVF"
        elif vtg > 0.25:
            vtg_tag = "WRN"
        else:
            vtg_tag = "OK"

        # --- Durum ikonu ---
        if is_new_best:
            status_icon = "**"; status_msg = "NEW BEST"
        elif psteps < 5:
            status_icon = ".."; status_msg = f"Wait({psteps}ep)"
        elif psteps < 8 * (self.total_epochs / 50.0):
            status_icon = "!!"; status_msg = f"Stag({psteps}ep)"
        else:
            if vtg_tag == "OK":
                status_icon = "^^"; status_msg = f"Explore({psteps}ep)"
            elif vtg_tag == "WRN":
                status_icon = "~~"; status_msg = f"Caution({psteps}ep)"
            else:
                status_icon = "vv"; status_msg = f"Brake({psteps}ep)"

        # --- mAP50 gosterimi ---
        if is_new_best:
            map_display = f"{current_map50:.4f} (^best)"
        else:
            gap = current_map50 - best_map50 if best_map50 > 0 else 0.0
            map_display = f"{current_map50:.4f} (best:{best_map50:.4f} {gap:+.4f})"

        # --- Cikti ---
        mode_tag = "FZv4" if self.use_fuzzy else "ABL"
        display_epoch = self.current_epoch + 1  # 1-indexed

        # Satir 1: Kompakt tek satir
        print(
            f"[{mode_tag}] Ep {display_epoch:>2}/{self.total_epochs} "
            f"[{status_icon}] {status_msg:<16}| "
            f"LR {effective_lr:.6f} (s:x{avg_scale:.3f}) | "
            f"mAP50 {map_display} | "
            f"P:{psteps} VTG:{vtg_tag} cos:{progress*100:.0f}%"
        )

        # --- Kosullu 2. satir ---
        show_second = False
        detail_parts = []

        # Yeni best (saf mAP50)
        if is_new_best:
            detail_parts.append(f"New best mAP50: {current_map50:.4f}")
            show_second = True

        # VTG durum degisimi
        if self._prev_vtg_tag is not None and vtg_tag != self._prev_vtg_tag:
            detail_parts.append(f"VTG: {self._prev_vtg_tag}->{vtg_tag}")
            show_second = True

        if show_second:
            print(f"       >>> {' | '.join(detail_parts)}")

        # State guncelle (sonraki epoch icin karsilastirma)
        self._prev_vtg_tag = vtg_tag

    def on_train_end(self, trainer):
        """Save summary at training end."""
        try:
            self.logger.save_summary({
                "total_steps": self.global_step,
                "total_epochs": self.current_epoch,
                "best_map50": self.best_val_metric,
                "final_plateau_steps": self.plateau.steps,
                "final_progress": self.scheduler.progress if self.scheduler else 0.0,
                "events_up": self.logger.events_up,
                "events_down": self.logger.events_down,
                "use_fuzzy": self.use_fuzzy,
                "mode": "fuzzy_augmented" if self.use_fuzzy else "ablation",
                "architecture": "Phase-Aware Scale-Only (v4)",
                "use_warmup": self.cfg.use_warmup,
            })
            self.logger.close()
        except Exception as e:
            print(f"[FuzzyCB] save_summary error: {e}")

    close = on_train_end
