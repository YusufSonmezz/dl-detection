# train/scheduler/exp_logger.py
import csv, json, os, time, socket, uuid

class StepLogger:
    """
    Enhanced logger for step-level and epoch-level metrics

    New features:
    - Logs val/train gap for overfitting detection
    - Logs epoch progress ratio for phase-aware scheduling
    - Logs improvement rate for convergence monitoring
    - Logs phase_base_lr for 3-phase strategy tracking
    """

    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.step_path = os.path.join(run_dir, "steps.csv")
        self.epoch_path = os.path.join(run_dir, "epochs.csv")
        self.summary_path = os.path.join(run_dir, "run_summary.json")
        self._t0 = time.perf_counter()
        self._last_t = self._t0

        self._step_fh = open(self.step_path, "w", newline="", encoding="utf-8")
        self._epoch_fh = open(self.epoch_path, "w", newline="", encoding="utf-8")

        # Step log fields (v4: scale-only, no gate/speed)
        self.step_writer = csv.DictWriter(self._step_fh, fieldnames=[
            "step", "epoch", "epoch_ratio",
            "loss", "delta_loss", "grad_norm",
            "lr", "phase_base_lr", "scale",
            "cosine_progress",
            "plateau_steps", "val_train_gap", "improvement_rate",
            "accuracy", "imgs_per_sec", "step_time", "since_start",
            "events_up", "events_down"
        ])

        # Epoch log fields (enhanced)
        self.epoch_writer = csv.DictWriter(self._epoch_fh, fieldnames=[
            "epoch", "epoch_ratio",
            "map50", "map5095", "precision", "recall",
            "train_loss", "val_loss", "val_train_gap",
            "best_map50", "improvement_rate",
            "time_epoch_s"
        ])

        self.step_writer.writeheader()
        self.epoch_writer.writeheader()

        self.events_up = 0
        self.events_down = 0

        # Track best metrics for improvement rate calculation
        self.best_map50 = 0.0

    # ========================================================================
    #  STEP-LEVEL LOGGING
    # ========================================================================
    def log_step(self, step, epoch, loss, delta_loss, grad_norm, lr, scale,
                 plateau_steps, batch_size, n_imgs=None, accuracy=None,
                 total_epochs=50, val_train_gap=0.0, improvement_rate=0.0,
                 phase_base_lr=None, cosine_progress=0.0):
        """
        Log step-level metrics (v4: scale-only).

        Params:
            cosine_progress: Cosine decay progress [0, 1]
        """
        t = time.perf_counter()
        step_time = t - self._last_t
        self._last_t = t

        imgs = n_imgs if n_imgs is not None else batch_size
        imgs = max(1, imgs)
        ips = imgs / step_time if step_time > 1e-9 else 0.0

        # Scale event counters
        if scale > 1.01:
            self.events_up += 1
        if scale < 0.99:
            self.events_down += 1

        # Accuracy handling
        try:
            acc_val = "" if accuracy is None else float(accuracy)
        except Exception:
            acc_val = ""

        def safe_float(x):
            try:
                return float(x)
            except Exception:
                return 0.0

        # Epoch ratio
        epoch_ratio = epoch / total_epochs if total_epochs > 0 else 0.0

        row = dict(
            step=step,
            epoch=epoch,
            epoch_ratio=safe_float(epoch_ratio),
            loss=safe_float(loss),
            delta_loss=safe_float(delta_loss),
            grad_norm=safe_float(grad_norm),
            lr=safe_float(lr),
            phase_base_lr=safe_float(phase_base_lr) if phase_base_lr is not None else safe_float(lr),
            scale=safe_float(scale),
            cosine_progress=safe_float(cosine_progress),
            plateau_steps=safe_float(plateau_steps),
            val_train_gap=safe_float(val_train_gap),
            improvement_rate=safe_float(improvement_rate),
            accuracy=acc_val,
            imgs_per_sec=safe_float(ips),
            step_time=safe_float(step_time),
            since_start=safe_float(t - self._t0),
            events_up=self.events_up,
            events_down=self.events_down
        )

        self.step_writer.writerow(row)
        self._step_fh.flush()

    # ========================================================================
    #  EPOCH-LEVEL LOGGING
    # ========================================================================
    def log_epoch(self, epoch, metrics: dict, t_epoch, total_epochs=50,
                  train_loss=None, val_loss=None):
        """
        Log epoch-level metrics

        New params:
            total_epochs: For epoch_ratio calculation
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch
        """
        def safe_float(x):
            try:
                return float(x)
            except Exception:
                return 0.0

        # Calculate val/train gap
        val_train_gap = 0.0
        if train_loss is not None and val_loss is not None:
            val_train_gap = safe_float(val_loss) - safe_float(train_loss)

        # Update best and calculate improvement rate
        current_map50 = safe_float(metrics.get("map50", 0.0))
        if current_map50 > self.best_map50:
            self.best_map50 = current_map50

        improvement_rate = 0.0
        if self.best_map50 > 1e-6:
            improvement_rate = (self.best_map50 - current_map50) / self.best_map50

        epoch_ratio = epoch / total_epochs if total_epochs > 0 else 0.0

        row = dict(
            epoch=epoch,
            epoch_ratio=safe_float(epoch_ratio),
            map50=current_map50,
            map5095=safe_float(metrics.get("map5095", metrics.get("map", 0.0))),
            precision=safe_float(metrics.get("precision", 0.0)),
            recall=safe_float(metrics.get("recall", 0.0)),
            train_loss=safe_float(train_loss) if train_loss is not None else 0.0,
            val_loss=safe_float(val_loss) if val_loss is not None else 0.0,
            val_train_gap=safe_float(val_train_gap),
            best_map50=safe_float(self.best_map50),
            improvement_rate=safe_float(improvement_rate),
            time_epoch_s=safe_float(t_epoch)
        )

        self.epoch_writer.writerow(row)
        self._epoch_fh.flush()

    # ========================================================================
    #  PARAMETER & SUMMARY METHODS
    # ========================================================================
    def save_params(self, params: dict):
        """Save training parameters to params.json"""
        params = dict(params)
        params.update({"host": socket.gethostname(), "run_id": str(uuid.uuid4())})
        with open(os.path.join(self.run_dir, "params.json"), "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

        # Run history'ye otomatik kayıt (tracker crash ederse eğitim etkilenmez)
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from run_tracker import RunRegistry
            RunRegistry().register_run(name=os.path.basename(self.run_dir))
        except Exception:
            pass

    def save_summary(self, summary: dict):
        """Save training summary to run_summary.json"""
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def close(self):
        """Close file handles"""
        try:
            self._step_fh.flush()
            self._step_fh.close()
        except Exception:
            pass
        try:
            self._epoch_fh.flush()
            self._epoch_fh.close()
        except Exception:
            pass
