"""
Hybrid Metric Alpha Optimization Simulation.

Tests different alpha values for: plateau_metric = alpha*mAP50 + (1-alpha)*mAP50-95
to find optimal alignment with target metric (mAP50).

Methodology:
  For each alpha, simulate PlateauTracker with hybrid metric and compare
  decisions against a pure mAP50 tracker (ground truth).
  Mismatch types:
    Type A (False Plateau): mAP50 improved but hybrid didn't -> unnecessary plateau
    Type B (False Reset):   mAP50 didn't improve but hybrid did -> missed stagnation
"""

import csv
import os
import sys


# ---------------------------------------------------------------------------
#  Inline PlateauTracker (matches fuzzy_lr.py logic exactly)
# ---------------------------------------------------------------------------
class SimPlateauTracker:
    def __init__(self, rel_tol=0.002):
        self.best = None
        self.steps = 0
        self.rel_tol = rel_tol
        self.last_improved = False

    def update(self, metric):
        if self.best is None:
            self.best = metric
            self.steps = 0
            self.last_improved = True
        elif metric > self.best * (1.0 + self.rel_tol):
            self.best = metric
            self.steps = 0
            self.last_improved = True
        else:
            self.steps += 1
            self.last_improved = False
        return self.steps


# ---------------------------------------------------------------------------
#  Load epochs.csv
# ---------------------------------------------------------------------------
def load_epochs(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "epoch": int(float(row["epoch"])),
                    "map50": float(row["map50"]),
                    "map5095": float(row["map5095"]),
                })
            except (ValueError, KeyError):
                continue
    # Deduplicate: keep last entry per epoch (YOLO writes best model row at end)
    seen = {}
    for r in rows:
        seen[r["epoch"]] = r
    return [seen[k] for k in sorted(seen.keys())]


# ---------------------------------------------------------------------------
#  Simulate single alpha
# ---------------------------------------------------------------------------
def simulate_alpha(epochs, alpha, rel_tol=0.002):
    tracker_hybrid = SimPlateauTracker(rel_tol=rel_tol)
    tracker_map50 = SimPlateauTracker(rel_tol=rel_tol)

    type_a = 0  # mAP50 improved, hybrid didn't
    type_b = 0  # hybrid improved, mAP50 didn't
    agree = 0

    plateau_lengths = []  # hybrid tracker plateau durations
    current_plateau = 0

    epoch_details = []

    for row in epochs:
        hybrid_metric = alpha * row["map50"] + (1.0 - alpha) * row["map5095"]

        tracker_hybrid.update(hybrid_metric)
        tracker_map50.update(row["map50"])

        h_imp = tracker_hybrid.last_improved
        m_imp = tracker_map50.last_improved

        if m_imp and not h_imp:
            type_a += 1
            decision = "TypeA"
        elif h_imp and not m_imp:
            type_b += 1
            decision = "TypeB"
        else:
            agree += 1
            decision = "agree"

        # Track plateau lengths for hybrid tracker
        if tracker_hybrid.steps > 0:
            current_plateau = tracker_hybrid.steps
        else:
            if current_plateau > 0:
                plateau_lengths.append(current_plateau)
            current_plateau = 0

        epoch_details.append({
            "epoch": row["epoch"],
            "map50": row["map50"],
            "map5095": row["map5095"],
            "hybrid": hybrid_metric,
            "h_imp": h_imp,
            "m_imp": m_imp,
            "h_steps": tracker_hybrid.steps,
            "m_steps": tracker_map50.steps,
            "decision": decision,
        })

    # Final plateau if still running
    if current_plateau > 0:
        plateau_lengths.append(current_plateau)

    total = len(epochs)
    total_mismatch = type_a + type_b
    alignment_pct = (agree / total * 100.0) if total > 0 else 0.0
    avg_plateau = sum(plateau_lengths) / len(plateau_lengths) if plateau_lengths else 0.0
    max_plateau = max(plateau_lengths) if plateau_lengths else 0

    return {
        "alpha": alpha,
        "type_a": type_a,
        "type_b": type_b,
        "total_mismatch": total_mismatch,
        "alignment_pct": alignment_pct,
        "avg_plateau": avg_plateau,
        "max_plateau": max_plateau,
        "total": total,
        "details": epoch_details,
    }


# ---------------------------------------------------------------------------
#  Effective weight analysis
# ---------------------------------------------------------------------------
def effective_weight(alpha, avg_map50, avg_map5095):
    """Compute effective mAP50 share considering magnitude difference."""
    contrib_50 = alpha * avg_map50
    contrib_5095 = (1.0 - alpha) * avg_map5095
    total = contrib_50 + contrib_5095
    if total == 0:
        return 0.0
    return contrib_50 / total * 100.0


# ---------------------------------------------------------------------------
#  Run simulation for one run
# ---------------------------------------------------------------------------
ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
REL_TOL = 0.002


def run_simulation(csv_path, run_name):
    epochs = load_epochs(csv_path)
    if not epochs:
        print(f"  [SKIP] No data in {csv_path}")
        return None

    # Compute averages for effective weight analysis
    avg_map50 = sum(e["map50"] for e in epochs) / len(epochs)
    avg_map5095 = sum(e["map5095"] for e in epochs) / len(epochs)

    print("=" * 74)
    print(f"  HYBRID METRIC ALPHA SIMULATION: {run_name}")
    print(f"  Total epochs: {len(epochs)}, rel_tol: {REL_TOL}")
    print(f"  Avg mAP50: {avg_map50:.4f}, Avg mAP50-95: {avg_map5095:.4f}")
    print("=" * 74)
    print()

    # Effective weight table
    print("  Effective mAP50 weight (accounting for magnitude difference):")
    print("  Alpha   Nominal   Effective")
    print("  ------  --------  ---------")
    for a in ALPHAS:
        eff = effective_weight(a, avg_map50, avg_map5095)
        print(f"  {a:.1f}     {a*100:5.0f}%     {eff:5.1f}%")
    print()

    # Simulation results
    results = []
    for alpha in ALPHAS:
        r = simulate_alpha(epochs, alpha, REL_TOL)
        results.append(r)

    # Summary table
    print("  Alpha   TypeA   TypeB   Total   Align%   AvgPlat   MaxPlat")
    print("  ------  ------  ------  ------  ------   -------   -------")
    for r in results:
        print(
            f"  {r['alpha']:.1f}     "
            f"{r['type_a']:>4}    "
            f"{r['type_b']:>4}    "
            f"{r['total_mismatch']:>4}    "
            f"{r['alignment_pct']:>5.1f}%   "
            f"{r['avg_plateau']:>6.1f}    "
            f"{r['max_plateau']:>6}"
        )
    print()

    # Find optimal
    best = min(results, key=lambda r: (r["total_mismatch"], -r["alignment_pct"], r["max_plateau"]))
    print(f"  >>> Optimal alpha: {best['alpha']:.1f} "
          f"(alignment: {best['alignment_pct']:.1f}%, "
          f"total mismatch: {best['total_mismatch']})")
    print()

    # Show mismatch details for current (0.4) vs optimal
    current = [r for r in results if abs(r["alpha"] - 0.4) < 0.01][0]
    if abs(best["alpha"] - 0.4) > 0.01:
        print(f"  --- Mismatch comparison: alpha=0.4 (current) vs alpha={best['alpha']:.1f} (optimal) ---")
        print(f"  {'Epoch':<6} {'mAP50':>8} {'mAP5095':>8}  {'a=0.4':>10}  {'a='+str(best['alpha']):>10}")
        print(f"  {'-----':<6} {'--------':>8} {'--------':>8}  {'----------':>10}  {'----------':>10}")

        for i in range(len(current["details"])):
            dc = current["details"][i]
            db = best["details"][i]
            # Only show epochs with at least one mismatch
            if dc["decision"] != "agree" or db["decision"] != "agree":
                print(
                    f"  {dc['epoch']:<6} "
                    f"{dc['map50']:>8.4f} "
                    f"{dc['map5095']:>8.4f}  "
                    f"{dc['decision']:>10}  "
                    f"{db['decision']:>10}"
                )
        print()

    # Sensitivity analysis: how much does alignment change around optimal?
    print("  --- Sensitivity Analysis ---")
    for r in results:
        bar_len = int(r["alignment_pct"] / 2)  # scale to ~50 chars
        bar = "#" * bar_len
        marker = " <-- current" if abs(r["alpha"] - 0.4) < 0.01 else ""
        marker = " <-- OPTIMAL" if abs(r["alpha"] - best["alpha"]) < 0.01 else marker
        if abs(r["alpha"] - 0.4) < 0.01 and abs(best["alpha"] - 0.4) < 0.01:
            marker = " <-- current & OPTIMAL"
        print(f"  a={r['alpha']:.1f} [{r['alignment_pct']:>5.1f}%] {bar}{marker}")
    print()

    return {
        "run_name": run_name,
        "optimal_alpha": best["alpha"],
        "alignment_pct": best["alignment_pct"],
        "total_mismatch": best["total_mismatch"],
        "all_results": results,
    }


# ---------------------------------------------------------------------------
#  Volatility & False Improvement Analysis
# ---------------------------------------------------------------------------
def volatility_analysis(csv_path, run_name):
    """Analyze epoch-to-epoch metric volatility for different alphas.

    A 'false improvement' is when metric improves (new best) but drops back
    below previous best within 2 epochs — indicating noise, not real progress.
    Pure mAP50 (alpha=1.0) may have more of these than smoothed hybrid.
    """
    epochs = load_epochs(csv_path)
    if len(epochs) < 5:
        return

    print("=" * 74)
    print(f"  VOLATILITY & FALSE IMPROVEMENT ANALYSIS: {run_name}")
    print("=" * 74)
    print()

    for alpha in [0.4, 0.6, 0.7, 0.8, 1.0]:
        # Compute metric series
        metrics = [alpha * e["map50"] + (1.0 - alpha) * e["map5095"] for e in epochs]

        # Epoch-to-epoch deltas
        deltas = [metrics[i] - metrics[i-1] for i in range(1, len(metrics))]
        std_delta = (sum(d**2 for d in deltas) / len(deltas)
                     - (sum(deltas) / len(deltas))**2) ** 0.5

        # Count sign changes in deltas (oscillation)
        sign_changes = sum(1 for i in range(1, len(deltas))
                          if deltas[i] * deltas[i-1] < 0)
        osc_rate = sign_changes / (len(deltas) - 1) * 100.0 if len(deltas) > 1 else 0

        # False improvements: new best that drops back within 2 epochs
        tracker = SimPlateauTracker(rel_tol=REL_TOL)
        false_imp = 0
        for i, m in enumerate(metrics):
            tracker.update(m)
            if tracker.last_improved and i >= 2:
                # Check if within next 2 epochs, metric drops below previous best
                prev_best = tracker.best  # current best after improvement
                future = metrics[i+1:i+3] if i+2 < len(metrics) else metrics[i+1:]
                if future and min(future) < prev_best * (1.0 - REL_TOL):
                    false_imp += 1

        # Plateau resets (total number of improvements detected)
        tracker2 = SimPlateauTracker(rel_tol=REL_TOL)
        total_resets = 0
        for m in metrics:
            tracker2.update(m)
            if tracker2.last_improved:
                total_resets += 1

        print(f"  alpha={alpha:.1f}: "
              f"delta_std={std_delta:.5f}, "
              f"osc={osc_rate:.0f}%, "
              f"false_imp={false_imp}, "
              f"total_resets={total_resets}")

    print()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(base, "..", "artifacts", "runs")

    target_runs = [
        "neudet_fuzzy_improved_v1",
        "neudet_fuzzy_hybrid",
        "gc10_hybrid_v2",
        "neudet_baseline",
    ]

    cross_results = []

    for run_name in target_runs:
        csv_path = os.path.join(runs_dir, run_name, "epochs.csv")
        if not os.path.exists(csv_path):
            print(f"[SKIP] {run_name}: epochs.csv not found")
            continue
        result = run_simulation(csv_path, run_name)
        if result:
            cross_results.append(result)
        print()

    # Volatility analysis
    print("\n" + "=" * 74)
    print("  VOLATILITY & FALSE IMPROVEMENT ANALYSIS")
    print("=" * 74 + "\n")
    for run_name in target_runs:
        csv_path = os.path.join(runs_dir, run_name, "epochs.csv")
        if os.path.exists(csv_path):
            volatility_analysis(csv_path, run_name)

    # Cross-run summary
    if len(cross_results) > 1:
        print("=" * 74)
        print("  CROSS-RUN SUMMARY")
        print("=" * 74)
        print()
        print(f"  {'Run':<30} {'Optimal':>8} {'Align%':>8} {'Mismatch':>9}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*9}")
        for cr in cross_results:
            print(
                f"  {cr['run_name']:<30} "
                f"{cr['optimal_alpha']:>7.1f} "
                f"{cr['alignment_pct']:>7.1f}% "
                f"{cr['total_mismatch']:>8}"
            )
        print()

        # Consensus: weighted by alignment
        # Simple: most common optimal alpha
        alpha_counts = {}
        for cr in cross_results:
            a = cr["optimal_alpha"]
            alpha_counts[a] = alpha_counts.get(a, 0) + 1
        consensus_alpha = max(alpha_counts, key=alpha_counts.get)

        # If no clear consensus, use average
        if max(alpha_counts.values()) == 1:
            avg_optimal = sum(cr["optimal_alpha"] for cr in cross_results) / len(cross_results)
            # Round to nearest 0.1
            consensus_alpha = round(avg_optimal * 10) / 10
            print(f"  No clear consensus. Average optimal: {avg_optimal:.2f}")
            print(f"  Rounded consensus alpha: {consensus_alpha:.1f}")
        else:
            print(f"  >>> Consensus optimal alpha: {consensus_alpha:.1f} "
                  f"(chosen by {alpha_counts[consensus_alpha]}/{len(cross_results)} runs)")

        # Show what this means in effective weight terms
        # Use overall average metrics
        all_map50 = []
        all_map5095 = []
        for cr in cross_results:
            for r in cr["all_results"]:
                if abs(r["alpha"] - 0.4) < 0.01:
                    # Get from details
                    for d in r["details"]:
                        all_map50.append(d["map50"])
                        all_map5095.append(d["map5095"])
                    break

        if all_map50:
            gm50 = sum(all_map50) / len(all_map50)
            gm5095 = sum(all_map5095) / len(all_map5095)
            eff_current = effective_weight(0.4, gm50, gm5095)
            eff_optimal = effective_weight(consensus_alpha, gm50, gm5095)
            print()
            print(f"  Effective mAP50 influence:")
            print(f"    Current (alpha=0.4):   {eff_current:.1f}%")
            print(f"    Optimal (alpha={consensus_alpha:.1f}):   {eff_optimal:.1f}%")

        print()
