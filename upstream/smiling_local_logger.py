"""
LocalLogger for SMILING training.

Saves metrics as CSV files and generates line-plot PNGs.

Directory layout::

    {log_dir}/
        metrics/
            train.csv               # train/... keys
            eval.csv                # eval/...  keys
            diffusion.csv          # diffusion/... keys
            general.csv             # un-prefixed keys (epoch, rb_len, …)
        plots/
            eval_episode_reward.png
            train_episode_reward.png
            diffusion_loss.png
            ...

Usage::

    logger = LocalLogger(Path("logs/smiling/gymnasium_Ant-v2/seed0_0330_123456"))
    logger.log({"eval/episode_reward": 120.0, "train/episode_reward": 80.0}, step=1000)
    logger.plot()   # call periodically (e.g. once per outer epoch)
    logger.close()  # final plot + close file handles
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")  # non-interactive, server-safe
import matplotlib.pyplot as plt
import numpy as np


PLOT_METRICS = {
    "eval": {"episode_reward": "Evaluation Return"},
    "eval_compare": {"episode_reward": "Fixed-Seed Evaluation Return"},
    "train": {"episode_reward": "Training Return", "episode_length": "Episode Length"},
    "rl_reward": {"mean": "Mean Imitation Reward"},
    "train_actor": {"loss": "SAC Actor Loss", "entropy": "SAC Policy Entropy"},
    "train_critic": {"loss": "SAC Critic Loss"},
    "train_alpha": {"value": "SAC Temperature"},
    "diffusion": {"loss": "Learner Diffusion Loss"},
    "diffusion_expert": {"loss": "Expert Diffusion Training Loss"},
    "diffusion_expert_val": {"loss": "Expert Diffusion Validation Loss"},
    "dac": {"loss": "Discriminator Loss"},
    "bc": {"loss": "Behavior Cloning Loss"},
}


class LocalLogger:

    def __init__(
        self,
        log_dir: Path,
        environment_name: Optional[str] = None,
        expert_return: Optional[float] = None,
        eval_curve_label: Optional[str] = None,
        live_plots: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.metrics_dir = self.log_dir / "metrics"
        self.plots_dir = self.log_dir / "plots"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Per-group CSV state
        self._csv_files: dict = {}
        self._csv_writers: dict = {}
        self._awaiting_first_plot = set(PLOT_METRICS)

        self.environment_name = environment_name
        self.expert_return = expert_return
        self.eval_curve_label = eval_curve_label
        self.live_plots = live_plots

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, metrics: dict, step: int) -> None:
        """
        Write a flat metrics dict to the appropriate CSV files.

        Keys can be:
          - ``"group/subkey"``  → ``metrics/group.csv``
          - ``"subkey"``        → ``metrics/general.csv``

        Non-numeric values are silently skipped.
        """
        grouped: dict = defaultdict(dict)
        for key, value in metrics.items():
            v = self._to_float(value)
            if v is None:
                continue
            if "/" in key:
                group, subkey = key.split("/", 1)
            else:
                group = "general"
                subkey = key
            grouped[group][subkey] = v

        for group, data in grouped.items():
            self._write_row(group, step, data)
            if (
                self.live_plots
                and (group in self._awaiting_first_plot or group in ("eval", "eval_compare"))
                and any(math.isfinite(data.get(key, math.nan)) for key in PLOT_METRICS[group])
            ):
                self.plot(groups=(group,))
                self._awaiting_first_plot.discard(group)

    def plot(self, groups=PLOT_METRICS) -> int:
        """Generate plots for the key metrics recorded in CSV files."""
        count = 0
        for group in groups:
            path = self.metrics_dir / f"{group}.csv"
            if not path.is_file():
                continue
            series = defaultdict(list)
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    step = self._to_float(row.get("step"))
                    if step is None or not math.isfinite(step):
                        continue
                    std = (
                        self._to_float(row.get("episode_reward_std"))
                        if group in ("eval", "eval_compare") else None
                    )
                    for key in PLOT_METRICS[group]:
                        value = self._to_float(row.get(key))
                        if value is not None and math.isfinite(value):
                            series[key].append((step, value, std))
            for key, points in series.items():
                self._save_plot(group, key, points)
                count += 1
        return count

    def close(self) -> None:
        """Generate final plots and close all file handles."""
        try:
            self.plot()
        finally:
            for f in self._csv_files.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._csv_files.clear()
            self._csv_writers.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float(value) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _write_row(self, group: str, step: int, data: dict) -> None:
        """Append and flush one row to the group's CSV."""
        if group not in self._csv_files:
            path = self.metrics_dir / f"{group}.csv"
            fieldnames = ["step"] + sorted(data.keys())
            f = path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
                restval="",
            )
            writer.writeheader()
            self._csv_files[group] = f
            self._csv_writers[group] = writer

        self._csv_writers[group].writerow({"step": step, **data})
        self._csv_files[group].flush()

    def _save_plot(
        self,
        group: str,
        key: str,
        points,
    ) -> None:
        steps_arr, values_arr, std_arr = np.asarray(points, dtype=float).T
        is_eval_return = group in ("eval", "eval_compare") and key == "episode_reward"
        curve_label = "Fixed-seed mean" if group == "eval_compare" else self.eval_curve_label

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            steps_arr, values_arr,
            linewidth=2.0 if is_eval_return else 1.0,
            marker="o" if len(steps_arr) == 1 else None,
            color="#1d4ed8",
            alpha=1.0 if is_eval_return else 0.7,
            label=(curve_label or "Evaluation return") if is_eval_return else None,
        )

        if is_eval_return and np.any(np.isfinite(std_arr) & (std_arr > 0)):
            ax.fill_between(
                steps_arr,
                values_arr - std_arr,
                values_arr + std_arr,
                color="#93c5fd",
                alpha=0.35,
                label="±1 SD",
            )

        if is_eval_return and self.expert_return is not None:
            ax.axhline(
                self.expert_return,
                color="#6b7280",
                linestyle="--",
                linewidth=1.5,
                label="Expert",
            )

        if is_eval_return:
            ax.legend(fontsize=9)

        x_label = "Environment Steps"
        if group in ("diffusion_expert", "diffusion_expert_val"):
            x_label = "Expert Diffusion Epoch"
        elif group == "bc" or self.log_dir.parent.parent.name == "bc":
            x_label = "Training Epoch"
        ax.set_xlabel(x_label)
        metric_title = PLOT_METRICS[group][key]
        ax.set_ylabel("Episode Return" if key == "episode_reward" else metric_title)
        title = f"{self.environment_name} {metric_title}" if self.environment_name else metric_title
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.25, linestyle="--")
        fig.tight_layout()

        fname = f"{group}_{key.replace('/', '_')}.png"
        temporary_plot = self.plots_dir / f".{uuid4().hex}.png"
        try:
            fig.savefig(temporary_plot, dpi=120)
            temporary_plot.replace(self.plots_dir / fname)
        finally:
            plt.close(fig)
            temporary_plot.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot key SMILING metrics from an existing run.")
    parser.add_argument("run_dir", type=Path, help="Run directory containing metrics/*.csv.")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("Run directory must be inside the SMILING subproject.")
    if not (run_dir / "plots").resolve().is_relative_to(run_dir):
        parser.error("Plot directory must be inside the run directory.")
    if not (run_dir / "metrics").is_dir():
        parser.error(f"Missing metrics directory in {run_dir}")
    logger = LocalLogger(run_dir, environment_name=run_dir.parent.name.split("_", 1)[-1])
    count = logger.plot()
    print(f"Saved {count} plots to {logger.plots_dir}")
