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

import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive, server-safe
import matplotlib.pyplot as plt
import numpy as np


class LocalLogger:

    def __init__(
        self,
        log_dir: Path,
        environment_name: Optional[str] = None,
        expert_return: Optional[float] = None,
        evaluation_seed_count: int = 3,
        eval_curve_label: Optional[str] = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.metrics_dir = self.log_dir / "metrics"
        self.plots_dir = self.log_dir / "plots"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Per-group CSV state
        self._csv_files: dict = {}
        self._csv_writers: dict = {}

        # In-memory buffer: group -> { key: [float, ...], "step": [int, ...] }
        self._data: dict = defaultdict(lambda: defaultdict(list))
        self.environment_name = environment_name
        self.expert_return = expert_return
        self.evaluation_seed_count = evaluation_seed_count
        self.eval_curve_label = eval_curve_label

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

    def plot(self) -> None:
        """Regenerate all PNG plots from in-memory data."""
        for group, group_data in self._data.items():
            steps = group_data.get("step", [])
            if not steps:
                continue
            for key, values in group_data.items():
                if (
                    key == "step"
                    or key.endswith("_std")
                    or key.endswith("_n")
                    or not values
                ):
                    continue
                n = min(len(steps), len(values))
                std_values = None
                if group == "eval" and key == "episode_reward":
                    std_values = group_data.get("episode_reward_std")
                self._save_plot(
                    group,
                    key,
                    steps[:n],
                    values[:n],
                    std_values=std_values[:n] if std_values is not None else None,
                )

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
        """Append one row to the group's CSV and accumulate in-memory."""
        self._data[group]["step"].append(step)
        for k, v in data.items():
            self._data[group][k].append(v)

        if group not in self._csv_files:
            path = self.metrics_dir / f"{group}.csv"
            fieldnames = ["step"] + sorted(data.keys())
            f = open(path, "w", newline="")
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
        steps,
        values,
        std_values=None,
    ) -> None:
        steps_arr = np.asarray(steps, dtype=float)
        values_arr = np.asarray(values, dtype=float)
        is_eval_return = group == "eval" and key == "episode_reward"

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            steps_arr, values_arr,
            linewidth=2.0 if is_eval_return else 1.0,
            marker="o" if len(steps_arr) == 1 else None,
            color="#1d4ed8" if is_eval_return else "#1d4ed8",
            alpha=1.0 if is_eval_return else 0.7,
            label=(
                (
                    self.eval_curve_label
                    if self.eval_curve_label is not None
                    else f"Mean ({self.evaluation_seed_count} Seeds)"
                )
                if is_eval_return
                else None
            ),
        )

        if is_eval_return and std_values is not None:
            std_arr = np.asarray(std_values, dtype=float)
            n = min(len(steps_arr), len(values_arr), len(std_arr))
            steps_arr = steps_arr[:n]
            values_arr = values_arr[:n]
            std_arr = std_arr[:n]
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

        ax.set_xlabel("Environment Steps" if is_eval_return else "Step")
        ax.set_ylabel("Episode Return" if is_eval_return else key.replace("_", " ").title())
        if is_eval_return and self.environment_name:
            title = f"{self.environment_name} Evaluation Return"
        elif self.environment_name:
            title = f"{self.environment_name}: {group.replace('_', ' ').title()} / {key.replace('_', ' ').title()}"
        else:
            title = f"{group.replace('_', ' ').title()} / {key.replace('_', ' ').title()}"
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.25, linestyle="--")
        fig.tight_layout()

        fname = f"{group}_{key.replace('/', '_')}.png"
        fig.savefig(self.plots_dir / fname, dpi=120)
        plt.close(fig)
