"""Write training results inside the SMILING subproject."""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from smiling_local_logger import LocalLogger


SMILING_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def training_log(cfg, algorithm):
    log_root = (SMILING_ROOT / Path(cfg.local_log_root or "logs").expanduser()).resolve()
    if not log_root.is_relative_to(SMILING_ROOT):
        raise ValueError(f"local_log_root must be inside {SMILING_ROOT}")

    run_id = f"seed{cfg.seed}_{datetime.now():%m%d_%H%M%S_%f}"
    run_dir = (log_root / algorithm / cfg.env / run_id).resolve()
    if not run_dir.is_relative_to(log_root):
        raise ValueError(f"Run directory must be inside {log_root}: {run_dir}")
    if cfg.run_name in ("", ".", "..") or Path(cfg.run_name).name != cfg.run_name:
        raise ValueError("run_name must be a single directory name")

    run_dir.mkdir(parents=True)
    cfg.log_wandb = False  # Keep legacy Hydra overrides accepted.
    OmegaConf.save(cfg, run_dir / "config.yaml")
    episode_count = cfg.num_eval_episodes if cfg.agent.name == "sac" else cfg.agent.run.eval_eps
    episode_label = (
        "Evaluation return" if episode_count == 1 else f"Mean ({episode_count} episodes)"
    )
    logger = LocalLogger(
        run_dir,
        environment_name=cfg.env.split("_", 1)[-1],
        eval_curve_label=episode_label,
    )
    print(f"Local logs: {run_dir}", flush=True)
    try:
        yield logger
    finally:
        logger.close()
