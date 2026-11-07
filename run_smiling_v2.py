"""Run the official SMILING configuration on the project's MuJoCo v2 tasks."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
TASKS = ("Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Humanoid-v2", "Walker2d-v2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("all", *TASKS), default="all", nargs="?")
    parser.add_argument("hydra_args", nargs=argparse.REMAINDER, help="Arguments for upstream main.py.")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.exit(1, "Run with the existing uv Python on the Linux training machine.\n")

    env = os.environ.copy()
    mujoco_dir = Path(
        env.get("MUJOCO_PY_MUJOCO_PATH") or Path.home() / ".mujoco" / "mujoco210"
    ).expanduser().resolve()
    if not (mujoco_dir / "bin").is_dir():
        parser.exit(1, "Set MUJOCO_PY_MUJOCO_PATH to the existing MuJoCo 2.1.0 directory.\n")
    env["MUJOCO_PY_MUJOCO_PATH"] = str(mujoco_dir)
    mujoco_bin = str(mujoco_dir / "bin")
    library_path = env.get("LD_LIBRARY_PATH", "")
    if library_path.split(os.pathsep, 1)[0] != mujoco_bin:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (mujoco_bin, library_path)))

    for task in TASKS if args.task == "all" else (args.task,):
        expert_dir = ROOT / "experts" / task
        for split in ("train", "val"):
            path = expert_dir / f"{split}_obs_5000.npy"
            if not path.is_file():
                parser.exit(1, f"Missing expert observations: {path}\n")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "smiling_v2_adapter.py"),
                f"env=gymnasium_{task}",
                f"expert_path={json.dumps(expert_dir.as_posix(), ensure_ascii=False)}",
                *args.hydra_args,
            ],
            cwd=ROOT / "upstream",
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
