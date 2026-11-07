#!/usr/bin/env python3
"""Convert trajectory pickle expert data to train/val .npy files for main.py."""

import argparse
import pickle
from pathlib import Path

import numpy as np


def _flatten_states(data: dict) -> np.ndarray:
    if "states" not in data:
        raise KeyError(
            f"Expected key 'states' in pickle dict, got keys: {list(data.keys())}"
        )
    states = data["states"]
    if isinstance(states, np.ndarray) and states.ndim == 2:
        return states.astype(np.float32)
    if isinstance(states, (list, tuple)):
        return np.concatenate([np.asarray(s, dtype=np.float32) for s in states], axis=0)
    raise TypeError(f"Unsupported states type: {type(states)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for train_obs_<N>.npy and val_obs_<N>.npy",
    )
    parser.add_argument("--num-obs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .npy files",
    )
    args = parser.parse_args()

    if not args.pkl_path.is_file():
        raise FileNotFoundError(f"Expert pickle not found: {args.pkl_path}")

    suffix = ""  # state-only demos
    train_path = args.output_dir / f"train_obs_{args.num_obs}{suffix}.npy"
    val_path = args.output_dir / f"val_obs_{args.num_obs}{suffix}.npy"

    if not args.force and train_path.is_file() and val_path.is_file():
        print(f"Expert npy already exists under {args.output_dir}, skipping conversion")
        return

    with args.pkl_path.open("rb") as f:
        data = pickle.load(f)

    obs = _flatten_states(data)
    need = 2 * args.num_obs
    if obs.shape[0] < need:
        raise ValueError(
            f"Need at least {need} expert states, got {obs.shape[0]} from {args.pkl_path}"
        )

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(obs.shape[0], size=need, replace=False)
    train_obs = obs[indices[: args.num_obs]]
    val_obs = obs[indices[args.num_obs :]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(train_path, train_obs)
    np.save(val_path, val_obs)
    print(f"Saved {train_path} shape={train_obs.shape}")
    print(f"Saved {val_path} shape={val_obs.shape}")


if __name__ == "__main__":
    main()
