#!/usr/bin/env bash
# Gymnasium classic MuJoCo: Ant-v2 + SAC with the official SMILING defaults.
# Expert demos: experts/Ant-v2/train_obs_5000.npy and val_obs_5000.npy.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Ant-v2 needs mujoco-py + MuJoCo 2.1.0.
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-${HOME}/.mujoco/mujoco210}"
if [[ ! -d "${MUJOCO_PY_MUJOCO_PATH}" && -d /home/ubuntu/tianruijie/.mujoco/mujoco210 ]]; then
  export MUJOCO_PY_MUJOCO_PATH=/home/ubuntu/tianruijie/.mujoco/mujoco210
fi
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${LD_LIBRARY_PATH:-}"
if [[ ! -v CUDA_VISIBLE_DEVICES ]]; then
  DEVICE_IDS="${NVIDIA_VISIBLE_DEVICES:-}"
  if [[ -z "${DEVICE_IDS}" && -r /proc/1/environ ]]; then
    DEVICE_IDS="$(strings /proc/1/environ 2>/dev/null | awk -F= '$1 == "NVIDIA_VISIBLE_DEVICES" { print $2 }' || true)"
  fi
  case "${DEVICE_IDS}" in
    ""|all|none|void) ;;
    *) CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "${DEVICE_IDS}" --query-gpu=index --format=csv,noheader | paste -sd ',')"
       export CUDA_VISIBLE_DEVICES ;;
  esac
fi

if [[ -z "${UV_VENV:-}" ]]; then
  UV_VENV="${HERE}/.venv"
  if [[ ! -f "${UV_VENV}/bin/activate" ]]; then
    UV_VENV="${HERE}/../.venv"
  fi
  FISHERADV_VENV=/home/ubuntu/tianruijie/wangjian/fisheradv/.venv
  if [[ ! -f "${UV_VENV}/bin/activate" && -f "${FISHERADV_VENV}/bin/activate" ]]; then
    UV_VENV="${FISHERADV_VENV}"
  fi
fi
if [[ ! -f "${UV_VENV}/bin/activate" ]]; then
  echo "Missing ${UV_VENV}/bin/activate; set UV_VENV to your existing uv environment." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${UV_VENV}/bin/activate"

exec python "${HERE}/run_smiling_v2.py" "Ant-v2" "$@"
