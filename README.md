# SMILING 官方基线 · 五个 MuJoCo v2 任务

`upstream/` 是官方 SMILING `main` 的完整快照。五任务启动器、Gymnasium 兼容脚本和专家观测独立放在本仓库内。 [官方提交 c655d34](https://github.com/ziqian2000/SMILING/tree/c655d341575c559250681567eed7885ca4c0a69c)

## 在 Linux 训练机运行

在本仓库根目录按任务执行对应的 Bash 脚本。脚本沿用原项目从 `/proc/1/environ` 读取 NVIDIA 设备并通过 `nvidia-smi` 映射 CUDA 编号的方式。默认依次查找本目录和上一级目录的 `.venv`。uv 环境位于其他目录时，用 `UV_VENV` 指向现有的 `.venv`。

```bash
bash run_smiling_ant_v2_full.sh
bash run_smiling_halfcheetah_v2_full.sh
bash run_smiling_hopper_v2_full.sh
bash run_smiling_humanoid_v2_full.sh
bash run_smiling_walker2d_v2_full.sh
```

例如，`UV_VENV=/path/to/existing/.venv bash run_smiling_ant_v2_full.sh` 会使用指定的环境。uv 环境需能运行 CUDA、Gymnasium 1.1.1、`mujoco-py` 和 MuJoCo 2.1.0。MuJoCo 装在其他路径时设置 `MUJOCO_PY_MUJOCO_PATH`。

默认只为官方入口指定 `env` 和 `expert_path`。正式运行沿用官方的 seed 0、100,000 轮、每轮 1,001 步，以及 W&B 和视频设置。脚本末尾的显式参数会传给官方 Hydra 入口，例如 `bash run_smiling_ant_v2_full.sh --cfg job` 可打印配置。模型和训练输出保存在 `upstream/` 下相应的官方输出目录。

`experts/` 包含 Ant-v2、HalfCheetah-v2、Hopper-v2、Humanoid-v2 和 Walker2d-v2 的训练与验证观测，每份 5,000 条，使用原项目的转换脚本以 seed 2026 生成。克隆后可直接读取这些观测。若需要用自己的轨迹文件重新生成，可调用 `scripts/prepare_expert_obs_from_pkl.py`。

## 推送到自己的 GitHub

在本仓库根目录执行，将远端地址替换成你创建的空仓库地址。

```bash
git init -b main
git add .
git update-index --chmod=+x upstream/train.py
git commit -m "Add official SMILING baseline for five MuJoCo v2 tasks"
git remote add origin <your-repository-url>
git push -u origin main
```
