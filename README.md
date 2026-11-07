# SMILING 官方基线 · 五个 MuJoCo v2 任务

`upstream/` 是官方 SMILING `main` 的完整快照。五任务启动器、Gymnasium 兼容脚本和专家观测独立放在本仓库内。 [官方提交 c655d34](https://github.com/ziqian2000/SMILING/tree/c655d341575c559250681567eed7885ca4c0a69c)

## 在 Linux 训练机运行

在本仓库根目录用已有 uv 环境的 Python 启动。该环境需能运行 CUDA、Gymnasium 1.1.1、`mujoco-py` 和 MuJoCo 2.1.0。MuJoCo 装在其他路径时设置 `MUJOCO_PY_MUJOCO_PATH`。

```bash
/path/to/existing/.venv/bin/python run_smiling_v2.py
/path/to/existing/.venv/bin/python run_smiling_v2.py HalfCheetah-v2
```

启动器只为官方入口指定 `env` 和 `expert_path`。正式运行沿用官方的 seed 0、100,000 轮、每轮 1,001 步，以及 W&B 和视频设置。模型和训练输出保存在 `upstream/` 下相应的官方输出目录。

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
