# SMILING 官方基线 · 五个 MuJoCo v2 任务

`upstream/` 基于官方 SMILING 快照。五个任务的启动器、Gymnasium 适配器、专家观测和本地日志实现都在 `smiling_baseline/` 下，本目录可单独部署。 [官方提交 c655d34](https://github.com/ziqian2000/SMILING/tree/c655d341575c559250681567eed7885ca4c0a69c)

## 在 Linux 训练机运行

在 `smiling_baseline/` 目录执行对应的 Bash 脚本。脚本沿用原项目的 NVIDIA 设备映射方式。默认依次查找本目录、上一级目录的 `.venv`，以及云端的 `/home/ubuntu/tianruijie/wangjian/fisheradv/.venv`。显式设置的 `UV_VENV` 优先。

```bash
bash run_smiling_ant_v2_full.sh
bash run_smiling_halfcheetah_v2_full.sh
bash run_smiling_hopper_v2_full.sh
bash run_smiling_humanoid_v2_full.sh
bash run_smiling_walker2d_v2_full.sh
```

例如，`UV_VENV=/path/to/existing/.venv bash run_smiling_ant_v2_full.sh` 会使用指定的环境。uv 环境需能运行 CUDA、Gymnasium 1.1.1、`mujoco-py` 和 MuJoCo 2.1.0。MuJoCo 装在其他路径时设置 `MUJOCO_PY_MUJOCO_PATH`。

正式运行使用 seed 0、100,000 轮、每轮 1,001 步。训练只写本地日志，W&B 始终禁用。视频默认关闭，可传 `save_video=true` 恢复 SAC 评估录像。脚本末尾的参数会传给 Hydra，例如 `bash run_smiling_ant_v2_full.sh --cfg job` 可打印配置。

`experts/` 包含 Ant-v2、HalfCheetah-v2、Hopper-v2、Humanoid-v2 和 Walker2d-v2 的训练与验证观测，每份 5,000 条，使用原项目的转换脚本以 seed 2026 生成。克隆后可直接读取这些观测。若需要用自己的轨迹文件重新生成，可调用 `scripts/prepare_expert_obs_from_pkl.py`。

## 本地训练产物

每次运行在 `smiling_baseline/logs/smiling/<env>/seed<seed>_<时间戳>/` 写入结果。可用 `local_log_root=logs/experiment_a` 指定子项目内的其他目录，相对路径以 `smiling_baseline/` 为起点。

- `config.yaml` 保存本次 Hydra 参数，包括实际的观测维度、动作维度和专家数据路径。
- `metrics/eval.csv` 和 `plots/eval_episode_reward.png` 保存评估回报。SAC 在跨过 `eval_frequency` 后于下一个回合边界评估，CSV 的 `step` 为实际训练步数。
- `metrics/train.csv` 保存训练回报和回合长度。SAC 按 `log_frequency` 采样写入 actor、critic、温度和模仿奖励等指标。
- `metrics/diffusion_expert.csv`、`metrics/diffusion_expert_val.csv` 和 `metrics/diffusion.csv` 分别保存专家扩散训练、验证和策略扩散训练损失。专家扩散曲线以扩散训练轮次为横轴，策略扩散曲线以环境步数为横轴。
- `plots/` 按指标生成 PNG。训练中每 `local_log_plot_every` 轮刷新，训练结束时再保存一次。设为 `0` 可仅在结束时绘图。
- `models/` 保存专家扩散模型和按 `save_model_every` 步写入的 SAC 策略检查点。继续使用已有扩散模型时，可为 `alg.diff_model_name` 传入模型目录的绝对路径。

单独执行 `upstream/train.py` 训练专家时，输出位于 `smiling_baseline/logs/expert/`。
