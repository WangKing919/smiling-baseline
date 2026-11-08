# SMILING 官方基线 · 五个 MuJoCo v2 任务

`upstream/` 固定到官方 SMILING 提交 `c655d34`。五个 Gymnasium MuJoCo v2 任务使用官方 SAC 和状态模仿奖励，输入为主项目的环境与专家观测。官方论文使用 DeepMind Control Suite 和 HumanoidBench，这里的结果是官方算法在本项目任务上的基线。 [官方提交](https://github.com/ziqian2000/SMILING/tree/c655d341575c559250681567eed7885ca4c0a69c) · [论文实验设置](https://arxiv.org/html/2410.13855v2#S6)

## 在 Linux 训练机运行

在 `smiling_baseline/` 下可以直接使用云主机已有的环境，包括 Python 3.10.16、Gymnasium 1.1.1 和 PyTorch 2.13.0。MuJoCo v2 任务需要 `mujoco-py` 和 MuJoCo 2.1.0。MuJoCo 装在其他路径时设置 `MUJOCO_PY_MUJOCO_PATH`。

```bash
UV_VENV=/home/ubuntu/tianruijie/SMILING/.venv bash run_smiling_ant_v2_full.sh save_video=false
```

启动器优先使用 `UV_VENV` 指定的环境，其次查找本目录、上一级目录和云端 FisherAdv 的 `.venv`。实际依赖版本写入 `runtime.json`，同一张对比图中的运行应标注使用的环境版本。Hugging Face 和 Matplotlib 的缓存默认写入子项目的 `logs/.cache/`。

需要单独创建官方锁定版本的 uv 环境时，可使用以下命令。

```bash
uv venv --python 3.10.14 --seed .venv
.venv/bin/python -m pip install -r upstream/requirements.txt \
  'mujoco-py==2.1.2.14' 'Cython==0.29.37' 'huggingface-hub==0.25.2'
```

```bash
bash run_smiling_ant_v2_full.sh
bash run_smiling_halfcheetah_v2_full.sh
bash run_smiling_hopper_v2_full.sh
bash run_smiling_humanoid_v2_full.sh
bash run_smiling_walker2d_v2_full.sh
```

正式运行继承官方 `config/train.yaml`、`config/agent/sac.yaml` 和 `config/alg/smiling.yaml`。默认使用 seed 0、100,000 轮、每轮 1,001 步，专家扩散预训练 4,000 轮。训练日志写到本地，视频默认开启。无渲染后端时可以显式传入 `save_video=false`。可用 `bash run_smiling_ant_v2_full.sh --cfg job` 检查实际配置。任务、SAC 和 SMILING 入口由启动器固定。

如工作区内有 `SMILING first ant/`，其中的日志和模型属于旧版适配器。该次运行的评估和保存时机与当前基线有差异。正式基线应从头启动，使用新运行目录下的结果。

官方训练内评估在训练步数恰好是 `eval_frequency` 的倍数且回合结束时触发，每次评估一个回合。提前终止会使这条曲线的采样点较少。与主项目默认评估口径比较时，可以指定相同的训练种子，另外输出三个固定评估种子的平均回报。

```bash
bash run_smiling_ant_v2_full.sh seed=2026 'eval_seeds=[0,1,2]'
```

固定种子评估在跨过评估间隔后的首个回合边界运行，使用独立环境，结果写入 `metrics/eval_compare.csv` 和 `plots/eval_compare_episode_reward.png`。阴影表示该次评估的回合回报标准差。论文曲线汇总 seed 0 至 4 的独立训练结果，复用论文统计口径时应分别运行这些训练种子。 [官方运行说明](https://github.com/ziqian2000/SMILING#run-smiling)

`experts/` 包含 Ant-v2、HalfCheetah-v2、Hopper-v2、Humanoid-v2 和 Walker2d-v2 的训练与验证观测，每份 5,000 条，使用原项目的转换脚本以 seed 2026 生成。克隆后可直接读取这些观测。若需要用自己的轨迹文件重新生成，可调用 `scripts/prepare_expert_obs_from_pkl.py`。

## 本地训练产物

每次运行在 `smiling_baseline/logs/smiling/<env>/seed<seed>_<时间戳>/` 写入结果。可用 `local_log_root=logs/experiment_a` 指定子项目内的其他目录，相对路径以 `smiling_baseline/` 为起点。

- `config.yaml` 保存本次 Hydra 参数，包括实际的观测维度、动作维度和专家数据路径。
- `runtime.json` 保存对应的官方提交、实际环境入口、依赖版本、SAC 实现和奖励函数的来源。SMILING 的奖励为 `algorithm.diffusion.Diff.reward_fn`，文件位于子项目的 `upstream/algorithm/diffusion.py`。
- `metrics/eval.csv` 和 `plots/eval_episode_reward.png` 保存官方训练内评估回报，CSV 的 `step` 为实际训练步数。
- `metrics/train.csv` 保存训练回报和回合长度。SAC 按 `log_frequency` 采样写入 actor、critic、温度和模仿奖励等指标。
- `metrics/diffusion_expert.csv`、`metrics/diffusion_expert_val.csv` 和 `metrics/diffusion.csv` 分别保存专家扩散训练、验证和策略扩散训练损失。专家扩散曲线以扩散训练轮次为横轴，策略扩散曲线以环境步数为横轴。
- `plots/` 优先保存评估回报、训练回报、回合长度、模仿奖励、SAC actor/critic 损失、策略熵、温度和扩散模型损失。指标首次有数据就出图，评估回报在每次评估后更新，其余曲线每 `local_log_plot_every` 轮刷新。设为 `0` 可仅在结束时绘图。学习率和轮次等辅助字段保存在 CSV 中。
- `models/` 保存专家扩散模型和 SAC 策略检查点。SAC 在回合结束且步数恰好是 `save_model_every` 的倍数时保存。显式复用本子项目训练的专家扩散模型时，可将模型目录的绝对路径传给 `alg.diff_model_name`。

单独执行 `upstream/train.py` 训练专家时，输出位于 `smiling_baseline/logs/expert/`。

在 `smiling_baseline/` 目录用已有运行的 CSV 补画关键曲线：

```bash
python upstream/smiling_local_logger.py logs/smiling/gymnasium_Ant-v2/seed0_1107_120000_000000
```
