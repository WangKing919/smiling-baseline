import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import hydra
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from collections import namedtuple

from video import VideoRecorder
from replay_buffer import ReplayBuffer
import utils

import agent.sac.sac
from agent import Agent, ObservationBuffer


def make_alg(cfg):
    return agent.sac.sac.SACAlgorithm(**cfg.agent.params)


Snapshot = namedtuple(
    "Snapshot", ["obs", "done", "episode_reward", "episode_step", "info"]
)


class SACAgent(Agent):
    def __init__(self, cfg, local_logger=None):
        super().__init__(cfg=cfg, local_logger=local_logger)

        self.eval_seeds = cfg.get("eval_seeds")
        if self.eval_seeds is not None and (
            not cfg.env.startswith("gymnasium_")
            or not self.eval_seeds
            or any(type(seed) is not int or seed < 0 for seed in self.eval_seeds)
        ):
            raise ValueError(
                "eval_seeds requires Gymnasium SAC and a nonempty list of nonnegative integers"
            )
        self.eval_env = None
        self.env = utils.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]
        self.alg = make_alg(cfg)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
        )

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0
        self.episode = 0
        self.next_compare_eval_step = int(cfg.eval_frequency)
        self.next_update_log_step = cfg.agent.params.num_seed_steps

        self.save_action = cfg.use_action
        self.obs_buffer = ObservationBuffer(capacity=cfg.obs_buffer_capacity)

        self.snapshot = Snapshot(
            obs=None, done=True, episode_reward=None, episode_step=None, info=dict()
        )

    def get_obs_shape(self):
        return self.env.observation_space.shape[0]

    def get_action_shape(self):
        return self.env.action_space.shape[0]

    def prepare(self, reward_fn=None):
        self.reward_fn = reward_fn
        self._prepared = True

    def evaluate(self, video_prefix=None, return_statistics=False, *, env=None, seeds=None):
        episode_count = int(self.cfg.num_eval_episodes)
        if episode_count < 1:
            raise ValueError("num_eval_episodes must be positive")
        env = self.env if env is None else env
        reset_seeds = [None] * episode_count if seeds is None else seeds
        episode_rewards = []
        for episode, seed in enumerate(reset_seeds):
            if seed is None:
                obs, _ = env.reset()
            else:
                obs, _ = env.reset(seed=seed)
            self.alg.reset()
            self.video_recorder.init(enabled=(episode == 0 and env is self.env))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.alg):
                    action = self.alg.act(obs, sample=False)
                    # action = self.env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.video_recorder.record(env)
                episode_reward += reward

            episode_rewards.append(episode_reward)
            self.video_recorder.save(
                f"{self.step if video_prefix is None else video_prefix}.mp4"
            )
        mean = float(np.mean(episode_rewards))
        if return_statistics:
            n = len(episode_rewards)
            std = float(np.std(episode_rewards, ddof=1)) if n > 1 else 0.0
            return mean, std, n
        return mean

    def _log_evaluation(self, group, result):
        if self.local_logger is not None:
            mean, std, n = result
            self.local_logger.log(
                {
                    f"{group}/episode": self.episode,
                    f"{group}/episode_reward": mean,
                    f"{group}/episode_reward_std": std,
                    f"{group}/episode_reward_n": n,
                },
                step=self.step,
            )

    def _evaluate_for_comparison(self):
        if self.eval_env is None:
            self.eval_env = utils.make_env(self.cfg, render=False)
        seeds = [
            seed + episode * len(self.eval_seeds)
            for episode in range(self.cfg.num_eval_episodes)
            for seed in self.eval_seeds
        ]
        with torch.no_grad():
            return self.evaluate(return_statistics=True, env=self.eval_env, seeds=seeds)

    def run_bc(self, dataset, num_train_epochs):
        for i in range(num_train_epochs):

            metrics = dict()

            if i % 2000 == 0:
                ret = self.evaluate()
                if self.local_logger is not None:
                    self.local_logger.log({"eval/episode_reward": ret}, step=i)
                print(f"eval: {ret}")

            obs = torch.tensor(dataset[:, : self.get_obs_shape()]).to(self.device)
            action = torch.tensor(dataset[:, self.get_obs_shape() :]).to(self.device)

            metrics.update(self.alg.bc_update(obs, action, loss_type=self.cfg.alg.loss))

            if i % 100 == 0:
                if self.local_logger is not None:
                    self.local_logger.log({"bc/loss": metrics["loss"]}, step=i)
                print(f"epoch {i}, loss: {metrics['loss']}")

    def run(self, num_train_steps):
        assert self._prepared, "Call prepare() before run()"

        reward_fn = self.reward_fn

        obs = self.snapshot.obs
        done = self.snapshot.done
        episode_reward = self.snapshot.episode_reward
        episode_step = self.snapshot.episode_step
        info = self.snapshot.info

        start_time = time.time()

        end_step = self.step + num_train_steps
        eval_frequency = int(self.cfg.eval_frequency)
        log_frequency = int(self.cfg.log_frequency)
        save_frequency = int(self.cfg.save_model_every)

        while self.step < end_step:
            if done:
                if self.step > 0:
                    duration = time.time() - start_time
                    start_time = time.time()
                    self.episode += 1
                    if self.local_logger is not None:
                        self.local_logger.log(
                            {
                                "train/episode_reward": episode_reward,
                                "train/episode": self.episode,
                                "train/episode_length": episode_step,
                                "train/duration": duration,
                            },
                            step=self.step,
                        )

                    print(
                        f"Step: {self.step}, Episode: {self.episode}, Reward: {episode_reward}"
                    )

                    if eval_frequency > 0 and self.step % eval_frequency == 0:
                        self._log_evaluation("eval", self.evaluate(return_statistics=True))

                    # Log the project's comparison protocol on a separate simulator.
                    if (
                        self.eval_seeds is not None
                        and eval_frequency > 0
                        and self.step >= self.next_compare_eval_step
                    ):
                        self.next_compare_eval_step = (
                            self.step // eval_frequency + 1
                        ) * eval_frequency
                        self._log_evaluation("eval_compare", self._evaluate_for_comparison())

                    if save_frequency > 0 and self.step % save_frequency == 0:
                        path = self.work_dir / "models" / self.cfg.run_name / str(self.step)
                        path.mkdir(parents=True, exist_ok=True)
                        self.alg.save(path)

                obs = self.env.reset()[0]
                done = False
                episode_reward = 0
                episode_step = 0

                self.alg.reset()

            # sample action for data collection
            if self.step < self.cfg.agent.params.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.alg):
                    action = self.alg.act(obs, sample=True)

            if self.save_action:
                self.obs_buffer.add(np.concatenate([obs, action]))
            else:
                self.obs_buffer.add(obs.copy())

            # run training update
            if self.step >= self.cfg.agent.params.num_seed_steps:
                metrics = self.alg.update(
                    self.replay_buffer,
                    self.step,
                    reward_fn,
                    use_action=self.save_action,
                )
                if (
                    self.local_logger is not None
                    and log_frequency > 0
                    and self.step >= self.next_update_log_step
                    and "train_actor/loss" in metrics
                ):
                    self.next_update_log_step = self.step + log_frequency
                    # Batch metrics and completed episodes occur on different steps.
                    metrics["train_batch/reward"] = metrics.pop("train/batch_reward")
                    metrics = {
                        key: value.detach().item() if isinstance(value, torch.Tensor) else value
                        for key, value in metrics.items()
                    }
                    self.local_logger.log(metrics, step=self.step)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            done = float(done)
            # The upstream SAC bootstraps transitions at the episode time limit.
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs

            episode_step += 1
            self.step += 1

        self.snapshot = Snapshot(
            obs=obs,
            done=done,
            episode_reward=episode_reward,
            episode_step=episode_step,
            info=info,
        )

    def collect_obs(self, num_obs, save_action, verbose=False):
        obs_list = []
        episode, done = -1, True

        for i in tqdm(range(num_obs), disable=not verbose, desc="Collecting obs"):
            if done:
                obs = self.env.reset()[0]
                self.alg.reset()
                done = False
                episode += 1

            with utils.eval_mode(self.alg):
                action = self.alg.act(obs, sample=True)
                obs_list.append(
                    obs if not save_action else np.concatenate([obs, action])
                )
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            obs = next_obs

        return obs_list, episode

    def load_agent(self, path):
        self.alg.load(path)
