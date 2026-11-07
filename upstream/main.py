import time
import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
from pathlib import Path

import utils
from run_logging import training_log


@hydra.main(version_base="1.2", config_path="./config", config_name="train")
def main(cfg):
    with training_log(cfg, cfg.alg.name) as local_logger:
        train(cfg, local_logger)


def train(cfg, local_logger):
    utils.set_seed_everywhere(cfg.seed)
    agent = utils.make_agent(cfg, local_logger=local_logger)
    OmegaConf.save(cfg, local_logger.log_dir / "config.yaml", resolve=True)

    # init alg
    obs_dim = agent.get_obs_shape()
    action_dim = agent.get_action_shape() if cfg.use_action else 0
    alg = utils.make_alg(
        cfg,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    # expert stuff
    obs_save_dir = Path.cwd() / "models" / cfg.expert_path
    if cfg.alg.name == "smiling":
        if cfg.alg.diff_model_name is None:
            retain_every = cfg.expert_dataset_retain_every
            assert retain_every >= 1
            train_obs_batch = utils.load_or_collect_obs(
                obs_save_dir
                / f"train_obs_{cfg.expert_dataset_num_obs}{'_a' if cfg.use_action else ''}.npy",
                agents=None,
                num_obs=cfg.expert_dataset_num_obs,
                retain_every=retain_every,
                use_action=cfg.use_action,
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_obs_batch,
                batch_size=cfg.alg.batch_size,
                shuffle=True,
            )
            val_obs_batch = utils.load_or_collect_obs(
                obs_save_dir
                / f"val_obs_{cfg.expert_dataset_num_obs}{'_a' if cfg.use_action else ''}.npy",
                agents=None,
                num_obs=cfg.expert_dataset_num_obs,
                retain_every=retain_every,
                use_action=cfg.use_action,
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_obs_batch,
                batch_size=cfg.alg.batch_size,
                shuffle=True,
            )
            print(
                f"training diffusion model --- num obs: {train_dataloader.dataset.shape[0]} \t batch size: {cfg.alg.batch_size} \t dataset len {len(train_dataloader)}",
                flush=True,
            )
            alg.update(
                dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                num_epochs=cfg.alg.expert_train_num_epochs,
                verbose=True,
                local_logger=local_logger,
                global_epoch=0,
                train_expert=True,
            )
            alg.save_expert_diff(local_logger.log_dir / "models" / f"diff_{cfg.run_name}")
            if cfg.local_log_plot_every > 0:
                local_logger.plot()
        else:
            alg.load_expert_diff(cfg.alg.diff_model_name)

    elif cfg.alg.name in ["dac", "bc"]:
        expert_obs_dataset = utils.load_or_collect_obs(
            path=obs_save_dir
            / f"train_obs_{cfg.expert_dataset_num_obs}{'_a' if cfg.use_action else ''}.npy",
            agents=None,
            num_obs=cfg.expert_dataset_num_obs,
            retain_every=cfg.expert_dataset_retain_every,
            use_action=cfg.use_action,
        )

        if cfg.alg.name == "bc":
            assert cfg.use_action
            agent.run_bc(expert_obs_dataset, cfg.alg.num_train_epochs)
            return

    else:
        raise NotImplementedError

    # prepare agent with reward function
    agent.prepare(reward_fn=alg.reward_fn)

    # training loop
    for epoch in range(cfg.epochs):
        metrics = dict()
        metrics["epoch"] = epoch

        print(f"====== Training RL Agent by {cfg.agent.name} ======")
        start_time = time.time()
        agent.run(num_train_steps=cfg.rl_train_steps)
        metrics["rb/rb_len"] = agent.obs_buffer_len()
        print(f"Training RL Agent took {time.time() - start_time:.2f} seconds")

        print("====== Collecting Data ======")
        start_time = time.time()

        obs_data = agent.sample_from_obs_buffer(cfg.data_size)

        # incorporate expert data for DAC
        if cfg.alg.name == "dac":
            indices = np.random.randint(0, expert_obs_dataset.shape[0], cfg.data_size)
            expert_obs_batch = expert_obs_dataset[indices]
            assert expert_obs_batch.shape == obs_data.shape, (
                obs_data.shape,
                expert_obs_batch.shape,
            )
            obs_data = np.concatenate((obs_data, expert_obs_batch), axis=1)
            assert obs_data.shape[1] == 2 * expert_obs_batch.shape[1]

        assert (
            len(obs_data.shape) == 2 or len(obs_data.shape) == 4
        ) and obs_data.shape[0] == cfg.data_size

        dataloader = torch.utils.data.DataLoader(
            obs_data, batch_size=cfg.alg.batch_size
        )
        print(f"Collecting Data took {time.time() - start_time:.2f} seconds")

        print(f"====== Updating {cfg.alg.name} ======")
        start_time = time.time()
        alg.update(
            dataloader=dataloader,
            val_dataloader=None,
            num_epochs=cfg.alg.epochs,
            verbose=False,
            local_logger=local_logger,
            global_epoch=epoch,
            step=int(agent.step),
        )
        print(f"Updating {cfg.alg.name} took {time.time() - start_time:.2f} seconds")

        local_logger.log(metrics, step=int(agent.step))
        if cfg.local_log_plot_every > 0 and (epoch + 1) % cfg.local_log_plot_every == 0:
            local_logger.plot()


if __name__ == "__main__":
    main()
