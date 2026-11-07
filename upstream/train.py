import hydra
from omegaconf import OmegaConf

import utils
from run_logging import training_log


@hydra.main(version_base="1.2", config_path="./config", config_name="train")
def main(cfg):
    with training_log(cfg, "expert") as local_logger:
        utils.set_seed_everywhere(cfg.seed)
        agent = utils.make_agent(cfg, local_logger=local_logger)
        OmegaConf.save(cfg, local_logger.log_dir / "config.yaml", resolve=True)
        agent.prepare()
        agent.run(num_train_steps=int(cfg.expert_num_train_steps))


if __name__ == "__main__":
    main()
