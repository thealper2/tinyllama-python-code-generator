#!/usr/bin/env python3
import argparse

import hydra
from omegaconf import DictConfig, OmegaConf

from training.trainer import CodeModelTrainer


@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    trainer = CodeModelTrainer(cfg)
    trainer.train()
    eval_results = trainer.evaluate()

    print("Training completed!")
    print("Evaluation results:", eval_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Override config file")
    args = parser.parse_args()

    if args.config:
        main(config_name=args.config)
    else:
        main()
