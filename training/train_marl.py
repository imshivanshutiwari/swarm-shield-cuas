"""
Main entry point for SWARM-SHIELD MARL training.
Usage: python training/train_marl.py [--config configs/]
"""

import argparse
import os
import sys

# Ensure repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_all_configs  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

log = get_logger("train_marl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SWARM-SHIELD MARL Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs",
        help="Path to configs directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total training timesteps",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Override max training episodes (for quick testing)",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in checkpoint-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Set seed
    set_seed(args.seed)

    # 2. Load all 4 YAML configs
    configs = load_all_configs(args.config)
    log.info(f"Loaded configs: {list(configs.keys())}")

    if args.total_timesteps is not None:
        configs["marl"]["total_timesteps"] = args.total_timesteps

    use_wandb = not args.no_wandb

    # 3. Initialize WandB
    if use_wandb:
        try:
            import wandb
            from dotenv import load_dotenv

            load_dotenv()
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "swarm-shield-cuas"),
                entity=os.getenv("WANDB_ENTITY", None),
                config={**configs["marl"], **configs["swarm"], **configs["snn"]},
                name=f"swarm-shield-seed{args.seed}",
            )
            log.info("WandB initialized")
        except Exception as e:
            log.warning(f"WandB init failed: {e}. Continuing without WandB.")
            use_wandb = False

    # 4. Initialize Curriculum Scheduler
    from adversarial.curriculum import CurriculumScheduler

    curriculum = CurriculumScheduler()

    # 5. Initialize agents and environment via trainer
    from training.trainer import MARLTrainer

    trainer_config = {
        **configs,
        "use_wandb": use_wandb,
        "checkpoint_dir": args.checkpoint_dir,
        "resume": args.resume,
    }
    trainer = MARLTrainer(config=trainer_config)

    # 6. Initialize Digital Twin for adversarial training
    from adversarial.digital_twin import DigitalTwin

    dt_config = curriculum.get_config(0)
    DigitalTwin(
        env_config=dt_config,
        attacker=trainer.attacker,
        commander=trainer.commander,
        interceptors=trainer.interceptors,
    )

    # 7. Run training loop
    max_episodes = args.max_episodes
    trainer.run(max_episodes=max_episodes)

    # 8. Final evaluation
    log.info("Running final evaluation...")
    from evaluation.evaluate import Evaluator

    evaluator = Evaluator(
        commander=trainer.commander,
        interceptors=trainer.interceptors,
        config=trainer_config,
    )
    eval_results = evaluator.evaluate(n_episodes=5)
    log.info(f"Final evaluation results: {eval_results}")

    # 9. Log final results to WandB
    if use_wandb:
        try:
            import wandb

            wandb.log({"final_eval": eval_results})
            wandb.finish()
        except Exception:
            pass

    log.info("Training pipeline complete.")


if __name__ == "__main__":
    main()
