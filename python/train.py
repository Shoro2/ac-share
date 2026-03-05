"""
PPO Training on Live WoW Server — sim-parity edition.

Uses MaskablePPO (sb3_contrib) with action masking, matching the sim training
setup.  Hyperparameters, reward design, and observation/action spaces are
aligned with train_sim.py for model transfer.

Usage:
    python train.py                      # default: 5 bots, 100k steps
    python train.py --bots 5 --steps 500000
    python train.py --resume models/PPO/wow_bot_sim_v1.zip --lr 1e-4

Transfer from sim:
    python train.py --resume models/PPO/wow_bot_sim_v1.zip --lr 1e-4 --steps 100000
"""

import traceback
import os
import sys
import time
import argparse
from multiprocessing import freeze_support

print(">>> Importing modules... <<<")
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from torch.utils.tensorboard import SummaryWriter
    from wow_env import WoWEnv
except Exception as e:
    print(f"IMPORT ERROR: {e}")
    traceback.print_exc()
    raise


def _mask_fn(env: WoWEnv):
    """Extract action masks from the environment for MaskablePPO."""
    return env.action_masks()


class GameplayMetricsCallback(BaseCallback):
    """Log gameplay metrics per episode to TensorBoard.

    Uses a dedicated SummaryWriter for guaranteed immediate writes.
    Matches sim callback for metric parity.
    """

    _TRAIN_KEYS = [
        "train/approx_kl", "train/clip_fraction", "train/clip_range",
        "train/entropy_loss", "train/explained_variance",
        "train/learning_rate", "train/loss",
        "train/policy_gradient_loss", "train/value_loss",
    ]

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._writer = None
        self._episode_count = 0
        self._total_kills = 0
        self._total_deaths = 0
        self._last_iter_time = None
        self._last_iter_steps = 0

    def _on_step(self) -> bool:
        if self._writer is None:
            log_dir = getattr(self.logger, 'dir', None)
            if log_dir:
                self._writer = SummaryWriter(log_dir=log_dir)
                print(f"  [Gameplay-Callback] Writing to: {log_dir}")
                self._last_iter_time = time.time()
                self._last_iter_steps = 0

        step = self.num_timesteps
        if (self._writer and self._last_iter_time is not None
                and step - self._last_iter_steps >= 5120):
            now = time.time()
            dt = now - self._last_iter_time
            if dt > 0:
                real_fps = (step - self._last_iter_steps) / dt
                self._writer.add_scalar("perf/real_fps", real_fps, step)
            self._last_iter_time = now
            self._last_iter_steps = step

        infos = self.locals.get("infos", [])
        for info in infos:
            stats = info.get("episode_stats")
            if stats is None:
                continue

            self._episode_count += 1
            self._total_kills += stats["kills"]
            self._total_deaths += stats["death"]

            if self._writer:
                self._writer.add_scalar("gameplay/ep_reward", stats["reward"], step)
                self._writer.add_scalar("gameplay/ep_length", stats["length"], step)
                self._writer.add_scalar("gameplay/ep_kills", stats["kills"], step)
                self._writer.add_scalar("gameplay/ep_xp", stats["xp"], step)
                self._writer.add_scalar("gameplay/ep_damage_dealt", stats["damage_dealt"], step)
                self._writer.add_scalar("gameplay/ep_death", stats["death"], step)
                self._writer.add_scalar("gameplay/ep_idle_ratio", stats.get("idle_ratio", 0), step)
                self._writer.add_scalar("gameplay/ep_areas_explored", stats.get("areas_explored", 0), step)
                self._writer.add_scalar("gameplay/ep_zones_explored", stats.get("zones_explored", 0), step)
                self._writer.add_scalar("gameplay/ep_levels_gained", stats.get("levels_gained", 0), step)
                self._writer.add_scalar("gameplay/ep_final_level", stats.get("final_level", 1), step)
                self._writer.add_scalar("gameplay/ep_loot_items", stats.get("loot_items", 0), step)
                self._writer.add_scalar("gameplay/ep_loot_copper", stats.get("loot", 0), step)
                self._writer.add_scalar("gameplay/ep_sell_copper", stats.get("sell_copper", 0), step)
                self._writer.add_scalar("gameplay/ep_quests_completed", stats.get("quests_completed", 0), step)
                self._writer.add_scalar("gameplay/ep_quest_xp", stats.get("quest_xp", 0), step)
                self._writer.add_scalar("gameplay/ep_equipment_upgrades", stats.get("equipment_upgrades", 0), step)

                self._writer.add_scalar("reward_breakdown/explore", stats.get("rw_explore", 0), step)

                self._writer.add_scalar("gameplay/total_episodes", self._episode_count, step)
                self._writer.add_scalar("gameplay/total_kills", self._total_kills, step)
                self._writer.add_scalar("gameplay/total_deaths", self._total_deaths, step)
                self._writer.add_scalar(
                    "gameplay/kill_death_ratio",
                    self._total_kills / max(1, self._total_deaths),
                    step,
                )
                if self._episode_count % 20 == 0:
                    self._writer.flush()

            if self.verbose:
                areas = stats.get('areas_explored', 0)
                quests = stats.get('quests_completed', 0)
                idle = stats.get('idle_ratio', 0)
                upgrades = stats.get('equipment_upgrades', 0)
                print(f"  [Episode {self._episode_count}] "
                      f"reward={stats['reward']:.1f} kills={stats['kills']} "
                      f"xp={stats['xp']} deaths={stats['death']} "
                      f"areas={areas} idle={idle:.0%} "
                      f"quests={quests} upgrades={upgrades} "
                      f"len={stats['length']}")

        return True

    def _on_rollout_end(self) -> None:
        if not self._writer:
            return
        step = self.num_timesteps
        name_to_value = getattr(self.logger, "name_to_value", {})
        for key in self._TRAIN_KEYS:
            if key in name_to_value:
                self._writer.add_scalar(key, name_to_value[key], step)
        for tkey in ("time/fps", "time/iterations", "time/time_elapsed",
                      "time/total_timesteps"):
            if tkey in name_to_value:
                self._writer.add_scalar(tkey, name_to_value[tkey], step)

    def _on_training_end(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None


# Ensure wow_env is importable from subprocesses
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

BOT_NAMES = ["Bota", "Botb", "Botc", "Botd", "Bote"]


def make_env(bot_name: str, enable_quests: bool = False):
    def _init():
        env = WoWEnv(bot_name=bot_name, enable_quests=enable_quests)
        return ActionMasker(env, _mask_fn)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Live WoW Server")
    parser.add_argument("--bots", type=int, default=5, help="Number of parallel bots")
    parser.add_argument("--steps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--n-steps", type=int, default=512, help="Steps per rollout per env")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to model to resume from")
    parser.add_argument("--output", type=str, default=None, help="Output model path")
    parser.add_argument("--enable-quests", action="store_true", help="Enable quest system")
    args = parser.parse_args()

    models_dir = os.path.join("models", "PPO")
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    bot_names = BOT_NAMES[:args.bots]
    print(f">>> Starting live training: {args.bots} bots, {args.steps} timesteps <<<")
    print(f">>> n_steps={args.n_steps}, batch_size={args.batch_size}, lr={args.lr} <<<")
    print(f">>> MaskablePPO with action masking (sim parity) <<<")

    try:
        env = SubprocVecEnv(
            [make_env(name, enable_quests=args.enable_quests) for name in bot_names],
            start_method="spawn",
        )
    except Exception as e:
        print(f"ENV INIT ERROR: {e}")
        traceback.print_exc()
        raise

    try:
        if args.resume:
            print(f">>> Resuming from {args.resume} <<<")
            print(f">>> Overriding: lr={args.lr}, n_steps={args.n_steps}, "
                  f"batch_size={args.batch_size} <<<")
            model = MaskablePPO.load(
                args.resume, env=env, tensorboard_log=log_dir,
                learning_rate=args.lr,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
            )
        else:
            model = MaskablePPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=log_dir,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                ent_coef=0.01,
                clip_range=0.2,
                n_epochs=8,
                gamma=0.97,
                gae_lambda=0.95,
            )
    except Exception as e:
        print(f"MODEL ERROR: {e}")
        traceback.print_exc()
        raise

    start_time = time.time()
    print(">>> TRAINING STARTS... (Ctrl+C to interrupt) <<<")

    metrics_callback = GameplayMetricsCallback(verbose=1)

    try:
        model.learn(total_timesteps=args.steps, callback=metrics_callback)

        version = 1
        while os.path.exists(os.path.join(models_dir, f"wow_bot_v{version}.zip")):
            version += 1
        output_path = args.output or os.path.join(models_dir, f"wow_bot_v{version}")
        model.save(output_path)
        elapsed = time.time() - start_time
        print(f">>> Training complete! {elapsed:.1f}s <<<")
        print(f">>> Model saved: {output_path} <<<")

    except KeyboardInterrupt:
        interrupt_path = os.path.join(models_dir, "wow_bot_interrupted")
        model.save(interrupt_path)
        elapsed = time.time() - start_time
        print(f"\n>>> Interrupted after {elapsed:.1f}s. Saved: {interrupt_path} <<<")

    except Exception as e:
        print(f"\n!!! CRASH: {e}")
        traceback.print_exc()

    finally:
        if metrics_callback._writer:
            metrics_callback._writer.close()
        try:
            env.close()
        except Exception:
            pass

    print("Done.")


if __name__ == "__main__":
    freeze_support()
    main()
