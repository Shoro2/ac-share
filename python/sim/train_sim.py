"""
PPO Training on WoW Combat Simulation — runs ~1000x faster than real server.

Usage:
    python -m sim.train_sim                    # default: 5 bots, 100k steps
    python -m sim.train_sim --bots 10 --steps 500000

The trained model is compatible with wow_env.py (same obs/action space).
Transfer: load the saved model in run_model.py or auto_grind.py.
"""

import os
import sys
import time
import argparse
import traceback
from multiprocessing import freeze_support

# Ensure parent dir is on path for sim package imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from sim.wow_sim_env import WoWSimEnv


class GameplayMetricsCallback(BaseCallback):
    """Loggt Gameplay-Metriken pro Episode direkt nach TensorBoard.

    Nutzt einen eigenen SummaryWriter im selben Verzeichnis wie SB3,
    damit die Werte garantiert sofort geschrieben werden.
    Der SummaryWriter wird lazy erstellt sobald SB3 seinen Logger hat.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._writer = None
        self._episode_count = 0
        self._total_kills = 0
        self._total_deaths = 0
        # Per-iteration FPS tracking (shows REAL speed, not cumulative avg)
        self._last_iter_time = None
        self._last_iter_steps = 0

    def _on_step(self) -> bool:
        # Lazy init: SummaryWriter im selben Ordner wie SB3's Logger
        if self._writer is None:
            log_dir = getattr(self.logger, 'dir', None)
            if log_dir:
                self._writer = SummaryWriter(log_dir=log_dir)
                print(f"  [Gameplay-Callback] Schreibe nach: {log_dir}")
                self._last_iter_time = time.time()
                self._last_iter_steps = 0

        # Log real per-iteration FPS every 5120 steps (~4 iterations)
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
                self._writer.add_scalar("gameplay/ep_loot_copper", stats["loot"], step)
                self._writer.add_scalar("gameplay/ep_damage_dealt", stats["damage_dealt"], step)
                self._writer.add_scalar("gameplay/ep_death", stats["death"], step)
                self._writer.add_scalar("gameplay/ep_areas_explored", stats.get("areas_explored", 0), step)
                self._writer.add_scalar("gameplay/ep_zones_explored", stats.get("zones_explored", 0), step)
                self._writer.add_scalar("gameplay/ep_maps_explored", stats.get("maps_explored", 0), step)
                self._writer.add_scalar("gameplay/ep_levels_gained", stats.get("levels_gained", 0), step)
                self._writer.add_scalar("gameplay/ep_final_level", stats.get("final_level", 1), step)

                # Exploration reward breakdown
                self._writer.add_scalar("reward_breakdown/explore", stats.get("rw_explore", 0), step)

                self._writer.add_scalar("gameplay/total_episodes", self._episode_count, step)
                self._writer.add_scalar("gameplay/total_kills", self._total_kills, step)
                self._writer.add_scalar("gameplay/total_deaths", self._total_deaths, step)
                self._writer.add_scalar(
                    "gameplay/kill_death_ratio",
                    self._total_kills / max(1, self._total_deaths),
                    step,
                )
                # Flush only every 20 episodes (not every episode)
                if self._episode_count % 20 == 0:
                    self._writer.flush()

            if self.verbose:
                areas = stats.get('areas_explored', 0)
                zones = stats.get('zones_explored', 0)
                maps = stats.get('maps_explored', 0)
                dmg = stats.get('damage_dealt', 0)
                print(f"  [Episode {self._episode_count}] "
                      f"reward={stats['reward']:.1f} kills={stats['kills']} "
                      f"xp={stats['xp']} deaths={stats['death']} "
                      f"dmg={dmg} areas={areas} "
                      f"len={stats['length']}")

        return True

    def _on_training_end(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None


def make_env(bot_name: str, seed: int, data_root: str = None,
             creature_csv_dir: str = None):
    def _init():
        return WoWSimEnv(bot_name=bot_name, seed=seed, data_root=data_root,
                         creature_csv_dir=creature_csv_dir)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO on WoW Combat Sim")
    parser.add_argument("--bots", type=int, default=5, help="Number of parallel bots")
    parser.add_argument("--steps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--n-steps", type=int, default=256, help="Steps per rollout per env")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to model to resume from")
    parser.add_argument("--output", type=str, default=None, help="Output model path")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Path to Data/ directory (maps/, vmaps/) for 3D terrain")
    parser.add_argument("--creature-data", type=str, default=None,
                        help="Path to directory with creature.csv and creature_template.csv "
                             "(enables full-world creature spawning via spatial chunks)")
    args = parser.parse_args()

    models_dir = os.path.join(PARENT_DIR, "models", "PPO")
    log_dir = os.path.join(PARENT_DIR, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    bot_names = [f"SimBot{i}" for i in range(args.bots)]
    data_root = args.data_root
    creature_csv_dir = args.creature_data
    print(f">>> Starting sim training: {args.bots} bots, {args.steps} timesteps <<<")
    print(f">>> n_steps={args.n_steps}, batch_size={args.batch_size}, lr={args.lr} <<<")
    if data_root:
        print(f">>> 3D terrain enabled: {data_root} <<<")
    if creature_csv_dir:
        print(f">>> Full-world creatures enabled: {creature_csv_dir} <<<")

    start_method = "fork" if sys.platform != "win32" else "spawn"

    try:
        env = SubprocVecEnv(
            [make_env(name, seed=i * 1000, data_root=data_root,
                      creature_csv_dir=creature_csv_dir)
             for i, name in enumerate(bot_names)],
            start_method=start_method,
        )
    except Exception as e:
        print(f"ENV INIT ERROR: {e}")
        traceback.print_exc()
        raise

    try:
        if args.resume:
            print(f">>> Resuming from {args.resume} <<<")
            model = PPO.load(args.resume, env=env, tensorboard_log=log_dir)
        else:
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=log_dir,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                ent_coef=0.05,
                clip_range=0.2,
                n_epochs=4,
                gamma=0.99,
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
        # reset_num_timesteps=True (default) → SB3 erstellt PPO_1, PPO_2, ...
        model.learn(total_timesteps=args.steps, callback=metrics_callback)

        # Modell-Versionierung: wow_bot_sim_v1, v2, v3, ...
        version = 1
        while os.path.exists(os.path.join(models_dir, f"wow_bot_sim_v{version}.zip")):
            version += 1
        output_path = args.output or os.path.join(models_dir, f"wow_bot_sim_v{version}")
        model.save(output_path)
        elapsed = time.time() - start_time
        fps = args.steps / elapsed
        print(f">>> Training complete! {elapsed:.1f}s, {fps:.0f} FPS <<<")
        print(f">>> Model saved: {output_path} <<<")

    except KeyboardInterrupt:
        interrupt_path = os.path.join(models_dir, "wow_bot_sim_interrupted")
        model.save(interrupt_path)
        elapsed = time.time() - start_time
        print(f"\n>>> Interrupted after {elapsed:.1f}s. Saved: {interrupt_path} <<<")

    except Exception as e:
        print(f"\n!!! CRASH: {e}")
        traceback.print_exc()

    finally:
        # Gameplay-Writer sicher schliessen (falls _on_training_end nicht lief)
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
