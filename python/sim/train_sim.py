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
from sim.wow_sim_env import WoWSimEnv


def make_env(bot_name: str, seed: int):
    def _init():
        return WoWSimEnv(bot_name=bot_name, num_mobs=15, seed=seed)
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
    args = parser.parse_args()

    models_dir = os.path.join(PARENT_DIR, "models", "PPO")
    log_dir = os.path.join(PARENT_DIR, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    bot_names = [f"SimBot{i}" for i in range(args.bots)]
    print(f">>> Starting sim training: {args.bots} bots, {args.steps} timesteps <<<")
    print(f">>> n_steps={args.n_steps}, batch_size={args.batch_size}, lr={args.lr} <<<")

    start_method = "fork" if sys.platform != "win32" else "spawn"

    try:
        env = SubprocVecEnv(
            [make_env(name, seed=i * 1000) for i, name in enumerate(bot_names)],
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
                ent_coef=0.01,
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

    try:
        # reset_num_timesteps=True (default) → SB3 erstellt PPO_1, PPO_2, ...
        model.learn(total_timesteps=args.steps)

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
        try:
            env.close()
        except Exception:
            pass

    print("Done.")


if __name__ == "__main__":
    freeze_support()
    main()
