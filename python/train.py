import traceback
import os
import sys
from datetime import datetime
from multiprocessing import freeze_support

print(">>> Importiere Module... <<<")
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure as configure_logger
    from torch.utils.tensorboard import SummaryWriter
    from wow_env import WoWEnv
except Exception as e:
    print(f"IMPORT FEHLER: {e}")
    traceback.print_exc()
    raise


class GameplayMetricsCallback(BaseCallback):
    """Loggt Gameplay-Metriken pro Episode direkt nach TensorBoard.

    Nutzt einen eigenen SummaryWriter statt SB3's record_mean(),
    damit die Werte garantiert sofort geschrieben werden — unabhaengig
    von SB3's internem dump()-Timing.
    """

    def __init__(self, tb_writer: SummaryWriter, verbose=0):
        super().__init__(verbose)
        self._writer = tb_writer
        self._episode_count = 0
        self._total_kills = 0
        self._total_deaths = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            stats = info.get("episode_stats")
            if stats is None:
                continue

            self._episode_count += 1
            self._total_kills += stats["kills"]
            self._total_deaths += stats["death"]
            step = self.num_timesteps

            # Pro-Episode Metriken (jede beendete Episode = 1 Datenpunkt)
            self._writer.add_scalar("gameplay/ep_reward", stats["reward"], step)
            self._writer.add_scalar("gameplay/ep_length", stats["length"], step)
            self._writer.add_scalar("gameplay/ep_kills", stats["kills"], step)
            self._writer.add_scalar("gameplay/ep_xp", stats["xp"], step)
            self._writer.add_scalar("gameplay/ep_loot_copper", stats["loot"], step)
            self._writer.add_scalar("gameplay/ep_damage_dealt", stats["damage_dealt"], step)
            self._writer.add_scalar("gameplay/ep_death", stats["death"], step)

            # Kumulative Zaehler
            self._writer.add_scalar("gameplay/total_episodes", self._episode_count, step)
            self._writer.add_scalar("gameplay/total_kills", self._total_kills, step)
            self._writer.add_scalar("gameplay/total_deaths", self._total_deaths, step)
            self._writer.add_scalar(
                "gameplay/kill_death_ratio",
                self._total_kills / max(1, self._total_deaths),
                step,
            )
            self._writer.flush()

            print(f"  [Episode {self._episode_count}] "
                  f"reward={stats['reward']:.1f} kills={stats['kills']} "
                  f"xp={stats['xp']} deaths={stats['death']} "
                  f"len={stats['length']}")

        return True

    def _on_training_end(self) -> None:
        self._writer.close()

# Damit Subprocesses wow_env sicher finden (Windows + spawn)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

models_dir = os.path.join(“models”, “PPO”)
base_log_dir = “logs”

os.makedirs(models_dir, exist_ok=True)

BOT_NAMES = [“Bota”, “Botb”, “Botc”, “Botd”, “Bote”]

def make_env(bot_name: str):
    def _init():
        return WoWEnv(bot_name=bot_name)
    return _init

def main():
    # Eindeutiger Run-Ordner: logs/PPO_20260301_143022/
    # Kein SB3 _0 Suffix, jeder Run hat sein eigenes Verzeichnis
    run_name = f”PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}”
    run_log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    print(f”>>> TensorBoard Run: {run_log_dir} <<<”)

    print(“>>> Lade Environments... <<<”)
    try:
        env = SubprocVecEnv([make_env(name) for name in BOT_NAMES], start_method=”spawn”)
    except Exception as e:
        print(f”ENV INIT FEHLER: {e}”)
        traceback.print_exc()
        raise

    print(“>>> Erstelle Modell... <<<”)
    try:
        model = PPO(
            “MlpPolicy”,
            env,
            verbose=1,
            n_steps=128,
            batch_size=64
            # KEIN tensorboard_log hier — wir konfigurieren den Logger selbst
        )
    except Exception as e:
        print(f”MODEL FEHLER: {e}”)
        traceback.print_exc()
        raise

    # Eigenen Logger setzen: schreibt direkt nach run_log_dir
    # ohne SB3's _0/_1 Suffix-Logik
    custom_logger = configure_logger(run_log_dir, [“stdout”, “tensorboard”])
    model.set_logger(custom_logger)

    # SummaryWriter fuer Gameplay-Metriken — schreibt ins selbe Verzeichnis
    tb_writer = SummaryWriter(log_dir=run_log_dir)

    print(“>>> TRAINING STARTET... (Drücke Strg+C zum Abbrechen) <<<”)
    TIMESTEPS = 10000
    metrics_callback = GameplayMetricsCallback(tb_writer=tb_writer, verbose=1)

    try:
        model.learn(
            total_timesteps=TIMESTEPS,
            callback=metrics_callback,
        )

        model_path = os.path.join(models_dir, “wow_bot_v1”)
        model.save(model_path)
        print(f”>>> Modell gespeichert unter {model_path} <<<”)

    except KeyboardInterrupt:
        print(“Training manuell abgebrochen. Speichere Zwischenstand...”)
        model.save(os.path.join(models_dir, “wow_bot_interrupted”))
    except Exception as e:
        print(“\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!”)
        print(f”CRASH REPORT: {e}”)
        traceback.print_exc()
        print(“!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n”)
    finally:
        tb_writer.close()
        try:
            env.close()
        except Exception:
            pass

    print(“Fertig.”)

if __name__ == "__main__":
    freeze_support()  # wichtig für Windows multiprocessing
    main()
