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
    from wow_env import WoWEnv
except Exception as e:
    print(f"IMPORT FEHLER: {e}")
    traceback.print_exc()
    raise


class GameplayMetricsCallback(BaseCallback):
    """Loggt Gameplay-Metriken pro Episode nach TensorBoard.

    Standard-PPO-Metriken (entropy, value_loss, etc.) sagen nur ob PPO
    intern konvergiert. Diese Callback trackt ob der Bot tatsächlich
    besser SPIELT: mehr Kills, mehr XP, weniger Tode, länger überlebt.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_count = 0
        self._total_kills = 0
        self._total_deaths = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            # SubprocVecEnv kann terminal info wrappen
            stats = info.get("episode_stats")
            if stats is None:
                terminal_info = info.get("terminal_info")
                if terminal_info:
                    stats = terminal_info.get("episode_stats")
            if stats is None:
                continue

            self._episode_count += 1
            self._total_kills += stats["kills"]
            self._total_deaths += stats["death"]

            # record_mean: mittelt korrekt wenn mehrere Episoden pro Rollout enden
            self.logger.record_mean("gameplay/ep_reward", stats["reward"])
            self.logger.record_mean("gameplay/ep_length", stats["length"])
            self.logger.record_mean("gameplay/ep_kills", stats["kills"])
            self.logger.record_mean("gameplay/ep_xp", stats["xp"])
            self.logger.record_mean("gameplay/ep_loot_copper", stats["loot"])
            self.logger.record_mean("gameplay/ep_damage_dealt", stats["damage_dealt"])
            self.logger.record_mean("gameplay/ep_death", stats["death"])
            # Kumulative Zähler: record() (letzter Wert = aktuellster)
            self.logger.record("gameplay/total_episodes", self._episode_count)
            self.logger.record("gameplay/total_kills", self._total_kills)
            self.logger.record("gameplay/total_deaths", self._total_deaths)
            if self._episode_count > 0:
                self.logger.record(
                    "gameplay/kill_death_ratio",
                    self._total_kills / max(1, self._total_deaths),
                )
        return True

# Damit Subprocesses wow_env sicher finden (Windows + spawn)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

models_dir = os.path.join("models", "PPO")
log_dir = "logs"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

BOT_NAMES = ["Bota", "Botb", "Botc", "Botd", "Bote"]

def make_env(bot_name: str):
    def _init():
        return WoWEnv(bot_name=bot_name)
    return _init

def main():
    print(">>> Lade Environments... <<<")
    try:
        # Parallel pro Prozess -> jeder Bot läuft “gleichzeitig”
        env = SubprocVecEnv([make_env(name) for name in BOT_NAMES], start_method="spawn")
    except Exception as e:
        print(f"ENV INIT FEHLER: {e}")
        traceback.print_exc()
        raise

    print(">>> Erstelle Modell... <<<")
    try:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            n_steps=128,      # kleiner = häufigere Updates, fühlt sich meist “snappier” an
            batch_size=64
        )
    except Exception as e:
        print(f"MODEL FEHLER: {e}")
        traceback.print_exc()
        raise

    print(">>> TRAINING STARTET... (Drücke Strg+C zum Abbrechen) <<<")
    TIMESTEPS = 10000
    metrics_callback = GameplayMetricsCallback(verbose=1)
    run_name = f"PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f">>> TensorBoard Run: {run_name} <<<")

    try:
        model.learn(
            total_timesteps=TIMESTEPS,
            tb_log_name=run_name,
            callback=metrics_callback,
        )

        model_path = os.path.join(models_dir, "wow_bot_v1")
        model.save(model_path)
        print(f">>> Modell gespeichert unter {model_path} <<<")

    except KeyboardInterrupt:
        print("Training manuell abgebrochen. Speichere Zwischenstand...")
        model.save(os.path.join(models_dir, "wow_bot_interrupted"))
    except Exception as e:
        print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"CRASH REPORT: {e}")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    finally:
        try:
            env.close()
        except Exception:
            pass

    print("Fertig.")

if __name__ == "__main__":
    freeze_support()  # wichtig für Windows multiprocessing
    main()
