import traceback
import os
import sys
from multiprocessing import freeze_support

print(">>> Importiere Module... <<<")
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from wow_env import WoWEnv
except Exception as e:
    print(f"IMPORT FEHLER: {e}")
    traceback.print_exc()
    raise

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

    try:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)

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
