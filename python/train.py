import traceback # NEU: Für Fehlerverfolgung

print(">>> Importiere Module... <<<")
try:
    from stable_baselines3 import PPO
    from wow_env import WoWEnv
    import os
except Exception as e:
    print(f"IMPORT FEHLER: {e}")
    traceback.print_exc()
    exit()

# Verzeichnis
models_dir = "models/PPO"
log_dir = "logs"

if not os.path.exists(models_dir): os.makedirs(models_dir)
if not os.path.exists(log_dir): os.makedirs(log_dir)

print(">>> Lade Environment... <<<")
try:
    env = WoWEnv()
except Exception as e:
    print(f"ENV INIT FEHLER: {e}")
    traceback.print_exc()
    exit()

print(">>> Erstelle Modell... <<<")
try:
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, n_steps=256, batch_size=64)
except Exception as e:
    print(f"MODEL FEHLER: {e}")
    traceback.print_exc()
    exit()

print(">>> TRAINING STARTET... (Drücke Strg+C zum Abbrechen) <<<")
TIMESTEPS = 10000 

try:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    
    model_path = f"{models_dir}/wow_bot_v1"
    model.save(model_path)
    print(f">>> Modell gespeichert unter {model_path} <<<")

except KeyboardInterrupt:
    print("Training manuell abgebrochen. Speichere Zwischenstand...")
    model.save(f"{models_dir}/wow_bot_interrupted")
except Exception as e:
    # HIER IST DER WICHTIGE TEIL:
    print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"CRASH REPORT: {e}")
    traceback.print_exc() # Zeigt genau die Zeile
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

print("Fertig.")