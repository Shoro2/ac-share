from stable_baselines3 import PPO
from wow_env import WoWEnv

# Umgebung laden
env = WoWEnv()

# Modell laden (Pfad anpassen, falls nötig!)
model_path = "models/PPO/wow_bot_v1"

try:
    model = PPO.load(model_path, env=env)
    print(">>> Modell geladen! Starte Vorführung... <<<")
except:
    print("Kein Modell gefunden. Bitte erst train.py ausführen!")
    exit()

# Der Loop
obs, _ = env.reset()
while True:
    # Das Modell fragen: "Was soll ich in dieser Situation (obs) tun?"
    action, _states = model.predict(obs)
    
    # Ausführen
    obs, rewards, done, truncated, info = env.step(action)
    
    # Wenn Episode vorbei (Tot oder Kill), resetten
    if done:
        obs, _ = env.reset()