from stable_baselines3 import PPO
from wow_env import WoWEnv
import time

# Umgebung laden
env = WoWEnv()

# Modell laden (Achte auf den Dateinamen! PPO speichert oft als .zip)
# Wenn du abgebrochen hast, nimm wow_bot_interrupted
model_path = modelsPPOwow_bot_v1 

try
    model = PPO.load(model_path)
    print( Gehirn geladen! Bot übernimmt die Kontrolle... )
except
    print(fFehler Konnte {model_path} nicht finden. Hast du trainiert)
    exit()

# Der Endlos-Loop
obs, _ = env.reset()
while True
    # Die KI entscheidet (deterministisch = kein Zufall mehr, sondern Skill)
    action, _states = model.predict(obs, deterministic=True)
    
    # Ausführen
    obs, rewards, done, truncated, info = env.step(action)
    
    # Feedback im Fenster
    if action == 5 print(KI FEUER!)
    if action == 1 print(KI Vorwärts!)
    
    if done
        obs, _ = env.reset()