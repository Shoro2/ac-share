from wow_env import WoWEnv
import random

# Umgebung starten
env = WoWEnv()

# Resetten
obs, info = env.reset()
print(f"Start Observation: {obs}")

# 10 Schritte zufällig machen
for i in range(10):
    # Zufällige Aktion wählen (0 bis 5)
    action = random.randint(0, 5)
    
    # Schritt ausführen
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"Schritt {i}: Action={action} -> Reward={reward:.2f} | HP={obs[0]:.2f}")
    
    if done:
        print("Episode beendet!")
        env.reset()