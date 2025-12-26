from stable_baselines3 import PPO
from wow_env import WoWEnv
import numpy as np
import time
import math
import socket

# --- KONFIGURATION ---
# Pfad ggf. anpassen
MODEL_PATH = "models/PPO/wow_bot_interrupted" 
MAX_STEP_DISTANCE = 50.0

# DEINE FARM ROUTE
FARM_ROUTE = [
    {"x": -8843.562, "y": -104.231, "z": 82.538}, # Punkt 1
    {"x": -8772.383, "y": -122.333, "z": 83.304}, # Punkt 2
    {"x": -8756.731, "y": -184.207, "z": 84.996}  # Punkt 3
]
# ---------------------

def get_distance(p1, p2):
    dx = p1['x'] - p2['x']
    dy = p1['y'] - p2['y']
    return math.sqrt(dx*dx + dy*dy)

def run():
    print(">>> Starte Auto-Grind Bot v3 (Final) ... <<<")
    env = WoWEnv()
    
    # WICHTIG: Timeout, damit das Skript nicht einfriert
    env.sock.settimeout(0.1) 
    
    try:
        model = PPO.load(MODEL_PATH)
        print(">>> Kampf-Modell geladen! <<<")
    except:
        print("FEHLER: Modell nicht gefunden. Pfad prüfen!")
        return

    current_wp_index = 0
    last_action_time = 0
    
    # Initialer Reset
    obs, _ = env.reset()
    
    print(">>> Loop startet. Drücke Strg+C zum Beenden. <<<")

    while True:
        try:
            # 1. State Update (Non-Blocking)
            try:
                data = env._get_state_from_server()
                if data:
                    env.last_state = data
            except socket.timeout:
                pass 
            except Exception as e:
                print(f"Verbindungsfehler: {e}")
                break

            if env.last_state is None:
                time.sleep(0.1)
                continue

            raw_data = env.last_state
            current_time = time.time()

            # Taktung: Nur alle 0.5s eine Entscheidung treffen
            if current_time - last_action_time < 0.5:
                continue
            
            last_action_time = current_time

            # --- ENTSCHEIDUNG ---
            target_alive = (raw_data['target_status'] == 'alive')
            in_combat = (raw_data['combat'] == 'true')
            
            # MODUS A: KAMPF (Priorität!)
            if target_alive or in_combat:
                # RL Modell fragen
                action, _ = model.predict(obs, deterministic=True)
                
                # Schritt ausführen
                obs, reward, done, truncated, info = env.step(action)
                
                # Feedback (nur bei Cast)
                if action == 5: print(f"[KAMPF] Zaubere... (HP: {raw_data['hp']})")
                if action == 6: print(f"[KAMPF] Heile mich... (HP: {raw_data['hp']})")
                
                if done:
                    print(">>> Kampf vorbei. Reset. <<<")
                    obs, _ = env.reset()

            # MODUS B: REISE (Smart Search)
            else:
                # 1. Haben wir ein Ziel im Gedächtnis?
                # Wir fragen das Environment nach dem besten bekannten (lebenden) Mob
                memory_target = env.get_best_memory_target()
                
                if memory_target:
                    target_wp = memory_target # Das ist unser dynamisches Ziel!
                    print(f"[MEMORY] Laufe zu Mob {memory_target['name']}...")
                else:
                    # Fallback: Wenn das Gedächtnis leer ist, laufen wir die Farm-Route
                    target_wp = FARM_ROUTE[current_wp_index]
                
                # --- Ab hier normale Travel-Logik ---
                me_pos = {'x': raw_data['x'], 'y': raw_data['y'], 'z': raw_data['z']}
                dist = get_distance(me_pos, target_wp)
                
                tick = int(current_time * 2)
                
                if dist < 3.0:
                    if not memory_target: # Nur bei Route weiterzählen
                        print(f">>> WP {current_wp_index} erreicht! <<<")
                        current_wp_index = (current_wp_index + 1) % len(FARM_ROUTE)
                    else:
                        print(">>> Beim Mob angekommen! (Sollte gleich kämpfen) <<<")
                
                elif tick % 5 == 0: 
                    print("[SUCHE] Scanne Umgebung...")
                    env.sock.sendall(f"{env.my_name}:target_nearest:0\n".encode('utf-8'))
                
                else:
                    # Bewegung (Salami-Taktik)
                    move_target = target_wp
                    if dist > MAX_STEP_DISTANCE:
                        dx = target_wp['x'] - me_pos['x']
                        dy = target_wp['y'] - me_pos['y']
                        scale = MAX_STEP_DISTANCE / dist
                        temp_x = me_pos['x'] + (dx * scale)
                        temp_y = me_pos['y'] + (dy * scale)
                        move_target = {"x": temp_x, "y": temp_y, "z": target_wp['z']}
                    
                    cmd = f"{env.my_name}:move_to:{move_target['x']:.2f}:{move_target['y']:.2f}:{move_target['z']:.2f}"
                    env.sock.sendall(f"{cmd}\n".encode('utf-8'))
                    
                    # Update Cache (sonst blockiert es)
                    env._get_state_from_server()

        except KeyboardInterrupt:
            print("\n>>> Bot manuell gestoppt. <<<")
            break
        except Exception as e:
            print(f"Fehler im Loop: {e}")
            time.sleep(1)

if __name__ == "__main__":
    run()
