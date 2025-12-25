import socket
import json
import time
import math
from wow_env import SPELL_SMITE, SPELL_HEAL # Nutze Spells aus deiner Env

HOST = '127.0.0.1'
PORT = 5000

# Name deines Bots (muss exakt so heißen wie im Spiel!)
BOT_NAME = "BotA" 

def run_bot_controller():
    print(f">>> Verbinde zu Server... Suche nach '{BOT_NAME}' <<<")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((HOST, PORT))
    except:
        print("Server nicht erreichbar!")
        return

    while True:
        try:
            data = sock.recv(8192)
            if not data: break
            
            # JSON Stream parsen
            text_chunk = data.decode('utf-8')
            lines = text_chunk.split("\n")
            
            # Wir suchen die letzte vollständige JSON-Zeile
            last_valid_json = None
            for line in reversed(lines):
                if line.strip().startswith("{") and line.strip().endswith("}"):
                    last_valid_json = line
                    break
            
            if not last_valid_json: continue

            try:
                state = json.loads(last_valid_json)
            except: continue

            # --- MULTI-BOT LOGIK ---
            players = state.get('players', [])
            
            # Suchen wir unseren Bot in der Liste
            my_bot = None
            for p in players:
                if p['name'] == BOT_NAME:
                    my_bot = p
                    break
            
            if not my_bot:
                print(f"Warte auf Bot '{BOT_NAME}'... (Gefunden: {[p['name'] for p in players]})")
                time.sleep(1)
                continue

            # --- BOT GEFUNDEN: STEUERUNG ---
            # Einfache Test-Logik: Wenn Ziel da -> Angriff, sonst -> Folgen/Warten
            
            target_guid = my_bot.get('target_guid', '0') # Oder aus target_status ableiten
            target_alive = (my_bot['target_status'] == 'alive')
            in_combat = (my_bot['combat'] == 'true')
            hp_pct = my_bot['hp'] / max(1, my_bot['max_hp'])
            
            cmd = ""
            
            if hp_pct < 0.5:
                print(f"[{BOT_NAME}] Kritisch! Heile mich...")
                cmd = f"cast:{SPELL_HEAL}"
                
            elif target_alive:
                print(f"[{BOT_NAME}] Kämpfe gegen Ziel!")
                cmd = f"cast:{SPELL_SMITE}"
                
            else:
                # Idle Mode: Suche Ziel
                print(f"[{BOT_NAME}] Suche Ziel...")
                cmd = "target_nearest:0"

            # Befehl senden: "Name:Befehl"
            if cmd:
                full_msg = f"{BOT_NAME}:{cmd}"
                sock.sendall(full_msg.encode('utf-8'))

            time.sleep(0.5) # Reaktionszeit

        except KeyboardInterrupt:
            print("Beendet.")
            break
        except Exception as e:
            print(f"Fehler: {e}")
            time.sleep(1)

if __name__ == "__main__":
    run_bot_controller()