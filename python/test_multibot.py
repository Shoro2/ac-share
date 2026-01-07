import socket
import json
import time
import math
from wow_env import SPELL_SMITE, SPELL_HEAL # Nutze Spells aus deiner Env

HOST = '127.0.0.1'
PORT = 5000

# Bot-Namen (müssen exakt so heißen wie im Spiel!)
BOT_NAMES = ["Autoai", "Bota", "Botb", "Botc", "Botd", "Bote"]

# Testaktion für alle Bots gleichzeitig
ALL_BOTS_TEST_ACTION = True
TEST_ACTION = "target_nearest:0"

def run_bot_controller():
    print(f">>> Verbinde zu Server... Suche nach {BOT_NAMES} <<<", flush=True)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((HOST, PORT))
    except:
        print("Server nicht erreichbar!", flush=True)
        return

    sock.settimeout(2.0)
    buffer = ""
    last_status = 0.0
    last_raw_status = 0.0

    while True:
        try:
            try:
                data = sock.recv(8192)
            except socket.timeout:
                now = time.time()
                if now - last_status >= 2.0:
                    print("Warte auf Serverdaten...", flush=True)
                    last_status = now
                continue
            if not data:
                print("Verbindung getrennt.", flush=True)
                break
            
            # JSON Stream parsen (newline-delimited JSON)
            buffer += data.decode('utf-8', errors='ignore')
            state = None
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not (line.startswith("{") and line.endswith("}")):
                    continue
                try:
                    state = json.loads(line)
                except json.JSONDecodeError:
                    continue

            if state is None:
                continue

            # --- MULTI-BOT LOGIK ---
            players = state.get('players', [])
            if not players and time.time() - last_raw_status >= 2.0:
                print("DEBUG raw JSON vom Server:", line, flush=True)
                last_raw_status = time.time()
            if time.time() - last_status >= 2.0:
                names = [p.get('name') for p in players if p.get('name')]
                print(f"Empfangen: {len(players)} Player -> {names}", flush=True)
                last_status = time.time()
            
            bots_by_name = {p.get('name'): p for p in players}
            missing = [name for name in BOT_NAMES if name not in bots_by_name]
            if missing:
                print(f"Warte auf Bots: {missing}... (Gefunden: {[p['name'] for p in players]})", flush=True)
                time.sleep(1)
                continue

            if ALL_BOTS_TEST_ACTION:
                for name in BOT_NAMES:
                    full_msg = f"{name}:{TEST_ACTION}"
                    sock.sendall(f"{full_msg}\n".encode('utf-8'))
                time.sleep(0.5)
                continue

            for name in BOT_NAMES:
                my_bot = bots_by_name[name]

                # --- BOT GEFUNDEN: STEUERUNG ---
                # Einfache Test-Logik: Wenn Ziel da -> Angriff, sonst -> Folgen/Warten
                target_guid = my_bot.get('target_guid', '0')  # Oder aus target_status ableiten
                target_alive = (my_bot['target_status'] == 'alive')
                in_combat = (my_bot['combat'] == 'true')
                hp_pct = my_bot['hp'] / max(1, my_bot['max_hp'])

                cmd = ""

                if hp_pct < 0.5:
                    print(f"[{name}] Kritisch! Heile mich...", flush=True)
                    cmd = f"cast:{SPELL_HEAL}"

                elif target_alive:
                    print(f"[{name}] Kämpfe gegen Ziel!", flush=True)
                    cmd = f"cast:{SPELL_SMITE}"

                else:
                    # Idle Mode: Suche Ziel
                    print(f"[{name}] Suche Ziel...", flush=True)
                    cmd = "target_nearest:0"

                # Befehl senden: "Name:Befehl"
                if cmd:
                    full_msg = f"{name}:{cmd}"
                    sock.sendall(f"{full_msg}\n".encode('utf-8'))
            time.sleep(0.5) # Reaktionszeit

        except KeyboardInterrupt:
            print("Beendet.", flush=True)
            break
        except Exception as e:
            print(f"Fehler: {e}", flush=True)
            time.sleep(1)

if __name__ == "__main__":
    run_bot_controller()
