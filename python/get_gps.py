import socket
import json
import time

HOST = '127.0.0.1'
PORT = 5000

def run_gps_tool():
    print(f"Verbinde GPS-Tool mit {HOST}:{PORT}...")
    
    # Verbindung aufbauen
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        print(">>> VERBUNDEN! <<<")
        print("Lauf im Spiel zu deinen Punkten. Kopiere die Zeilen hier raus:")
        print("-" * 60)
    except Exception as e:
        print(f"Verbindungsfehler: {e}")
        return

    buffer = ""
    
    # Hauptschleife
    try:
        while True:
            try:
                data = s.recv(4096)
                if not data: break
                
                buffer += data.decode('utf-8')
                
                if "\n" in buffer:
                    lines = buffer.split("\n")
                    buffer = lines[-1] # Rest behalten für nächsten Tick
                    
                    # Wir nehmen die letzte vollständige Zeile
                    if len(lines) > 1:
                        raw_json = lines[-2]
                        if raw_json.strip():
                            # Innerer Try-Block für JSON Parsing
                            try:
                                state = json.loads(raw_json)
                                players = state.get("players", [])
                                
                                if players:
                                    me = players[0]
                                    # Output formatieren
                                    output = f'{{"x": {me["x"]:.3f}, "y": {me["y"]:.3f}, "z": {me["z"]:.3f}}},'
                                    print(output)
                            except json.JSONDecodeError:
                                pass # Fehlerhaftes JSON ignorieren

            except socket.error:
                print("Verbindung verloren.")
                break
            
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nBeendet durch Benutzer.")
    finally:
        s.close()
        print("Verbindung geschlossen.")

if __name__ == "__main__":
    run_gps_tool()