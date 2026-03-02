# CLAUDE.md — Projektdokumentation für ac-share

Dieses Repository ist ein experimentelles WoW-Bot-Training-Setup. Der **aktuelle Hauptfokus** liegt auf der Python-Simulation (`python/sim/`), die eine vollständige Trainingsumgebung ohne laufenden WoW-Server bereitstellt. Parallel dazu existiert ein C++-Modul für AzerothCore (Live-Server-Anbindung), das später zum Einsatz kommt.

## Aktueller Fokus: Python-Simulation

Die Sim-Umgebung (`python/sim/`) repliziert das WoW-Kampfsystem in reinem Python:
- **~1000x schneller** als Live-Server-Training (kein TCP, kein Server nötig)
- **Identische Schnittstelle** zu `wow_env.py` (gleicher Obs/Action Space, gleiche Rewards)
- **Optionale 3D-Terrain-Daten** aus echten WoW-Dateien (maps/vmaps) via `test_3d_env.py`
- **Ziel**: Alle Grundfunktionen (Combat, Targeting, Loot, Spells, Movement) in der Sim validieren, bevor auf den Live-Server übertragen wird

## Repository-Struktur

```
ac-share/
├── CLAUDE.md                    ← diese Datei
├── README.md                    ← ausführliche Projekt-Doku (Protokoll, Architektur, Workflow)
├── .gitattributes
├── python/                      ← Python RL-Training, Inference & Utilities
│   ├── sim/                     ← ★ HAUPTFOKUS: Offline-Simulation ★
│   │   ├── combat_sim.py        ← Kampfsystem-Simulation (Mobs, Spells, Loot, Movement)
│   │   ├── wow_sim_env.py       ← Gymnasium-Environment für die Sim
│   │   ├── train_sim.py         ← PPO-Training auf der Sim (5 Bots, kein Server nötig)
│   │   └── __init__.py
│   ├── test_3d_env.py           ← 3D-Terrain/VMAP/LOS aus echten WoW-Daten
│   ├── wow_env.py               ← Gymnasium-Environment (Live-Server via TCP)
│   ├── train.py                 ← Multi-Bot PPO-Training (Live-Server)
│   ├── run_model.py             ← Inference-Loop (trained model)
│   ├── auto_grind.py            ← Hybrid-Runner: Route + RL-Policy
│   ├── get_gps.py               ← GPS-Koordinaten-Logger (für Routen)
│   ├── check_env.py             ← Schneller Env-Validierungstest
│   ├── test_multibot.py         ← Multi-Bot-Steuerung mit Scripted-Logic
│   ├── run_bot.py               ← BROKEN — Syntax-Fehler, nicht nutzbar
│   ├── npc_memory.json          ← Gemeinsame NPC-Datenbank (Baseline)
│   ├── npc_memory_*.json        ← Bot-spezifische NPC-Memory-Dateien
│   ├── models/PPO/              ← Gespeicherte PPO-Modelle (.zip)
│   │   └── wow_bot_interrupted.zip  ← letzter Checkpoint (Training abgebrochen)
│   └── logs/                    ← TensorBoard-Logs (PPO_0=live, PPO_2=sim)
├── src_module-ai-controller/    ← C++ AzerothCore-Modul (2 Dateien)
│   ├── AIControllerHook.cpp     ← Gesamte Logik (1008 Zeilen): TCP-Server,
│   │                               Bot-Spawning, Kommando-Verarbeitung, State-Publishing
│   └── AIControllerLoader.cpp   ← Modul-Registrierung (14 Zeilen)
└── src_azeroth_core/            ← Vollständiger AzerothCore-Source-Tree
    ├── cmake/                   ← Build-System
    ├── common/                  ← Shared Libraries (Threading, Crypto, Config)
    ├── server/                  ← Server-Kern
    │   ├── apps/                ← worldserver & authserver
    │   ├── game/                ← 51 Subsysteme (Entities, AI, Spells, Maps, ...)
    │   ├── scripts/             ← Content-Scripts
    │   ├── shared/              ← Netzwerk/Protokoll
    │   └── database/            ← DB-Abstraktion
    ├── test/                    ← Tests
    └── tools/                   ← Externe Tools
```

## Architektur-Überblick

### Dual-Pfad: Sim (Hauptfokus) + Live-Server (später)

```
  ★ AKTUELLER FOKUS ★                    │  SPÄTERE PHASE
                                          │
  ┌──────────────────────────┐            │  ┌──────────────────────────────────┐
  │    CombatSimulation      │            │  │   AzerothCore worldserver        │
  │    python/sim/combat_sim │            │  │   (C++, AI-Controller-Modul)     │
  ├──────────────────────────┤            │  │   TCP :5000, JSON State-Stream   │
  │ 15 Mobs, Spell-System   │            │  └────────────────┬─────────────────┘
  │ Loot, XP, Movement      │            │                   │ TCP
  │ Optional: 3D-Terrain    │            │                   │
  │ (test_3d_env.py)        │            │  ┌────────────────▼─────────────────┐
  └────────────┬─────────────┘            │  │   WoWEnv (python/wow_env.py)     │
               │ direkt (in-process)      │  │   Action: Discrete(11)           │
               │                          │  │   Obs:    Box(17,)               │
  ┌────────────▼─────────────┐            │  └────────────────┬─────────────────┘
  │  WoWSimEnv (Gymnasium)   │            │                   │
  │  python/sim/wow_sim_env  │            │          ┌────────▼──────────┐
  ├──────────────────────────┤            │          │ train.py / etc.   │
  │ Action: Discrete(11)    │            │          └───────────────────┘
  │ Obs:    Box(17,)        │            │
  │ Gleiche Override-Logik   │            │
  │ Gleiche Rewards          │◄───────────┤  ★ Identische Schnittstelle ★
  └──────────┬───────────────┘            │
             │                            │
    ┌────────▼──────────┐                 │
    │  train_sim.py     │                 │
    │  5 Bots, PPO      │                 │
    │  ~5000 FPS        │                 │
    └───────────────────┘                 │
```

### Ziel: Modelle in der Sim trainieren, dann auf Live-Server transferieren.

## TCP-Protokoll

### Server → Python (State-Stream)

Alle 400ms (oder 500ms Keepalive) sendet der Server eine JSON-Zeile:

```json
{
  "players": [
    {
      "name": "Bota",
      "hp": 100, "max_hp": 100,
      "power": 200, "max_power": 200,
      "level": 1,
      "x": -8921.0, "y": -120.5, "z": 82.0, "o": 3.3,
      "combat": "false",
      "casting": "false",
      "free_slots": 14,
      "equipped_upgrade": "false",
      "target_status": "alive",
      "target_hp": 42,
      "xp_gained": 0,
      "loot_copper": 0,
      "loot_score": 0,
      "leveled_up": "false",
      "tx": -8900.0, "ty": -110.0, "tz": 81.0,
      "nearby_mobs": [
        {
          "guid": "12345678",
          "name": "Diseased Young Wolf",
          "level": 1,
          "attackable": 1,
          "vendor": 0,
          "target": "0",
          "hp": 42,
          "x": -8900.0, "y": -110.0, "z": 81.0
        }
      ]
    }
  ]
}
```

**Wichtig**: `combat`, `casting`, `equipped_upgrade`, `leveled_up` sind Strings (`"true"`/`"false"`), keine JSON-Booleans. `xp_gained`, `loot_copper`, `loot_score` werden nach dem Senden zurückgesetzt (consume-on-read).

### Python → Server (Kommandos)

Format: `<playerName>:<actionType>:<value>\n`

| Kommando | Beschreibung |
|---|---|
| `say:<text>` | Spieler sagt Text im Chat |
| `stop:0` | Stoppt Bewegung |
| `turn_left:0` / `turn_right:0` | Dreht Orientierung um ±0.5 rad |
| `move_forward:0` | Bewegt 3 Units nach vorne (Ground-Z-Korrektur) |
| `move_to:<x>:<y>:<z>` | Bewegt zu Koordinaten |
| `target_nearest:<range>` | Wählt nächstes gültiges Ziel (default 30) |
| `target_guid:<guid>` | Wählt Einheit per GUID |
| `cast:<spellId>` | Zaubert Spell (585=Smite auto-targets Feind, 2050=Heal auto-targets self) |
| `loot_guid:<guid>` | Lootet tote Kreatur (≤10 Units), auto-equip wenn besser |
| `sell_grey:<vendorGuid>` | Verkauft alle Items mit SellPrice>0 (außer Hearthstone 6948) |
| `reset:0` | Volle Heilung, Cooldown-Reset, Teleport zu Homebind |

## Python-Komponenten im Detail

### wow_env.py — Gymnasium-Environment

**Klasse**: `WoWEnv(gym.Env)`

**Initialisierung**: `WoWEnv(host='127.0.0.1', port=5000, bot_name=None)`
- `bot_name=None`: adoptiert ersten Spieler im Stream
- `bot_name="Bota"`: filtert explizit nach diesem Namen

**Action Space** — `Discrete(9)`:
| ID | Aktion |
|---|---|
| 0 | No-op (warten) |
| 1 | move_forward |
| 2 | turn_left |
| 3 | turn_right |
| 4 | Target Mob (nächster aus nearby_mobs per target_guid) |
| 5 | Cast Smite (Spell 585) |
| 6 | Cast Heal (Spell 2050) |
| 7 | Loot (nächste tote Kreatur per loot_guid) |
| 8 | Sell (zum Vendor, nur im Vendor-Modus) |

**Observation Vector** — `Box(shape=(10,), dtype=float32)`:
| Index | Wert | Bereich |
|---|---|---|
| 0 | hp_pct (HP/MaxHP) | 0–1 |
| 1 | mana_pct (Mana/MaxMana) | 0–1 |
| 2 | target_hp / 100 | 0–∞ |
| 3 | target_exists (1=alive, 0=sonst) | 0/1 |
| 4 | in_combat | 0/1 |
| 5 | target_distance / 40 (clamped) | 0–1 |
| 6 | relative_angle / π | -1–1 |
| 7 | is_casting | 0/1 |
| 8 | reserviert (immer 0) | 0 |
| 9 | free_slots / 20 | 0–1 |

**Reward-Shaping**: Siehe einheitliche Reward-Tabelle oben (gilt für Sim und Live identisch).

**Override-Logik** (überlagert RL-Entscheidungen):
1. **Vendor-Modus**: Bei `free_slots < 2` und nicht im Kampf → navigiert zum nächsten Vendor aus NPC-Memory, verkauft automatisch
2. **Aggro-Recovery**: Im Kampf ohne Target → sucht Mob der den Bot angreift, targeted ihn
3. **Cast-Schutz**: Während Casting werden Bewegung/Drehen/andere Casts unterdrückt
4. **Loot-Automatik**: Totes Target → nähert sich an, lootet automatisch bei ≤3 Units
5. **Range-Management**: Stoppt Vorwärtsbewegung bei <25 Units zum Target
6. **Heal-Sperre**: Heal wird bei HP > 85% unterdrückt
7. **Sell-Sperre**: Action 8 nur im Vendor-Modus erlaubt

**NPC-Memory-System**:
- Datei: `npc_memory_{bot_name}.json` (pro Bot isoliert)
- Speichert alle gesichteten Mobs mit Position, Level, Vendor-Flag etc.
- Blacklist: Tote/gelootete Mobs werden 15 Minuten ignoriert
- Auto-Save alle 30 Sekunden (atomar via .tmp-Datei)
- Wird von `auto_grind.py` für Memory-basiertes Targeting genutzt

**Deterministische Startdrehung** (`_initial_heading_kick`):
Jeder Bot dreht beim Reset in eine andere Richtung, um Verteilung zu verbessern:
- Autoai: 0 Schritte, Bota: 2, Botb: 4, Botc: 6, Botd: 8, Bote: 10 (je ~0.5 rad)

## Python-Simulation im Detail (Hauptfokus)

### combat_sim.py — Kampfsystem-Simulation

**Klasse**: `CombatSimulation`

Simuliert das komplette WoW-Kampfsystem in reinem Python:
- **15 Mobs** um den Spawnpunkt mit zufälligen Positionen und Stats
- **Priest-Spells**: Smite (585), Heal (2050), SW:Pain (589), PW:Shield (17)
- **Mob-AI**: Aggro-Range (8 Units), Chase, Melee-Angriff, Leash (40 Units)
- **Loot-System**: Copper + Items mit Score, Auto-Equip wenn besser
- **Respawn**: Tote Mobs respawnen nach 30s an neuer Position
- **XP**: Formelbasiert nach Mob-Level
- **State-Dict**: Identisch zum TCP-JSON des Live-Servers

### wow_sim_env.py — Gymnasium Sim-Environment

Drop-in Replacement für `wow_env.py`:
- **Gleicher Action Space**: `Discrete(11)` — No-op, Move, Turn×2, Target, Smite, Heal, Loot, Sell, SW:Pain, PW:Shield
- **Gleicher Obs Space**: `Box(17,)` — HP%, Mana%, Target-HP, Combat, Distance, Angle, etc.
- **Gleiche Rewards**: Synchronisiert mit `wow_env.py` (Skala ~[-5, +15])
- **Gleiche Override-Logik**: Vendor, Aggro, Cast-Guard, Loot-Automatik, Range-Management
- **max_episode_steps**: 4000

### train_sim.py — Sim-Training

- **5 Bots** in `SubprocVecEnv`
- **PPO** mit `ent_coef=0.01`, `n_steps=256`, `batch_size=128`
- **TensorBoard-Logs** in `logs/PPO_2/`
- **Episode-Callbacks** mit Kills, XP, Deaths
- **~5000+ FPS** (ohne 3D-Terrain)

### test_3d_env.py — 3D-Terrain aus echten WoW-Daten

Liest die originalen WoW-Dateien (maps/, vmaps/):
- **Terrain-Höhen**: 129×129 Height-Grid pro Tile, Triangle-Interpolation
- **LOS (Line of Sight)**: VMAP-Spawns (Gebäude, Bäume) mit AABB-Ray-Intersection
- **HeightCache**: Vorberechnetes numpy-Grid für O(1) Höhen-Lookups (~100x schneller)
- **SpatialLOSChecker**: Räumlich indizierter LOS-Check (~100-500x schneller als brute-force)

### Reward-Tabelle (gilt für Sim UND Live identisch)

| Signal | Wert | Anmerkung |
|---|---|---|
| Step-Penalty | -0.01 | pro Tick |
| Idle-Penalty | -0.03 | Noop ohne Casting |
| Mob entdeckt | +0.25 | 0.5 × 0.5 Skalierung |
| Approach | clip(delta×0.05, -0.2, +0.3) | näher an Target |
| Damage dealt | min(dmg×0.03, 1.0) | Schaden am Target |
| Facing Target | facing_quality × 0.08 | im Kampf |
| XP/Kill | 3.0 + min(xp×0.05, 2.0) | ~3–5 pro Kill |
| Level-Up | +15.0 | terminal |
| Equipment-Upgrade | +3.0 | |
| Loot | min((copper×0.01)+(score×0.2), 3.0) | gedeckelt |
| Verkauf | +2.0 | Slots freigeräumt |
| Smite mit Target | +0.3 | |
| Smite ohne Target | -0.1 | |
| Heal bei HP<50% | +0.5 | |
| Heal bei HP>80% | -0.3 | |
| SW:Pain (frisch) | +0.5 | |
| SW:Pain (doppelt) | -0.2 | |
| PW:Shield (Kampf) | +0.4 | |
| PW:Shield (aktiv) | -0.2 | |
| Bewegen im Kampf | -0.3 | |
| Drehen zu Target | +0.4 | im Kampf |
| Tod | -5.0 | terminal, überschreibt |
| OOM (<5% Mana) | -2.0 | terminal |

### train.py — PPO-Training (Live-Server)

- **Bots**: `["Bota", "Botb", "Botc", "Botd", "Bote"]` (5 parallele Environments)
- **Vectorization**: `SubprocVecEnv` (separate Prozesse pro Bot)
- **Algorithmus**: PPO mit `MlpPolicy` (2-Layer FC-Netz)
- **Hyperparameter**: `n_steps=128`, `batch_size=64`, `total_timesteps=10000`
- **Logs**: TensorBoard in `logs/PPO_0/`
- **Modell-Speicherung**: `models/PPO/wow_bot_v1.zip` (bei Abschluss), `wow_bot_interrupted.zip` (bei Ctrl+C)
- **Status**: Nur `wow_bot_interrupted.zip` existiert — Training wurde nie vollständig abgeschlossen

### run_model.py — Inference

- Lädt `models/PPO/wow_bot_v1` (existiert aktuell nicht!)
- Endlos-Loop: `model.predict(obs)` → `env.step(action)` → bei `done` reset
- Stochastische Policy (nicht deterministisch)

### auto_grind.py — Hybrid-Runner

- Lädt `models/PPO/wow_bot_interrupted`
- **Farm-Route**: 3 Waypoints (Koordinaten in Northshire/Elwynn)
- **Entscheidungslogik**:
  - Im Kampf / Target alive → RL-Policy (deterministisch)
  - Kein Kampf → prüft NPC-Memory nach nächstem bekannten Mob
  - Kein Mob bekannt → folgt der Farm-Route zum nächsten Waypoint
- **Navigation**: `move_to` mit Salami-Slicing (max 50 Units pro Schritt)
- **Scan**: Alle 0.5s `target_nearest:0` als Hintergrund-Scan
- **Tick-Rate**: 0.5s Entscheidungsintervall

### Weitere Scripts

| Script | Zweck |
|---|---|
| `get_gps.py` | Verbindet zum Server, gibt laufend `{"x", "y", "z"}` des ersten Spielers aus. Nützlich zum Erstellen von Farm-Routen. |
| `check_env.py` | Führt 10 zufällige Schritte aus (Actions 0–5), validiert Socket-Verbindung und Reward-Signale. |
| `test_multibot.py` | Steuert 6 Bots (`Autoai` + 5) mit einfacher Scripted-Logic (Heal wenn HP<50%, Smite wenn Target, sonst Target suchen). |
| `run_bot.py` | **BROKEN** — enthält Syntax-Fehler (fehlende Anführungszeichen, Doppelpunkte). Nicht nutzbar. |

## C++ Modul im Detail

### Architektur

Das Modul besteht aus 2 Dateien ohne eigene Header oder CMakeLists:

- **AIControllerLoader.cpp**: Exportiert `Addmod_ai_controllerScripts()` → ruft `AddAIControllerScripts()` auf
- **AIControllerHook.cpp**: Enthält die gesamte Logik in 1008 Zeilen

### Klassen und Komponenten

**`BotLoginQueryHolder`** — Async-DB-Query-Holder für Bot-Login:
- Lädt 16 PreparedStatements (Character-Daten, Spells, Inventory, Talents, Homebind, etc.)
- Pattern analog zum normalen AzerothCore-LoginQueryHolder

**`AIControllerWorldScript`** (erbt `WorldScript`) — Haupt-Update-Loop:
- `OnStartup()`: Startet TCP-Server-Thread
- `OnUpdate(diff)`: Drei Timer-gesteuerte Pfade:
  - **150ms** (`_faceTimer`): Dreht Spieler im Kampf/Casting zum Target
  - **400ms** (`_fastTimer`): Baut JSON-State aus allen Online-Spielern, publisht via `g_CurrentJsonState`
  - **2000ms** (`_slowTimer`): Scannt nearby_mobs per `Cell::VisitObjects` (50 Units Radius)
- Verarbeitet `g_CommandQueue` synchron im Game-Thread

**`AIControllerPlayerScript`** (erbt `PlayerScript`) — Chat-Commands & Event-Hooks:
- `#spawn <Name>`: Spawnt einzelnen Bot
- `#spawnbots`: Spawnt Bota–Bote
- `OnPlayerGiveXP()`: Sammelt XP-Events
- `OnPlayerLevelChanged()`: Setzt Level auf 1 zurück bei Level-Up
- `OnPlayerMoneyChanged()`: Sammelt Copper-Events

### Bot-Spawning-Prozess

1. Character-GUID aus `sCharacterCache` per Name laden
2. Account-ID ermitteln, prüfen dass nicht bereits eingeloggt
3. `WorldSession` erstellen (ohne echten Client/Socket)
4. `BotLoginQueryHolder` mit 16 DB-Queries async ausführen
5. Im Callback: `Player` erstellen, `LoadFromDB()`, zur Map hinzufügen
6. Initial-Packets senden, in DB als online markieren
7. Teleport zum Hardcoded-Spawnpunkt: Map 0 (Eastern Kingdoms), (-8921.037, -120.485, 82.025)

### Globale Variablen (Thread-Safety)

| Variable | Mutex | Zweck |
|---|---|---|
| `g_CurrentJsonState` | `g_Mutex` | Aktueller JSON-State-String |
| `g_CommandQueue` | `g_Mutex` | Warteschlange eingehender Python-Kommandos |
| `g_StateVersion` | atomic | Versionszähler für State-Updates |
| `g_PlayerEvents` | `g_EventMutex` | Per-Player XP/Loot/Level-Event-Akkumulator |
| `g_BotSessions` | `g_BotSessionsMutex` | Account-ID → WorldSession Mapping |

### Hilfs-Funktionen

- **`GetItemScore(ItemTemplate*)`**: Berechnet Score = (Quality×10) + ItemLevel + Armor + Weapon-DPS + (Stats×2)
- **`TryEquipIfBetter(Player*, srcPos)`**: Vergleicht neues Item mit ausgerüstetem, tauscht wenn besser
- **`CreatureCollector`**: GridNotifier der Kreaturen im Umkreis sammelt (filtert Totems, Pets, Critters)
- **`GetFreeBagSlots(Player*)`**: Zählt freie Inventory-Slots (Rucksack + Taschen)
- **`IsBotControlledPlayer(Player*)`**: Prüft ob Spieler ein Bot ist (via `g_BotSessions`)

## Bekannte Probleme & Einschränkungen

### Kritisch
- **nearby_mobs Cache-Bug**: `_cachedNearbyMobsJson` wird nur für den **ersten** Online-Spieler berechnet (das `break` in der Schleife). Alle Spieler im JSON erhalten dieselbe Mob-Liste, auch wenn sie auf verschiedenen Teilen der Map sind.
- **Timer-Bug (behoben?)**: Der README erwähnt, dass `_fastTimer` und `_faceTimer` sich früher gegenseitig blockiert haben. Im aktuellen Code sind sie getrennt (`_faceTimer` für 150ms, `_fastTimer` für 400ms).

### Design-Entscheidungen
- **Level-1-Sandbox**: Bots werden bei Level-Up auf Level 1 zurückgesetzt — gewollt für wiederholtes Niedrigstufentraining
- **sell_grey Fehlbenennung**: Verkauft alle Items mit `SellPrice > 0`, nicht nur graue Items. Hearthstone (6948) ist ausgenommen.
- **Kein Security**: TCP ist Klartext ohne Authentifizierung — nur auf localhost verwenden
- **Eine TCP-Verbindung**: Server akzeptiert mehrere Clients (je ein Thread), aber der State ist global gleich
- **run_bot.py ist kaputt**: Enthält mehrere Syntax-Fehler und ist nicht ausführbar

### Limitierungen
- Hardcoded Spawn-Position (Northshire Abbey / Elwynn Forest)
- Hardcoded Bot-Namen (Bota–Bote, plus Autoai im Test-Script)
- Character muss in der DB existieren bevor `#spawn` funktioniert
- Python-Environment ist teilweise gescriptet (Override-Logik) — gelernte Policy ist abhängig davon

## Workflow

### Sim-Training (Hauptfokus — kein Server nötig)

**Voraussetzungen**: Python 3.x mit `gymnasium`, `numpy`, `stable-baselines3`

```bash
# Standard-Training (reine Sim)
python -m sim.train_sim --steps 500000

# Mit 3D-Terrain aus echten WoW-Daten
python -m sim.train_sim --data-root /pfad/zu/Data --steps 500000

# TensorBoard
tensorboard --logdir python/logs/
```

Logs landen in `logs/PPO_2/`. Zeigt: FPS, Rewards, KL, Entropy, Value/Policy-Loss.

### Live-Server-Training (spätere Phase)

**Voraussetzungen**:
1. AzerothCore aus `src_azeroth_core/` bauen (Standard-CMake-Build)
2. AI-Controller-Modul aus `src_module-ai-controller/` in die AzerothCore-Module integrieren
3. Bot-Characters in der Character-DB anlegen (Namen müssen matchen)
4. Python 3.x mit `gymnasium`, `numpy`, `stable-baselines3`

**Ablauf**:
1. `worldserver` starten → Modul startet TCP auf Port 5000
2. GM-Character einloggen, `#spawnbots` oder `#spawn <Name>` eingeben
3. Python starten:
   - **Training**: `python python/train.py`
   - **Inference**: `python python/run_model.py` (benötigt `wow_bot_v1.zip`)
   - **Hybrid-Grind**: `python python/auto_grind.py` (nutzt `wow_bot_interrupted.zip`)
   - **GPS-Logger**: `python python/get_gps.py` (zum Erstellen neuer Routen)
   - **Multi-Bot-Test**: `python python/test_multibot.py`

## Coding-Konventionen

- **C++**: AzerothCore-Standard (camelCase Methoden, UPPER_CASE Konstanten, Boost.Asio für Netzwerk)
- **Python**: Standard-Python mit `snake_case`, Type Hints fehlen weitgehend
- **Kein Build-System im Modul**: `src_module-ai-controller/` hat keine eigene CMakeLists.txt — muss manuell in den AzerothCore-Module-Build integriert werden
- **Sprache**: Code-Kommentare teilweise auf Deutsch ("Lausche auf Port 5000", "WICHTIG", "ACHTUNG")
- **Keine Tests**: Kein Unit-Test-Framework, nur `check_env.py` als Smoke-Test
