# CLAUDE.md — Project Documentation for ac-share

This repository is an experimental WoW bot training setup. The **current main focus** is on the Python simulation (`python/sim/`), which provides a complete training environment without a running WoW server. A C++ module for AzerothCore (live server integration) exists in parallel and will be used in a later phase.

## Current Focus: Python Simulation

The sim environment (`python/sim/`) replicates the WoW combat system in pure Python:
- **~1000x faster** than live server training (no TCP, no server needed)
- **Identical interface** to `wow_env.py` (same Obs/Action Space)
- **Optional 3D terrain data** from real WoW files (maps/vmaps) via `test_3d_env.py`
- **Full-world creature spawning** from AzerothCore CSV exports via `creature_db.py`
- **Episode logging & visualization** via `sim_logger.py` and `visualize.py`
- **Goal**: Validate all core features (Combat, Targeting, Loot, Spells, Movement, Leveling) in the sim before transferring to the live server

## Repository Structure

```
ac-share/
├── CLAUDE.md                    <- this file
├── README.md                    <- project overview (architecture, workflow)
├── EVALUATION.md                <- project evaluation & improvement suggestions
├── .gitattributes
├── data/                        <- WoW game data files
│   ├── creature.csv             <- All NPC/mob spawns (AzerothCore DB export, 11 MB)
│   ├── creature_template.csv    <- NPC stat templates (3.9 MB)
│   ├── spell_dbc.csv            <- Spell data export (30 MB)
│   ├── map_dbc.csv              <- Map metadata from DBC files
│   ├── 000.vmtree               <- VMAP binary index for collision/LOS
│   ├── 000_27_29.vmtile         <- Sample terrain tile VMAP
│   ├── 0002035.map              <- Binary map tile data
│   └── 000.mmap                 <- Map heightfield data
├── python/                      <- Python RL training, inference & utilities
│   ├── sim/                     <- ** MAIN FOCUS: Offline Simulation **
│   │   ├── combat_sim.py        <- Combat system simulation (Mobs, Spells, Loot, Movement, Exploration, Leveling)
│   │   ├── wow_sim_env.py       <- Gymnasium environment for the sim
│   │   ├── train_sim.py         <- PPO training on the sim (5 bots, no server needed)
│   │   ├── test_sim.py          <- Validation tests (6 tests: engine, spaces, episode, benchmark, combat, levels)
│   │   ├── terrain.py           <- SimTerrain wrapper for 3D terrain in the sim
│   │   ├── creature_db.py       <- AzerothCore CSV creature loader with spatial indexing
│   │   ├── loot_db.py           <- AzerothCore CSV loot table loader (creature_loot_template + item_template)
│   │   ├── sim_logger.py        <- Episode logging for visualization (zero I/O during sim)
│   │   ├── visualize.py         <- Interactive map viewer with episode browser
│   │   └── __init__.py
│   ├── test_3d_env.py           <- 3D terrain/VMAP/LOS/AreaTable from real WoW data
│   ├── wow_env.py               <- Gymnasium environment (live server via TCP)
│   ├── train.py                 <- Multi-bot PPO training (live server)
│   ├── run_model.py             <- Inference loop (trained model)
│   ├── auto_grind.py            <- Hybrid runner: Route + RL policy
│   ├── get_gps.py               <- GPS coordinate logger (for routes)
│   ├── check_env.py             <- Quick env validation test
│   ├── test_multibot.py         <- Multi-bot control with scripted logic
│   ├── run_bot.py               <- BROKEN — syntax errors, not usable
│   ├── npc_memory.json          <- Shared NPC database (baseline)
│   └── npc_memory_*.json        <- Bot-specific NPC memory files
├── src_module-ai-controller/    <- C++ AzerothCore module (2 files)
│   ├── AIControllerHook.cpp     <- All logic (~1025 lines): TCP server,
│   │                               bot spawning, command processing, state publishing
│   └── AIControllerLoader.cpp   <- Module registration (14 lines)
└── src_azeroth_core/            <- Full AzerothCore source tree
    ├── cmake/                   <- Build system
    ├── common/                  <- Shared libraries (Threading, Crypto, Config)
    ├── server/                  <- Server core
    │   ├── apps/                <- worldserver & authserver
    │   ├── game/                <- 51 subsystems (Entities, AI, Spells, Maps, ...)
    │   ├── scripts/             <- Content scripts
    │   ├── shared/              <- Network/protocol
    │   └── database/            <- DB abstraction
    ├── test/                    <- Tests
    └── tools/                   <- External tools
```

**Note**: `models/` and `logs/` directories are created dynamically during training and are not stored in the repository.

## Architecture Overview

### Dual-Path: Sim (Main Focus) + Live Server (Later)

```
  ** CURRENT FOCUS **                     |  LATER PHASE
                                          |
  +----------------------------+          |  +----------------------------------+
  |    CombatSimulation        |          |  |   AzerothCore worldserver        |
  |    python/sim/combat_sim   |          |  |   (C++, AI controller module)    |
  +----------------------------+          |  |   TCP :5000, JSON state stream   |
  | 119 hardcoded spawns       |          |  +----------------+-----------------+
  | + CreatureDB (full world)  |          |                   | TCP
  | 4 Spells, Mob-AI, Loot, XP |          |                   |
  | Level 1-80, Exploration    |          |  +----------------v-----------------+
  | Optional: 3D terrain       |          |  |   WoWEnv (python/wow_env.py)     |
  +------------+---------------+          |  |   Action: Discrete(11)           |
               | direct (in-process)      |  |   Obs:    Box(17,)               |
               |                          |  +----------------+-----------------+
  +------------v---------------+          |                   |
  |  WoWSimEnv (Gymnasium)     |          |          +--------v----------+
  |  python/sim/wow_sim_env    |          |          | train.py / etc.   |
  +----------------------------+          |          +-------------------+
  | Action: Discrete(11)      |          |
  | Obs:    Box(17,)           |          |
  | Similar override logic     |          |
  | Sparse reward design       |<---------+  ** Same interface **
  +----------+-----------------+          |
             |                            |
    +--------v----------+                 |
    |  train_sim.py     |                 |
    |  5 bots, PPO      |                 |
    |  ~5000 FPS        |                 |
    +-------------------+                 |
```

### Goal: Train models in the sim, then transfer to the live server.

## TCP Protocol

### Server -> Python (State Stream)

Every 400ms (or 500ms keepalive) the server sends a JSON line:

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
      "target_level": 1,
      "xp_gained": 0,
      "loot_copper": 0,
      "loot_score": 0,
      "leveled_up": "false",
      "has_shield": "false",
      "target_has_sw_pain": "false",
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

**Important**: `combat`, `casting`, `equipped_upgrade`, `leveled_up`, `has_shield`, `target_has_sw_pain` are strings (`"true"`/`"false"`), not JSON booleans. `xp_gained`, `loot_copper`, `loot_score` are reset after sending (consume-on-read).

### Python -> Server (Commands)

Format: `<playerName>:<actionType>:<value>\n`

| Command | Description |
|---|---|
| `say:<text>` | Player says text in chat |
| `stop:0` | Stops movement |
| `turn_left:0` / `turn_right:0` | Rotates orientation by +/-0.5 rad |
| `move_forward:0` | Moves 3 units forward (ground Z correction) |
| `move_to:<x>:<y>:<z>` | Moves to coordinates |
| `target_nearest:<range>` | Selects nearest valid target (default 30) |
| `target_guid:<guid>` | Selects unit by GUID |
| `cast:<spellId>` | Casts spell (585=Smite auto-targets enemy, 2050=Heal auto-targets self, 17=PW:Shield auto-targets self) |
| `loot_guid:<guid>` | Loots dead creature (<=10 units), auto-equip if better |
| `sell_grey:<vendorGuid>` | Sells all items with SellPrice>0 (except Hearthstone 6948) |
| `reset:0` | Full heal, cooldown reset, teleport to homebind |

## Python Components in Detail

### wow_env.py — Gymnasium Environment (Live Server)

**Class**: `WoWEnv(gym.Env)`

**Initialization**: `WoWEnv(host='127.0.0.1', port=5000, bot_name=None)`
- `bot_name=None`: adopts the first player in the stream
- `bot_name="Bota"`: explicitly filters for this name

**Action Space** — `Discrete(11)`:
| ID | Action |
|---|---|
| 0 | No-op (wait) |
| 1 | move_forward |
| 2 | turn_left |
| 3 | turn_right |
| 4 | Target mob (nearest from nearby_mobs via target_guid) |
| 5 | Cast Smite (Spell 585) |
| 6 | Cast Heal (Spell 2050) |
| 7 | Loot (nearest dead creature via loot_guid) |
| 8 | Sell (to vendor, only in vendor mode) |
| 9 | Cast SW:Pain (Spell 589) |
| 10 | Cast PW:Shield (Spell 17) |

**Observation Vector** — `Box(shape=(17,), dtype=float32)`:
| Index | Value | Range |
|---|---|---|
| 0 | hp_pct (HP/MaxHP) | 0-1 |
| 1 | mana_pct (Mana/MaxMana) | 0-1 |
| 2 | target_hp / 100 | 0-inf |
| 3 | target_exists (1=alive, 0=else) | 0/1 |
| 4 | in_combat | 0/1 |
| 5 | target_distance / 40 (clamped) | 0-1 |
| 6 | relative_angle / pi | -1 to 1 |
| 7 | is_casting | 0/1 |
| 8 | mob_count (nearby mobs / 10) | 0-inf |
| 9 | free_slots / 20 | 0-1 |
| 10 | closest_mob_distance / 40 | 0-1 |
| 11 | closest_mob_angle / pi | -1 to 1 |
| 12 | num_attackers / 5 | 0-inf |
| 13 | target_level / 10 | 0-inf |
| 14 | player_level / 10 | 0-inf |
| 15 | has_shield | 0/1 |
| 16 | target_has_sw_pain | 0/1 |

**Override Logic** (overrides RL decisions):
1. **Vendor Mode**: If `free_slots < 2` and not in combat -> navigates to nearest vendor from NPC memory, sells automatically
2. **Aggro Recovery**: In combat without target -> finds mob attacking the bot, targets it
3. **Cast Guard**: During casting, movement/turning/other casts are suppressed
4. **Loot Automation**: Dead target -> approaches, loots automatically at <=6 units
5. **Range Management**: Stops forward movement at <25 units to target
6. **Heal Block**: Heal is blocked at HP > 85%
7. **Shield Block**: PW:Shield blocked if already shielded
8. **SW:Pain Block**: SW:Pain blocked if already active on target
9. **Sell Block**: Action 8 only allowed in vendor mode

**NPC Memory System**:
- File: `npc_memory_{bot_name}.json` (isolated per bot)
- Stores all encountered mobs with position, level, vendor flag, etc.
- Blacklist: Dead/looted mobs are ignored for 15 minutes
- Auto-save every 30 seconds (atomic via .tmp file)
- Used by `auto_grind.py` for memory-based targeting

**Deterministic Initial Heading** (`_initial_heading_kick`):
Each bot rotates in a different direction on reset to improve distribution:
- Autoai: 0 steps, Bota: 2, Botb: 4, Botc: 6, Botd: 8, Bote: 10 (each ~0.5 rad)

## Python Simulation in Detail (Main Focus)

### combat_sim.py — Combat System Simulation

**Class**: `CombatSimulation`

Simulates the complete WoW combat system in pure Python:
- **119 hardcoded mob spawns** from real AzerothCore DB spawn positions (4 mob types, Level 1-3)
- **Full-world creature spawning** via `CreatureDB` from CSV exports (chunk-based spatial indexing, 100-unit chunks)
- **Natural difficulty gradient**: Wolves (L1) in the north -> Kobolds (L1-3) in the south/east
- **Priest Spells**: Smite (585), Heal (2050), SW:Pain (589), PW:Shield (17)
- **Mob AI**: Aggro range (10-20 units), chase, melee attack, leash (60 units)
- **Loot System**: Copper drops, simplified item score system
- **Respawn**: Dead mobs respawn after 60s at original spawn point
- **XP**: AzerothCore formula `BaseGain()` with gray level, ZeroDifference — mobs below gray level give 0 XP
- **Level System**: Level 1-80 with XP table, per level +10 Smite damage, +5 Heal, +50 HP, +5 Mana
- **Exploration**: Three-tier tracking (`visited_areas`, `visited_zones`, `visited_maps`)
  - Real WoW Area/Zone/Map IDs from AreaTable.dbc when `env3d` + DBC available
  - Grid fallback without 3D data: Areas=50x50 units, Zones=200x200 units
  - `_new_areas`/`_new_zones`/`_new_maps` counters (consume-on-read like XP/Loot)
- **3D Terrain** (optional via `terrain` parameter): Z coordinates, walkability checks, LOS checks for spells
- **State Dict**: Identical to the TCP JSON of the live server
- **Regen System**: HP regen 0.67/tick OOC (after 6s combat delay), Mana regen 2.75/tick while not casting

**XP Formula** (from AzerothCore `Formulas.h`/`.cpp`):
- **Mob >= Player Level**: `((pl*5 + 45) * (20 + min(diff, 4)) / 10 + 1) / 2`
- **Mob > Gray Level**: `(pl*5 + 45) * (ZD + mob - pl) / ZD`
- **Mob <= Gray Level**: `0 XP`

**Stat Scaling per Level**: HP: 72 + (L-1)*50, Mana: 123 + (L-1)*5, Smite: (13-17) + (L-1)*10, Heal: (46-56) + (L-1)*5

**Initialization**: `CombatSimulation(num_mobs=None, seed=None, terrain=None, env3d=None, creature_db=None)`
- `num_mobs=None`: Uses all 119 hardcoded spawn positions
- `terrain`: `SimTerrain` instance for heights, LOS, walkability
- `env3d`: `WoW3DEnvironment` instance for Area/Zone lookups via AreaTable.dbc
- `creature_db`: `CreatureDB` instance for full-world chunk-based mob spawning

### creature_db.py — Creature Database Loader

Loads AzerothCore CSV exports for full-world creature spawning with spatial indexing:
- **Loads**: `creature_template.csv` (stat templates) and `creature.csv` (spawn positions)
- **Spatial Index**: Dict `(map, chunk_x, chunk_y) -> [SpawnPoint]` for O(1) chunk lookups
- **Chunk Size**: 100 world-units per chunk
- **Stat Interpolation**: HP, damage, XP anchored by level (1-83), interpolated for arbitrary levels
- **Filters**: Skips friendly factions (Stormwind, etc.), critters, totems, non-combat pets, gas clouds
- **Data Classes**: `CreatureTemplate` (entry, name, level range, faction, npc flags, stats) and `SpawnPoint` (guid, entry, map, position)

### loot_db.py — Loot Table Loader

Loads AzerothCore CSV exports for realistic item drops with full group/reference logic:
- **Loads**: `item_template.csv` (item data, scores), `creature_loot_template.csv` (drop tables), `reference_loot_template.csv` (optional, shared loot references)
- **Item Score**: Precomputed via `GetItemScore` formula: `(Quality*10) + ItemLevel + Armor + WeaponDPS + (TotalStats*2)`
- **Group System** (AzerothCore standard):
  - **Group 0**: Each entry rolls independently (chance %)
  - **Group N>0**: Exactly one entry wins per group (weighted selection)
  - **chance=0**: In grouped entries, equal share of remaining probability
- **Reference Resolution**: Recursive processing from `reference_loot_template`, with max depth 5 to prevent loops
- **Data Classes**: `ItemData` (entry, name, quality, sell_price, inventory_type, item_level, score), `LootEntry` (item, reference, chance, group_id, counts), `LootResult` (item + count)
- **Graceful Degradation**: Auto-discovers CSV files, missing files silently skipped — check `loaded` property
- **Integration**: When loaded, `CombatSimulation.do_loot()` uses real loot tables; otherwise falls back to random loot

**Required CSV Exports** (semicolon-delimited, double-quote enclosed):
- `creature_loot_template.csv`: Entry, Item, Reference, Chance, QuestRequired, LootMode, GroupId, MinCount, MaxCount, Comment
- `item_template.csv`: entry, name, class, subclass, Quality, SellPrice, InventoryType, ItemLevel, armor, dmg_min1, dmg_max1, delay, stat_type1..10, stat_value1..10
- `reference_loot_template.csv` (optional): same schema as creature_loot_template
- `creature_template.csv` (updated): add `lootid` column for creature→loot table mapping (defaults to entry if missing)

### wow_sim_env.py — Gymnasium Sim Environment

Drop-in replacement for `wow_env.py`:
- **Same Action Space**: `Discrete(11)` — No-op, Move, Turn x2, Target, Smite, Heal, Loot, Sell, SW:Pain, PW:Shield
- **Same Obs Space**: `Box(17,)` — HP%, Mana%, Target-HP, Combat, Distance, Angle, etc.
- **Sparse Reward Design**: Focused on real outcomes only (see reward table below)
- **Similar Override Logic**: Aggro, Cast-Guard, Range-Management, Heal/Shield/DoT blocks
- **No Episode Step Limit**: Episode runs until death (bot should level as far as possible)
- **Stall Detection**: Truncates episode after 30,000 steps without XP gain
- **OOM is NOT terminal**: Bot must learn to wait for mana regen

**Initialization**: `WoWSimEnv(bot_name="SimBot", num_mobs=None, seed=None, data_root=None, creature_csv_dir=None, log_dir=None, log_interval=1)`
- `data_root`: Path to WoW `Data/` directory -> enables 3D terrain (`SimTerrain`) + area lookups (`WoW3DEnvironment` with AreaTable.dbc)
- `creature_csv_dir`: Path to directory containing `creature.csv` + `creature_template.csv` -> enables full-world creature spawning
- `log_dir`: Path for episode JSONL logs (used by `visualize.py`)
- Without `data_root`: Flat terrain, grid-based exploration detection

### Reward Tables

#### Sim Rewards (wow_sim_env.py — Sparse Design)

| Signal | Value | Notes |
|---|---|---|
| Step Penalty | -0.01 | per tick |
| Idle Penalty | -0.05 | Noop without casting |
| Damage Dealt | min(dmg * 0.03, 1.0) | damage to target |
| XP/Kill | 10.0 + xp * 0.5 | ~35 per 50-XP kill, scales with XP |
| Level-Up | +15.0 * levels | per level gained |
| Equipment Upgrade | +3.0 | |
| Loot | min((copper * 0.01) + (score * 0.2), 3.0) | capped |
| Sell | +2.0 | slots freed |
| New Area Entered | +1.0 | real WoW Area ID or grid fallback (once per episode) |
| New Zone Entered | +3.0 | real WoW Zone ID (once per episode) |
| New Map Entered | +10.0 | real WoW Map ID (once per episode) |
| Death | -15.0 | terminal, overrides all other rewards |

#### Live Rewards (wow_env.py — More Shaped)

| Signal | Value | Notes |
|---|---|---|
| Step Penalty | -0.01 | per tick |
| Idle Penalty | -0.03 | Noop without casting |
| Discovery | +0.5 | new mob GUID added to memory |
| Approach | clip(delta * 0.05, -0.2, +0.3) | potential-based, closer to target |
| Damage Dealt | min(dmg * 0.03, 1.0) | damage to target |
| Facing | quality * 0.08 | in combat, facing target |
| XP/Kill | 3.0 + min(xp * 0.05, 2.0) | ~5 per kill |
| Level-Up | +15.0 | terminal |
| Equipment Upgrade | +3.0 | |
| Loot | min((copper * 0.01) + (score * 0.2), 3.0) | capped |
| Sell | +2.0 | slots freed |
| Action-Specific | +/-0.1 to 0.5 | context-based bonuses for Smite/Heal/SW:Pain/PW:Shield |
| Death | -5.0 | terminal |
| OOM (<5% Mana) | -2.0 | terminal |

**Key Differences**: The sim uses a **sparse reward** design (only real outcomes matter), while the live env uses **more reward shaping** (approach, facing, discovery, action-specific bonuses). The sim has a harsher death penalty (-15 vs -5) and higher XP/kill reward (10+xp*0.5 vs 3+xp*0.05). OOM is only terminal in the live env.

### sim_logger.py — Episode Logging System

Lightweight episode logger for training visualization:
- **Zero I/O during simulation**: All data buffered in memory
- **JSONL format**: One JSON object per episode, written at episode end
- **Trail data**: Step-by-step bot position, HP%, level, combat state, orientation
- **Event data**: Kills, deaths, level-ups with position and step number
- **Mob snapshot**: All mob spawn positions for map overlay
- **Configurable interval**: Record every N steps (default 1)
- **Atomic writes**: Uses file append, no temp files needed
- **`load_episodes()`**: Utility function for reading JSONL files

### visualize.py — Interactive Map Viewer

Interactive map visualization for analyzing training episodes:
- **Primary mode**: Reads from JSONL log files (`--log-dir`)
- **Fallback mode**: `--run` flag runs simulation directly
- **Interactive controls**: Episode slider, bot checkboxes, zoom slider (0.1-10x), right-click drag panning, scroll-wheel zoom
- **Keyboard**: Arrow keys navigate episodes, 'r' resets view
- **Visualization**: Color-coded trails with time progression, mob spawn overlays, event markers (kills=red X, level-ups=gold star, deaths=red X)
- **Log panel**: Toggleable event log showing kills/deaths/level-ups
- **Static export**: `--output` flag saves map to PNG

### train_sim.py — Sim Training

- **5 bots** in `SubprocVecEnv`
- **PPO** with `ent_coef=0.1`, `n_steps=256`, `batch_size=128`, `learning_rate=3e-4`
- **TensorBoard Logs** in `logs/PPO_2/`
- **Episode Callbacks** with kills, XP, deaths, areas/zones/maps explored, levels gained, final level
- **TensorBoard Metrics**: `gameplay/ep_areas_explored`, `gameplay/ep_zones_explored`, `gameplay/ep_maps_explored`, `gameplay/ep_levels_gained`, `gameplay/ep_final_level`
- **Real per-iteration FPS tracking** (not cumulative)
- **~5000+ FPS** (without 3D terrain)
- **Model versioning**: Auto-increments `wow_bot_sim_v1.zip`, `v2.zip`, etc. Interrupt save: `wow_bot_sim_interrupted.zip`
- **`--data-root`**: Optional, enables 3D terrain + real WoW area IDs
- **`--creature-data`**: Optional, enables full-world creature spawning from CSV
- **`--log-dir`**: Optional, enables episode trail logging for visualization
- **`--log-interval`**: How often to write episode logs (default: every episode)

### test_sim.py — Validation Tests

7 test functions:
1. **test_combat_engine()**: Basic engine initialization, movement, targeting, spell casting
2. **test_gym_env()**: Gymnasium spaces validation — Box(17,) obs, Discrete(11) actions
3. **test_random_episode()**: 1000-step episode with random actions
4. **test_performance()**: FPS benchmark (~5000+ FPS single-env)
5. **test_combat_scenario()**: Scripted combat with targeting and spell rotation
6. **test_level_system()**: XP formulas, level-up mechanics, stat scaling, multi-level-up
7. **test_loot_tables()**: LootDB loading, item score computation, group rolling distribution, sim integration, upgrade detection, fallback without DB

### test_3d_env.py — 3D Terrain + Area System from Real WoW Data

Reads original WoW files (maps/, vmaps/, dbc/):
- **Terrain Heights**: 129x129 height grid per tile, triangle interpolation
- **LOS (Line of Sight)**: VMAP spawns (buildings, trees) with AABB ray intersection
- **AreaTable.dbc Parser** (`parse_area_table_dbc()`): Reads binary WDBC file -> Dict `{area_id: AreaTableEntry}` with name, zone, map, level, explore flag
- **`AreaTableEntry`**: Dataclass with `id`, `map_id`, `zone`, `explore_flag`, `flags`, `area_level`, `name`
- **Area Lookup**: `get_area_id(map, x, y)` -> real WoW Area ID from 16x16 grid per tile
- **Zone Lookup**: `get_zone_id(map, x, y)` -> parent zone via AreaTable hierarchy
- **Area Info**: `get_area_info(map, x, y)` -> full dict with `area_name`, `zone_name`, `area_level` etc.
- **Dynamic Tile Loading**: Tiles loaded on-demand as the bot enters new areas — AI is not limited to pre-loaded regions
- **HeightCache**: Pre-computed numpy grid for O(1) height lookups (~100x faster)
- **SpatialLOSChecker**: Spatially indexed LOS check (~100-500x faster than brute force)

### terrain.py — SimTerrain Wrapper

Lightweight wrapper around `WoW3DEnvironment` for the sim:
- **`SimTerrain(data_root)`**: Loads vmtree (BIH index, once) + initial tiles around spawn
- **`ensure_loaded(x, y)`**: Loads map tiles + VMAPs on-demand when the player enters a new tile (3x3 around position). Cheap no-op when on the same tile.
- **`get_height(x, y)`**: Terrain height at world coordinates (fallback to 82.025 without data)
- **`check_los(x1,y1,z1, x2,y2,z2)`**: Line-of-sight check with eye height offset (+1.7)
- **`check_walkable(x1,y1,z1, x2,y2,z2)`**: Terrain walkability (slope/step check)
- **Dynamic Tile Loading**: Tiles loaded on-demand as the bot enters new areas. Loaded tiles remain in cache (never unloaded).

**Exploration Hierarchy** (real WoW data):
```
Map 0 (Eastern Kingdoms)
  +- Zone 12 (Elwynn Forest)
       +- Area 9 (Northshire Valley)
       +- Area 87 (Goldshire)
       +- Area 57 (Crystal Lake)
       +- ...
  +- Zone 40 (Westfall)
       +- Area 108 (Sentinel Hill)
       +- ...
```

### train.py — PPO Training (Live Server)

- **Bots**: `["Bota", "Botb", "Botc", "Botd", "Bote"]` (5 parallel environments)
- **Vectorization**: `SubprocVecEnv` (separate processes per bot)
- **Algorithm**: PPO with `MlpPolicy` (2-layer FC network)
- **Hyperparameters**: `n_steps=128`, `batch_size=64`, `ent_coef=0.01`, `total_timesteps=10000`
- **Logs**: TensorBoard in `logs/PPO_0/`
- **Model Saving**: `models/PPO/wow_bot_v1.zip` (on completion), `wow_bot_interrupted.zip` (on Ctrl+C)
- **Status**: Only `wow_bot_interrupted.zip` would exist — training was never fully completed

### run_model.py — Inference

- Loads `models/PPO/wow_bot_v1` (**does not exist!**)
- Infinite loop: `model.predict(obs)` -> `env.step(action)` -> reset on `done`
- Stochastic policy (not deterministic)

### auto_grind.py — Hybrid Runner

- Loads `models/PPO/wow_bot_interrupted`
- **Farm Route**: 3 waypoints (coordinates in Northshire/Elwynn)
- **Decision Logic**:
  - In combat / target alive -> RL policy (deterministic)
  - No combat -> checks NPC memory for nearest known mob
  - No mob known -> follows farm route to next waypoint
- **Navigation**: `move_to` with salami-slicing (max 50 units per step)
- **Scan**: Every 0.5s `target_nearest:0` as background scan
- **Tick Rate**: 0.5s decision interval

### Other Scripts

| Script | Purpose |
|---|---|
| `get_gps.py` | Connects to server, continuously outputs `{"x", "y", "z"}` of the first player. Useful for creating farm routes. |
| `check_env.py` | Runs 10 random steps (actions 0-5), validates socket connection and reward signals. |
| `test_multibot.py` | Controls 6 bots (`Autoai` + 5) with simple scripted logic (Heal if HP<50%, PW:Shield if in combat, SW:Pain if no DoT, Smite if target, else target search). |
| `run_bot.py` | **BROKEN** — contains syntax errors (missing quotes, colons). Not usable. |

## C++ Module in Detail

### Architecture

The module consists of 2 files without its own header or CMakeLists:

- **AIControllerLoader.cpp**: Exports `Addmod_ai_controllerScripts()` -> calls `AddAIControllerScripts()`
- **AIControllerHook.cpp**: Contains all logic in ~1025 lines

### Classes and Components

**`BotLoginQueryHolder`** — Async DB query holder for bot login:
- Loads 16 PreparedStatements (character data, spells, inventory, talents, homebind, etc.)
- Pattern analogous to the normal AzerothCore LoginQueryHolder

**`AIControllerWorldScript`** (inherits `WorldScript`) — Main update loop:
- `OnStartup()`: Starts TCP server thread
- `OnUpdate(diff)`: Three timer-driven paths:
  - **150ms** (`_faceTimer`): Rotates player in combat/casting to face target
  - **400ms** (`_fastTimer`): Builds JSON state from all online players, publishes via `g_CurrentJsonState`
  - **2000ms** (`_slowTimer`): Scans nearby_mobs per player via `Cell::VisitObjects` (50 units radius)
- Processes `g_CommandQueue` synchronously in the game thread

**`AIControllerPlayerScript`** (inherits `PlayerScript`) — Chat commands & event hooks:
- `#spawn <Name>`: Spawns a single bot
- `#spawnbots`: Spawns Bota-Bote
- `OnPlayerGiveXP()`: Collects XP events
- `OnPlayerLevelChanged()`: Resets level to 1 on level-up
- `OnPlayerMoneyChanged()`: Collects copper events

### Bot Spawning Process

1. Load character GUID from `sCharacterCache` by name
2. Determine account ID, check not already logged in
3. Create `WorldSession` (without real client/socket)
4. Execute `BotLoginQueryHolder` with 16 DB queries async
5. In callback: Create `Player`, `LoadFromDB()`, add to map
6. Send initial packets, mark as online in DB
7. Teleport to hardcoded spawn point: Map 0 (Eastern Kingdoms), (-8921.037, -120.485, 82.025)

### Global Variables (Thread Safety)

| Variable | Mutex | Purpose |
|---|---|---|
| `g_CurrentJsonState` | `g_Mutex` | Current JSON state string |
| `g_CommandQueue` | `g_Mutex` | Queue of incoming Python commands |
| `g_StateVersion` | atomic | Version counter for state updates |
| `g_PlayerEvents` | `g_EventMutex` | Per-player XP/loot/level event accumulator |
| `g_BotSessions` | `g_BotSessionsMutex` | Account-ID -> WorldSession mapping |

### Helper Functions

- **`GetItemScore(ItemTemplate*)`**: Calculates score = (Quality*10) + ItemLevel + Armor + Weapon-DPS + (Stats*2)
- **`TryEquipIfBetter(Player*, srcPos)`**: Compares new item with equipped, swaps if better
- **`CreatureCollector`**: GridNotifier that collects creatures in radius (filters totems, pets, critters)
- **`GetFreeBagSlots(Player*)`**: Counts free inventory slots (backpack + bags)
- **`IsBotControlledPlayer(Player*)`**: Checks if player is a bot (via `g_BotSessions`)

## Known Issues & Limitations

### Design Decisions
- **Level-1 Sandbox**: Bots are reset to level 1 on level-up on the live server — intended for repeated low-level training
- **sell_grey Misnomer**: Sells all items with `SellPrice > 0`, not just gray items. Hearthstone (6948) is excluded.
- **No Security**: TCP is plaintext without authentication — use only on localhost
- **Multiple TCP Connections**: Server accepts multiple clients (one thread each), but the state is globally the same
- **run_bot.py is Broken**: Contains multiple syntax errors and is not executable

### Reward Parity Gap
- The sim uses **sparse reward design** (only real outcomes: XP, kills, deaths, exploration)
- The live env uses **more shaping** (approach, facing, discovery, action-specific bonuses)
- XP/kill reward differs: sim=10+xp*0.2, live=3+xp*0.05
- Death penalty differs: sim=-30, live=-5
- OOM: sim=not terminal, live=-2 terminal
- When transferring sim-trained models to live, reward behavior will differ

### Limitations
- Hardcoded spawn position (Northshire Abbey / Elwynn Forest) — sim always starts there
- Hardcoded bot names (Bota-Bote, plus Autoai in test script)
- Character must exist in the DB before `#spawn` works
- Python environment is partially scripted (override logic) — learned policy depends on it
- Terrain tiles are loaded on-demand, but only for Map 0 (Eastern Kingdoms) — map transfer not yet implemented
- Exploration rewards not yet implemented in `wow_env.py` (only in sim)

## Workflow

### Sim Training (Main Focus — No Server Needed)

**Prerequisites**: Python 3.x with `gymnasium`, `numpy`, `stable-baselines3`

```bash
# Standard training (pure sim)
python -m sim.train_sim --steps 500000

# With 3D terrain from real WoW data
python -m sim.train_sim --data-root /path/to/Data --steps 500000

# With full-world creature spawning
python -m sim.train_sim --creature-data /path/to/data --steps 500000

# With episode logging for visualization
python -m sim.train_sim --log-dir logs/episodes --steps 500000

# Visualize training episodes
python -m sim.visualize --log-dir logs/episodes

# TensorBoard
tensorboard --logdir python/logs/
```

Logs go to `logs/PPO_2/`. Shows: FPS, Rewards, KL, Entropy, Value/Policy Loss + Gameplay Metrics (Kills, XP, Deaths, Areas/Zones/Maps explored, Levels gained, Final level).

### Live Server Training (Later Phase)

**Prerequisites**:
1. Build AzerothCore from `src_azeroth_core/` (standard CMake build)
2. Integrate AI controller module from `src_module-ai-controller/` into AzerothCore modules
3. Create bot characters in the character DB (names must match)
4. Python 3.x with `gymnasium`, `numpy`, `stable-baselines3`

**Process**:
1. Start `worldserver` -> module starts TCP on port 5000
2. Log in with GM character, type `#spawnbots` or `#spawn <Name>`
3. Start Python:
   - **Training**: `python python/train.py`
   - **Inference**: `python python/run_model.py` (needs `wow_bot_v1.zip`)
   - **Hybrid Grind**: `python python/auto_grind.py` (uses `wow_bot_interrupted.zip`)
   - **GPS Logger**: `python python/get_gps.py` (for creating new routes)
   - **Multi-Bot Test**: `python python/test_multibot.py`

## Coding Conventions

- **C++**: AzerothCore standard (camelCase methods, UPPER_CASE constants, Boost.Asio for networking)
- **Python**: Standard Python with `snake_case`, type hints are mostly missing
- **No Build System in Module**: `src_module-ai-controller/` has no CMakeLists.txt — must be manually integrated into the AzerothCore module build
- **Language**: Code comments partially in German ("Lausche auf Port 5000", "WICHTIG", "ACHTUNG")
- **Tests**: `sim/test_sim.py` (6 tests for sim validation), `check_env.py` (live env smoke test)

## Progress & Status

### What Works (Completed)

| Component | Status | Details |
|---|---|---|
| **CombatSimulation Engine** | done | 119 spawns + CreatureDB, 4 spells, mob AI, loot, XP, respawn, exploration, leveling (1-80) |
| **WoWSimEnv (Gym Interface)** | done | Discrete(11) actions, Box(17) obs, sparse rewards, stall detection |
| **train_sim.py (PPO Training)** | done | 5 bots, SubprocVecEnv, TensorBoard, gameplay metrics, episode logging |
| **Loot Table System** | done | LootDB CSV loader, AzerothCore group/reference logic, item scores, upgrade detection, sim integration with fallback |
| **test_sim.py (Validation)** | done | 7 tests: engine, gym spaces, random episode, benchmark, scripted combat, level system, loot tables |
| **3D Terrain System** | done | Maps/VMAPs parser, HeightCache, SpatialLOSChecker, SimTerrain wrapper |
| **AreaTable.dbc Parser** | done | Reads all areas/zones/maps of the WoW world, on-demand tile loading |
| **Exploration System** | done | 3-tier tracking (Area/Zone/Map), rewards, TensorBoard metrics |
| **CreatureDB (Full World)** | done | CSV loader, spatial index, stat interpolation, attackability checks |
| **Episode Logger** | done | Zero-I/O JSONL logger, trail data, events, mob snapshots |
| **Visualization** | done | Interactive map viewer with episode slider, zoom, bot filters, event log |
| **Override Logic** | done | Vendor, aggro, cast guard, loot, range mgmt — in both envs |
| **wow_env.py (Live Server)** | done | TCP connection, NPC memory, blacklist, override logic, shaped rewards |
| **C++ AI Controller Module** | done | Bot spawning, TCP server, state publishing, command processing, per-player mob lists |
| **auto_grind.py** | done | Hybrid runner with farm route + RL policy |
| **train.py (Live Training)** | done | Multi-bot PPO, but only interrupted runs so far |

### Known Gaps & Parity Differences

| Problem | Area | Severity | Details |
|---|---|---|---|
| **Exploration missing in wow_env.py** | Live Env | medium | Sim has Area/Zone/Map exploration rewards, live env does not yet — these rewards will be missing during sim->live transfer |
| **Reward parity gap** | Both Envs | medium | Sim uses sparse design (XP=10+xp*0.5, death=-15), live uses more shaping (XP=3+xp*0.05, death=-5) — trained model may not transfer cleanly |
| **run_bot.py broken** | Script | low | Syntax errors (missing quotes, colons, brackets) — not usable |
| **run_model.py references wow_bot_v1** | Script | low | Model does not exist, only wow_bot_interrupted.zip available |
| **Vendor system simplified** | Sim | low | Sim has no real vendors — sell action only frees slots, without copper gain. Loot tables provide real item data but sell copper is not tracked. |
| **Loot table CSVs not yet exported** | Sim | low | LootDB system ready but needs `creature_loot_template.csv`, `item_template.csv` CSV exports from AzerothCore DB. Falls back to random loot without them. `creature_template.csv` should be re-exported with `lootid` column. |
| **No training artifacts** | Training | info | Neither models/ nor logs/ directories exist currently — no completed training run stored |

## Roadmap

### Phase 1: Validate Sim Training (Current)

**Goal**: Train a stable PPO model in the sim that demonstrates basic combat skills and leveling.

1. **First complete training run**
   - Run `python -m sim.train_sim --steps 500000`
   - Check TensorBoard metrics: are kills/XP per episode rising? Is the death rate falling?
   - Save checkpoint as `wow_bot_sim_v1.zip`

2. **Hyperparameter tuning**
   - Vary `ent_coef` (0.01-0.1) — too little exploration vs. too much randomness
   - Adjust `n_steps` and `batch_size` based on reward curve
   - Increase `total_timesteps` to 1M-5M if reward is still rising

3. **Evaluate training metrics**
   - Kill rate per episode should reach >2
   - Death rate should fall below 30%
   - Areas explored as indicator for movement behavior
   - Final level as indicator for sustained survival

### Phase 2: Improve Sim Quality

4. **Test 3D terrain in training**
   - Run `--data-root` training, measure FPS impact
   - Compare: does the bot learn better/differently with terrain?
   - Use LOS blockages and walkability as additional learning signals

5. **Synchronize rewards between sim and live**
   - Either bring live env closer to sparse design, or vice versa
   - Add exploration tracking to `wow_env.py`
   - Ensure reward parity for sim->live transfer

6. **Check combat balancing**
   - Mob damage, spell damage, mana costs vs. real WoW values
   - Fine-tune aggro range and leash distance
   - Test multi-aggro situations (2+ mobs simultaneously)

### Phase 3: Transfer to Live Server

7. **Test sim model on live server**
   - Run trained `wow_bot_sim_v1.zip` with `auto_grind.py` or `run_model.py` against real server
   - Observe: which behaviors transfer, which don't?
   - Delta analysis: where does sim behavior deviate from live behavior?

8. **Live training with sim-pretrained model**
   - `train.py --resume models/PPO/wow_bot_sim_v1.zip` — fine-tuning on real server
   - Lower learning rate for fine-tuning (1e-4 instead of 3e-4)
   - Compare: sim-pretrained vs. from-scratch on live

### Phase 4: Extensions (Later)

9. **More spells / higher levels**
    - Additional Priest spells from level 4+ (Renew, Mind Blast, Fade)
    - Mob types with special abilities (ranged, caster, runners)

10. **Multi-zone navigation**
    - Bot should independently explore Elwynn Forest (not just Northshire)
    - Waypoint system or curiosity-driven exploration
    - Zone-specific mob scaling

11. **run_bot.py repair or replace**
    - Fix syntax errors or replace with clean script
    - Simple inference runner that supports both sim and live

12. **Automated tests**
    - CI pipeline with `test_sim.py` as minimum requirement
    - Regression tests for reward parity (sim vs. live)
    - Performance benchmark as gate (FPS must not fall below threshold)
