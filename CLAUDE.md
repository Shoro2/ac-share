# CLAUDE.md — Project Documentation for ac-share

This repository is an experimental WoW bot training setup. The **current main focus** is on the Python simulation (`python/sim/`), which provides a complete training environment without a running WoW server. A C++ module for AzerothCore (live server integration) exists in parallel and will be used in a later phase.

## Local Data Paths (Hardcoded Reference)

These are the actual paths on the dev machine. Use them as defaults for `--data-root`, `--creature-data`, etc.:

| Data | Path |
|---|---|
| **WoW Data** (maps, vmaps, mmaps, DBC) | `C:\wowstuff\WoWKI_serv\Data` |
| **DB Exports / CSVs** (creature.csv, quest_template.csv, etc.) | `C:\wowstuff\WoWKI_serv\python\dbexport` |

- `--data-root C:\wowstuff\WoWKI_serv\Data` — enables 3D terrain, LOS, AreaTable.dbc
- `--creature-data C:\wowstuff\WoWKI_serv\python\dbexport` — enables full-world creature spawning, loot tables, quest CSV loading

**Note**: The `data/` directory in the repo contains a subset of these exports (creature.csv, quest_template.csv, etc.) and is auto-detected as fallback.

## Current Focus: Python Simulation

The sim environment (`python/sim/`) replicates the WoW combat system in pure Python:
- **~1000x faster** than live server training (no TCP, no server needed)
- **WotLK 3.3.5 stat system** — all 10 classes, 5 primary stats, full combat rating conversions from DBC tables
- **19-slot equipment system** — equip/unequip with automatic stat recalculation, combat-locked
- **9 Priest spells** with spell power coefficients from AzerothCore `spell_bonus_data`
- **Armor mitigation** using WotLK formula (Unit.cpp), spell crit from Intellect + rating
- **Optional 3D terrain data** from real WoW files (maps/vmaps) via `test_3d_env.py`
- **Full-world creature spawning** from AzerothCore CSV exports via `creature_db.py`
- **Episode logging & visualization** via `sim_logger.py` and `visualize.py`
- **Goal**: Validate all core features (Combat, Targeting, Loot, Spells, Stats, Gearing, Movement, Leveling) in the sim before transferring to the live server

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
│   ├── creature_loot_template.csv <- Creature loot tables (AzerothCore DB export)
│   ├── creature_queststarterr.csv <- Quest giver NPC->quest mapping
│   ├── creature_questender.csv  <- Quest ender NPC->quest mapping
│   ├── item_template.csv        <- Item data (quality, sell price, stats)
│   ├── quest_template.csv       <- Quest definitions (~9.5K quests)
│   ├── quest_template_addon.csv <- Quest chain info (PrevQuestID, NextQuestID)
│   ├── reference_loot_template.csv <- Shared loot reference tables
│   ├── QuestXP.dbc              <- Quest XP rewards per level/difficulty (binary DBC)
│   ├── spell_dbc.csv            <- Spell data export (30 MB)
│   ├── map_dbc.csv              <- Map metadata from DBC files
│   ├── 000.vmtree               <- VMAP binary index for collision/LOS
│   ├── 000_27_29.vmtile         <- Sample terrain tile VMAP
│   ├── 0002035.map              <- Binary map tile data
│   └── 000.mmap                 <- Map heightfield data
├── python/                      <- Python RL training, inference & utilities
│   ├── sim/                     <- ** MAIN FOCUS: Offline Simulation **
│   │   ├── combat_sim.py        <- Combat system simulation (Mobs, Spells, Loot, Movement, Exploration, Leveling, Quests)
│   │   ├── wow_sim_env.py       <- Gymnasium environment for the sim (Box(23), Discrete(12))
│   │   ├── train_sim.py         <- PPO training on the sim (5 bots, no server needed)
│   │   ├── test_sim.py          <- Validation tests (10 tests: engine, spaces, episode, benchmark, combat, levels, loot, vendor, quests, quest CSV loading)
│   │   ├── quest_db.py          <- Quest system: CSV loader + hardcoded fallback, objectives, NPC data, quest chains
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
  | 84 hardcoded spawns        |          |  +----------------+-----------------+
  | + CreatureDB (full world)  |          |                   | TCP
  | 9 Spells, WotLK Stats     |          |                   |
  | 19-slot Gearing, Leveling  |          |  +----------------v-----------------+
  | Optional: 3D terrain       |          |  |   WoWEnv (python/wow_env.py)     |
  +------------+---------------+          |  |   Action: Discrete(12)           |
               | direct (in-process)      |  |   Obs:    Box(23,)               |
               |                          |  +----------------+-----------------+
  +------------v---------------+          |                   |
  |  WoWSimEnv (Gymnasium)     |          |          +--------v----------+
  |  python/sim/wow_sim_env    |          |          | train.py / etc.   |
  +----------------------------+          |          +-------------------+
  | Action: Discrete(17)      |          |
  | Obs:    Box(38,)           |          |
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

**Action Space** — `Discrete(12)`:
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
| 11 | Quest NPC Interact (accept/turn-in, with auto-navigation) |

**Observation Vector** — `Box(shape=(23,), dtype=float32)`:
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
| 17 | has_active_quest | 0/1 |
| 18 | quest_progress (objectives ratio) | 0-1 |
| 19 | quest_npc_nearby | 0/1 |
| 20 | quest_npc_distance / 40 | 0-1 |
| 21 | quest_npc_angle / pi | -1 to 1 |
| 22 | quests_completed / 10 | 0-inf |

Quest dims (17-22) are always present but zero when quests are disabled (`enable_quests=False`).

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

Simulates the complete WoW 3.3.5 WotLK combat system in pure Python. All formulas derived from AzerothCore C++ source (StatSystem.cpp, Player.cpp, Unit.cpp, DBC game tables). Supports all 10 classes (stat framework), leveling 1–80. Currently only Priest has spell implementations.

- **84 hardcoded mob spawns** from real AzerothCore DB spawn positions (4 mob types, Level 1-3)
- **Full-world creature spawning** via `CreatureDB` from CSV exports (chunk-based spatial indexing, 100-unit chunks)
- **Natural difficulty gradient**: Wolves (L1) in the north -> Kobolds (L1-3) in the south/east
- **WotLK 3.3.5 Attribute System** (see details below): 5 primary stats, combat ratings, spell power, armor mitigation
- **19 Equipment Slots**: Full WoW gearing with stat recalculation on equip/unequip, combat-locked
- **9 Priest Spells**: Smite (585), Lesser Heal (2050), SW:Pain (589), PW:Shield (17), Mind Blast (8092), Renew (139), Holy Fire (14914), Inner Fire (588), PW:Fortitude (1243)
- **Spell Power Scaling**: All spells scale with `total_spell_power` via WotLK coefficients from `spell_bonus_data`
- **Spell Crit**: All damage/heal spells can crit (150% multiplier) based on `total_spell_crit` from Intellect + rating
- **Armor Mitigation**: WotLK formula from `Unit.cpp:CalcArmorReducedDamage`, capped at 75%
- **Mob AI**: Aggro range (10-20 units), chase, melee attack, leash (60 units)
- **Loot System**: Copper drops, item score system, gear stats parsed from CSV (10 stat slots per item)
- **Respawn**: Dead mobs respawn after 60s at original spawn point
- **XP**: AzerothCore formula `BaseGain()` with gray level, ZeroDifference — mobs below gray level give 0 XP
- **Level System**: Level 1-80 with XP table, class-specific HP/Mana per level, stat recalculation on level-up
- **Exploration**: Three-tier tracking (`visited_areas`, `visited_zones`, `visited_maps`)
  - Real WoW Area/Zone/Map IDs from AreaTable.dbc when `env3d` + DBC available
  - Grid fallback without 3D data: Areas=50x50 units, Zones=200x200 units
  - `_new_areas`/`_new_zones`/`_new_maps` counters (consume-on-read like XP/Loot)
- **3D Terrain** (optional via `terrain` parameter): Z coordinates, walkability checks, LOS checks for spells
- **State Dict**: Extends TCP JSON format with primary stats, combat ratings, and equipment summary
- **Regen System**: HP regen 0.67/tick OOC (after 6s combat delay), Mana regen 2% of max_mana/tick while not casting, Spirit-based mana regen (5-second rule)

**WotLK 3.3.5 Attribute System**:

All 10 WoW classes are supported with correct base stats from PlayerClassLevelInfo DBC:

| Class | STR | AGI | STA | INT | SPI | Base HP | Power Type |
|---|---|---|---|---|---|---|---|
| Warrior | 23 | 20 | 22 | 17 | 19 | 60 | Rage |
| Paladin | 22 | 17 | 21 | 20 | 20 | 68 | Mana |
| Hunter | 16 | 24 | 21 | 17 | 20 | 56 | Mana |
| Rogue | 18 | 24 | 20 | 17 | 19 | 55 | Energy |
| Priest | 15 | 17 | 20 | 22 | 23 | 72 | Mana |
| Death Knight | 24 | 16 | 23 | 11 | 18 | 130 | Runic |
| Shaman | 18 | 16 | 21 | 20 | 22 | 57 | Mana |
| Mage | 15 | 17 | 18 | 24 | 22 | 52 | Mana |
| Warlock | 15 | 17 | 20 | 22 | 22 | 58 | Mana |
| Druid | 17 | 17 | 19 | 22 | 22 | 56 | Mana |

Primary stat scaling: `base_stat + (level - 1)` per level for each stat.

**Stamina → HP** (StatSystem.cpp:GetHealthBonusFromStamina): First 20 stamina = 1 HP each, above 20 = 10 HP each.
**Intellect → Mana** (StatSystem.cpp:GetManaBonusFromIntellect): First 20 int = 1 mana each, above 20 = 15 mana each.
**Spirit → Mana Regen**: `sqrt(intellect) * spirit * coeff` per second (OOC, 5-second rule). Coefficients from GtRegenMPPerSpt.dbc.

**Combat Rating System** (from GtCombatRatings.dbc):

Non-linear per-level scaling with known WotLK keypoints. Rating needed for 1% at L80:

| Rating | L80 Value | Effect |
|---|---|---|
| Hit (Melee/Ranged) | 32.79 | +1% hit chance |
| Hit (Spell) | 26.23 | +1% spell hit |
| Crit (all) | 45.91 | +1% crit chance |
| Haste (all) | 32.79 | +1% haste |
| Dodge | 39.35 | +1% dodge (with DR) |
| Parry | 39.35 | +1% parry (with DR) |
| Block | 16.39 | +1% block |
| Defense | 4.92 | +1 defense skill |
| Expertise | 8.20 | -0.25% dodge/parry |
| Armor Penetration | 13.99 | +1% ArP |
| Resilience | 81.97 | -1% crit damage |

Dodge and Parry use **diminishing returns**: `dr * cap / (dr + cap * k)` with class-specific k values (0.956–0.988) and caps.

**Spell Power Coefficients** (from spell_bonus_data):

| Spell | SP Coefficient | Notes |
|---|---|---|
| Smite | 0.7143 | 2.5s cast / 3.5 base |
| Lesser Heal | 0.8571 | 3.0s / 3.5 |
| Mind Blast | 0.4286 | 1.5s / 3.5 |
| SW:Pain (per tick) | 0.1833 | ~1.1 total over 6 ticks |
| PW:Shield | 0.8068 | absorb amount |
| Renew (per tick) | 0.1 | ~0.5 total over 5 ticks |
| Holy Fire (direct) | 0.5711 | direct damage |
| Holy Fire (DoT tick) | 0.024 | per tick |

**Armor Mitigation** (Unit.cpp:CalcArmorReducedDamage):
```
eff_level = mob_level + 4.5 * max(0, mob_level - 59)
dr = 0.1 * armor / (8.5 * eff_level + 40)
mitigation = min(dr / (1 + dr), 0.75)
```

**Attack Power Formulas** (StatSystem.cpp:UpdateAttackPowerAndDamage):
- Warrior/Paladin/DK: `level*3 + str*2 - 20`
- Hunter/Shaman/Rogue: `level*2 + str + agi - 20`
- Mage/Priest/Warlock: `str - 10`
- Druid (caster): `str*2 - 20`

**Equipment System** (19 slots):

| Slot IDs | Slots |
|---|---|
| 0-9 | Head, Neck, Shoulders, Shirt, Chest, Waist, Legs, Feet, Wrists, Hands |
| 10-14 | Finger 1, Finger 2, Trinket 1, Trinket 2, Back |
| 15-18 | Main Hand, Off Hand, Ranged, Tabard |

- Items map to slots via `INVTYPE_TO_SLOTS` (WoW InventoryType → valid equipment slots)
- Dual-slot items (rings, trinkets): fills empty slot first, then replaces lowest-score item
- Two-hand weapons automatically clear the offhand slot
- **Equipment changes blocked during combat** (WoW behaviour)
- On equip/unequip: `recalculate_stats()` recomputes all derived stats from gear + level + buffs
- Gear stat accumulation: 25 individual gear fields (gear_strength through gear_hp5) summed from all equipped items
- Item stats parsed from 10 `stat_type/stat_value` pairs per item (WotLK ITEM_MOD enum)

**Stat Recalculation** (`recalculate_stats()`):
Called on equip, unequip, level-up, and buff changes. Computes:
- Primary stat totals: base(class, level) + gear bonuses
- Max HP/Mana with Stamina/Intellect formulas (preserves current HP/Mana ratio)
- Armor: gear + agility*2 + Inner Fire buff
- Attack Power: class-specific formula + gear AP
- Spell Power: gear SP + Inner Fire buff
- Crit (melee/ranged/spell): from Agility/Intellect + crit rating (GtChanceToMeleeCrit/SpellCrit.dbc)
- Haste (melee/ranged/spell): from haste rating
- Hit (melee/ranged/spell): from hit rating
- Dodge/Parry: with diminishing returns + defense rating contribution
- Block: shield-only, from block rating + defense (block value from str/2 + gear)
- Expertise, Armor Penetration, Resilience, Defense: from respective ratings

**XP Formula** (from AzerothCore `Formulas.h`/`.cpp`):
- **Mob >= Player Level**: `((pl*5 + 45) * (20 + min(diff, 4)) / 10 + 1) / 2`
- **Mob > Gray Level**: `(pl*5 + 45) * (ZD + mob - pl) / ZD`
- **Mob <= Gray Level**: `0 XP`

**Initialization**: `CombatSimulation(num_mobs=None, seed=None, terrain=None, env3d=None, creature_db=None)`
- `num_mobs=None`: Uses all 84 hardcoded spawn positions
- `terrain`: `SimTerrain` instance for heights, LOS, walkability
- `env3d`: `WoW3DEnvironment` instance for Area/Zone lookups via AreaTable.dbc
- `creature_db`: `CreatureDB` instance for full-world chunk-based mob spawning

### quest_db.py — Quest System

Loads quest definitions from AzerothCore CSV exports with hardcoded fallback:
- **Quest Types**: KILL (kill N creatures), COLLECT (loot N items from creatures), EXPLORE (visit location)
- **CSV Loading** (4 CSV files): `quest_template.csv`, `quest_template_addon.csv`, `creature_queststarter[r].csv`, `creature_questender.csv`
- **~9500 quests** loaded from CSV (1862 kill, 4961 collect objectives), ~3170 quest NPCs with positions from creature.csv
- **QuestXP from DBC**: Exact quest XP from `QuestXP.dbc` (100 levels × 10 difficulty columns), fallback to anchor-based approximation
- **Objective Parsing**: KILL from `RequiredNpcOrGo1-4`, COLLECT from `RequiredItemId1-6` (source creature = heuristic from RequiredNpcOrGo)
- **Chain Info**: `PrevQuestID`/`NextQuestID` from `quest_template_addon.csv`
- **NPC Positions**: Auto-loaded from `creature.csv` + names from `creature_template.csv`
- **Hardcoded Fallback** (3 quests, 2 NPCs) when CSVs not available:
  - Quest 33: "Wolves Across the Border" — Kill 10 Young Wolves (XP: 250)
  - Quest 7: "Kobold Camp Cleanup" — Kill 10 Kobold Vermin (XP: 450, chain from 33)
  - Quest 15: "Investigate Echo Ridge" — Kill 10 Kobold Workers (XP: 675, chain from 7)
- **Data Classes**: `QuestTemplate`, `QuestObjective`, `QuestReward`, `QuestProgress`, `QuestNPCData`
- **`QuestDB`**: Follows CreatureDB/LootDB pattern — CSV loading + hardcoded fallback
- **Integration**: `CombatSimulation(quest_db=qdb)` enables quest NPCs, objective tracking, and quest events

**Initialization**: `QuestDB(data_dir=None, quiet=False)`
- Without `data_dir`: Uses 3 hardcoded Northshire quests (backwards compatible)
- With `data_dir`: Loads from CSV, replaces hardcoded quests entirely — check `loaded` property

**Required CSV Exports** (semicolon-delimited, double-quote enclosed):
- `quest_template.csv`: ID, QuestType, QuestLevel, MinLevel, ..., RequiredNpcOrGo1-4, RequiredNpcOrGoCount1-4, RequiredItemId1-6, RequiredItemCount1-6, ...
- `quest_template_addon.csv`: ID, MaxLevel, AllowableClasses, ..., PrevQuestID, NextQuestID, ...
- `creature_queststarter.csv` (or `creature_queststarterr.csv`): id, quest
- `creature_questender.csv`: id, quest

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
- **Item Score**: Base score via `GetItemScore` formula: `(Quality*10) + ItemLevel + Armor + WeaponDPS + (TotalStats*2)`. Equipment decisions use `class_aware_score()` which replaces the flat `TotalStats*2` with class-specific stat weights (e.g., Priest values INT at 3x, STR at 0.1x)
- **Group System** (AzerothCore standard):
  - **Group 0**: Each entry rolls independently (chance %)
  - **Group N>0**: Exactly one entry wins per group (weighted selection)
  - **chance=0**: In grouped entries, equal share of remaining probability
- **Reference Resolution**: Recursive processing from `reference_loot_template`, with max depth 5 to prevent loops
- **Item Stats**: Each `ItemData` carries a `stats` dict (`{ITEM_MOD_*: value}`) parsed from 10 `stat_type/stat_value` CSV columns, plus `armor` and `weapon_dps` — used by the equipment system for stat recalculation
- **Data Classes**: `ItemData` (entry, name, quality, sell_price, inventory_type, item_level, score, stats, armor, weapon_dps), `LootEntry` (item, reference, chance, group_id, counts), `LootResult` (item + count)
- **Graceful Degradation**: Auto-discovers CSV files, missing files silently skipped — check `loaded` property
- **Integration**: When loaded, `CombatSimulation.do_loot()` uses real loot tables; otherwise falls back to random loot

**Required CSV Exports** (semicolon-delimited, double-quote enclosed):
- `creature_loot_template.csv`: Entry, Item, Reference, Chance, QuestRequired, LootMode, GroupId, MinCount, MaxCount, Comment
- `item_template.csv`: entry, name, class, subclass, Quality, SellPrice, InventoryType, ItemLevel, armor, dmg_min1, dmg_max1, delay, stat_type1..10, stat_value1..10
- `reference_loot_template.csv` (optional): same schema as creature_loot_template
- `creature_template.csv` (updated): add `lootid` column for creature→loot table mapping (defaults to entry if missing)

### wow_sim_env.py — Gymnasium Sim Environment

Extended replacement for `wow_env.py` with optional quest system:
- **Action Space**: `Discrete(17)` — 16 base actions + Quest Interact
- **Obs Space**: `Box(38,)` — 22 base dims + 10 stat dims + 6 quest dims
- **Sparse Reward Design**: Focused on real outcomes only (see reward table below)
- **Similar Override Logic**: Aggro, Cast-Guard, Range-Management, Heal/Shield/DoT/Buff blocks
- **No Episode Step Limit**: Episode runs until death (bot should level as far as possible)
- **Stall Detection**: Truncates episode after 3,000 steps without kill XP (quest XP alone does not reset the counter)
- **OOM is NOT terminal**: Bot must learn to wait for mana regen

**Action Space** — `Discrete(17)`:
| ID | Action |
|---|---|
| 0 | No-op (wait) |
| 1 | move_forward |
| 2 | turn_left |
| 3 | turn_right |
| 4 | Target mob (nearest) |
| 5 | Cast Smite (585) |
| 6 | Cast Lesser Heal (2050) |
| 7 | Loot (nearest dead creature) |
| 8 | Sell (to vendor) |
| 9 | Cast SW:Pain (589) |
| 10 | Cast PW:Shield (17) |
| 11 | Quest NPC Interact |
| 12 | Cast Mind Blast (8092) |
| 13 | Cast Renew (139) |
| 14 | Cast Holy Fire (14914) |
| 15 | Cast Inner Fire (588) — self-buff: armor + spell power |
| 16 | Cast PW:Fortitude (1243) — self-buff: +HP |

**Observation Vector** — `Box(shape=(38,), dtype=float32)`:

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
| 17 | has_renew | 0/1 |
| 18 | has_inner_fire | 0/1 |
| 19 | has_fortitude | 0/1 |
| 20 | mind_blast_ready | 0/1 |
| 21 | target_has_holy_fire | 0/1 |
| 22 | spell_power / 200 | 0-inf |
| 23 | spell_crit / 50 | 0-inf |
| 24 | spell_haste / 50 | 0-inf |
| 25 | total_armor / 2000 | 0-inf |
| 26 | attack_power / 500 | 0-inf |
| 27 | melee_crit / 50 | 0-inf |
| 28 | dodge / 50 | 0-inf |
| 29 | hit_spell / 50 | 0-inf |
| 30 | expertise / 50 | 0-inf |
| 31 | armor_pen / 100 | 0-inf |
| 32 | has_active_quest | 0/1 |
| 33 | quest_progress (objectives ratio) | 0-1 |
| 34 | quest_npc_nearby | 0/1 |
| 35 | quest_npc_distance / 40 | 0-1 |
| 36 | quest_npc_angle / pi | -1 to 1 |
| 37 | quests_completed / 10 | 0-inf |

Stat dims (22-31) reflect gear + buffs and update as the bot equips items or levels. Quest dims (32-37) are always present but zero when quests are disabled (`enable_quests=False`).

**Initialization**: `WoWSimEnv(bot_name="SimBot", num_mobs=None, seed=None, data_root=None, creature_csv_dir=None, log_dir=None, log_interval=1, enable_quests=False)`
- `data_root`: Path to WoW `Data/` directory -> enables 3D terrain (`SimTerrain`) + area lookups (`WoW3DEnvironment` with AreaTable.dbc)
- `creature_csv_dir`: Path to directory containing `creature.csv` + `creature_template.csv` -> enables full-world creature spawning
- `log_dir`: Path for episode JSONL logs (used by `visualize.py`)
- Without `data_root`: Flat terrain, grid-based exploration detection

### Reward Tables

#### Sim Rewards (wow_sim_env.py — Sparse Design)

| Signal | Value | Notes |
|---|---|---|
| Step Penalty | -0.001 | per tick |
| Idle Penalty | -0.005 | Noop without casting |
| Approach | clip(delta * 0.03, -0.1, +0.15) | potential-based, closer to target |
| Damage Dealt | min(dmg * 0.03, 1.0) | damage to target |
| XP/Kill | 10.0 + xp * 0.5 | ~35 per 50-XP kill, scales with XP |
| Level-Up | +15.0 * levels | per level gained |
| Equipment Upgrade | min(1.0 + diff * 0.15, 5.0) | class-aware scoring, scaled by score improvement |
| Loot | per-item quality reward (0.1 grey to 5.0 epic) + min(copper * 0.01, 1.0) | penalty if inventory full |
| Sell | 1.0 + 7.0 * fullness + min(copper * 0.005, 2.0) | scales with inventory fill (1-8) + copper bonus |
| New Area Entered | +1.0 | real WoW Area ID or grid fallback (once per episode) |
| New Zone Entered | +3.0 | real WoW Zone ID (once per episode) |
| New Map Entered | +10.0 | real WoW Map ID (once per episode) |
| Quest Completion | +20.0 * quests | per quest turned in (+ quest XP via kill signal) |
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

**Key Differences**: The sim uses a **mostly sparse reward** design with light approach shaping, while the live env uses **more reward shaping** (facing, discovery, action-specific bonuses). Both have approach shaping (sim: clip to [-0.1, +0.15], live: clip to [-0.2, +0.3]). The sim has a harsher death penalty (-15 vs -5) and higher XP/kill reward (10+xp*0.5 vs 3+xp*0.05). OOM is only terminal in the live env. Stall detection: sim truncates after 3k steps without kill XP (quest XP does not count).

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
- **PPO** with `ent_coef=0.01`, `n_steps=512`, `batch_size=128`, `learning_rate=3e-4`, `gamma=0.97`, `n_epochs=8`
- **TensorBoard Logs** in `logs/PPO_2/`
- **Episode Callbacks** with kills, XP, deaths, areas/zones/maps explored, levels gained, final level
- **TensorBoard Metrics**: `gameplay/ep_areas_explored`, `gameplay/ep_zones_explored`, `gameplay/ep_maps_explored`, `gameplay/ep_levels_gained`, `gameplay/ep_final_level`, `gameplay/ep_quests_completed`, `gameplay/ep_quest_xp`
- **Real per-iteration FPS tracking** (not cumulative)
- **~5000+ FPS** (without 3D terrain)
- **Model versioning**: Auto-increments `wow_bot_sim_v1.zip`, `v2.zip`, etc. Interrupt save: `wow_bot_sim_interrupted.zip`
- **`--data-root`**: Optional, enables 3D terrain + real WoW area IDs
- **`--creature-data`**: Optional, enables full-world creature spawning from CSV
- **`--log-dir`**: Optional, enables episode trail logging for visualization
- **`--log-interval`**: How often to write episode logs (default: every episode)
- **`--enable-quests`**: Optional, enables quest system (Northshire quests with kill/collect/explore objectives)

### test_sim.py — Validation Tests

12 test functions:
1. **test_combat_engine()**: Basic engine initialization, movement, targeting, spell casting (all 9 spells)
2. **test_gym_env()**: Gymnasium spaces validation — Box(38,) obs, Discrete(17) actions
3. **test_random_episode()**: 1000-step episode with random actions
4. **test_performance()**: FPS benchmark (~40000+ FPS single-env)
5. **test_combat_scenario()**: Scripted combat with targeting and spell rotation
6. **test_level_system()**: XP formulas, level-up mechanics, stat scaling, multi-level-up
7. **test_loot_tables()**: LootDB loading, item score computation, group rolling distribution, sim integration, upgrade detection, fallback without DB
8. **test_vendor_system()**: Vendor NPCs, navigation, sell mechanics, dynamic spawning, sell rewards
9. **test_quest_system()**: QuestDB loading (hardcoded), chain prerequisites, level requirements, kill objectives, quest NPC interaction, turn-in rewards, consume_events, reset, env integration
10. **test_quest_csv_loading()**: QuestDB CSV loading from AzerothCore exports, objective parsing (kill/collect), chain info, QuestXP.dbc parsing, NPC position loading, fallback behavior
11. **test_attribute_system()**: WotLK base stats, HP/Mana formulas, spell crit from Intellect, Spirit mana regen, spell power scaling, armor mitigation, combat rating conversions, state_dict stat fields, observation vector stat dims, stat persistence across reset
12. **test_equipment_system()**: Equipment slots, equip/unequip, stat recalculation, dual-slot items (rings/trinkets), two-hand offhand clearing, combat-lock, item stats accumulation, upgrade detection with gear stats

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
- XP/kill reward differs: sim=10+xp*0.5, live=3+xp*0.05
- Death penalty differs: sim=-15, live=-5
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

# With quest system (Northshire quests with kill/collect/explore objectives)
python -m sim.train_sim --enable-quests --steps 500000

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
- **Tests**: `sim/test_sim.py` (10 tests for sim validation), `check_env.py` (live env smoke test)

## Progress & Status

### What Works (Completed)

| Component | Status | Details |
|---|---|---|
| **CombatSimulation Engine** | done | 84 spawns + CreatureDB, 9 Priest spells, WotLK stat system (all 10 classes), 19-slot equipment, armor mitigation, mob AI, loot, XP, respawn, exploration, leveling (1-80) |
| **WotLK Attribute System** | done | 5 primary stats, all combat ratings (hit/crit/haste/dodge/parry/block/expertise/ArP/resilience), spell power coefficients, DBC-derived formulas, diminishing returns |
| **Equipment System** | done | 19 WoW equipment slots, equip/unequip with stat recalculation, combat-locked, two-hand offhand clearing, dual-slot logic (rings/trinkets), item stats from CSV |
| **WoWSimEnv (Gym Interface)** | done | Discrete(17) actions, Box(38) obs (22 base + 10 stat + 6 quest), sparse rewards, stall detection |
| **train_sim.py (PPO Training)** | done | 5 bots, SubprocVecEnv, TensorBoard, gameplay metrics, episode logging |
| **Loot Table System** | done | LootDB CSV loader, AzerothCore group/reference logic, item scores + individual stat types, upgrade detection, sim integration with fallback |
| **test_sim.py (Validation)** | done | 10+ tests: engine, gym spaces, random episode, benchmark, scripted combat, level system, loot tables, vendor system, quest system, quest CSV loading, attribute system, equipment |
| **3D Terrain System** | done | Maps/VMAPs parser, HeightCache, SpatialLOSChecker, SimTerrain wrapper |
| **AreaTable.dbc Parser** | done | Reads all areas/zones/maps of the WoW world, on-demand tile loading |
| **Exploration System** | done | 3-tier tracking (Area/Zone/Map), rewards, TensorBoard metrics |
| **CreatureDB (Full World)** | done | CSV loader, spatial index, stat interpolation, attackability checks |
| **Episode Logger** | done | Zero-I/O JSONL logger, trail data, events, mob snapshots |
| **Quest System** | done | CSV loading (~9500 quests from AzerothCore DB) + 3 hardcoded fallback quests, quest NPCs (~3170 from CSV), quest chains, obs/action space extended, rewards, TensorBoard metrics |
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
| **COLLECT quest source creatures** | Sim | low | CSV-loaded COLLECT objectives use first RequiredNpcOrGo as source creature heuristic. Some quests may have wrong or missing source creatures — would need loot table cross-reference for accuracy. |
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

9. **More spells / additional classes** *(partially done — 9 Priest spells implemented, all 10 classes have stat frameworks)*
    - Additional Priest spells (Fade, Psychic Scream, Shadow Word: Death)
    - Spell implementations for non-Priest classes
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
