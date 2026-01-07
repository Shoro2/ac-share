# ac-share

This repository is a self-contained snapshot of a local AzerothCore-based WoW server plus a custom in-process module and the Python side used to train and run an agent.

Core idea: the server publishes game state over a local TCP socket, Python decides what to do next, and the server executes that action for a server-side spawned “bot” character.

## Repository layout

- `python/`
  Python control + training code:
  - A Gymnasium environment (`wow_env.py`) that connects to the server socket
  - Stable-Baselines3 PPO training (`train.py`)
  - Model runner (`run_model.py`)
  - Hybrid “route grind” runner (`auto_grind.py`)
  - Utilities (`get_gps.py`, `check_env.py`, `test_multibot.py`)
  - Training outputs (`logs/`, `models/`) and a persistent NPC memory file (`npc_memory.json`)

- `src_azeroth_core/`
  Full AzerothCore source tree of the local server this project was developed against. It is included so the module and hook code can be read and built against a known baseline.

- `src_module-ai-controller/`
  C++ sources for the AzerothCore module that:
  - spawns bot characters via chat commands
  - runs a TCP server (default port 5000)
  - streams newline-delimited JSON state to Python
  - receives newline-delimited commands from Python and applies them in-game

## High-level architecture

1. `worldserver` loads the AI controller module.
2. On startup, the module starts a TCP server thread on port 5000.
3. A Python script connects to `127.0.0.1:5000` and reads state snapshots (newline-delimited JSON).
4. Python converts the state into a fixed-size observation vector.
5. Python chooses an action (PPO policy or scripted override) and sends a command line back.
6. The module parses commands and executes movement/targeting/casting/looting/selling for the selected bot character.
7. XP, loot, upgrades, death, etc. are reflected in subsequent state snapshots and used as rewards/termination signals.

## Bot spawning (server-side players)

The module supports spawning characters without a real client connection (it creates a `WorldSession`, loads the player from the character database, adds it to the world map, and marks it online).

### Chat commands

- `#spawn <BotName>`
  Spawns a single bot character by name.

- `#spawnbots`
  Spawns a hardcoded list of bot names:
  `Bota`, `Botb`, `Botc`, `Botd`, `Bote`

Important: the character(s) must already exist in the character database with matching names.

## TCP protocol

### Transport

- Plain TCP
- Default: `127.0.0.1:5000`
- One client connection is handled at a time
- Messages are newline-delimited

### Server -> Python (state stream)

The server sends one JSON object per line:

- Top-level:
  - `{ "players": [ ... ] }`

- Player object fields:
  - `name` (string)
  - `hp`, `max_hp` (int)
  - `power`, `max_power` (int)
  - `level` (int)
  - `x`, `y`, `z` (float)
  - `o` (float) orientation
  - `combat` (string `"true"` / `"false"`, not a JSON boolean)
  - `casting` (string `"true"` / `"false"`, not a JSON boolean)
  - `free_slots` (int)
  - `equipped_upgrade` (string `"true"` / `"false"`)
  - `target_status` (string: `"alive"`, `"dead"`, `"none"`)
  - `target_hp` (int)
  - `xp_gained` (int, aggregated since last publish)
  - `loot_copper` (int, aggregated since last publish)
  - `loot_score` (int, aggregated since last publish)
  - `leveled_up` (string `"true"` / `"false"`)
  - `tx`, `ty`, `tz` (float target position; 0 when no target)
  - `nearby_mobs` (array)

- `nearby_mobs` entry fields:
  - `guid` (string, raw GUID value)
  - `name` (string)
  - `level` (int)
  - `attackable` (int 1/0)
  - `vendor` (int 1/0)
  - `target` (string, raw GUID value of the creature’s current target, or `"0"`)
  - `hp` (int)
  - `x`, `y`, `z` (float)

Notes:
- The server publishes updates using an internal timer and also sends a keepalive snapshot roughly every 500 ms when nothing new was produced.
- XP/loot/upgrade/level flags are collected from hooks and then reset after being reported.

### Python -> Server (commands)

Python sends one command per line using this grammar:

`<playerName>:<actionType>:<value>\n`

`value` is required syntactically but is `"0"` or empty for actions that do not need one.

Implemented `actionType` values:

- `say:<text>`
  Makes the player say `<text>`.

- `stop:0`
  Stops movement and idles.

- `turn_left:0` / `turn_right:0`
  Rotates orientation by a fixed step.

- `move_forward:0`
  Moves a short step forward by computing a point ~3 units ahead and issuing a `MovePoint`.

- `move_to:<x>:<y>:<z>`
  Moves to the given coordinates (the server adjusts Z against ground).

- `target_nearest:<range>`
  Selects a nearby valid target within `range` (defaults to ~30 if parsing fails).

- `target_guid:<rawGuid>`
  Selects the referenced unit/creature by GUID.

- `cast:<spellId>`
  Casts the spell. For Smite (`585`), the module tries to ensure a hostile target; otherwise it tends to fall back to self.

- `loot_guid:<rawGuid>`
  If the creature is dead and within ~10 units:
  loots money, attempts to take items, increments loot counters, and may auto-equip an upgrade.

- `sell_grey:<vendorGuid>`
  Requires a vendor creature within ~15 units and then removes sellable items from inventory and adds the computed money to the player.
  Despite the name, the current logic sells/destroys items based on `SellPrice > 0` (with a special-case exclusion for Hearthstone itemId `6948`).

- `reset:0`
  Combat stop, clear movement, heal to full, refill power, clear cooldowns/auras, and teleport to homebind.

## Python RL environment (Gymnasium + PPO)

### Environment

`python/wow_env.py` implements `WoWEnv(gym.Env)`.

- Connects to the TCP server
- Assumes it controls exactly one “main” player (see “Known limitations” below)
- Converts state to an observation vector
- Converts discrete actions to socket commands
- Adds practical override logic (vendor trips, aggro recovery, etc.)

### Action space

`Discrete(9)` with the following mapping:

- `0`: no-op
- `1`: `move_forward`
- `2`: `turn_left`
- `3`: `turn_right`
- `4`: target a mob (selects the nearest valid `nearby_mobs` GUID and sends `target_guid`)
- `5`: cast Smite (`spellId 585`)
- `6`: cast Heal (`spellId 2050`)
- `7`: loot a nearby dead creature (`loot_guid`)
- `8`: sell to a vendor from memory (`sell_grey`)

### Observation vector

`Box(shape=(10,), dtype=float32)`

Current layout:

0. `hp_pct` (0..1)
1. `mana_pct` / resource percent (0..1)
2. `target_hp_scaled` (`target_hp / 100.0`)
3. `target_exists` (1 if target alive else 0)
4. `in_combat` (1/0)
5. `target_distance_norm` (clamped to 40 and scaled to 0..1)
6. `relative_angle_norm` (target angle relative to facing, normalized to -1..1)
7. `is_casting` (1/0)
8. reserved constant `0.0` (placeholder)
9. `free_slots_norm` (`free_slots / 20.0`)

### Reward shaping and termination

Reward is a shaped signal on top of a small step penalty. Highlights:

- Small constant step penalty: `-0.01`
- Exploration/discovery: +0.5 when a previously unseen mob GUID is added to memory
- Equipment upgrade: +100 when the server reports `equipped_upgrade == "true"`
- XP gain: +100 + (2 * xp_gained)
- Level-up: +2000 and terminate (see level-reset behavior in the module)
- Loot: +(0.1 * loot_copper) + (2.0 * loot_score)
- Selling: +50 when free slots increased compared to the previous state
- Termination:
  - death (`hp == 0`): -100 and terminate
  - near-empty mana/resource (`mana_pct < 0.05`): -10 and terminate

### Practical override logic (important)

`wow_env.py` is not “pure RL”. It overrides actions to keep the bot functional:

- Vendor mode:
  - If `free_slots < 2` and not in combat, the bot navigates to the nearest vendor remembered in `npc_memory.json` using `move_to`, then triggers `sell_grey` when close enough.
  - Action `8` is blocked unless vendor mode is active.

- Aggro recovery:
  - If in combat but no alive target, the env searches `nearby_mobs` for an attackable mob that currently has a target (`target != "0"`) and forces a `target_guid` command directly.

- Casting guardrails:
  - Movement/turning/casting/looting can be suppressed while casting to reduce self-interrupting behavior.

- Loot logic:
  - If the current target is dead, the env moves closer until within loot distance and then loots.

- Healing guardrails:
  - Healing is suppressed at high HP to reduce useless casting.

This improves stability for training and demos, but it also means the learned policy is trained in a partially scripted environment.

### NPC memory and blacklist

- `npc_memory.json` stores the last known data for mobs encountered (including vendors).
- Dead mobs are removed from memory and added to a timed blacklist.
- A blacklist duration is used (15 minutes in code) to reduce repeated interaction attempts and oscillations.
- Memory is saved periodically (every ~30 seconds).

## Scripts

- `python/train.py`
  Trains PPO for a fixed number of timesteps (currently 10,000) and saves to `models/PPO/wow_bot_v1`.

- `python/run_model.py`
  Loads `models/PPO/wow_bot_v1` and runs inference in a loop, resetting on termination.

- `python/auto_grind.py`
  Hybrid runner:
  - follows a hardcoded farm route (`FARM_ROUTE`) via `move_to`
  - uses the trained policy for combat/loot decisions

- `python/get_gps.py`
  Utility to print your current in-game coordinates from the state stream (useful to build routes).

- `python/test_multibot.py`
  Demonstrates parsing the JSON stream and issuing commands for multiple bot names (separate control approach from `WoWEnv`).

- `python/run_bot.py`
  Scratch/experimental file. As committed, it is not valid Python and is not used by the main workflow.

## Typical workflow

1. Build and run `worldserver` from `src_azeroth_core/` (standard AzerothCore build process applies).
2. Integrate/build the module sources from `src_module-ai-controller/` into your AzerothCore modules setup.
3. Ensure the bot characters exist in your character DB.
4. Log in with a GM character and use:
   - `#spawn <BotName>` or
   - `#spawnbots`
5. Start Python:
   - Training: `python/train.py`
   - Inference: `python/run_model.py`
   - Hybrid grind: `python/auto_grind.py`

## Known limitations and sharp edges (read this before debugging for hours)

- No security:
  The TCP socket is plaintext and unauthenticated. Keep it on localhost. Do not expose port 5000 to untrusted networks.

- One client connection:
  The server thread accepts and serves one TCP client at a time.

- Player selection in Python:
  The environment effectively assumes “the controlled player” is the one it receives and tracks as `self.my_name` and state.
  If multiple players/bots are online, you should explicitly select by name in Python instead of relying on list/order behavior.

- `nearby_mobs` is cached per update and can reflect only one player’s surroundings:
  The module builds `_cachedNearbyMobsJson` by scanning around a player and then reuses that cached list for all players in the JSON.

- Timer bug in the module update loop:
  `_fastTimer` is reset at ~150 ms for facing updates and then also used as the gate for the ~400 ms JSON publish path.
  As written, the 400 ms publish block will never execute. If you see the Python side receiving `{}` or stale state forever, split the timers (use `_faceTimer` for facing, `_fastTimer` for publishing) or otherwise fix the reset logic.

- Level behavior is intentionally “training sandbox” style:
  On level change, the module sets the player back to level 1 when reaching level 2 and clears XP. This is useful for repeated low-level training loops, but it is not normal gameplay behavior.

- Selling behavior is misnamed:
  `sell_grey` currently deletes/sells items based on `SellPrice > 0` (except Hearthstone), not by “grey quality only”.
  If you want true grey-only selling, filter by item quality.

## Dependencies (Python side)

You will need at least:
- Python 3.x
- `gymnasium`
- `numpy`
- `stable-baselines3`

Optional but useful:
- TensorBoard (for viewing `python/logs/`)

## Scope

This is an experimental local-server research/engineering setup intended for controlled environments you own. It is not intended for public servers.
