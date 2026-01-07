Purpose
This repository is a working snapshot of a local AzerothCore setup plus an in-process “AI controller” module and the Python side used to train and run an agent. The core idea is simple: the WoW server publishes the game state over a TCP socket, Python decides an action, and the server executes that action on a player (typically a server-side spawned bot).

Repository layout
python/
Contains the Gymnasium environment plus training and control scripts (Stable-Baselines3 PPO). It connects to the server module via TCP and implements observation building, reward shaping, action-to-command mapping, and some safety/override logic (looting, vendor trips, aggro retargeting).

src_azeroth_core/
A full AzerothCore source snapshot from the local server this project was developed against. It is included so the module code can be understood in context (types, APIs, prepared statements, etc.) and so you can build against a known baseline.

src_module-ai-controller/
The AzerothCore module source that:

Spawns server-side “bot players” via a chat command (no real client connection required).

Runs a TCP server on port 5000.

Streams newline-delimited JSON state to the Python side.

Accepts newline-delimited commands from Python and applies them in-game (movement, targeting, casting, looting, selling).

High-level data flow

Worldserver loads the AI controller module.

On server startup, the module starts a TCP server (default port 5000).

A Python script connects to 127.0.0.1:5000 and continuously reads state snapshots.

Python converts the server state into a fixed-size observation vector for reinforcement learning.

Python chooses an action (either from a learned PPO policy or from scripted logic) and sends a command line to the server.

The server parses the command and executes the requested in-game action for the named player.

Server-side events (XP gain, loot, equipment upgrades, death) are reflected back into the next state snapshot and become rewards/termination signals in Python.

AI socket protocol (server <-> python)
Transport
Plain TCP. One client connection is handled at a time.

Server to client messages
The server sends newline-delimited JSON. Each line is one JSON object. The module sends the “current” state immediately after a client connects, then continues sending either when a new state is produced or as a keepalive roughly every 500 ms.

Top-level schema
{ "players": [ ... ] }

Player object fields currently produced by the module
name: character name (string)
hp, max_hp: current and maximum health (integers)
power, max_power: current and maximum resource (integers)
level: player level (integer)
x, y, z: player position (floats)
o: orientation (float)
combat: "true" or "false" (string, not boolean)
casting: "true" or "false" (string, not boolean)
free_slots: free bag slots (integer)
equipped_upgrade: "true" or "false" (string) set when the module auto-equips a better item after looting
target_status: "alive", "dead", or "none" (string)
target_hp: target health (integer)
tx, ty, tz: target position (floats; 0 if no target)
xp_gained: XP gained since last report (integer, aggregated)
loot_copper: money gained since last report (integer, aggregated; copper units)
loot_score: count-like loot signal since last report (integer; currently increments per looted item)
leveled_up: "true" or "false" (string) set when the player levels (see leveling note below)
nearby_mobs: JSON array of nearby creatures (see below)

nearby_mobs entry fields
guid: creature GUID raw value as a string
name: creature name (string)
level: creature level (integer)
attackable: 1 or 0 (integer)
vendor: 1 or 0 (integer)
target: GUID raw value of the creature’s current target as a string (or "0")
hp: creature health (integer)
x, y, z: creature position (floats)

Client to server messages
The client sends newline-delimited command lines:
playerName:actionType:value

actionType values implemented in the module
say
stop
turn_left
turn_right
move_forward
move_to
target_nearest
target_guid
cast
loot_guid
sell_grey
reset

Command semantics
say:value
Makes the player say the given text.

stop
Clears movement and idles.

turn_left / turn_right
Adjusts orientation by a fixed step.

move_forward
Moves a short step forward by computing a point 3 units ahead and issuing a MovePoint.

move_to:value
value must be formatted as x:y:z (floats). The module ground-adjusts Z and issues a MovePoint to that position.

target_nearest:value
value is an optional range float (defaults to 30). Selects a nearby valid attack target and sets selection/target.

target_guid:value
value is the raw GUID value (string containing an integer). Sets selection/target to the referenced unit if found.

cast:value
value is a spellId integer. The module tries to choose a target based on spell id and current selection and calls CastSpell.

loot_guid:value
value is a creature GUID raw value. If the creature is dead and within 10 units, the module loots money, loots items it can store, may auto-equip upgrades, and clears the corpse loot flags.

sell_grey:value
Despite the name, this currently destroys and “sells” every item with a SellPrice > 0 (except Hearthstone itemId 6948), from bags and backpack, and adds the computed money to the player. It requires a vendor creature within 15 units and uses the vendor GUID as value.

reset
Stops combat, resurrects if needed, heals to full, refills power, clears cooldowns/auras, and teleports the player to their homebind position.

Bot spawning (server-side players)
The module implements chat commands handled in OnPlayerBeforeSendChatMessage:
#spawn <BotName>
Spawns a single bot character by name.
#spawnbots
Spawns a hardcoded list of bot names: Bota, Botb, Botc, Botd, Bote.

How spawning works internally
The module looks up the character GUID by name, resolves its account id, creates a WorldSession without a real socket, loads the Player from the character database using an async query holder, adds the Player to ObjectAccessor and the world map, marks the character online in the database, and finally teleports the bot to a fixed spawn point.

Fixed spawn point currently hardcoded
Map 0, X -8921.037, Y -120.484985, Z 82.02542, O 3.299

Python environment and training
Core environment
python/wow_env.py implements a Gymnasium Env (WoWEnv) that connects to the AI socket and turns the server JSON into a 10-float observation vector. It also maps a discrete action id into one of the server commands above.

Action space
Discrete(9), values 0..8:
0 no-op
1 move_forward
2 turn_left
3 turn_right
4 target (implemented by selecting a specific guid from nearby_mobs and sending target_guid)
5 cast Smite (spellId 585)
6 cast Heal (spellId 2050)
7 loot (chooses a nearby dead mob and sends loot_guid)
8 sell (chooses nearest remembered vendor and sends sell_grey)

Observation vector (shape 10)
Index 0: hp_pct (0..1)
Index 1: mana/resource pct (0..1)
Index 2: target hp scaled (target_hp / 100.0)
Index 3: target exists flag (1 if target alive else 0)
Index 4: in_combat flag (1/0)
Index 5: target distance normalized (clamped to 40m and scaled to 0..1)
Index 6: relative angle to target normalized (-1..1)
Index 7: is_casting flag (1/0)
Index 8: reserved constant 0.0 in this snapshot
Index 9: free slots normalized (free_slots / 20.0)

Reward shaping (high-level)
Base step penalty plus exploration reward when discovering new mobs into memory.
Large positive reward when an equipment upgrade is auto-equipped by the server.
Large reward for XP gain and an even larger reward for “leveled_up”.
Loot rewards based on copper gained and number of looted items.
Penalty for dying or for running out of mana/resources.

NPC memory and blacklist
wow_env.py keeps a persistent npc_memory.json mapping creature guid to last seen info. Dead or already processed mobs are blacklisted for 15 minutes to reduce repeated looting attempts and oscillations.

Important behavior: action overrides
wow_env.py is not a pure “take action and observe” loop. It can override the model’s chosen action for practical reasons:
It will force vendor travel/selling when inventory is nearly full.
It can auto-retarget a mob if the player is in combat but currently lacks a valid alive target (aggro recovery).
It blocks actions while casting and applies some guardrails around looting and healing.
This makes training more stable, but it also means the learned policy is trained in a partly scripted environment.

Training script
python/train.py trains a PPO policy (Stable-Baselines3) against WoWEnv and saves to python/models/PPO/.

Running an already trained model
python/run_model.py loads a saved PPO policy and runs it in an infinite loop with env.reset() on episode termination.

Hybrid “auto grind” runner
python/auto_grind.py combines scripted travel (a waypoint route and memory-based roaming) with RL-based combat decisions. It uses move_to for navigation and periodically issues target_nearest scans.

Multi-bot control
python/test_multibot.py demonstrates sending commands for multiple bot names over one socket. This is a separate controller approach and does not use WoWEnv’s “players[0]” assumption.

How to use this project (typical workflow)
Step 1: Build and run your AzerothCore worldserver with the AI controller module enabled.
Step 2: Create the bot characters in the character database (names must match what you spawn).
Step 3: Log in with a GM character and use #spawn <BotName> (or #spawnbots).
Step 4: Log out the GM character if you want Python to control only the bot without ambiguity.
Step 5: Start Python:
For training: run python/train.py
For inference: run python/run_model.py or python/auto_grind.py

Notes and limitations in this snapshot
State streaming timer bug
In AIControllerHook.cpp, the same timer variable is reset at 150 ms for facing updates and later checked for a 400 ms state publish interval. As written, the 400 ms state publish block will never execute because the timer is reset earlier. If you see the Python side stuck with empty or stale state, split the timers (one for facing updates, one for JSON publishing) or remove the early reset.

Player ordering and “players[0]”
WoWEnv reads only the first entry in the players array and treats it as “me”. Object iteration order is not guaranteed. For reliable training, keep only one player online (the bot), or modify wow_env.py to select a specific player by name.

nearby_mobs cache is computed for one player
The module updates nearby_mobs by scanning around the first online player it finds and reuses that cached list for all players in the JSON. This works best when exactly one bot is online.

No authentication and no encryption
The TCP socket is plain text and unauthenticated. Do not expose port 5000 to untrusted networks.

“sell_grey” does not only sell grey items
The server-side sell routine deletes any item with a SellPrice > 0 (except Hearthstone). If you want true “grey only”, filter by item quality.

Forced leveling behavior
OnPlayerLevelChanged forces the character back to level 1 when reaching level 2 and clears XP. This is useful for keeping training in a low-level sandbox but will break normal gameplay if used on real players.

Files committed that are usually local artifacts
python/logs/ and python/models/ contain training outputs and a saved policy snapshot. python/pycache/ is also present. You may want to remove these from version control depending on how you share the project.

What this project is and is not
This is an experimental research/engineering setup to control a WoW character through an RL environment. It is not an anti-cheat evasion project and it is not meant for use on public servers. It assumes a controlled local environment where you own the server and database.

If you adapt this project, the two places you will touch most are AIControllerHook.cpp (state/commands/events) and python/wow_env.py (observation/reward/action mapping).
