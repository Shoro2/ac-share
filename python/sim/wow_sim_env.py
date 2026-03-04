"""
WoW Simulation Gymnasium Environment — Drop-in replacement for WoWEnv.

Observation space: Box(39,) — 23 base dims + 10 stat dims + 6 quest dims.
Action space: Discrete(18) — 17 base actions + 1 quest action.
Runs ~1000x faster than the real server.

Uses **action masking** instead of override logic: invalid actions are masked
out so the bot can only choose from valid actions.  Game-mechanic constraints
(casting lock, GCD, cooldowns, buff duplication, etc.) are hard-masked.
Strategic decisions (when to loot, heal timing, range management, aggro
recovery) are left to the bot to learn.

Vendor and quest NPC navigation remain as multi-step overrides (too complex
for the bot to learn pathfinding from scratch at this stage).

Usage:
    env = WoWSimEnv()                        # single bot (no quests)
    env = WoWSimEnv(enable_quests=True)      # with quest system
    obs, info = env.reset()
    obs, reward, done, trunc, info = env.step(action)

    # Action masking (for MaskablePPO from sb3_contrib):
    masks = env.action_masks()               # np.ndarray(18,) bool
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import os
import random
from typing import Optional

from sim.combat_sim import CombatSimulation, SPELLS
from sim.formulas import spell_mana_cost

# Reward per successfully looted item, indexed by WoW item quality
QUALITY_LOOT_REWARD = {
    0: 0.1,   # Poor (grey)
    1: 0.3,   # Common (white)
    2: 1.0,   # Uncommon (green)
    3: 3.0,   # Rare (blue)
    4: 5.0,   # Epic (purple)
}

# Penalty when an item can't be picked up (inventory full), same scale
QUALITY_FAIL_PENALTY = {
    0: 0.1,
    1: 0.3,
    2: 1.0,
    3: 3.0,
    4: 5.0,
}


class WoWSimEnv(gym.Env):
    """
    Simulated WoW environment with optional quest system.

    Observation Space: Box(39,) — 23 base + 10 stat + 6 quest dimensions
    Action Space: Discrete(18) — 17 base actions + quest interact

    Stat dimensions (indices 23-32):
      [23] spell_power/200, [24] spell_crit/50, [25] spell_haste/50,
      [26] total_armor/2000, [27] attack_power/500, [28] melee_crit/50,
      [29] dodge/50, [30] hit_spell/50, [31] expertise/50, [32] armor_pen/100
    Quest dimensions (indices 33-38) are always present but zero when
    quests are disabled, keeping the interface stable for model transfer.
    """

    metadata = {"render_modes": []}

    def __init__(self, bot_name: str = "SimBot", num_mobs: int = None,
                 seed: int = None, data_root: str = None,
                 creature_csv_dir: str = None, log_dir: str = None,
                 log_interval: int = 1, enable_quests: bool = False):
        super().__init__()

        self.action_space = spaces.Discrete(18)
        self.observation_space = spaces.Box(
            low=-1.0, high=float('inf'), shape=(39,), dtype=np.float32
        )

        self.bot_name = bot_name
        self.num_mobs = num_mobs
        self._seed = seed
        self._data_root = data_root
        self._creature_csv_dir = creature_csv_dir

        # Episode logger (optional — zero overhead when disabled)
        self._logger = None
        if log_dir:
            from sim.sim_logger import SimEpisodeLogger
            self._logger = SimEpisodeLogger(log_dir, bot_name,
                                            record_interval=log_interval)

        # Load 3D terrain + area lookup if data_root provided
        self._terrain = None
        self._env3d = None
        if data_root:
            from sim.terrain import SimTerrain
            self._terrain = SimTerrain(data_root, quiet=True)
            # Reuse the WoW3DEnvironment already loaded by SimTerrain
            # (avoids loading 545K BIH-nodes + 27K VMAP spawns a second time)
            try:
                self._env3d = self._terrain.env
                if not self._env3d.area_table:
                    self._env3d.load_area_table()
            except Exception as e:
                print(f"  [WARN] Area lookup not available: {e}")
                self._env3d = None

        # Load creature DB from CSVs if provided
        self._creature_db = None
        if creature_csv_dir:
            from sim.creature_db import CreatureDB
            self._creature_db = CreatureDB(creature_csv_dir, quiet=True)

        # Load loot tables (auto-discover from creature_csv_dir or data/)
        self._loot_db = None
        loot_dir = creature_csv_dir or (os.path.dirname(data_root) if data_root else None)
        if loot_dir:
            from sim.loot_db import LootDB
            loot_db = LootDB(loot_dir, quiet=True)
            if loot_db.loaded:
                self._loot_db = loot_db

        # Load quest system if enabled
        self._quest_db = None
        self._enable_quests = enable_quests
        if enable_quests:
            from sim.quest_db import QuestDB
            quest_data_dir = creature_csv_dir or (
                os.path.dirname(data_root) if data_root else None)
            self._quest_db = QuestDB(data_dir=quest_data_dir, quiet=True)

        self.sim = CombatSimulation(num_mobs=num_mobs, seed=seed,
                                    terrain=self._terrain, env3d=self._env3d,
                                    creature_db=self._creature_db,
                                    loot_db=self._loot_db,
                                    quest_db=self._quest_db)
        self.last_state = None
        self._step_count = 0
        # No step limit — episode runs until death (bot should level as far as possible)
        self._ep_reward = 0.0
        self._ep_xp = 0
        self._ep_loot = 0
        self._ep_kills = 0
        self._ep_damage_dealt = 0
        self._ep_loot_items = 0
        self._ep_loot_failed = 0
        self._ep_sell_copper = 0
        self._ep_areas = 0
        self._ep_zones = 0
        self._ep_maps = 0
        self._prev_target_hp = None
        self._ep_exploration_reward = 0.0
        self._ep_levels_gained = 0
        self._ep_quests_completed = 0
        self._ep_quest_xp = 0
        self._ep_equipment_upgrades = 0
        self._steps_since_kill_xp = 0      # stall detector: reset episode after 3k steps without kill XP
        self._idle_steps = 0              # noop-without-casting steps (idle time tracking)
        self._vendor_nav_active = False    # True while bot is walking to vendor / selling
        self._quest_nav_active = False     # True while bot is walking to quest NPC

    # Spell ID -> action ID mapping for mask building
    _SPELL_ACTION = {
        585: 5,     # Smite
        2050: 6,    # Lesser Heal
        589: 9,     # SW:Pain
        17: 10,     # PW:Shield
        8092: 12,   # Mind Blast
        139: 13,    # Renew
        14914: 14,  # Holy Fire
        588: 15,    # Inner Fire
        1243: 16,   # PW:Fortitude
    }
    _OFFENSIVE_SPELLS = {585, 589, 8092, 14914}   # need alive target + range
    _SELF_SPELLS = {2050, 17, 139, 588, 1243}     # self-cast, no target needed

    def action_masks(self) -> np.ndarray:
        """Return boolean mask: True = action allowed, False = masked.

        Game-mechanic masks (action is physically impossible):
        - Casting → only noop allowed
        - Eating → only noop allowed (eating auto-ticks, movement interrupts)
        - GCD active → all spells masked
        - Insufficient mana → that spell masked
        - Spell on cooldown → that spell masked
        - No alive target / out of range → offensive spells masked
        - Buff already active → that buff masked
        - No dead mob in loot range → loot masked
        - In combat → loot masked (don't run to corpse mid-fight)
        - Vendor/quest nav not active → sell/quest interact masked

        Strategic decisions (bot must learn):
        - When to loot vs keep fighting (loot only available OOC + in range)
        - Heal timing (no HP threshold block)
        - Range management (no auto-stop at 25 units)
        - Aggro recovery (targeting in combat)
        - Walking to corpse after kill
        - When to eat/drink vs keep going
        """
        mask = np.ones(18, dtype=bool)
        p = self.sim.player

        # ── While casting: ONLY noop is valid ──
        if p.is_casting:
            mask[:] = False
            mask[0] = True  # noop
            return mask

        # ── While eating: ONLY noop is valid (regen ticks automatically) ──
        # Any other action would interrupt eating — masking prevents accidental cancel
        if p.is_eating:
            mask[:] = False
            mask[0] = True  # noop (continue eating)
            return mask

        in_combat = p.in_combat
        target = self.sim.target
        target_alive = target is not None and target.alive

        # ── Movement (1-3): always allowed when not casting ──
        # (already True)

        # ── Target nearest (4): need alive mobs in range ──
        has_targetable = any(
            m.alive and self.sim._dist_to_mob(m) <= self.sim.TARGET_RANGE
            for m in self.sim.mobs
        )
        if not has_targetable:
            mask[4] = False

        # ── Spell masks (5,6,9,10,12,13,14,15,16) ──
        gcd_blocked = p.gcd_remaining > 0
        for spell_id, action_id in self._SPELL_ACTION.items():
            spell = SPELLS[spell_id]

            # Level gate: spell not yet learned
            if p.level < spell.level_req:
                mask[action_id] = False
                continue

            # GCD blocks all spells
            if gcd_blocked:
                mask[action_id] = False
                continue

            # Mana check (% of BaseMana from Spell.dbc)
            cost = spell_mana_cost(spell_id, p.level, p.class_id)
            if p.mana < cost:
                mask[action_id] = False
                continue

            # Spell-specific cooldown
            if p.spell_cooldowns.get(spell_id, 0) > 0:
                mask[action_id] = False
                continue

            # Offensive spells: need alive target in range
            if spell_id in self._OFFENSIVE_SPELLS:
                if not target_alive:
                    mask[action_id] = False
                    continue
                if self.sim._dist_to_mob(target) > spell.spell_range:
                    mask[action_id] = False
                    continue
                # LOS check
                if self.sim.terrain:
                    if not self.sim.terrain.check_los(
                        p.x, p.y, p.z, target.x, target.y, target.z
                    ):
                        mask[action_id] = False
                        continue

            # Buff/debuff duplication checks (game mechanic — can't double-apply)
            if spell_id == 17 and (p.shield_remaining > 0 or p.shield_cooldown > 0):
                mask[action_id] = False
            elif spell_id == 589 and target is not None and target.dot_remaining > 0:
                mask[action_id] = False
            elif spell_id == 139 and p.hot_remaining > 0:
                mask[action_id] = False
            elif spell_id == 588 and p.inner_fire_remaining > 0:
                mask[action_id] = False
            elif spell_id == 1243 and p.fortitude_remaining > 0:
                mask[action_id] = False

        # ── Loot (7): need dead unlootable mob in range AND not in combat ──
        # Key design: in combat → loot masked → bot fights first, loots later
        has_lootable = False
        if not in_combat:
            for mob in self.sim.mobs:
                if not mob.alive and not mob.looted:
                    if self.sim._dist_to_mob(mob) <= self.sim.LOOT_RANGE:
                        has_lootable = True
                        break
        if not has_lootable:
            mask[7] = False

        # ── Sell (8): only valid during vendor nav (override handles walk) ──
        # Bot triggers action 8 to START vendor nav; once active, override walks there.
        # Mask sell if: in combat, no inventory, or nav already active (wait for arrival)
        if in_combat or len(p.inventory) == 0:
            mask[8] = False
        elif self._vendor_nav_active:
            mask[8] = False  # already walking to vendor, wait

        # ── Quest interact (11): only when quests enabled and NPC reachable ──
        if not self._quest_db:
            mask[11] = False
        elif in_combat:
            mask[11] = False
        elif self._quest_nav_active:
            mask[11] = False  # already walking to quest NPC, wait
        else:
            qnpc, _ = self.sim.get_best_quest_npc()
            if qnpc is None:
                mask[11] = False

        # ── Eat/Drink (17): only OOC, not casting, not already eating, not full ──
        if in_combat or p.is_casting:
            mask[17] = False
        elif p.hp >= p.max_hp and p.mana >= p.max_mana:
            mask[17] = False  # already full, no point eating

        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self.sim.reset()
        if self._seed is not None:
            self.sim.rng = random.Random(self._seed)
            self._seed += 1  # vary each episode
        self._step_count = 0
        self._ep_reward = 0.0
        self._ep_xp = 0
        self._ep_loot = 0
        self._ep_kills = 0
        self._ep_damage_dealt = 0
        self._ep_loot_items = 0
        self._ep_loot_failed = 0
        self._ep_sell_copper = 0
        self._ep_areas = 0
        self._ep_zones = 0
        self._ep_maps = 0
        self._prev_target_hp = None
        self._ep_exploration_reward = 0.0
        self._ep_levels_gained = 0
        self._ep_quests_completed = 0
        self._ep_quest_xp = 0
        self._ep_equipment_upgrades = 0
        self._steps_since_kill_xp = 0
        self._idle_steps = 0
        self._prev_sim_kills = 0            # track sim.kills for event logging
        self._prev_target_dist = None       # for approach shaping
        self._vendor_nav_active = False
        self._quest_nav_active = False
        self.last_state = self.sim.get_state_dict()
        obs = self._build_obs(self.last_state)

        # Log initial state
        if self._logger:
            self._logger.reset()
            self._logger.record_mobs([
                {"entry": m.template.entry, "name": m.template.name,
                 "x": m.spawn_x, "y": m.spawn_y, "level": m.level}
                for m in self.sim.mobs
            ])
            p = self.sim.player
            self._logger.record_step(
                0, p.x, p.y, p.hp / max(1, p.max_hp),
                p.level, p.in_combat, p.orientation)

        return obs, {}

    def step(self, action):
        self._step_count += 1
        p = self.sim.player
        action = int(action)

        # ─── Vendor/Quest Navigation Overrides ──────────────────────
        # These are multi-step macro actions: bot triggers action 8/11,
        # then the override auto-walks to the NPC over multiple ticks.
        # Combat interrupts navigation (aggro takes priority).
        executed_action = action  # track what actually ran (for idle detection)

        if action == 8 and not p.in_combat and not self._vendor_nav_active:
            self._vendor_nav_active = True
        if action == 11 and not p.in_combat and not self._quest_nav_active:
            self._quest_nav_active = True

        if p.in_combat and self._vendor_nav_active:
            self._vendor_nav_active = False
        if p.in_combat and self._quest_nav_active:
            self._quest_nav_active = False

        nav_handled = False  # True when nav override consumed this tick

        if self._quest_nav_active and not p.in_combat:
            qnpc, _ = self.sim.get_best_quest_npc()
            if qnpc:
                qdist = math.sqrt((qnpc.x - p.x) ** 2 + (qnpc.y - p.y) ** 2)
                if qdist <= self.sim.QUEST_NPC_RANGE:
                    self.sim.do_quest_interact()
                    self._quest_nav_active = False
                    executed_action = 11
                else:
                    self.sim.do_move_to(qnpc.x, qnpc.y)
                    executed_action = 0
                nav_handled = True
            else:
                self._quest_nav_active = False

        if self._vendor_nav_active and not p.in_combat and not nav_handled:
            vendor = self.sim.get_nearest_vendor()
            if vendor:
                vdist = math.sqrt((vendor.x - p.x) ** 2 + (vendor.y - p.y) ** 2)
                if vdist <= self.sim.SELL_RANGE:
                    self.sim.do_sell()
                    self._vendor_nav_active = False
                    executed_action = 8
                else:
                    self.sim.do_move_to(vendor.x, vendor.y)
                    executed_action = 0
                nav_handled = True
            else:
                self._vendor_nav_active = False

        # ─── Execute Action (no overrides — masking handles validity) ──
        if not nav_handled:
            if action == 0:
                self.sim.do_noop()
            elif action == 1:
                self.sim.do_move_forward()
            elif action == 2:
                self.sim.do_turn_left()
            elif action == 3:
                self.sim.do_turn_right()
            elif action == 4:
                self.sim.do_target_nearest()
            elif action == 5:
                self.sim.do_cast_smite()
            elif action == 6:
                self.sim.do_cast_heal()
            elif action == 7:
                self.sim.do_loot()
            elif action == 8:
                self.sim.do_sell()
            elif action == 9:
                self.sim.do_cast_sw_pain()
            elif action == 10:
                self.sim.do_cast_pw_shield()
            elif action == 11:
                self.sim.do_quest_interact()
            elif action == 12:
                self.sim.do_cast_mind_blast()
            elif action == 13:
                self.sim.do_cast_renew()
            elif action == 14:
                self.sim.do_cast_holy_fire()
            elif action == 15:
                self.sim.do_cast_inner_fire()
            elif action == 16:
                self.sim.do_cast_fortitude()
            elif action == 17:
                self.sim.do_eat_drink()

        # ─── Advance Simulation ───────────────────────────────────
        self.sim.tick()

        # ─── Consume Events ───────────────────────────────────────
        events = self.sim.consume_events()
        state = self.sim.get_state_dict()

        # ─── Compute Rewards (sparse design — only real outcomes) ────
        reward = 0.0

        t_exists = 1.0 if state.get('target_status') == 'alive' else 0.0
        is_casting_now = state.get('casting') == 'true'
        hp_pct = self.sim.player.hp / max(1, self.sim.player.max_hp)
        mana_pct = self.sim.player.mana / max(1, self.sim.player.max_mana)

        # Current target tracking
        curr_target_hp = 0
        if t_exists > 0.5:
            curr_target_hp = state.get('target_hp', 0)

        # 1. Step-Penalty (time pressure — forces the bot to act)
        reward -= 0.001

        # 2. Idle-Penalty (Noop without casting = wasted time)
        if executed_action == 0 and not is_casting_now:
            reward -= 0.005
            self._idle_steps += 1

        # 3. Damage-Reward (gradient toward kills — can't be faked)
        if (t_exists > 0.5
                and self._prev_target_hp is not None
                and self._prev_target_hp > 0):
            damage = self._prev_target_hp - curr_target_hp
            if damage > 0:
                reward += min(damage * 0.03, 1.0)
                self._ep_damage_dealt += damage

        # 3b. Approach Shaping (potential-based — getting closer to target)
        if t_exists > 0.5:
            curr_dist = self.sim._dist_to_mob(self.sim.target) if self.sim.target else 9999.0
            if self._prev_target_dist is not None and self._prev_target_dist < 9000:
                delta = self._prev_target_dist - curr_dist  # positive = closer
                reward += max(-0.1, min(delta * 0.03, 0.15))
            self._prev_target_dist = curr_dist
        else:
            self._prev_target_dist = None

        # 4. XP/Kill (primary reward signal — must dominate step penalty)
        xp = events["xp_gained"]
        kill_xp = xp - events.get("quest_xp", 0)  # separate kill XP from quest XP
        new_kills = self.sim.kills - self._prev_sim_kills
        if new_kills > 0:
            self._ep_kills += new_kills
            self._prev_sim_kills = self.sim.kills
            if self._logger:
                for _ in range(new_kills):
                    self._logger.record_event(
                        self._step_count, p.x, p.y, "kill")
        if xp > 0:
            reward += 10.0 + xp * 0.5
            if kill_xp > 0:
                self._steps_since_kill_xp = 0
        else:
            self._steps_since_kill_xp += 1

        # 5. Level-Up
        levels = events.get("levels_gained", 0)
        if levels > 0:
            reward += 15.0 * levels
            self._ep_levels_gained += levels
            if self._logger:
                self._logger.record_event(
                    self._step_count, p.x, p.y, "levelup",
                    f"Lv{p.level}")

        # 6. Equipment-Upgrade — scaled by class-aware score improvement
        #    Small upgrades (~5 score diff) -> ~1.5 reward
        #    Medium upgrades (~20 score diff) -> ~4.0 reward (capped at 5.0)
        upgrade_score = events["equipped_upgrade"]
        if upgrade_score > 0:
            reward += min(1.0 + upgrade_score * 0.15, 5.0)
            self._ep_equipment_upgrades += 1

        # 7. Loot — quality-based reward per item, penalty when inventory full
        copper = events["loot_copper"]
        score = events["loot_score"]
        loot_items = events.get("loot_items", [])
        loot_failed = events.get("loot_failed", [])
        for q in loot_items:
            reward += QUALITY_LOOT_REWARD.get(q, 0.3)
        for q in loot_failed:
            reward -= QUALITY_FAIL_PENALTY.get(q, 0.3)
        if copper > 0:
            reward += min(copper * 0.01, 1.0)
        self._ep_loot_items += len(loot_items)
        self._ep_loot_failed += len(loot_failed)

        # 8. Sold items at vendor — reward scales with inventory fullness
        #    Selling a full inventory = massive bonus
        #    Selling nearly empty inventory = barely worth the trip
        sell_copper = events.get("sell_copper", 0)
        items_sold = events.get("items_sold", 0)
        if items_sold > 0:
            total_slots = self.sim.player.total_bag_slots
            fullness = items_sold / max(total_slots, 1)  # 0.0 to 1.0
            # Base reward 1.0 at minimal sell, up to 8.0 at full inventory
            sell_reward = 1.0 + 7.0 * fullness
            reward += sell_reward
        if sell_copper > 0:
            reward += min(sell_copper * 0.005, 2.0)
            self._ep_sell_copper += sell_copper

        # 9. Exploration (new area/zone/map — naturally capped, can't be farmed)
        new_areas = events.get("new_areas", 0)
        new_zones = events.get("new_zones", 0)
        new_maps = events.get("new_maps", 0)
        if new_areas > 0:
            r = new_areas * 1.0
            reward += r
            self._ep_areas += new_areas
            self._ep_exploration_reward += r
        if new_zones > 0:
            r = new_zones * 3.0
            reward += r
            self._ep_zones += new_zones
            self._ep_exploration_reward += r
        if new_maps > 0:
            r = new_maps * 10.0
            reward += r
            self._ep_maps += new_maps
            self._ep_exploration_reward += r

        # 10. Quest completion (significant reward — comparable to multiple kills)
        quests_done = events.get("quests_completed", 0)
        quest_xp = events.get("quest_xp", 0)
        if quests_done > 0:
            # Quest XP counts toward the kill XP signal (already in xp_gained)
            # Additional quest completion bonus
            reward += 20.0 * quests_done
            self._ep_quests_completed += quests_done
            self._ep_quest_xp += quest_xp
            # NOTE: quest XP does NOT reset stall counter — only kills do
            if self._logger:
                self._logger.record_event(
                    self._step_count, p.x, p.y, "quest",
                    f"Completed {quests_done} quest(s)")

        # Log trail point
        if self._logger:
            self._logger.record_step(
                self._step_count, p.x, p.y, hp_pct,
                p.level, p.in_combat, p.orientation)

        # 11. Terminal: Death (only terminal — bot must learn to survive)
        terminated = False
        if self.sim.player.hp <= 0:
            reward = -15.0
            terminated = True
            if self._logger:
                self._logger.record_event(
                    self._step_count, p.x, p.y, "death")
        # OOM is not terminal — bot must learn to wait for mana regen

        # Stall detection: if bot hasn't earned kill XP in 3k steps, it's stuck
        truncated = (self._steps_since_kill_xp >= 3_000)

        # State-Tracking
        self._prev_target_hp = curr_target_hp if t_exists > 0.5 else None

        # Accumulate episode stats
        self._ep_reward += reward
        self._ep_xp += xp
        self._ep_loot += copper

        # Build obs
        obs = self._build_obs(state)
        self.last_state = state

        info = {}
        if terminated or truncated:
            ep_stats = {
                "reward": self._ep_reward,
                "length": self._step_count,
                "kills": self._ep_kills,
                "xp": self._ep_xp,
                "loot": self._ep_loot,
                "damage_dealt": self.sim.damage_dealt,
                "death": 1 if self.sim.player.hp <= 0 else 0,
                "idle_ratio": self._idle_steps / max(1, self._step_count),
                "areas_explored": self._ep_areas,
                "zones_explored": self._ep_zones,
                "maps_explored": self._ep_maps,
                "rw_explore": self._ep_exploration_reward,
                "levels_gained": self._ep_levels_gained,
                "final_level": self.sim.player.level,
                "loot_items": self._ep_loot_items,
                "loot_failed": self._ep_loot_failed,
                "sell_copper": self._ep_sell_copper,
                "quests_completed": self._ep_quests_completed,
                "quest_xp": self._ep_quest_xp,
                "equipment_upgrades": self._ep_equipment_upgrades,
                "equipped_items": len(self.sim.player.equipment),
                "equipped_bags": len(self.sim.player.bags),
                "total_bag_slots": self.sim.player.total_bag_slots,
            }
            info["episode_stats"] = ep_stats

            # Flush episode log to disk
            if self._logger:
                self._logger.flush_episode(ep_stats)

        return obs, reward, terminated, truncated, info

    def _build_obs(self, data: dict) -> np.ndarray:
        """Build observation vector — 23 base + 10 stat + 6 quest = 39 total."""
        max_hp = max(1, data['max_hp'])
        hp_pct = data['hp'] / max_hp
        mana_pct = data['power'] / max(1, data['max_power'])

        t_hp, t_exists, dist_norm, angle_norm = 0.0, 0.0, 0.0, 0.0
        if data.get('target_status') == 'alive':
            t_hp = data.get('target_hp', 0) / 100.0
            t_exists = 1.0
            dx = data['tx'] - data['x']
            dy = data['ty'] - data['y']
            dist = math.sqrt(dx * dx + dy * dy)
            dist_norm = min(dist, 40.0) / 40.0
            target_angle = math.atan2(dy, dx)
            rel = target_angle - data['o']
            while rel > math.pi:
                rel -= 2 * math.pi
            while rel < -math.pi:
                rel += 2 * math.pi
            angle_norm = rel / math.pi

        is_casting = 1.0 if data.get('casting') == 'true' else 0.0
        in_combat = 1.0 if data.get('combat') == 'true' else 0.0
        slots_norm = data.get('free_slots', 0) / 20.0

        # Nearby mob features
        mob_count, closest_mob_dist, closest_mob_angle, num_attackers = \
            self._compute_nearby_mob_features(data)

        target_level = data.get('target_level', 0) / 10.0
        player_level = data.get('level', 1) / 10.0

        has_shield = 1.0 if data.get('has_shield') == 'true' else 0.0
        target_has_sw_pain = 1.0 if data.get('target_has_sw_pain') == 'true' else 0.0
        has_renew = 1.0 if data.get('has_renew') == 'true' else 0.0
        has_inner_fire = 1.0 if data.get('has_inner_fire') == 'true' else 0.0
        has_fortitude = 1.0 if data.get('has_fortitude') == 'true' else 0.0
        mind_blast_ready = 1.0 if data.get('mind_blast_ready') == 'true' else 0.0
        target_has_holy_fire = 1.0 if data.get('target_has_holy_fire') == 'true' else 0.0
        is_eating = 1.0 if data.get('is_eating') == 'true' else 0.0

        # Stat observations (dims 23-32) — comprehensive WotLK stats
        stat_sp = data.get('spell_power', 0) / 200.0           # spell power / 200
        stat_spell_crit = data.get('spell_crit', 0) / 50.0     # spell crit% / 50
        stat_spell_haste = data.get('spell_haste', 0) / 50.0   # spell haste% / 50
        stat_armor = data.get('total_armor', 0) / 2000.0       # armor / 2000
        stat_ap = data.get('attack_power', 0) / 500.0          # AP / 500
        stat_melee_crit = data.get('melee_crit', 0) / 50.0     # melee crit% / 50
        stat_dodge = data.get('dodge', 0) / 50.0               # dodge% / 50
        stat_hit = data.get('hit_spell', 0) / 50.0             # spell hit% / 50
        stat_expertise = data.get('expertise', 0) / 50.0       # expertise% / 50
        stat_arp = data.get('armor_pen', 0) / 100.0            # ArP% / 100

        # Quest observations (dims 33-38) — always present, zero when quests disabled
        quest_obs = self._compute_quest_obs(data)

        return np.array([
            hp_pct, mana_pct, t_hp, t_exists, in_combat,
            dist_norm, angle_norm, is_casting,
            mob_count, slots_norm,
            closest_mob_dist, closest_mob_angle, num_attackers,
            target_level, player_level,
            has_shield, target_has_sw_pain,
            has_renew, has_inner_fire, has_fortitude,
            mind_blast_ready, target_has_holy_fire,
            is_eating,
            stat_sp, stat_spell_crit, stat_spell_haste, stat_armor,
            stat_ap, stat_melee_crit, stat_dodge, stat_hit,
            stat_expertise, stat_arp,
            *quest_obs,
        ], dtype=np.float32)

    def _compute_quest_obs(self, data: dict):
        """Compute quest observation features (6 dimensions).

        [33] has_active_quest        (0/1)
        [34] quest_progress          (0-1, ratio of completed objectives)
        [35] quest_npc_nearby        (0/1, relevant quest NPC exists)
        [36] quest_npc_distance / 40 (0-1, normalized distance)
        [37] quest_npc_angle / pi    (-1 to 1, relative angle)
        [38] quests_completed / 10   (0-inf, total completed this episode)
        """
        if not self._quest_db:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        has_active = 1.0 if data.get('quest_active', False) else 0.0
        progress = float(data.get('quest_progress', 0.0))
        quests_done = data.get('quests_completed_total', 0) / 10.0

        # Find the most relevant quest NPC (turn-in > accept)
        qnpc, qtype = self.sim.get_best_quest_npc()
        qnpc_nearby = 0.0
        qnpc_dist = 0.0
        qnpc_angle = 0.0
        if qnpc:
            qnpc_nearby = 1.0
            dx = qnpc.x - data['x']
            dy = qnpc.y - data['y']
            dist = math.sqrt(dx * dx + dy * dy)
            qnpc_dist = min(dist, 40.0) / 40.0
            npc_angle = math.atan2(dy, dx)
            rel = npc_angle - data['o']
            while rel > math.pi:
                rel -= 2 * math.pi
            while rel < -math.pi:
                rel += 2 * math.pi
            qnpc_angle = rel / math.pi

        return (has_active, progress, qnpc_nearby, qnpc_dist, qnpc_angle,
                quests_done)

    def _compute_nearby_mob_features(self, data: dict):
        """Compute observation features from nearby_mobs — matches wow_env.py."""
        nearby_mobs = data.get('nearby_mobs', [])
        me_x, me_y = data['x'], data['y']
        orientation = data['o']

        num_alive = 0
        num_attackers = 0
        closest_dist = 40.0
        closest_angle = 0.0

        for mob in nearby_mobs:
            if mob['hp'] <= 0:
                continue
            if mob.get('attackable', 0) == 0:
                continue
            num_alive += 1
            if mob.get('target', '0') != '0':
                num_attackers += 1
            dx = mob['x'] - me_x
            dy = mob['y'] - me_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < closest_dist:
                closest_dist = dist
                mob_angle = math.atan2(dy, dx)
                rel = mob_angle - orientation
                while rel > math.pi:
                    rel -= 2 * math.pi
                while rel < -math.pi:
                    rel += 2 * math.pi
                closest_angle = rel

        return (
            min(num_alive, 10) / 10.0,
            min(closest_dist, 40.0) / 40.0,
            closest_angle / math.pi,
            min(num_attackers, 5) / 5.0,
        )
