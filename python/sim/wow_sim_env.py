"""
WoW Simulation Gymnasium Environment — Drop-in replacement for WoWEnv.

Observation space: Box(23,) — 17 base dims + 6 quest dims.
Action space: Discrete(12) — 11 base actions + 1 quest action.
Runs ~1000x faster than the real server.

Usage:
    env = WoWSimEnv()                        # single bot (no quests)
    env = WoWSimEnv(enable_quests=True)      # with quest system
    obs, info = env.reset()
    obs, reward, done, trunc, info = env.step(action)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import os
import random
from typing import Optional

from sim.combat_sim import CombatSimulation, SPELLS, INVENTORY_SLOTS

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

    Observation Space: Box(23,) — 17 base + 6 quest dimensions
    Action Space: Discrete(12) — 11 base actions + quest interact

    Quest dimensions (indices 17-22) are always present but zero when
    quests are disabled, keeping the interface stable for model transfer.
    """

    metadata = {"render_modes": []}

    def __init__(self, bot_name: str = "SimBot", num_mobs: int = None,
                 seed: int = None, data_root: str = None,
                 creature_csv_dir: str = None, log_dir: str = None,
                 log_interval: int = 1, enable_quests: bool = False):
        super().__init__()

        self.action_space = spaces.Discrete(12)
        self.observation_space = spaces.Box(
            low=-1.0, high=float('inf'), shape=(23,), dtype=np.float32
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
        self._steps_since_kill_xp = 0      # stall detector: reset episode after 3k steps without kill XP
        self._vendor_nav_active = False    # True while bot is walking to vendor / selling
        self._quest_nav_active = False     # True while bot is walking to quest NPC

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
        self._steps_since_kill_xp = 0
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

        # ─── Override Logic (matching wow_env.py exactly) ──────────
        override_action = int(action)
        nearby_mobs = self.last_state.get('nearby_mobs', []) if self.last_state else []
        hp_pct = p.hp / max(1, p.max_hp)
        dist_to_target = 9999.0

        if self.sim.target and self.sim.target.alive:
            dist_to_target = self.sim._dist_to_mob(self.sim.target)

        in_combat = p.in_combat
        target_alive = self.sim.target is not None and self.sim.target.alive
        target_dead = self.sim.target is not None and not self.sim.target.alive
        is_casting = p.is_casting

        # Vendor navigation: AI triggers action 8 → bot auto-walks to vendor and sells
        # Once _vendor_nav_active is set, override handles the multi-step journey.
        # Combat interrupts vendor navigation (aggro takes priority).
        if action == 8 and not in_combat and not self._vendor_nav_active:
            self._vendor_nav_active = True  # AI decided to sell

        # Quest NPC navigation: AI triggers action 11 → bot auto-walks to quest NPC
        # Same pattern as vendor navigation.
        if action == 11 and not in_combat and not self._quest_nav_active:
            self._quest_nav_active = True  # AI decided to interact with quest NPC

        if in_combat and self._vendor_nav_active:
            self._vendor_nav_active = False  # combat cancels vendor trip
        if in_combat and self._quest_nav_active:
            self._quest_nav_active = False  # combat cancels quest trip

        if self._quest_nav_active and not in_combat:
            qnpc, _ = self.sim.get_best_quest_npc()
            if qnpc:
                qdx = qnpc.x - p.x
                qdy = qnpc.y - p.y
                qnpc_dist = math.sqrt(qdx * qdx + qdy * qdy)
                if qnpc_dist <= self.sim.QUEST_NPC_RANGE:
                    override_action = 11  # close enough to interact
                    self._quest_nav_active = False
                else:
                    self.sim.do_move_to(qnpc.x, qnpc.y)
                    override_action = 0  # movement handled
            else:
                self._quest_nav_active = False  # no relevant quest NPC

        if self._vendor_nav_active and not in_combat:
            vendor = self.sim.get_nearest_vendor()
            if vendor:
                vdx = vendor.x - p.x
                vdy = vendor.y - p.y
                vendor_dist = math.sqrt(vdx * vdx + vdy * vdy)
                if vendor_dist <= self.sim.SELL_RANGE:
                    override_action = 8  # close enough to sell
                    self._vendor_nav_active = False  # will sell this tick
                else:
                    self.sim.do_move_to(vendor.x, vendor.y)
                    override_action = 0  # movement handled
            else:
                self._vendor_nav_active = False  # no vendor found, abort
        elif action == 8 and not self._vendor_nav_active:
            override_action = 0  # block raw sell action (must go through nav)

        if (not self._vendor_nav_active and not self._quest_nav_active) or in_combat:
            # Aggro check: in combat but no living target
            if in_combat and not target_alive:
                aggro_mob = None
                min_dist = 9999.0
                for mob_data in nearby_mobs:
                    if mob_data.get('target', '0') != '0' and mob_data.get('attackable', 0) == 1 and mob_data['hp'] > 0:
                        d = math.sqrt((mob_data['x'] - p.x) ** 2 + (mob_data['y'] - p.y) ** 2)
                        if d < min_dist:
                            min_dist = d
                            aggro_mob = mob_data
                if aggro_mob:
                    # Find the actual mob object and target it
                    uid = int(aggro_mob['guid'])
                    for mob in self.sim.mobs:
                        if mob.uid == uid:
                            self.sim.target = mob
                            break
                    override_action = 0  # wait for target

            elif is_casting and action in [1, 2, 3, 5, 6, 7, 9, 10, 11]:
                override_action = 0
            elif target_dead and dist_to_target < 3.0:
                override_action = 7  # loot
            elif target_dead and dist_to_target >= 3.0:
                override_action = 1  # walk to corpse
            elif target_alive and dist_to_target < 25.0:
                if action == 1:
                    override_action = 0  # stop moving when in range
            elif in_combat and target_alive and action == 4:
                override_action = 0  # don't re-target in combat
            elif action == 6 and hp_pct > 0.85:
                override_action = 0  # block heal at high HP
            elif action == 10 and p.shield_remaining > 0:
                override_action = 0  # block shield if already shielded
            elif action == 9 and self.sim.target and self.sim.target.dot_remaining > 0:
                override_action = 0  # block SW:Pain if already active

        # ─── Execute Action ─────────────────────────────────────────
        if override_action == 0:
            self.sim.do_noop()
        elif override_action == 1:
            self.sim.do_move_forward()
        elif override_action == 2:
            self.sim.do_turn_left()
        elif override_action == 3:
            self.sim.do_turn_right()
        elif override_action == 4:
            self.sim.do_target_nearest()
        elif override_action == 5:
            self.sim.do_cast_smite()
        elif override_action == 6:
            self.sim.do_cast_heal()
        elif override_action == 7:
            self.sim.do_loot()
        elif override_action == 8:
            self.sim.do_sell()
        elif override_action == 9:
            self.sim.do_cast_sw_pain()
        elif override_action == 10:
            self.sim.do_cast_pw_shield()
        elif override_action == 11:
            self.sim.do_quest_interact()

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
        if override_action == 0 and not is_casting_now:
            reward -= 0.005

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
            curr_dist = dist_to_target
            if self._prev_target_dist is not None and self._prev_target_dist < 9000:
                delta = self._prev_target_dist - curr_dist  # positive = closer
                reward += max(-0.1, min(delta * 0.03, 0.15))
            self._prev_target_dist = curr_dist
        else:
            self._prev_target_dist = None

        # 4. XP/Kill (primary reward signal — must dominate step penalty)
        xp = events["xp_gained"]
        kill_xp = xp - events.get("quest_xp", 0)  # separate kill XP from quest XP
        if xp > 0:
            reward += 10.0 + xp * 0.5
            if kill_xp > 0:
                self._ep_kills += 1
                self._steps_since_kill_xp = 0
            if self._logger:
                self._logger.record_event(
                    self._step_count, p.x, p.y, "kill")
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

        # 6. Equipment-Upgrade
        if events["equipped_upgrade"]:
            reward += 3.0

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
        #    Selling a full inventory (30/30 items) = massive bonus
        #    Selling nearly empty inventory (2/30 items) = barely worth the trip
        sell_copper = events.get("sell_copper", 0)
        items_sold = events.get("items_sold", 0)
        if items_sold > 0:
            fullness = items_sold / INVENTORY_SLOTS  # 0.0 to 1.0
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
                "kills": self.sim.kills,
                "xp": self._ep_xp,
                "loot": self._ep_loot,
                "damage_dealt": self.sim.damage_dealt,
                "death": 1 if self.sim.player.hp <= 0 else 0,
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
            }
            info["episode_stats"] = ep_stats

            # Flush episode log to disk
            if self._logger:
                self._logger.flush_episode(ep_stats)

        return obs, reward, terminated, truncated, info

    def _build_obs(self, data: dict) -> np.ndarray:
        """Build observation vector — 17 base dims + 6 quest dims = 23 total."""
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

        # Quest observations (dims 17-22) — always present, zero when quests disabled
        quest_obs = self._compute_quest_obs(data)

        return np.array([
            hp_pct, mana_pct, t_hp, t_exists, in_combat,
            dist_norm, angle_norm, is_casting,
            mob_count, slots_norm,
            closest_mob_dist, closest_mob_angle, num_attackers,
            target_level, player_level,
            has_shield, target_has_sw_pain,
            *quest_obs,
        ], dtype=np.float32)

    def _compute_quest_obs(self, data: dict):
        """Compute quest observation features (6 dimensions).

        [17] has_active_quest        (0/1)
        [18] quest_progress          (0-1, ratio of completed objectives)
        [19] quest_npc_nearby        (0/1, relevant quest NPC exists)
        [20] quest_npc_distance / 40 (0-1, normalized distance)
        [21] quest_npc_angle / pi    (-1 to 1, relative angle)
        [22] quests_completed / 10   (0-inf, total completed this episode)
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
