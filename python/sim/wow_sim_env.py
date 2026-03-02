"""
WoW Simulation Gymnasium Environment — Drop-in replacement for WoWEnv.

Same observation space (17,), same action space (Discrete(11)),
same reward function. Runs ~1000x faster than the real server.

Usage:
    env = WoWSimEnv()            # single bot
    obs, info = env.reset()
    obs, reward, done, trunc, info = env.step(action)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from typing import Optional

from sim.combat_sim import CombatSimulation, SPELLS


class WoWSimEnv(gym.Env):
    """
    Simulated WoW environment matching WoWEnv interface exactly.

    Observation Space: Box(17,) — identical to wow_env.py
    Action Space: Discrete(11) — identical to wow_env.py
    """

    metadata = {"render_modes": []}

    def __init__(self, bot_name: str = "SimBot", num_mobs: int = 15,
                 seed: int = None, data_root: Optional[str] = None):
        super().__init__()

        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(
            low=-1.0, high=float('inf'), shape=(17,), dtype=np.float32
        )

        self.bot_name = bot_name
        self.num_mobs = num_mobs
        self._seed = seed
        self._data_root = data_root

        # Load 3D terrain if data_root provided
        self._terrain = None
        if data_root:
            from sim.terrain import SimTerrain
            self._terrain = SimTerrain(data_root, quiet=True)

        self.sim = CombatSimulation(num_mobs=num_mobs, seed=seed, terrain=self._terrain)
        self.last_state = None
        self._step_count = 0
        self._max_steps = 2000  # episode timeout
        self._ep_reward = 0.0
        self._ep_xp = 0
        self._ep_loot = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self.sim = CombatSimulation(num_mobs=self.num_mobs, seed=self._seed,
                                    terrain=self._terrain)
        if self._seed is not None:
            self._seed += 1  # vary each episode
        self._step_count = 0
        self._ep_reward = 0.0
        self._ep_xp = 0
        self._ep_loot = 0
        self.last_state = self.sim.get_state_dict()
        obs = self._build_obs(self.last_state)
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

        # Vendor mode (simplified — no vendors in sim, just sell action)
        vendor_mode = False
        if p.free_slots < 2 and not in_combat:
            vendor_mode = True
            override_action = 8  # sell

        if not vendor_mode:
            if action == 8:
                override_action = 0  # block sell when not needed

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

            elif is_casting and action in [1, 2, 3, 5, 6, 7, 9, 10]:
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

        # ─── Execute Action ───────────────────────────────────────
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

        # ─── Advance Simulation ───────────────────────────────────
        self.sim.tick()

        # ─── Consume Events ───────────────────────────────────────
        events = self.sim.consume_events()
        state = self.sim.get_state_dict()

        # ─── Compute Rewards (matching wow_env.py exactly) ────────
        reward = -0.01  # step penalty

        # Discovery reward
        if self.last_state:
            old_nearby = {m['guid'] for m in self.last_state.get('nearby_mobs', [])}
            for m in state.get('nearby_mobs', []):
                if m['guid'] not in old_nearby and m['hp'] > 0:
                    reward += 0.5

        # Equipment upgrade
        if events["equipped_upgrade"]:
            reward += 100.0

        # XP
        xp = events["xp_gained"]
        if xp > 0:
            reward += 100.0 + (xp * 2.0)

        # Loot
        copper = events["loot_copper"]
        score = events["loot_score"]
        if copper > 0 or score > 0:
            reward += (copper * 0.1) + (score * 2.0)

        # Sold items (free_slots increased)
        if self.last_state and state.get('free_slots', 0) > self.last_state.get('free_slots', 0):
            reward += 50.0

        # Cast rewards
        t_exists = 1.0 if state.get('target_status') == 'alive' else 0.0
        in_combat_f = 1.0 if state.get('combat') == 'true' else 0.0
        new_hp_pct = self.sim.player.hp / max(1, self.sim.player.max_hp)

        if override_action == 5:  # Smite
            if t_exists > 0.5:
                reward += 0.5
            else:
                reward -= 0.1
        elif override_action == 6:  # Heal
            if new_hp_pct > 0.8:
                reward -= 0.5
        elif override_action == 9:  # SW:Pain
            if t_exists > 0.5:
                if state.get('target_has_sw_pain') != 'true':
                    reward += 1.0
                else:
                    reward -= 0.3
            else:
                reward -= 0.1
        elif override_action == 10:  # PW:Shield
            if in_combat_f and state.get('has_shield') != 'true':
                reward += 0.8
            elif state.get('has_shield') == 'true':
                reward -= 0.3
            else:
                reward += 0.2

        # Movement in combat penalty
        angle_norm = 0.0
        if t_exists > 0.5:
            dx = state['tx'] - state['x']
            dy = state['ty'] - state['y']
            target_angle = math.atan2(dy, dx)
            rel = target_angle - state['o']
            while rel > math.pi:
                rel -= 2 * math.pi
            while rel < -math.pi:
                rel += 2 * math.pi
            angle_norm = rel / math.pi

        if in_combat_f and t_exists > 0.5:
            if override_action in [1, 2, 3]:
                reward -= 0.5
                if abs(angle_norm) > 0.2 and override_action in [2, 3]:
                    reward += 0.6

        # Terminal conditions
        terminated = False
        mana_pct = self.sim.player.mana / max(1, self.sim.player.max_mana)

        if self.sim.player.hp <= 0:
            reward -= 100.0
            terminated = True
        elif mana_pct < 0.05:
            reward -= 10.0
            terminated = True

        truncated = self._step_count >= self._max_steps

        # Accumulate episode stats
        self._ep_reward += reward
        self._ep_xp += xp
        self._ep_loot += copper

        # Build obs
        obs = self._build_obs(state)
        self.last_state = state

        info = {}
        if terminated or truncated:
            info["episode_stats"] = {
                "reward": self._ep_reward,
                "length": self._step_count,
                "kills": self.sim.kills,
                "xp": self._ep_xp,
                "loot": self._ep_loot,
                "damage_dealt": self.sim.damage_dealt,
                "death": 1 if self.sim.player.hp <= 0 else 0,
            }

        return obs, reward, terminated, truncated, info

    def _build_obs(self, data: dict) -> np.ndarray:
        """Build observation vector — exactly matches wow_env.py._build_obs."""
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

        return np.array([
            hp_pct, mana_pct, t_hp, t_exists, in_combat,
            dist_norm, angle_norm, is_casting,
            mob_count, slots_norm,
            closest_mob_dist, closest_mob_angle, num_attackers,
            target_level, player_level,
            has_shield, target_has_sw_pain
        ], dtype=np.float32)

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
