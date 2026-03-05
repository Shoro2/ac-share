"""
WoW Live Server Gymnasium Environment — parity with WoWSimEnv.

Observation space: Box(52,) — 29 base dims + 10 stat dims + 3 vendor dims + 4 talent dims + 6 quest dims.
Action space: Discrete(30) — 25 base actions + 1 quest action + 4 talent actions.
Connects to AzerothCore via TCP on port 5000.

Uses **action masking** (same as sim): invalid actions are masked out so the
bot can only choose from valid actions.  Game-mechanic constraints (casting lock,
GCD, cooldowns, buff duplication, etc.) are hard-masked.  Strategic decisions
(when to loot, heal timing, range management, aggro recovery) are left to the bot.

Sell (action 8) is only unmasked when within SELL_RANGE of a known vendor.
Quest interact (action 11) is only unmasked when the server supports it.

Reward design matches sim (sparse):
  - Step penalty: -0.001
  - Idle penalty: -0.005
  - Approach shaping: clip(delta * 0.03, -0.1, +0.15)
  - Damage dealt: min(dmg * 0.03, 1.0)
  - XP/Kill: 10.0 + xp * 0.5
  - Level-up: +15.0 per level (NOT terminal)
  - Equipment upgrade: min(1.0 + diff * 0.15, 5.0)
  - Loot: quality-based per item
  - Sell: 1.0 + 7.0 * fullness + copper bonus
  - Exploration: +1.0/area, +3.0/zone, +10.0/map (grid-based)
  - Quest: +20.0 per quest completed
  - Death: -15.0 (terminal)
  - OOM: NOT terminal (bot must learn to wait for regen)

Usage:
    env = WoWEnv(bot_name="Bota")
    obs, info = env.reset()
    obs, reward, done, trunc, info = env.step(action)

    # Action masking (for MaskablePPO from sb3_contrib):
    masks = env.action_masks()  # np.ndarray(30,) bool
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import json
import time
import math
import os

# ── Spell IDs (rank 1 — server resolves to player's known rank) ──
SPELL_SMITE = 585
SPELL_HEAL = 2050
SPELL_SW_PAIN = 589
SPELL_PW_SHIELD = 17
SPELL_MIND_BLAST = 8092
SPELL_RENEW = 139
SPELL_HOLY_FIRE = 14914
SPELL_INNER_FIRE = 588
SPELL_FORTITUDE = 1243
SPELL_FLASH_HEAL = 2061
SPELL_DEVOURING_PLAGUE = 2944
SPELL_PSYCHIC_SCREAM = 8122
SPELL_SHADOW_PROTECTION = 976
SPELL_DIVINE_SPIRIT = 14752
SPELL_FEAR_WARD = 6346
SPELL_HOLY_NOVA = 15237
SPELL_DISPEL_MAGIC = 527
SPELL_MIND_FLAY = 15407
SPELL_VAMPIRIC_TOUCH = 34914
SPELL_DISPERSION = 47585

DEFAULT_MEMORY_FILE = "npc_memory.json"

# Sell / loot / quest range constants (matching sim)
SELL_RANGE = 6.0
LOOT_RANGE = 6.0
QUEST_NPC_RANGE = 10.0
TARGET_RANGE = 30.0
GCD_DURATION = 1.5  # seconds

# Exploration grid sizes (matching sim fallback mode)
AREA_GRID_SIZE = 50.0   # world units per area
ZONE_GRID_SIZE = 200.0   # world units per zone

# Reward per successfully looted item, indexed by WoW item quality
QUALITY_LOOT_REWARD = {
    0: 0.1,   # Poor (grey)
    1: 0.3,   # Common (white)
    2: 1.0,   # Uncommon (green)
    3: 3.0,   # Rare (blue)
    4: 5.0,   # Epic (purple)
}


class WoWEnv(gym.Env):
    """Live server WoW environment — space-compatible with WoWSimEnv.

    Observation Space: Box(52,) — 29 base + 10 stat + 3 vendor + 4 talent + 6 quest dims
    Action Space: Discrete(30)

    Dims that require extended C++ state fields (has_renew, spell_power, etc.)
    will be zero until AIControllerHook.cpp is updated to send them.
    """

    metadata = {"render_modes": []}

    # ── Action-to-spell mapping ──
    _ACTION_SPELL = {
        5: SPELL_SMITE,
        6: SPELL_HEAL,
        9: SPELL_SW_PAIN,
        10: SPELL_PW_SHIELD,
        12: SPELL_MIND_BLAST,
        13: SPELL_RENEW,
        14: SPELL_HOLY_FIRE,
        15: SPELL_INNER_FIRE,
        16: SPELL_FORTITUDE,
        18: SPELL_FLASH_HEAL,
        19: SPELL_DEVOURING_PLAGUE,
        20: SPELL_PSYCHIC_SCREAM,
        21: SPELL_SHADOW_PROTECTION,
        22: SPELL_DIVINE_SPIRIT,
        23: SPELL_FEAR_WARD,
        24: SPELL_HOLY_NOVA,
        25: SPELL_DISPEL_MAGIC,
        26: SPELL_MIND_FLAY,
        27: SPELL_VAMPIRIC_TOUCH,
        28: SPELL_DISPERSION,
    }

    # Offensive spells: require alive target
    _OFFENSIVE_ACTIONS = {5, 9, 12, 14, 19, 26, 27}
    # Self-buff actions: require NOT having the buff
    _SELF_BUFF_ACTIONS = {6, 10, 13, 15, 16, 18, 21, 22, 23, 28}
    # AoE actions: require alive mob in combat nearby
    _AOE_ACTIONS = {20, 24}

    def __init__(self, host='127.0.0.1', port=5000, bot_name=None,
                 enable_quests=False):
        super().__init__()
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((host, port))
            self.sock.setblocking(True)
            self.sock.settimeout(2.0)
            print(f">>> WoW Env v12 (Sim Parity) connected! <<<")
        except Exception as e:
            print(f"CONNECTION ERROR: {e}")
            raise e

        # ── Spaces: match WoWSimEnv exactly ──
        self.action_space = spaces.Discrete(30)
        self.observation_space = spaces.Box(
            low=-1.0, high=float('inf'), shape=(52,), dtype=np.float32
        )

        self.last_state = None
        self.my_name = ""
        self.bot_name = bot_name
        self._enable_quests = enable_quests

        # Per-bot memory file
        self.memory_file = None
        self._set_memory_file(self.bot_name)
        self._recv_buffer = ""

        self.npc_memory = {}
        self.blacklist = {}
        self.BLACKLIST_DURATION = 15 * 60

        self.load_memory()
        self.last_save_time = time.time()

        # ── Client-side state tracking (for masking/obs where server doesn't provide) ──
        self._last_cast_time = 0.0    # for GCD tracking
        self._last_action = 0

        # ── Episode tracking (matching sim) ──
        self._ep_kills = 0
        self._ep_xp = 0
        self._ep_loot = 0
        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_damage_dealt = 0.0
        self._ep_loot_items = 0
        self._ep_sell_copper = 0
        self._ep_areas = 0
        self._ep_zones = 0
        self._ep_maps = 0
        self._ep_exploration_reward = 0.0
        self._ep_levels_gained = 0
        self._ep_quests_completed = 0
        self._ep_quest_xp = 0
        self._ep_equipment_upgrades = 0
        self._idle_steps = 0

        # State tracking for rewards
        self._prev_dist_to_target = None
        self._prev_target_hp = None
        self._steps_since_kill_xp = 0

        # Exploration tracking (grid-based, matching sim fallback)
        self._visited_areas = set()
        self._visited_zones = set()
        self._visited_maps = set()

    # ─── Memory System ────────────────────────────────────────────

    def _set_memory_file(self, bot_name):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        safe = (bot_name or "default").replace(" ", "_")
        self.memory_file = os.path.join(base_dir, f"npc_memory_{safe}.json")

    def load_memory(self):
        path = self.memory_file or DEFAULT_MEMORY_FILE
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    self.npc_memory = json.load(f)
            except Exception:
                self.npc_memory = {}

    def save_memory(self):
        try:
            path = self.memory_file or DEFAULT_MEMORY_FILE
            tmp = path + ".tmp"
            with open(tmp, 'w') as f:
                json.dump(self.npc_memory, f, indent=4)
            os.replace(tmp, path)
        except Exception:
            pass

    def get_best_memory_target(self):
        if not self.last_state:
            return None
        me_x = self.last_state['x']
        me_y = self.last_state['y']
        best_mob = None
        min_dist = 9999.0

        for guid, mob in self.npc_memory.items():
            if guid in self.blacklist:
                continue
            if mob['hp'] == 0 and math.sqrt((mob['x'] - me_x)**2 + (mob['y'] - me_y)**2) > 40.0:
                continue
            dx = mob['x'] - me_x
            dy = mob['y'] - me_y
            dist = math.sqrt(dx * dx + dy * dy)
            if 5.0 < dist < min_dist:
                min_dist = dist
                best_mob = mob
        return best_mob

    # ─── TCP Communication ────────────────────────────────────────

    def _get_state_from_server(self, allow_any_player=False):
        while True:
            try:
                data = self.sock.recv(8192)
                if not data:
                    break
                self._recv_buffer += data.decode('utf-8')
                if "\n" in self._recv_buffer:
                    lines = self._recv_buffer.split("\n")
                    if len(lines) < 2:
                        continue
                    raw_json = lines[-2]
                    self._recv_buffer = lines[-1]
                    if not raw_json.strip():
                        continue
                    try:
                        data = json.loads(raw_json)
                        if not data['players']:
                            continue
                        player = None
                        if self.bot_name:
                            for candidate in data['players']:
                                if candidate.get('name') == self.bot_name:
                                    player = candidate
                                    break
                            if player is None:
                                return None
                        else:
                            player = data['players'][0]
                            self.bot_name = player.get('name', None)
                            self._set_memory_file(self.bot_name)
                            self.load_memory()

                        self.my_name = player['name']
                        return player
                    except Exception:
                        continue
            except socket.timeout:
                return None
            except Exception:
                continue
        return None

    def _send_cmd(self, cmd):
        if cmd and self.my_name:
            try:
                self.sock.sendall(f"{self.my_name}:{cmd}\n".encode('utf-8'))
            except Exception:
                pass

    # ─── Blacklist ────────────────────────────────────────────────

    def _manage_blacklist(self):
        now = time.time()
        self.blacklist = {g: exp for g, exp in self.blacklist.items() if exp > now}

    # ─── Action Masking ───────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """Return boolean mask: True = action allowed, False = masked.

        Mirrors sim masking logic using available server state.
        Fields not yet in the C++ state stream use client-side approximations.
        """
        mask = np.ones(self.action_space.n, dtype=bool)

        if not self.last_state:
            # No state yet — only allow noop
            mask[:] = False
            mask[0] = True
            return mask

        data = self.last_state
        in_combat = data.get('combat') == 'true'
        is_casting = data.get('casting') == 'true'
        target_alive = data.get('target_status') == 'alive'
        target_dead = data.get('target_status') == 'dead'
        mana_pct = data['power'] / max(1, data['max_power'])
        nearby_mobs = data.get('nearby_mobs', [])

        # ── While casting: ONLY noop is valid ──
        if is_casting:
            mask[:] = False
            mask[0] = True
            return mask

        # ── While eating: ONLY noop ── (if server reports is_eating)
        if data.get('is_eating') == 'true':
            mask[:] = False
            mask[0] = True
            return mask

        # ── GCD check (client-side approximation) ──
        gcd_blocked = (time.time() - self._last_cast_time) < GCD_DURATION

        # ── Target nearest (4): need alive attackable mobs ──
        has_targetable = any(
            m['hp'] > 0 and m.get('attackable', 0) == 1
            for m in nearby_mobs
        )
        if not has_targetable:
            mask[4] = False

        # ── Spell masks ──
        spell_actions = set(self._ACTION_SPELL.keys())
        for action_id in spell_actions:
            # GCD blocks all spells
            if gcd_blocked:
                mask[action_id] = False
                continue

            # OOM check (approximate: block if < 5% mana for most spells)
            if mana_pct < 0.05:
                mask[action_id] = False
                continue

            # Offensive spells: need alive target
            if action_id in self._OFFENSIVE_ACTIONS:
                if not target_alive:
                    mask[action_id] = False
                    continue

            # AoE spells: need alive mob in combat nearby
            if action_id in self._AOE_ACTIONS:
                has_aoe_target = any(
                    m['hp'] > 0 and m.get('target', '0') != '0'
                    for m in nearby_mobs
                )
                if not has_aoe_target:
                    mask[action_id] = False
                    continue

        # ── Buff duplication checks (from server state) ──
        if data.get('has_shield') == 'true':
            mask[10] = False  # PW:Shield already active
        if target_alive and data.get('target_has_sw_pain') == 'true':
            mask[9] = False   # SW:Pain already on target

        # Extended buff checks (from updated C++ state, zero if not available)
        if data.get('has_renew') == 'true':
            mask[13] = False
        if data.get('has_inner_fire') == 'true':
            mask[15] = False
        if data.get('has_fortitude') == 'true':
            mask[16] = False
        if data.get('has_shadow_protection') == 'true':
            mask[21] = False
        if data.get('has_divine_spirit') == 'true':
            mask[22] = False
        if data.get('has_fear_ward') == 'true':
            mask[23] = False
        if target_alive and data.get('target_has_holy_fire') == 'true':
            mask[14] = False
        if target_alive and data.get('target_has_devouring_plague') == 'true':
            mask[19] = False
        if target_alive and data.get('target_has_vampiric_touch') == 'true':
            mask[27] = False
        if data.get('is_channeling') == 'true':
            mask[26] = False  # Mind Flay already channeling
        if data.get('dispersion_active') == 'true':
            mask[28] = False

        # ── Loot (7): need dead mob in range, NOT in combat ──
        has_lootable = False
        if not in_combat:
            me_x, me_y = data['x'], data['y']
            for mob in nearby_mobs:
                if mob['hp'] == 0 and mob['guid'] not in self.blacklist:
                    d = math.sqrt((mob['x'] - me_x)**2 + (mob['y'] - me_y)**2)
                    if d <= LOOT_RANGE:
                        has_lootable = True
                        break
        if not has_lootable:
            mask[7] = False

        # ── Sell (8): proximity-based vendor ──
        if in_combat:
            mask[8] = False
        else:
            vendor = self._get_nearest_vendor()
            if vendor is None:
                mask[8] = False
            else:
                vdist = math.sqrt((vendor['x'] - data['x'])**2 + (vendor['y'] - data['y'])**2)
                if vdist > SELL_RANGE:
                    mask[8] = False

        # ── Quest interact (11): not implemented on live server yet ──
        if not self._enable_quests:
            mask[11] = False
        elif in_combat:
            mask[11] = False

        # ── Eat/Drink (17): OOC only, not casting, not full ──
        if in_combat or is_casting:
            mask[17] = False
        elif data['hp'] >= data['max_hp'] and data['power'] >= data['max_power']:
            mask[17] = False

        # ── Shadowform toggle (29): needs talent, not casting ──
        # Client can't reliably detect talent state, so allow unless casting/GCD
        if is_casting or gcd_blocked:
            mask[29] = False

        return mask

    # ─── Observation Builder ──────────────────────────────────────

    def _build_obs(self, data: dict) -> np.ndarray:
        """Build 52-dim observation vector matching WoWSimEnv exactly."""
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

        mob_count, closest_mob_dist, closest_mob_angle, num_attackers = \
            self._compute_nearby_mob_features(data)

        target_level = data.get('target_level', 0) / 10.0
        player_level = data.get('level', 1) / 10.0

        # Dims 15-16: known from original C++ state
        has_shield = 1.0 if data.get('has_shield') == 'true' else 0.0
        target_has_sw_pain = 1.0 if data.get('target_has_sw_pain') == 'true' else 0.0

        # Dims 17-22: extended buff states (from updated C++ state, 0 if not available)
        has_renew = 1.0 if data.get('has_renew') == 'true' else 0.0
        has_inner_fire = 1.0 if data.get('has_inner_fire') == 'true' else 0.0
        has_fortitude = 1.0 if data.get('has_fortitude') == 'true' else 0.0
        mind_blast_ready = 1.0 if data.get('mind_blast_ready') == 'true' else 0.0
        target_has_holy_fire = 1.0 if data.get('target_has_holy_fire') == 'true' else 0.0
        is_eating = 1.0 if data.get('is_eating') == 'true' else 0.0

        # Dims 23-28: more buff/debuff states
        target_has_dp = 1.0 if data.get('target_has_devouring_plague') == 'true' else 0.0
        has_shadow_prot = 1.0 if data.get('has_shadow_protection') == 'true' else 0.0
        has_divine_spirit = 1.0 if data.get('has_divine_spirit') == 'true' else 0.0
        has_fear_ward = 1.0 if data.get('has_fear_ward') == 'true' else 0.0
        psychic_scream_ready = 1.0 if data.get('psychic_scream_ready') == 'true' else 0.0
        # num_feared: not available from server, approximate as 0
        num_feared = 0.0

        # Dims 29-32: talent-related
        target_has_vt = 1.0 if data.get('target_has_vampiric_touch') == 'true' else 0.0
        shadowform_active = 1.0 if data.get('shadowform_active') == 'true' else 0.0
        dispersion_active = 1.0 if data.get('dispersion_active') == 'true' else 0.0
        is_channeling = 1.0 if data.get('is_channeling') == 'true' else 0.0

        # Dims 33-42: combat stats (from updated C++ state, 0 if not available)
        stat_sp = data.get('spell_power', 0) / 200.0
        stat_spell_crit = data.get('spell_crit', 0) / 50.0
        stat_spell_haste = data.get('spell_haste', 0) / 50.0
        stat_armor = data.get('total_armor', 0) / 2000.0
        stat_ap = data.get('attack_power', 0) / 500.0
        stat_melee_crit = data.get('melee_crit', 0) / 50.0
        stat_dodge = data.get('dodge', 0) / 50.0
        stat_hit = data.get('hit_spell', 0) / 50.0
        stat_expertise = data.get('expertise', 0) / 50.0
        stat_arp = data.get('armor_pen', 0) / 100.0

        # Dims 43-45: vendor observations
        vendor_obs = self._compute_vendor_obs(data)

        # Dims 46-51: quest observations
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
            target_has_dp, has_shadow_prot, has_divine_spirit,
            has_fear_ward, psychic_scream_ready, num_feared,
            target_has_vt, shadowform_active, dispersion_active, is_channeling,
            stat_sp, stat_spell_crit, stat_spell_haste, stat_armor,
            stat_ap, stat_melee_crit, stat_dodge, stat_hit,
            stat_expertise, stat_arp,
            *vendor_obs,
            *quest_obs,
        ], dtype=np.float32)

    def _compute_nearby_mob_features(self, data: dict):
        """Compute observation features from nearby_mobs list."""
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

    def _compute_vendor_obs(self, data: dict):
        """Compute vendor observation features (3 dimensions): nearby, distance, angle."""
        vendor = self._get_nearest_vendor()
        if vendor is None:
            return (0.0, 0.0, 0.0)

        dx = vendor['x'] - data['x']
        dy = vendor['y'] - data['y']
        dist = math.sqrt(dx * dx + dy * dy)
        vendor_dist = min(dist, 40.0) / 40.0
        vendor_angle_abs = math.atan2(dy, dx)
        rel = vendor_angle_abs - data['o']
        while rel > math.pi:
            rel -= 2 * math.pi
        while rel < -math.pi:
            rel += 2 * math.pi
        vendor_angle = rel / math.pi

        return (1.0, vendor_dist, vendor_angle)

    def _compute_quest_obs(self, data: dict):
        """Compute quest observation features (6 dims) — zero when quests disabled."""
        if not self._enable_quests:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        has_active = 1.0 if data.get('quest_active') == 'true' else 0.0
        progress = float(data.get('quest_progress', 0.0))
        quests_done = data.get('quests_completed_total', 0) / 10.0

        # Quest NPC info (from server state if available)
        qnpc_nearby = 1.0 if data.get('quest_npc_nearby') == 'true' else 0.0
        qnpc_dist = data.get('quest_npc_distance', 0) / 40.0
        qnpc_angle = data.get('quest_npc_angle', 0) / math.pi

        return (has_active, progress, qnpc_nearby, qnpc_dist, qnpc_angle,
                quests_done)

    def _get_nearest_vendor(self):
        """Find nearest vendor from NPC memory."""
        if not self.last_state:
            return None
        me_x = self.last_state['x']
        me_y = self.last_state['y']
        best = None
        best_dist = 9999.0
        for guid, mob in self.npc_memory.items():
            if mob.get('vendor', 0) != 1:
                continue
            dx = mob['x'] - me_x
            dy = mob['y'] - me_y
            d = math.sqrt(dx * dx + dy * dy)
            if d < best_dist:
                best_dist = d
                best = mob
        return best

    # ─── Exploration Tracking ─────────────────────────────────────

    def _update_exploration(self, data):
        """Track grid-based exploration (matches sim fallback mode)."""
        x, y = data['x'], data['y']
        map_id = 0  # Eastern Kingdoms (hardcoded for now)

        area_key = (map_id, int(x // AREA_GRID_SIZE), int(y // AREA_GRID_SIZE))
        zone_key = (map_id, int(x // ZONE_GRID_SIZE), int(y // ZONE_GRID_SIZE))

        new_areas = 0
        new_zones = 0
        new_maps = 0

        if area_key not in self._visited_areas:
            self._visited_areas.add(area_key)
            new_areas = 1

        if zone_key not in self._visited_zones:
            self._visited_zones.add(zone_key)
            new_zones = 1

        if map_id not in self._visited_maps:
            self._visited_maps.add(map_id)
            new_maps = 1

        return new_areas, new_zones, new_maps

    # ─── Heading Kick ─────────────────────────────────────────────

    def _initial_heading_kick(self):
        if not self.my_name:
            return
        mapping = {
            "Autoai": 0, "Bota": 2, "Botb": 4,
            "Botc": 6, "Botd": 8, "Bote": 10,
        }
        steps = mapping.get(self.my_name, 0)
        for _ in range(steps):
            self._send_cmd("turn_left:0")

    # ─── Step ─────────────────────────────────────────────────────

    def step(self, action):
        action = int(action)
        self._manage_blacklist()
        if time.time() - self.last_save_time > 30.0:
            self.save_memory()
            self.last_save_time = time.time()

        nearby_mobs = self.last_state.get('nearby_mobs', []) if self.last_state else []

        # ─── Execute Action (masking handles validity) ────────────
        cmd = ""
        if action == 0:
            pass  # noop
        elif action == 1:
            cmd = "move_forward:0"
        elif action == 2:
            cmd = "turn_left:0"
        elif action == 3:
            cmd = "turn_right:0"
        elif action == 4:
            # Target nearest alive attackable mob
            best_guid = None
            min_dist = 9999.0
            if self.last_state:
                for mob in nearby_mobs:
                    if mob['hp'] <= 0 or mob.get('attackable', 0) == 0:
                        continue
                    if mob['guid'] in self.blacklist:
                        continue
                    dx = mob['x'] - self.last_state['x']
                    dy = mob['y'] - self.last_state['y']
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < min_dist:
                        min_dist = dist
                        best_guid = mob['guid']
            if best_guid:
                cmd = f"target_guid:{best_guid}"
        elif action == 7:
            # Loot nearest dead mob
            loot_guid = None
            min_d = LOOT_RANGE
            if self.last_state:
                for mob in nearby_mobs:
                    if mob['hp'] == 0 and mob['guid'] not in self.blacklist:
                        d = math.sqrt(
                            (mob['x'] - self.last_state['x'])**2
                            + (mob['y'] - self.last_state['y'])**2
                        )
                        if d < min_d:
                            min_d = d
                            loot_guid = mob['guid']
            if loot_guid:
                cmd = f"loot_guid:{loot_guid}"
                self.blacklist[loot_guid] = time.time() + self.BLACKLIST_DURATION
                if loot_guid in self.npc_memory:
                    del self.npc_memory[loot_guid]
        elif action == 8:
            # Sell at nearest vendor
            vendor = self._get_nearest_vendor()
            if vendor and self.last_state:
                vdist = math.sqrt(
                    (vendor['x'] - self.last_state['x'])**2
                    + (vendor['y'] - self.last_state['y'])**2
                )
                if vdist < SELL_RANGE:
                    # Find vendor GUID from memory
                    for guid, mob in self.npc_memory.items():
                        if mob.get('vendor', 0) == 1:
                            d = math.sqrt(
                                (mob['x'] - self.last_state['x'])**2
                                + (mob['y'] - self.last_state['y'])**2
                            )
                            if d < SELL_RANGE:
                                cmd = f"sell_grey:{guid}"
                                break
        elif action == 11:
            # Quest NPC interact (requires server support)
            if self._enable_quests:
                cmd = "quest_interact:0"
        elif action == 17:
            # Eat/Drink (requires server support)
            cmd = "eat_drink:0"
        elif action == 29:
            # Toggle Shadowform
            cmd = "toggle_shadowform:0"
        elif action in self._ACTION_SPELL:
            # Cast spell
            spell_id = self._ACTION_SPELL[action]
            cmd = f"cast:{spell_id}"
            self._last_cast_time = time.time()

        self._send_cmd(cmd)

        # ─── Wait for next server state ───────────────────────────
        data = None
        deadline = time.time() + 1.0
        while data is None and time.time() < deadline:
            data = self._get_state_from_server()
            if data is None:
                time.sleep(0.01)

        if not data:
            return np.zeros(52, dtype=np.float32), -0.1, False, True, {"timeout": True}

        # ─── Memory Update ────────────────────────────────────────
        current_mobs = data.get('nearby_mobs', [])
        for mob in current_mobs:
            guid = mob['guid']
            if mob['hp'] == 0:
                if guid not in self.blacklist:
                    self.blacklist[guid] = time.time() + self.BLACKLIST_DURATION
                    if guid in self.npc_memory:
                        del self.npc_memory[guid]
                continue
            if guid in self.blacklist:
                continue
            self.npc_memory[guid] = mob

        # ─── Consume Events ───────────────────────────────────────
        xp_gained = data.get('xp_gained', 0)
        loot_money = data.get('loot_copper', 0)
        loot_score = data.get('loot_score', 0)
        free_slots_curr = data.get('free_slots', 0)
        max_hp = max(1, data['max_hp'])
        hp_pct = data['hp'] / max_hp
        mana_pct = data['power'] / max(1, data['max_power'])

        obs = self._build_obs(data)

        t_exists = 1.0 if data.get('target_status') == 'alive' else 0.0
        is_casting_now = data.get('casting') == 'true'

        # Current target tracking
        curr_target_hp = 0.0
        curr_dist_to_target = 9999.0
        if t_exists > 0.5:
            dx_t = data['tx'] - data['x']
            dy_t = data['ty'] - data['y']
            curr_dist_to_target = math.sqrt(dx_t * dx_t + dy_t * dy_t)
            curr_target_hp = data.get('target_hp', 0)

        # ─── Compute Rewards (sparse design — matching sim) ──────
        reward = 0.0

        # 1. Step-Penalty
        reward -= 0.001

        # 2. Idle-Penalty
        if action == 0 and not is_casting_now:
            reward -= 0.005
            self._idle_steps += 1

        # 3. Damage-Reward
        if (t_exists > 0.5
                and self._prev_target_hp is not None
                and self._prev_target_hp > 0):
            damage = self._prev_target_hp - curr_target_hp
            if damage > 0:
                reward += min(damage * 0.03, 1.0)
                self._ep_damage_dealt += damage

        # 3b. Approach Shaping
        if t_exists > 0.5:
            if self._prev_dist_to_target is not None and self._prev_dist_to_target < 9000:
                delta = self._prev_dist_to_target - curr_dist_to_target
                reward += max(-0.1, min(delta * 0.03, 0.15))
            self._prev_dist_to_target = curr_dist_to_target
        else:
            self._prev_dist_to_target = None

        # 4. XP/Kill
        if xp_gained > 0:
            reward += 10.0 + xp_gained * 0.5
            self._ep_kills += 1
            self._ep_xp += xp_gained
            self._steps_since_kill_xp = 0
        else:
            self._steps_since_kill_xp += 1

        # 5. Level-Up (NOT terminal — matches sim)
        if data.get('leveled_up') == 'true':
            reward += 15.0
            self._ep_levels_gained += 1

        # 6. Equipment-Upgrade (scaled by score improvement)
        if data.get('equipped_upgrade') == 'true':
            upgrade_score = data.get('upgrade_score', 10)  # fallback estimate
            reward += min(1.0 + upgrade_score * 0.15, 5.0)
            self._ep_equipment_upgrades += 1

        # 7. Loot (quality-based when available, fallback to combined score)
        loot_qualities = data.get('loot_qualities', [])
        if loot_qualities:
            for q in loot_qualities:
                reward += QUALITY_LOOT_REWARD.get(q, 0.3)
            self._ep_loot_items += len(loot_qualities)
        elif loot_money > 0 or loot_score > 0:
            # Fallback: approximate loot reward
            reward += min((loot_money * 0.01) + (loot_score * 0.05), 2.0)
            self._ep_loot_items += 1
        if loot_money > 0:
            reward += min(loot_money * 0.01, 1.0)
            self._ep_loot += loot_money

        # 8. Sold items — reward scales with inventory fullness
        items_sold = data.get('items_sold', 0)
        sell_copper = data.get('sell_copper', 0)
        if items_sold == 0 and self.last_state and free_slots_curr > self.last_state.get('free_slots', 0):
            # Approximation: slots freed = items sold
            items_sold = free_slots_curr - self.last_state.get('free_slots', 0)
        if items_sold > 0:
            total_slots = 16  # base backpack
            fullness = items_sold / max(total_slots, 1)
            sell_reward = 1.0 + 7.0 * fullness
            reward += sell_reward
        if sell_copper > 0:
            reward += min(sell_copper * 0.005, 2.0)
            self._ep_sell_copper += sell_copper

        # 9. Exploration (grid-based)
        new_areas, new_zones, new_maps = self._update_exploration(data)
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

        # 10. Quest completion
        quests_done = data.get('quests_completed', 0)
        quest_xp = data.get('quest_xp', 0)
        if quests_done > 0:
            reward += 20.0 * quests_done
            self._ep_quests_completed += quests_done
            self._ep_quest_xp += quest_xp

        # 11. Terminal: Death (only terminal signal — matches sim)
        terminated = False
        if data['hp'] == 0:
            reward = -15.0
            terminated = True
        # OOM is NOT terminal (bot must learn to wait for regen)

        # Stall detection: 3k steps without kill XP
        truncated = (self._steps_since_kill_xp >= 3_000)

        # ─── State tracking ───────────────────────────────────────
        self._prev_target_hp = curr_target_hp if t_exists > 0.5 else None
        self._ep_reward += reward
        self._ep_length += 1
        self._last_action = action

        # ─── Episode stats ────────────────────────────────────────
        info = {}
        if terminated or truncated:
            info["episode_stats"] = {
                "reward": self._ep_reward,
                "length": self._ep_length,
                "kills": self._ep_kills,
                "xp": self._ep_xp,
                "loot": self._ep_loot,
                "damage_dealt": self._ep_damage_dealt,
                "death": 1 if data['hp'] == 0 else 0,
                "idle_ratio": self._idle_steps / max(1, self._ep_length),
                "areas_explored": self._ep_areas,
                "zones_explored": self._ep_zones,
                "maps_explored": self._ep_maps,
                "rw_explore": self._ep_exploration_reward,
                "levels_gained": self._ep_levels_gained,
                "final_level": data.get('level', 1),
                "loot_items": self._ep_loot_items,
                "sell_copper": self._ep_sell_copper,
                "quests_completed": self._ep_quests_completed,
                "quest_xp": self._ep_quest_xp,
                "equipment_upgrades": self._ep_equipment_upgrades,
            }

        self.last_state = data
        return obs, reward, terminated, truncated, info

    # ─── Reset ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print(">>> RESETTING ENVIRONMENT... <<<")

        name_to_use = self.my_name or self.bot_name
        if name_to_use:
            self._send_cmd("reset:0")

        # Wait for data
        data = None
        start_time = time.time()
        allow_any_player = (self.bot_name is None)
        fallback_logged = False
        max_wait = 60.0 if self.bot_name else 10.0

        while data is None:
            data = self._get_state_from_server(allow_any_player=allow_any_player)
            if data is None:
                waited = time.time() - start_time
                if self.bot_name and waited > max_wait:
                    raise RuntimeError(
                        f"Reset Timeout: no data for bot '{self.bot_name}'. Is the bot online?")
                if not self.bot_name and waited > 10.0:
                    if not fallback_logged:
                        print("Reset Timeout: no bot data, trying fallback player.")
                        fallback_logged = True
                    allow_any_player = True
                print("Waiting for server response...")
                time.sleep(1.0)

        self.last_state = data
        self._initial_heading_kick()

        # Reset episode tracking
        self._ep_kills = 0
        self._ep_xp = 0
        self._ep_loot = 0
        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_damage_dealt = 0.0
        self._ep_loot_items = 0
        self._ep_sell_copper = 0
        self._ep_areas = 0
        self._ep_zones = 0
        self._ep_maps = 0
        self._ep_exploration_reward = 0.0
        self._ep_levels_gained = 0
        self._ep_quests_completed = 0
        self._ep_quest_xp = 0
        self._ep_equipment_upgrades = 0
        self._idle_steps = 0
        self._prev_dist_to_target = None
        self._prev_target_hp = None
        self._steps_since_kill_xp = 0
        self._last_cast_time = 0.0
        self._last_action = 0

        # Reset exploration
        self._visited_areas = set()
        self._visited_zones = set()
        self._visited_maps = set()

        obs = self._build_obs(data)
        return obs, {}
