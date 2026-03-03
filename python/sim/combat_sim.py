"""
WoW Combat Simulation Engine — Pure Python, no server needed.

Simulates a Human Priest in Northshire Valley fighting mobs.
All values derived from AzerothCore DB exports and game mechanics.
Supports leveling from 1–79 with AzerothCore XP formulas.

Tick-based: 1 tick = 0.5 seconds (matches WoWEnv decision interval).
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sim.terrain import SimTerrain
    from sim.creature_db import CreatureDB
    from sim.loot_db import LootDB


# ─── XP Table (cumulative XP needed to reach level N) ────────────────
# Index = level, value = total XP required.  XP_TABLE[1] = 0 (start).
XP_TABLE = [
    0,       # 0  (unused)
    0,       # 1  (start level)
    400,     # 2
    900,     # 3
    1400,    # 4
    2100,    # 5
    2800,    # 6
    3600,    # 7
    4500,    # 8
    5400,    # 9
    6500,    # 10
    7600,    # 11
    8700,    # 12
    9800,    # 13
    11000,   # 14
    12300,   # 15
    13600,   # 16
    15000,   # 17
    16400,   # 18
    17800,   # 19
    19300,   # 20
    20800,   # 21
    22400,   # 22
    24000,   # 23
    25500,   # 24
    27200,   # 25
    28900,   # 26
    30500,   # 27
    32200,   # 28
    33900,   # 29
    36300,   # 30
    38800,   # 31
    41600,   # 32
    44600,   # 33
    48000,   # 34
    51400,   # 35
    55000,   # 36
    58700,   # 37
    62400,   # 38
    66200,   # 39
    70200,   # 40
    74300,   # 41
    78500,   # 42
    82800,   # 43
    87100,   # 44
    91600,   # 45
    96300,   # 46
    101000,  # 47
    105800,  # 48
    110700,  # 49
    115700,  # 50
    120900,  # 51
    126100,  # 52
    131500,  # 53
    137000,  # 54
    142500,  # 55
    148200,  # 56
    154000,  # 57
    159900,  # 58
    165800,  # 59
    172000,  # 60
    290000,  # 61
    317000,  # 62
    349000,  # 63
    386000,  # 64
    428000,  # 65
    475000,  # 66
    527000,  # 67
    585000,  # 68
    648000,  # 69
    717000,  # 70
    1523800, # 71
    1539600, # 72
    1555700, # 73
    1571800, # 74
    1587900, # 75
    1604200, # 76
    1620700, # 77
    1637400, # 78
    1653900, # 79
    1670800, # 80 (cap)
]

MAX_LEVEL = len(XP_TABLE) - 1  # 80


# ─── AzerothCore XP-per-kill formulas ─────────────────────────────────

def get_gray_level(pl_level: int) -> int:
    """Mob level at or below which the mob gives zero XP."""
    if pl_level <= 5:
        return 0
    elif pl_level <= 39:
        return pl_level - 5 - pl_level // 10
    elif pl_level <= 59:
        return pl_level - 1 - pl_level // 5
    else:
        return pl_level - 9


def get_zero_difference(pl_level: int) -> int:
    """ZD value used in the lower-level XP formula."""
    if pl_level < 8:
        return 5
    elif pl_level < 10:
        return 6
    elif pl_level < 12:
        return 7
    elif pl_level < 16:
        return 8
    elif pl_level < 20:
        return 9
    elif pl_level < 30:
        return 11
    elif pl_level < 40:
        return 12
    elif pl_level < 45:
        return 13
    elif pl_level < 50:
        return 14
    elif pl_level < 55:
        return 15
    elif pl_level < 60:
        return 16
    else:
        return 17


def base_xp_gain(pl_level: int, mob_level: int) -> int:
    """Calculate base XP for killing a mob (AzerothCore formula).

    Uses CONTENT_1_60 (nBaseExp=45) for all levels — simplified for sim.
    """
    n_base_exp = 45

    if mob_level >= pl_level:
        level_diff = min(mob_level - pl_level, 4)
        return ((pl_level * 5 + n_base_exp) * (20 + level_diff) // 10 + 1) // 2
    else:
        gray = get_gray_level(pl_level)
        if mob_level > gray:
            zd = get_zero_difference(pl_level)
            return (pl_level * 5 + n_base_exp) * (zd + mob_level - pl_level) // zd
        else:
            return 0


# ─── Per-level stat scaling ───────────────────────────────────────────

def player_max_hp(level: int) -> int:
    """Base HP 72 + 50 per level gained."""
    return 72 + (level - 1) * 50


def player_max_mana(level: int) -> int:
    """Base mana 123 + 5 per level gained."""
    return 123 + (level - 1) * 5


def smite_damage(level: int) -> tuple[int, int]:
    """Smite damage range: base (13-17) + 10 per level gained."""
    bonus = (level - 1) * 10
    return (13 + bonus, 17 + bonus)


def heal_amount(level: int) -> tuple[int, int]:
    """Lesser Heal range: base (46-56) + 5 per level gained."""
    bonus = (level - 1) * 5
    return (46 + bonus, 56 + bonus)


# ─── Spell Definitions ───────────────────────────────────────────────

@dataclass
class SpellDef:
    id: int
    name: str
    cast_ticks: int       # cast time in ticks (1 tick = 0.5s)
    mana_cost: int
    min_damage: int = 0
    max_damage: int = 0
    min_heal: int = 0
    max_heal: int = 0
    spell_range: float = 0.0
    is_dot: bool = False
    dot_damage: int = 0
    dot_ticks: int = 0     # total duration in ticks
    dot_interval: int = 6  # ticks between dot ticks (3s = 6 ticks)
    is_shield: bool = False
    shield_absorb: int = 0
    shield_duration: int = 0  # ticks
    gcd_ticks: int = 3    # 1.5s = 3 ticks


SPELLS = {
    585: SpellDef(
        id=585, name="Smite",
        cast_ticks=3, mana_cost=6,
        min_damage=13, max_damage=17,
        spell_range=30.0,
    ),
    2050: SpellDef(
        id=2050, name="Lesser Heal",
        cast_ticks=3, mana_cost=11,
        min_heal=46, max_heal=56,
        spell_range=0.0,  # self-cast
    ),
    589: SpellDef(
        id=589, name="Shadow Word: Pain",
        cast_ticks=0, mana_cost=25,  # instant
        spell_range=30.0,
        is_dot=True,
        dot_damage=30,     # total 30 over 18s = ~5 per tick
        dot_ticks=36,      # 18s = 36 ticks
        dot_interval=6,    # tick every 3s
    ),
    17: SpellDef(
        id=17, name="Power Word: Shield",
        cast_ticks=0, mana_cost=25,  # instant
        is_shield=True,
        shield_absorb=44,
        shield_duration=60,  # 30s = 60 ticks
    ),
}


# ─── Mob Definitions (from creature_template DB) ─────────────────────

@dataclass
class MobTemplate:
    entry: int
    name: str
    min_level: int
    max_level: int
    base_hp: int          # already computed from basehp * HealthModifier
    min_damage: int
    max_damage: int
    attack_speed: int     # ticks between attacks (2000ms = 4 ticks)
    detect_range: float
    min_gold: int = 0
    max_gold: int = 0
    xp_reward: int = 50
    speed: float = 4.0    # units per tick at walk speed (~2 units/s → 1 unit/tick at 0.5s)
    loot_id: int = 0      # creature_template.lootid → creature_loot_template.Entry (0 = no table)


# AzerothCore base HP per level (unit_class=1, expansion=0):
# Level 1: 42, Level 2: 55, Level 3: 71
MOB_TEMPLATES = {
    299: MobTemplate(
        entry=299, name="Diseased Young Wolf",
        min_level=1, max_level=1, base_hp=42,
        min_damage=1, max_damage=2, attack_speed=4,
        detect_range=20.0, xp_reward=50, loot_id=299,
    ),
    6: MobTemplate(
        entry=6, name="Kobold Vermin",
        min_level=1, max_level=2, base_hp=42,  # avg of 42-55
        min_damage=1, max_damage=3, attack_speed=4,
        detect_range=10.0, min_gold=1, max_gold=5, xp_reward=70, loot_id=6,
    ),
    69: MobTemplate(
        entry=69, name="Diseased Timber Wolf",
        min_level=2, max_level=2, base_hp=55,
        min_damage=2, max_damage=3, attack_speed=4,
        detect_range=20.0, xp_reward=90, loot_id=69,
    ),
    257: MobTemplate(
        entry=257, name="Kobold Worker",
        min_level=3, max_level=3, base_hp=71,
        min_damage=3, max_damage=5, attack_speed=4,
        detect_range=10.0, min_gold=1, max_gold=5, xp_reward=120, loot_id=257,
    ),
}


# ─── Spawn Positions (from creature.csv, map=0 near Northshire) ──────

# Grouped by entry, only using positions close to spawn point
SPAWN_POSITIONS = {
    299: [  # Diseased Young Wolf — 37 spawns
        (-8953.6, -48.6), (-8971.2, -52.8), (-8979.7, -64.6),
        (-8970.2, -87.7), (-8952.1, -83.9), (-8938.0, -49.8),
        (-8925.8, -38.4), (-8919.4, -52.7), (-8867.8, -69.9),
        (-8879.2, -50.4), (-8918.7, -73.9), (-8883.5, -59.0),
        (-8887.0, -85.5), (-8876.1, -114.9), (-8826.9, -159.5),
        (-8860.8, -88.0), (-8815.7, -110.3), (-8820.1, -91.1),
        (-8844.3, -45.0), (-8854.9, -106.4), (-8827.2, -100.0),
        (-8820.4, -79.7), (-8828.9, -69.4), (-8856.5, -131.6),
        (-8838.5, -133.4), (-8810.3, -179.3), (-8806.2, -143.2),
        (-8824.5, -58.7), (-8808.5, -91.4), (-8799.3, -70.4),
    ],
    6: [  # Kobold Vermin — 31 spawns
        (-8783.0, -161.6), (-8774.1, -184.5), (-8794.5, -170.4),
        (-8795.0, -134.2), (-8789.9, -143.3), (-8768.5, -176.4),
        (-8753.0, -160.8), (-8779.8, -195.4), (-8775.9, -148.5),
        (-8785.5, -171.2), (-8765.3, -93.4), (-8771.5, -115.9),
        (-8794.0, -118.5), (-8778.8, -125.7), (-8781.3, -115.6),
        (-8767.0, -117.4), (-8761.0, -127.5), (-8780.0, -108.4),
        (-8772.9, -103.6), (-8749.1, -115.0),
    ],
    69: [  # Diseased Timber Wolf — 24 spawns
        (-8872.6, -58.0), (-8851.4, -84.1), (-8813.4, -179.6),
        (-8804.5, -136.6), (-8747.6, -135.8), (-8781.3, -59.8),
        (-8752.6, -82.0), (-8761.9, -66.3), (-8789.8, -69.8),
        (-8718.8, -148.4), (-8736.3, -72.8), (-8753.2, -33.4),
        (-8748.7, -50.3), (-8737.1, -94.3), (-8766.0, -232.2),
        (-8805.4, -205.5),
    ],
    257: [  # Kobold Worker — 27 spawns
        (-8763.3, -159.3), (-8756.8, -171.4), (-8769.9, -138.1),
        (-8768.6, -113.1), (-8786.9, -105.9), (-8770.1, -117.7),
        (-8767.7, -111.6), (-8752.4, -101.9), (-8721.7, -155.8),
        (-8717.5, -144.6), (-8742.2, -176.2), (-8727.4, -133.8),
        (-8706.3, -129.8), (-8701.4, -118.1), (-8718.7, -98.3),
        (-8713.4, -90.3), (-8707.6, -108.4), (-8725.9, -109.7),
    ],
}


# ─── Player State ─────────────────────────────────────────────────────

INVENTORY_SLOTS = 30  # default bag capacity (no bag logic yet)


@dataclass
class Player:
    hp: int = 72
    max_hp: int = 72
    mana: int = 123
    max_mana: int = 123
    level: int = 1
    xp: int = 0               # cumulative XP (persists across consume_events)
    x: float = -8921.09
    y: float = -119.135
    z: float = 82.025
    orientation: float = 5.82
    in_combat: bool = False
    is_casting: bool = False
    cast_remaining: int = 0     # ticks until cast finishes
    cast_spell_id: int = 0
    gcd_remaining: int = 0      # ticks until GCD expires
    free_slots: int = INVENTORY_SLOTS
    # Shield state
    shield_absorb: int = 0
    shield_remaining: int = 0   # ticks
    # Accumulated rewards (consumed on read like real server)
    xp_gained: int = 0
    loot_copper: int = 0
    loot_score: int = 0
    equipped_upgrade: bool = False
    leveled_up: bool = False    # set True on level-up, consumed on read
    levels_gained: int = 0      # how many levels gained this tick (consumed on read)
    # Quality of items successfully looted this tick (consume-on-read)
    loot_items: list = field(default_factory=list)
    # Quality of items that couldn't be picked up — inventory full (consume-on-read)
    loot_failed: list = field(default_factory=list)
    # Equipped item tracking (for loot table upgrade detection)
    equipped_scores: dict = field(default_factory=dict)  # inventory_type -> best score
    # Regen tracking
    combat_timer: int = 0       # ticks since last combat action (for OOC regen)
    ooc_regen_accumulator: float = 0.0
    mana_regen_accumulator: float = 0.0


# ─── Mob Instance ─────────────────────────────────────────────────────

@dataclass
class Mob:
    uid: int
    template: MobTemplate
    hp: int
    max_hp: int
    level: int
    x: float
    y: float
    z: float = 82.0
    alive: bool = True
    in_combat: bool = False
    target_player: bool = False
    attack_timer: int = 0       # ticks until next attack
    respawn_timer: int = 0      # ticks until respawn (0 = alive)
    # DoT tracking
    dot_remaining: int = 0
    dot_timer: int = 0          # ticks until next dot tick
    dot_damage_per_tick: int = 0
    looted: bool = False
    spawn_x: float = 0.0
    spawn_y: float = 0.0
    spawn_z: float = 82.0


# ─── Combat Simulation ───────────────────────────────────────────────

class CombatSimulation:
    """
    Pure Python combat simulation for Level 1 Priest in Northshire.

    Tick-based: 1 tick = 0.5 seconds.
    No TCP, no server, no real-time — runs as fast as Python allows.
    """

    TICK_DURATION = 0.5       # seconds per tick
    MOVE_SPEED = 3.0          # units per move_forward action
    TURN_AMOUNT = 0.5         # radians per turn action
    SCAN_RANGE = 50.0         # mob visibility range
    TARGET_RANGE = 30.0       # max targeting range
    LOOT_RANGE = 10.0         # max looting range
    MOB_LEASH_RANGE = 60.0    # mob returns home after this distance from spawn
    OOC_DELAY_TICKS = 12      # 6 seconds out of combat before regen starts
    HP_REGEN_PER_TICK = 0.67  # 8 HP per 6 seconds = ~0.67/tick (OOC only)
    MANA_REGEN_PER_TICK = 2.75  # 33 mana per 6 seconds ≈ 2.75/tick (not casting)
    RESPAWN_TICKS = 120       # 60 seconds = 120 ticks
    MOB_SPEED = 1.0           # units per tick when chasing
    LOOT_CHANCE = 0.7         # probability of getting loot
    ITEM_SCORE_RANGE = (5, 25)
    UPGRADE_CHANCE = 0.15     # chance that looted item is an upgrade
    # Exploration grid sizes (for area/zone discovery tracking)
    AREA_CELL_SIZE = 50.0     # ~50x50 units per area cell
    ZONE_CELL_SIZE = 200.0    # ~200x200 units per zone cell
    # Chunk management (for creature_db mode)
    CHUNK_SIZE = 100.0        # world-units per chunk (must match creature_db.CHUNK_SIZE)
    CHUNK_RADIUS = 2          # activate 5×5 = 25 chunks around player

    def __init__(self, num_mobs: int = None, seed: Optional[int] = None,
                 terrain: 'SimTerrain | None' = None, env3d=None,
                 creature_db: 'CreatureDB | None' = None,
                 loot_db: 'LootDB | None' = None):
        self.rng = random.Random(seed)
        self.num_mobs = num_mobs  # None = all spawns
        self.terrain = terrain
        self.env3d = env3d        # WoW3DEnvironment for area/zone lookups
        self.creature_db = creature_db
        self.loot_db = loot_db    # LootDB for item drops from CSV loot tables
        self.map_id = 0           # Eastern Kingdoms
        self.player = Player()
        if self.terrain:
            self.player.z = self.terrain.get_height(self.player.x, self.player.y)
        self.mobs: list[Mob] = []
        self.target: Optional[Mob] = None
        self.tick_count: int = 0
        self.damage_dealt: int = 0
        self.kills: int = 0
        self._next_uid = 1
        # Exploration tracking (uses real WoW area IDs if env3d available, grid fallback otherwise)
        self.visited_areas: set = set()   # set of area_id (or (x,y) cells as fallback)
        self.visited_zones: set = set()   # set of zone_id (or (x,y) cells as fallback)
        self.visited_maps: set = set()    # set of map_id
        self._new_areas: int = 0          # consumed on read
        self._new_zones: int = 0          # consumed on read
        self._new_maps: int = 0           # consumed on read
        # Chunk management (creature_db mode)
        self._player_chunk: Optional[tuple] = None  # (map, cx, cy)
        self._active_chunks: set[tuple] = set()
        self._chunk_mobs: dict[tuple, list[Mob]] = {}
        if self.creature_db:
            self._update_chunks()
        else:
            self._spawn_mobs()
        self._update_exploration()  # register spawn position

    def _new_uid(self) -> int:
        uid = self._next_uid
        self._next_uid += 1
        return uid

    def _spawn_mobs(self):
        """Spawn mobs from real DB positions. num_mobs=None uses all spawns."""
        self.mobs.clear()
        all_spawns = []
        for entry, positions in SPAWN_POSITIONS.items():
            template = MOB_TEMPLATES[entry]
            for (x, y) in positions:
                all_spawns.append((template, x, y))

        # Pick a subset (or all if num_mobs is None)
        if self.num_mobs is not None:
            selected = self.rng.sample(all_spawns, min(self.num_mobs, len(all_spawns)))
        else:
            selected = all_spawns
            self.rng.shuffle(selected)
        for template, x, y in selected:
            level = self.rng.randint(template.min_level, template.max_level)
            # Scale HP by level
            hp_by_level = {1: 42, 2: 55, 3: 71}
            base_hp = hp_by_level.get(level, 42)
            z = self.terrain.get_height(x, y) if self.terrain else 82.0
            mob = Mob(
                uid=self._new_uid(),
                template=template,
                hp=base_hp,
                max_hp=base_hp,
                level=level,
                x=x, y=y, z=z,
                spawn_x=x, spawn_y=y, spawn_z=z,
            )
            self.mobs.append(mob)

    def reset(self) -> None:
        """Reset player and mobs to initial state."""
        self.player = Player()
        if self.terrain:
            self.player.z = self.terrain.get_height(self.player.x, self.player.y)
        self.target = None
        self.tick_count = 0
        self.damage_dealt = 0
        self.kills = 0
        self._next_uid = 1
        self.visited_areas.clear()
        self.visited_zones.clear()
        self.visited_maps.clear()
        self._new_areas = 0
        self._new_zones = 0
        self._new_maps = 0
        # Reset chunk state
        self._player_chunk = None
        self._active_chunks.clear()
        self._chunk_mobs.clear()
        self.mobs.clear()
        if self.creature_db:
            self._update_chunks()
        else:
            self._spawn_mobs()
        self._update_exploration()

    def _update_exploration(self):
        """Track area/zone/map discovery based on player position.

        Uses real WoW area IDs from AreaTable.dbc if env3d is available,
        falls back to grid-based cells otherwise (also for tiles not pre-loaded).
        """
        p = self.player

        if self.env3d and self.env3d.area_table:
            area_id = self.env3d.get_area_id(self.map_id, p.x, p.y)
            zone_id = self.env3d.get_zone_id(self.map_id, p.x, p.y)

            if area_id > 0:
                # Real WoW area/zone from pre-loaded tiles
                if area_id not in self.visited_areas:
                    self.visited_areas.add(area_id)
                    self._new_areas += 1
                if zone_id > 0 and zone_id not in self.visited_zones:
                    self.visited_zones.add(zone_id)
                    self._new_zones += 1
            else:
                # Outside pre-loaded tiles — grid-based fallback
                area_key = (int(p.x // self.AREA_CELL_SIZE), int(p.y // self.AREA_CELL_SIZE))
                zone_key = (int(p.x // self.ZONE_CELL_SIZE), int(p.y // self.ZONE_CELL_SIZE))
                if area_key not in self.visited_areas:
                    self.visited_areas.add(area_key)
                    self._new_areas += 1
                if zone_key not in self.visited_zones:
                    self.visited_zones.add(zone_key)
                    self._new_zones += 1

            if self.map_id not in self.visited_maps:
                self.visited_maps.add(self.map_id)
                self._new_maps += 1
        else:
            # Grid-based fallback (no 3D data)
            area_key = (int(p.x // self.AREA_CELL_SIZE), int(p.y // self.AREA_CELL_SIZE))
            zone_key = (int(p.x // self.ZONE_CELL_SIZE), int(p.y // self.ZONE_CELL_SIZE))

            if area_key not in self.visited_areas:
                self.visited_areas.add(area_key)
                self._new_areas += 1
            if zone_key not in self.visited_zones:
                self.visited_zones.add(zone_key)
                self._new_zones += 1

    # ─── Chunk Management (creature_db mode) ───────────────────────

    def _update_chunks(self):
        """Activate/deactivate chunks based on player position.

        Only runs when creature_db is set. Checks if the player moved to a
        new chunk and updates the active chunk set accordingly.
        """
        if not self.creature_db:
            return

        p = self.player
        cs = self.CHUNK_SIZE
        cx = int(p.x // cs) if p.x >= 0 else int(p.x // cs) - 1
        cy = int(p.y // cs) if p.y >= 0 else int(p.y // cs) - 1
        current_chunk = (self.map_id, cx, cy)

        if current_chunk == self._player_chunk:
            return  # player hasn't moved to a new chunk
        self._player_chunk = current_chunk

        # Determine which chunks should be active
        r = self.CHUNK_RADIUS
        needed: set[tuple] = set()
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                needed.add((self.map_id, cx + dx, cy + dy))

        # Deactivate old chunks
        for key in self._active_chunks - needed:
            for mob in self._chunk_mobs.pop(key, []):
                if self.target is mob:
                    self.target = None

        # Activate new chunks
        for key in needed - self._active_chunks:
            self._activate_chunk(key)

        self._active_chunks = needed

        # Rebuild mobs list from active chunks (mob objects persist)
        self.mobs = []
        for key in self._active_chunks:
            self.mobs.extend(self._chunk_mobs.get(key, []))

    def _activate_chunk(self, chunk_key: tuple):
        """Spawn mobs for a newly activated chunk from creature_db."""
        db = self.creature_db
        spawns = db.spatial_index.get(chunk_key, [])
        chunk_mobs: list[Mob] = []

        for sp in spawns:
            tmpl = db.templates.get(sp.entry)
            if tmpl is None:
                continue

            level = self.rng.randint(tmpl.min_level, tmpl.max_level)
            stats = db.get_mob_stats(tmpl, level)

            mob_template = MobTemplate(
                entry=tmpl.entry,
                name=tmpl.name,
                min_level=tmpl.min_level,
                max_level=tmpl.max_level,
                base_hp=stats['hp'],
                min_damage=stats['min_damage'],
                max_damage=stats['max_damage'],
                attack_speed=tmpl.attack_speed_ticks,
                detect_range=tmpl.detection_range,
                min_gold=tmpl.min_gold,
                max_gold=tmpl.max_gold,
                xp_reward=stats['xp'],
                loot_id=tmpl.lootid,
            )

            z = self.terrain.get_height(sp.x, sp.y) if self.terrain else sp.z
            mob = Mob(
                uid=self._new_uid(),
                template=mob_template,
                hp=stats['hp'],
                max_hp=stats['hp'],
                level=level,
                x=sp.x, y=sp.y, z=z,
                spawn_x=sp.x, spawn_y=sp.y, spawn_z=z,
            )
            chunk_mobs.append(mob)

        self._chunk_mobs[chunk_key] = chunk_mobs

    def _dist(self, x1: float, y1: float, x2: float, y2: float) -> float:
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx * dx + dy * dy)

    def _dist_to_mob(self, mob: Mob) -> float:
        return self._dist(self.player.x, self.player.y, mob.x, mob.y)

    def _angle_to(self, tx: float, ty: float) -> float:
        """Relative angle from player orientation to target. [-pi, pi]"""
        dx = tx - self.player.x
        dy = ty - self.player.y
        abs_angle = math.atan2(dy, dx)
        rel = abs_angle - self.player.orientation
        while rel > math.pi:
            rel -= 2 * math.pi
        while rel < -math.pi:
            rel += 2 * math.pi
        return rel

    # ─── Actions ──────────────────────────────────────────────────

    def do_noop(self):
        pass

    def do_move_forward(self):
        """Move 3 units in current orientation direction."""
        if self.player.is_casting:
            return
        p = self.player
        new_x = p.x + math.cos(p.orientation) * self.MOVE_SPEED
        new_y = p.y + math.sin(p.orientation) * self.MOVE_SPEED

        if self.terrain:
            self.terrain.ensure_loaded(new_x, new_y)
            new_z = self.terrain.get_height(new_x, new_y)
            if not self.terrain.check_walkable(p.x, p.y, p.z, new_x, new_y, new_z):
                return  # blocked by terrain slope/step
            p.z = new_z

        p.x = new_x
        p.y = new_y

    def do_turn_left(self):
        if self.player.is_casting:
            return
        self.player.orientation += self.TURN_AMOUNT
        if self.player.orientation > math.pi:
            self.player.orientation -= 2 * math.pi

    def do_turn_right(self):
        if self.player.is_casting:
            return
        self.player.orientation -= self.TURN_AMOUNT
        if self.player.orientation < -math.pi:
            self.player.orientation += 2 * math.pi

    def do_target_nearest(self):
        """Target the nearest alive, attackable mob in range."""
        if self.player.is_casting:
            return
        best = None
        best_dist = self.TARGET_RANGE
        for mob in self.mobs:
            if not mob.alive:
                continue
            d = self._dist_to_mob(mob)
            if d < best_dist:
                best_dist = d
                best = mob
        self.target = best

    def do_cast_smite(self) -> bool:
        """Start casting Smite. Returns True if cast started."""
        return self._start_cast(585)

    def do_cast_heal(self) -> bool:
        """Start casting Lesser Heal."""
        return self._start_cast(2050)

    def do_cast_sw_pain(self) -> bool:
        """Cast Shadow Word: Pain (instant)."""
        return self._start_cast(589)

    def do_cast_pw_shield(self) -> bool:
        """Cast Power Word: Shield (instant)."""
        return self._start_cast(17)

    def do_loot(self) -> bool:
        """Loot nearest dead mob within range.

        Uses loot tables from LootDB when loaded (creature_loot_template +
        item_template CSVs). Falls back to the original random loot system
        when no loot data is available.

        Each item requires 1 free inventory slot. Items that don't fit are
        tracked in player.loot_failed (quality list) for penalty signals.
        Gold never requires inventory space.
        """
        best = None
        best_dist = self.LOOT_RANGE
        for mob in self.mobs:
            if mob.alive or mob.looted:
                continue
            d = self._dist_to_mob(mob)
            if d < best_dist:
                best_dist = d
                best = mob
        if best is None:
            return False
        best.looted = True

        # Gold (always from creature_template min/max gold — separate from item loot)
        gold = self.rng.randint(best.template.min_gold,
                                max(best.template.min_gold, best.template.max_gold))
        self.player.loot_copper += gold

        # Item loot: use loot tables if available, else random fallback
        loot_id = best.template.loot_id
        if self.loot_db and self.loot_db.loaded and loot_id > 0:
            results = self.loot_db.roll_loot(loot_id, self.rng)
            for result in results:
                quality = result.item.quality
                if self.player.free_slots > 0:
                    self.player.loot_score += int(result.item.score * result.count)
                    self.player.free_slots -= 1
                    self.player.loot_items.append(quality)
                    # Check if equippable item is an upgrade
                    if result.item.inventory_type > 0:
                        current = self.player.equipped_scores.get(
                            result.item.inventory_type, 0.0)
                        if result.item.score > current:
                            self.player.equipped_scores[result.item.inventory_type] = result.item.score
                            self.player.equipped_upgrade = True
                else:
                    self.player.loot_failed.append(quality)
        else:
            # Fallback: random loot (no loot_db loaded)
            if self.rng.random() < self.LOOT_CHANCE:
                score = self.rng.randint(*self.ITEM_SCORE_RANGE)
                quality = 1  # assume Common for fallback items
                if self.player.free_slots > 0:
                    self.player.loot_score += score
                    self.player.free_slots -= 1
                    self.player.loot_items.append(quality)
                    if self.rng.random() < self.UPGRADE_CHANCE:
                        self.player.equipped_upgrade = True
                else:
                    self.player.loot_failed.append(quality)
        return True

    def do_sell(self) -> bool:
        """Sell items (simplified — just restores free_slots)."""
        if self.player.free_slots >= INVENTORY_SLOTS:
            return False
        self.player.free_slots = INVENTORY_SLOTS
        return True

    def _start_cast(self, spell_id: int) -> bool:
        """Attempt to start casting a spell."""
        if self.player.is_casting:
            return False
        if self.player.gcd_remaining > 0:
            return False

        spell = SPELLS.get(spell_id)
        if spell is None:
            return False
        if self.player.mana < spell.mana_cost:
            return False

        # Range check for offensive spells
        if spell_id in (585, 589):
            if self.target is None or not self.target.alive:
                return False
            if self._dist_to_mob(self.target) > spell.spell_range:
                return False
            # LOS check
            if self.terrain:
                if not self.terrain.check_los(
                    self.player.x, self.player.y, self.player.z,
                    self.target.x, self.target.y, self.target.z
                ):
                    return False

        # Shield: check if already shielded
        if spell_id == 17 and self.player.shield_remaining > 0:
            return False

        # Spend mana
        self.player.mana -= spell.mana_cost

        # GCD
        self.player.gcd_remaining = spell.gcd_ticks

        if spell.cast_ticks > 0:
            # Channeled/Cast time spell
            self.player.is_casting = True
            self.player.cast_remaining = spell.cast_ticks
            self.player.cast_spell_id = spell_id
        else:
            # Instant cast — apply immediately
            self._apply_spell(spell_id)

        return True

    def _apply_spell(self, spell_id: int):
        """Apply spell effect when cast completes."""
        spell = SPELLS[spell_id]

        if spell_id == 585:  # Smite — level-scaled damage
            if self.target and self.target.alive:
                min_dmg, max_dmg = smite_damage(self.player.level)
                dmg = self.rng.randint(min_dmg, max_dmg)
                self._damage_mob(self.target, dmg)

        elif spell_id == 2050:  # Lesser Heal — level-scaled
            min_h, max_h = heal_amount(self.player.level)
            heal = self.rng.randint(min_h, max_h)
            self.player.hp = min(self.player.max_hp, self.player.hp + heal)

        elif spell_id == 589:  # SW:Pain
            if self.target and self.target.alive:
                total_ticks = spell.dot_ticks // spell.dot_interval
                dmg_per_tick = spell.dot_damage // max(1, total_ticks)
                self.target.dot_remaining = spell.dot_ticks
                self.target.dot_timer = spell.dot_interval
                self.target.dot_damage_per_tick = dmg_per_tick

        elif spell_id == 17:  # PW:Shield
            self.player.shield_absorb = spell.shield_absorb
            self.player.shield_remaining = spell.shield_duration

    def _damage_mob(self, mob: Mob, damage: int):
        """Apply damage to a mob, handle death."""
        old_hp = mob.hp
        mob.hp = max(0, mob.hp - damage)
        self.damage_dealt += old_hp - mob.hp
        if not mob.in_combat:
            mob.in_combat = True
            mob.target_player = True
        if mob.hp <= 0:
            mob.alive = False
            mob.in_combat = False
            mob.target_player = False
            mob.respawn_timer = self.RESPAWN_TICKS
            self.kills += 1
            # XP reward — AzerothCore formula based on level difference
            xp = base_xp_gain(self.player.level, mob.level)
            self.player.xp_gained += xp
            self.player.xp += xp
            # Level-up check
            self._check_level_up()
            # Check if player leaves combat
            self._check_combat_end()

    def _check_combat_end(self):
        """Check if all mobs targeting player are dead."""
        for mob in self.mobs:
            if mob.alive and mob.target_player:
                return
        self.player.in_combat = False

    def _check_level_up(self):
        """Check if accumulated XP is enough for a level-up. May level multiple times."""
        p = self.player
        while p.level < MAX_LEVEL and p.xp >= XP_TABLE[p.level + 1]:
            p.level += 1
            p.leveled_up = True
            p.levels_gained += 1
            self._apply_level_stats()

    def _apply_level_stats(self):
        """Update player stats after level-up. Heals to full."""
        p = self.player
        p.max_hp = player_max_hp(p.level)
        p.max_mana = player_max_mana(p.level)
        # Full heal on level-up (matches WoW behaviour)
        p.hp = p.max_hp
        p.mana = p.max_mana

    # ─── Tick Processing ──────────────────────────────────────────

    def tick(self) -> None:
        """Advance simulation by one tick (0.5 seconds)."""
        self.tick_count += 1
        p = self.player
        self._update_chunks()
        self._update_exploration()

        # --- Cast completion ---
        if p.is_casting:
            p.cast_remaining -= 1
            if p.cast_remaining <= 0:
                p.is_casting = False
                self._apply_spell(p.cast_spell_id)
                p.cast_spell_id = 0

        # --- GCD ---
        if p.gcd_remaining > 0:
            p.gcd_remaining -= 1

        # --- Mob AI (inlined distance to avoid method-call overhead) ---
        px, py = p.x, p.y
        _sqrt = math.sqrt
        for mob in self.mobs:
            if not mob.alive:
                if mob.respawn_timer > 0:
                    mob.respawn_timer -= 1
                    if mob.respawn_timer <= 0:
                        self._respawn_mob(mob)
                continue

            # Inline distance (saves 2 method calls per mob)
            _dx = mob.x - px
            _dy = mob.y - py
            dist = _sqrt(_dx * _dx + _dy * _dy)

            if not mob.in_combat:
                # Aggro check
                if dist <= mob.template.detect_range:
                    mob.in_combat = True
                    mob.target_player = True
                    p.in_combat = True
                else:
                    # Far non-combat mob: only process DoT then skip
                    if mob.dot_remaining > 0:
                        mob.dot_remaining -= 1
                        mob.dot_timer -= 1
                        if mob.dot_timer <= 0:
                            self._damage_mob(mob, mob.dot_damage_per_tick)
                            mob.dot_timer = 6
                    continue

            if mob.target_player:
                # Leash check (inlined)
                sdx = mob.x - mob.spawn_x
                sdy = mob.y - mob.spawn_y
                if _sqrt(sdx * sdx + sdy * sdy) > self.MOB_LEASH_RANGE:
                    self._evade_mob(mob)
                    continue

                # Chase player
                if dist > 2.0:
                    move = min(self.MOB_SPEED, dist - 1.5)
                    new_mx = mob.x + (-_dx / dist) * move
                    new_my = mob.y + (-_dy / dist) * move
                    if self.terrain:
                        new_mz = self.terrain.get_height(new_mx, new_my)
                        if self.terrain.check_walkable(mob.x, mob.y, mob.z, new_mx, new_my, new_mz):
                            mob.x = new_mx
                            mob.y = new_my
                            mob.z = new_mz
                    else:
                        mob.x = new_mx
                        mob.y = new_my
                    # Recompute distance after move
                    _dx = mob.x - px
                    _dy = mob.y - py
                    dist = _sqrt(_dx * _dx + _dy * _dy)

                # Melee attack
                if dist <= 5.0:
                    mob.attack_timer -= 1
                    if mob.attack_timer <= 0:
                        dmg = self.rng.randint(mob.template.min_damage, mob.template.max_damage)
                        self._damage_player(dmg)
                        mob.attack_timer = mob.template.attack_speed
                        p.combat_timer = 0

            # DoT processing
            if mob.dot_remaining > 0:
                mob.dot_remaining -= 1
                mob.dot_timer -= 1
                if mob.dot_timer <= 0:
                    self._damage_mob(mob, mob.dot_damage_per_tick)
                    mob.dot_timer = 6

        # --- Shield decay ---
        if p.shield_remaining > 0:
            p.shield_remaining -= 1
            if p.shield_remaining <= 0:
                p.shield_absorb = 0

        # --- Regen ---
        if p.in_combat:
            p.combat_timer += 1
        else:
            p.combat_timer += 1

        # HP regen: only out of combat, after OOC delay
        if not p.in_combat and p.combat_timer >= self.OOC_DELAY_TICKS:
            p.ooc_regen_accumulator += self.HP_REGEN_PER_TICK
            if p.ooc_regen_accumulator >= 1.0:
                heal = int(p.ooc_regen_accumulator)
                p.hp = min(p.max_hp, p.hp + heal)
                p.ooc_regen_accumulator -= heal

        # Mana regen: when not casting (both in and out of combat)
        if not p.is_casting:
            p.mana_regen_accumulator += self.MANA_REGEN_PER_TICK
            if p.mana_regen_accumulator >= 1.0:
                regen = int(p.mana_regen_accumulator)
                p.mana = min(p.max_mana, p.mana + regen)
                p.mana_regen_accumulator -= regen

    def _damage_player(self, damage: int):
        """Apply damage to player, considering shield."""
        p = self.player
        if p.shield_absorb > 0:
            absorbed = min(p.shield_absorb, damage)
            p.shield_absorb -= absorbed
            damage -= absorbed
            if p.shield_absorb <= 0:
                p.shield_remaining = 0
        p.hp = max(0, p.hp - damage)

    def _respawn_mob(self, mob: Mob):
        """Respawn a dead mob at its spawn point."""
        level = self.rng.randint(mob.template.min_level, mob.template.max_level)
        hp_by_level = {1: 42, 2: 55, 3: 71}
        mob.hp = hp_by_level.get(level, 42)
        mob.max_hp = mob.hp
        mob.level = level
        mob.alive = True
        mob.in_combat = False
        mob.target_player = False
        mob.looted = False
        mob.x = mob.spawn_x
        mob.y = mob.spawn_y
        mob.z = mob.spawn_z
        mob.attack_timer = 0
        mob.dot_remaining = 0
        mob.dot_timer = 0
        mob.dot_damage_per_tick = 0
        mob.respawn_timer = 0

    def _evade_mob(self, mob: Mob):
        """Mob evades and returns to spawn."""
        mob.in_combat = False
        mob.target_player = False
        mob.hp = mob.max_hp
        mob.x = mob.spawn_x
        mob.y = mob.spawn_y
        mob.z = mob.spawn_z
        mob.attack_timer = 0
        mob.dot_remaining = 0
        self._check_combat_end()

    # ─── State Query ─────────────────────────────────────────────

    def get_nearby_mobs(self, scan_range: Optional[float] = None) -> list[dict]:
        """Get list of nearby mobs (alive and dead, within scan range)."""
        r = scan_range or self.SCAN_RANGE
        r_sq = r * r  # squared comparison avoids sqrt for far mobs
        px, py = self.player.x, self.player.y
        _sqrt = math.sqrt
        result = []
        for mob in self.mobs:
            dx = mob.x - px
            dy = mob.y - py
            dsq = dx * dx + dy * dy
            if dsq <= r_sq:
                d = _sqrt(dsq)
                result.append({
                    "uid": mob.uid,
                    "name": mob.template.name,
                    "level": mob.level,
                    "hp": mob.hp,
                    "max_hp": mob.max_hp,
                    "alive": mob.alive,
                    "x": mob.x,
                    "y": mob.y,
                    "z": mob.z,
                    "dist": d,
                    "target_player": mob.target_player,
                    "looted": mob.looted,
                    "attackable": 1 if mob.alive else 0,
                })
        return result

    def get_target_info(self) -> dict:
        """Get info about current target."""
        if self.target is None:
            return {"status": "none", "hp": 0, "x": 0, "y": 0, "z": 82.0, "level": 0}
        status = "alive" if self.target.alive else "dead"
        return {
            "status": status,
            "hp": self.target.hp,
            "max_hp": self.target.max_hp,
            "x": self.target.x,
            "y": self.target.y,
            "z": self.target.z,
            "level": self.target.level,
            "has_sw_pain": self.target.dot_remaining > 0,
        }

    def get_state_dict(self) -> dict:
        """Get full state dict matching WoWEnv format."""
        p = self.player
        t_info = self.get_target_info()
        nearby = self.get_nearby_mobs()
        return {
            "name": "SimBot",
            "hp": p.hp,
            "max_hp": p.max_hp,
            "power": p.mana,
            "max_power": p.max_mana,
            "level": p.level,
            "x": p.x,
            "y": p.y,
            "z": p.z,
            "o": p.orientation,
            "combat": "true" if p.in_combat else "false",
            "casting": "true" if p.is_casting else "false",
            "free_slots": p.free_slots,
            "equipped_upgrade": "true" if p.equipped_upgrade else "false",
            "target_status": t_info["status"],
            "target_hp": t_info["hp"],
            "target_level": t_info.get("level", 0),
            "xp_gained": p.xp_gained,
            "loot_copper": p.loot_copper,
            "loot_score": p.loot_score,
            "leveled_up": "true" if p.leveled_up else "false",
            "tx": t_info["x"],
            "ty": t_info["y"],
            "tz": t_info["z"],
            "has_shield": "true" if p.shield_remaining > 0 else "false",
            "target_has_sw_pain": "true" if t_info.get("has_sw_pain") else "false",
            "nearby_mobs": [
                {
                    "guid": str(m["uid"]),
                    "name": m["name"],
                    "level": m["level"],
                    "attackable": m["attackable"],
                    "vendor": 0,
                    "target": "1" if m["target_player"] else "0",
                    "hp": m["hp"],
                    "x": m["x"],
                    "y": m["y"],
                }
                for m in nearby
            ],
        }

    def consume_events(self) -> dict:
        """Consume and reset accumulated event values (like real server)."""
        p = self.player
        events = {
            "xp_gained": p.xp_gained,
            "loot_copper": p.loot_copper,
            "loot_score": p.loot_score,
            "equipped_upgrade": p.equipped_upgrade,
            "leveled_up": p.leveled_up,
            "levels_gained": p.levels_gained,
            "loot_items": list(p.loot_items),
            "loot_failed": list(p.loot_failed),
            "new_areas": self._new_areas,
            "new_zones": self._new_zones,
            "new_maps": self._new_maps,
        }
        p.xp_gained = 0
        p.loot_copper = 0
        p.loot_score = 0
        p.equipped_upgrade = False
        p.leveled_up = False
        p.levels_gained = 0
        p.loot_items.clear()
        p.loot_failed.clear()
        self._new_areas = 0
        self._new_zones = 0
        self._new_maps = 0
        return events
