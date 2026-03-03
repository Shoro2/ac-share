"""
Loads creature data from AzerothCore CSV exports (creature.csv, creature_template.csv).
Builds a spatial index for chunk-based mob spawning in the simulation.

Usage:
    db = CreatureDB("/path/to/data")
    # db.spatial_index[(map, chunk_x, chunk_y)] -> [SpawnPoint, ...]
    # db.templates[entry] -> CreatureTemplate
    # db.get_mob_stats(template, level) -> {hp, min_damage, max_damage, xp}
"""

import csv
import os
from dataclasses import dataclass

CHUNK_SIZE = 100.0  # 100x100 world-units per chunk

# ─── Base Stats by Level ─────────────────────────────────────────────
# Approximated from AzerothCore creature_classlevelstats (expansion 0, unit_class=1).
# Used as base values, then multiplied by template HealthModifier / DamageModifier.

_HP_ANCHORS_X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
                 65, 70, 75, 80, 83]
_HP_ANCHORS_Y = [42, 55, 71, 86, 104, 116, 130, 146, 160, 177,
                 279, 454, 684, 984, 1350, 1803, 2345, 2954, 3500, 4120,
                 5500, 7300, 9000, 12000, 14000]

_DMG_MIN_ANCHORS_X = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 83]
_DMG_MIN_ANCHORS_Y = [1, 2, 3, 3, 4, 9, 15, 22, 30, 38, 46, 55, 75, 95, 135, 185, 210]

_DMG_MAX_ANCHORS_X = _DMG_MIN_ANCHORS_X
_DMG_MAX_ANCHORS_Y = [2, 3, 5, 5, 7, 13, 21, 30, 42, 52, 64, 75, 100, 130, 185, 255, 290]

_XP_ANCHORS_X = [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 83]
_XP_ANCHORS_Y = [50, 90, 120, 200, 400, 600, 800, 1050, 1300, 1550,
                 1800, 2050, 2300, 2550, 2800, 3100, 3400, 3700, 4000, 4200]


def _interpolate(x: int, xs: list, ys: list) -> int:
    """Linear interpolation between anchor points."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return max(1, int(ys[i] + t * (ys[i + 1] - ys[i])))
    return ys[-1]


# Unit class multipliers (relative to class 1 = Warrior)
_CLASS_HP_MULT = {1: 1.0, 2: 0.75, 8: 0.55}
_CLASS_DMG_MULT = {1: 1.0, 2: 0.80, 8: 0.65}

# Faction 35 = friendly to all. Other Alliance-friendly factions:
FRIENDLY_FACTIONS = frozenset({
    1,    # Human (Player)
    3,    # Dwarf (Player)
    4,    # Night Elf (Player)
    11,   # Stormwind
    12,   # Stormwind related
    35,   # Friendly to all
    55,   # Ironforge
    56,   # Gnomeregan
    57,   # Gnomeregan 2
    79,   # Darnassus
    80,   # Gnomeregan Exiles
    104,  # Alliance Generic
    105,  # Alliance Theramore
    148,  # Stormwind extended
    1575, # Valiance Expedition (WotLK Alliance)
    1078, # Stormwind Champions
})

# Creature types to skip (non-combat)
SKIP_CREATURE_TYPES = frozenset({
    8,    # Critter (deer, squirrels, rabbits)
    11,   # Totem
    12,   # Non-combat Pet
    13,   # Gas Cloud
})

UNIT_FLAG_NON_ATTACKABLE = 0x00000002
UNIT_NPC_FLAG_VENDOR = 0x00000080     # 128 — NPC sells items


# ─── Data Classes ────────────────────────────────────────────────────

@dataclass(slots=True)
class CreatureTemplate:
    entry: int
    name: str
    min_level: int
    max_level: int
    faction: int
    npcflag: int
    detection_range: float
    rank: int
    base_attack_time: int    # ms
    min_gold: int
    max_gold: int
    health_modifier: float
    damage_modifier: float
    experience_modifier: float
    unit_class: int
    unit_flags: int
    creature_type: int
    lootid: int = 0       # creature_template.lootid → creature_loot_template.Entry

    @property
    def is_attackable(self) -> bool:
        if self.unit_flags & UNIT_FLAG_NON_ATTACKABLE:
            return False
        if self.faction in FRIENDLY_FACTIONS:
            return False
        if self.npcflag != 0:
            return False
        if self.creature_type in SKIP_CREATURE_TYPES:
            return False
        if self.min_level <= 0:
            return False
        # Skip decorative/event mobs with extremely low HP (turkeys, swarms, stalkers, dummies)
        if self.health_modifier < 0.5:
            return False
        return True

    @property
    def is_vendor(self) -> bool:
        """Check if this NPC is a vendor the player can sell items to."""
        if not (self.npcflag & UNIT_NPC_FLAG_VENDOR):
            return False
        if self.faction not in FRIENDLY_FACTIONS:
            return False
        if self.creature_type in SKIP_CREATURE_TYPES:
            return False
        if self.min_level <= 0:
            return False
        return True

    @property
    def attack_speed_ticks(self) -> int:
        """Convert BaseAttackTime (ms) to sim ticks (1 tick = 500ms)."""
        return max(1, self.base_attack_time // 500)


@dataclass(slots=True)
class SpawnPoint:
    guid: int
    entry: int
    map_id: int
    x: float
    y: float
    z: float
    orientation: float


# ─── CreatureDB ──────────────────────────────────────────────────────

class CreatureDB:
    """Loads creature data from CSV and builds a spatial index for chunk-based spawning."""

    def __init__(self, data_dir: str, quiet: bool = False):
        self.templates: dict[int, CreatureTemplate] = {}
        self.spatial_index: dict[tuple[int, int, int], list[SpawnPoint]] = {}
        self.vendor_index: dict[tuple[int, int, int], list[SpawnPoint]] = {}
        self._quiet = quiet

        tmpl_path = os.path.join(data_dir, 'creature_template.csv')
        spawn_path = os.path.join(data_dir, 'creature.csv')

        self._load_templates(tmpl_path)
        self._load_spawns(spawn_path)

        if not quiet:
            total_spawns = sum(len(v) for v in self.spatial_index.values())
            total_vendors = sum(len(v) for v in self.vendor_index.values())
            print(f"  [CreatureDB] {len(self.templates)} templates, "
                  f"{total_spawns} attackable spawns, "
                  f"{total_vendors} vendor spawns, "
                  f"{len(self.spatial_index)} chunks")

    def _load_templates(self, path: str):
        with open(path, 'r') as f:
            reader = csv.DictReader(f, delimiter=';', quotechar='"')
            for row in reader:
                entry = int(row['entry'])
                # lootid: use CSV column if present, otherwise default to entry
                lootid = int(row['lootid']) if 'lootid' in row else entry
                self.templates[entry] = CreatureTemplate(
                    entry=entry,
                    name=row['name'],
                    min_level=int(row['minlevel']),
                    max_level=int(row['maxlevel']),
                    faction=int(row['faction']),
                    npcflag=int(row['npcflag']),
                    detection_range=float(row['detection_range']),
                    rank=int(row['rank']),
                    base_attack_time=int(row['BaseAttackTime']),
                    min_gold=int(row['mingold']),
                    max_gold=int(row['maxgold']),
                    health_modifier=float(row['HealthModifier']),
                    damage_modifier=float(row['DamageModifier']),
                    experience_modifier=float(row['ExperienceModifier']),
                    unit_class=int(row['unit_class']),
                    unit_flags=int(row['unit_flags']),
                    creature_type=int(row['type']),
                    lootid=lootid,
                )

    def _load_spawns(self, path: str):
        with open(path, 'r') as f:
            reader = csv.DictReader(f, delimiter=';', quotechar='"')
            for row in reader:
                entry = int(row['id1'])
                tmpl = self.templates.get(entry)
                if tmpl is None:
                    continue

                map_id = int(row['map'])
                x = float(row['position_x'])
                y = float(row['position_y'])
                z = float(row['position_z'])
                orientation = float(row['orientation'])

                sp = SpawnPoint(
                    guid=int(row['guid']),
                    entry=entry,
                    map_id=map_id,
                    x=x, y=y, z=z,
                    orientation=orientation,
                )

                chunk_key = (map_id, int(x // CHUNK_SIZE), int(y // CHUNK_SIZE))

                if tmpl.is_vendor:
                    self.vendor_index.setdefault(chunk_key, []).append(sp)
                elif tmpl.is_attackable:
                    # Spawn-level overrides: skip if spawn has NPC flags or non-attackable
                    spawn_npcflag = int(row.get('npcflag', 0))
                    spawn_unit_flags = int(row.get('unit_flags', 0))
                    if spawn_npcflag != 0:
                        continue
                    if spawn_unit_flags & UNIT_FLAG_NON_ATTACKABLE:
                        continue
                    self.spatial_index.setdefault(chunk_key, []).append(sp)

    # ─── Stat Calculation ────────────────────────────────────────

    @staticmethod
    def get_base_hp(level: int, unit_class: int = 1) -> int:
        base = _interpolate(level, _HP_ANCHORS_X, _HP_ANCHORS_Y)
        mult = _CLASS_HP_MULT.get(unit_class, 1.0)
        return max(1, int(base * mult))

    @staticmethod
    def get_base_damage(level: int, unit_class: int = 1) -> tuple[int, int]:
        base_min = _interpolate(level, _DMG_MIN_ANCHORS_X, _DMG_MIN_ANCHORS_Y)
        base_max = _interpolate(level, _DMG_MAX_ANCHORS_X, _DMG_MAX_ANCHORS_Y)
        mult = _CLASS_DMG_MULT.get(unit_class, 1.0)
        return max(1, int(base_min * mult)), max(1, int(base_max * mult))

    @staticmethod
    def get_base_xp(level: int) -> int:
        return _interpolate(level, _XP_ANCHORS_X, _XP_ANCHORS_Y)

    def get_mob_stats(self, tmpl: CreatureTemplate, level: int) -> dict:
        """Compute concrete HP, damage, and XP for a mob at a specific level."""
        base_hp = self.get_base_hp(level, tmpl.unit_class)
        hp = max(1, int(base_hp * tmpl.health_modifier))

        base_min, base_max = self.get_base_damage(level, tmpl.unit_class)
        min_dmg = max(1, int(base_min * tmpl.damage_modifier))
        max_dmg = max(min_dmg, int(base_max * tmpl.damage_modifier))

        base_xp = self.get_base_xp(level)
        xp = max(1, int(base_xp * tmpl.experience_modifier))

        return {
            'hp': hp,
            'min_damage': min_dmg,
            'max_damage': max_dmg,
            'xp': xp,
        }
