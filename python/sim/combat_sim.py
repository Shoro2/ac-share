"""
WoW Combat Simulation Engine — Pure Python, no server needed.

Simulates WoW 3.3.5 WotLK characters with full stat system.
All formulas derived from AzerothCore C++ source (StatSystem.cpp,
Player.cpp, Unit.cpp, DBC game tables).

Supports all 10 classes (stat framework), leveling 1–80.
Currently only Priest has spell implementations.

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
    from sim.quest_db import QuestDB


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


# ─── WotLK 3.3.5 Class Constants (from SharedDefines.h) ──────────────

CLASS_WARRIOR = 1
CLASS_PALADIN = 2
CLASS_HUNTER = 3
CLASS_ROGUE = 4
CLASS_PRIEST = 5
CLASS_DEATH_KNIGHT = 6
CLASS_SHAMAN = 7
CLASS_MAGE = 8
CLASS_WARLOCK = 9
CLASS_DRUID = 11
MAX_CLASSES = 12

CLASS_NAMES = {
    CLASS_WARRIOR: "Warrior", CLASS_PALADIN: "Paladin", CLASS_HUNTER: "Hunter",
    CLASS_ROGUE: "Rogue", CLASS_PRIEST: "Priest", CLASS_DEATH_KNIGHT: "Death Knight",
    CLASS_SHAMAN: "Shaman", CLASS_MAGE: "Mage", CLASS_WARLOCK: "Warlock",
    CLASS_DRUID: "Druid",
}

# Power types per class
POWER_MANA = 0
POWER_RAGE = 1
POWER_ENERGY = 3
POWER_RUNIC = 6

CLASS_POWER_TYPE = {
    CLASS_WARRIOR: POWER_RAGE, CLASS_PALADIN: POWER_MANA,
    CLASS_HUNTER: POWER_MANA, CLASS_ROGUE: POWER_ENERGY,
    CLASS_PRIEST: POWER_MANA, CLASS_DEATH_KNIGHT: POWER_RUNIC,
    CLASS_SHAMAN: POWER_MANA, CLASS_MAGE: POWER_MANA,
    CLASS_WARLOCK: POWER_MANA, CLASS_DRUID: POWER_MANA,
}

# ─── WotLK 3.3.5 Item Stat Type Constants (from ItemTemplate.h) ──────

ITEM_MOD_MANA = 0
ITEM_MOD_HEALTH = 1
ITEM_MOD_AGILITY = 3
ITEM_MOD_STRENGTH = 4
ITEM_MOD_INTELLECT = 5
ITEM_MOD_SPIRIT = 6
ITEM_MOD_STAMINA = 7
ITEM_MOD_DEFENSE_SKILL_RATING = 12
ITEM_MOD_DODGE_RATING = 13
ITEM_MOD_PARRY_RATING = 14
ITEM_MOD_BLOCK_RATING = 15
ITEM_MOD_HIT_MELEE_RATING = 16
ITEM_MOD_HIT_RANGED_RATING = 17
ITEM_MOD_HIT_SPELL_RATING = 18
ITEM_MOD_CRIT_MELEE_RATING = 19
ITEM_MOD_CRIT_RANGED_RATING = 20
ITEM_MOD_CRIT_SPELL_RATING = 21
ITEM_MOD_HASTE_MELEE_RATING = 28
ITEM_MOD_HASTE_RANGED_RATING = 29
ITEM_MOD_HASTE_SPELL_RATING = 30
ITEM_MOD_HIT_RATING = 31          # all types
ITEM_MOD_CRIT_RATING = 32         # all types
ITEM_MOD_RESILIENCE_RATING = 35
ITEM_MOD_HASTE_RATING = 36        # all types
ITEM_MOD_EXPERTISE_RATING = 37
ITEM_MOD_ATTACK_POWER = 38
ITEM_MOD_RANGED_ATTACK_POWER = 39
ITEM_MOD_SPELL_HEALING_DONE = 41  # deprecated, old items
ITEM_MOD_SPELL_DAMAGE_DONE = 42   # deprecated, old items
ITEM_MOD_MANA_REGENERATION = 43   # MP5
ITEM_MOD_ARMOR_PENETRATION_RATING = 44
ITEM_MOD_SPELL_POWER = 45
ITEM_MOD_HEALTH_REGEN = 46        # HP5
ITEM_MOD_SPELL_PENETRATION = 47
ITEM_MOD_BLOCK_VALUE = 48


# ─── WotLK Base Stats per Class (from PlayerClassLevelInfo at L1) ────
# {class_id: (strength, agility, stamina, intellect, spirit)}
# Values from Human race, level 1. Other races differ by ~1-3.
CLASS_BASE_STATS = {
    CLASS_WARRIOR:      (23, 20, 22, 17, 19),
    CLASS_PALADIN:      (22, 17, 21, 20, 20),
    CLASS_HUNTER:       (16, 24, 21, 17, 20),
    CLASS_ROGUE:        (18, 24, 20, 17, 19),
    CLASS_PRIEST:       (15, 17, 20, 22, 23),
    CLASS_DEATH_KNIGHT: (24, 16, 23, 11, 18),
    CLASS_SHAMAN:       (18, 16, 21, 20, 22),
    CLASS_MAGE:         (15, 17, 18, 24, 22),
    CLASS_WARLOCK:      (15, 17, 20, 22, 22),
    CLASS_DRUID:        (17, 17, 19, 22, 22),
}

# Base HP / Mana at level 1 per class (from PlayerClassInfo)
# (base_hp, base_mana_or_power)
CLASS_BASE_HP_MANA = {
    CLASS_WARRIOR:      (60, 0),      # rage = 0 base
    CLASS_PALADIN:      (68, 80),
    CLASS_HUNTER:       (56, 85),
    CLASS_ROGUE:        (55, 100),    # energy = 100
    CLASS_PRIEST:       (72, 123),
    CLASS_DEATH_KNIGHT: (130, 100),   # runic = 100
    CLASS_SHAMAN:       (57, 78),
    CLASS_MAGE:         (52, 130),
    CLASS_WARLOCK:      (58, 110),
    CLASS_DRUID:        (56, 90),
}

# HP gain per level (simplified — real values from DBC vary slightly)
CLASS_HP_PER_LEVEL = {
    CLASS_WARRIOR: 60, CLASS_PALADIN: 55, CLASS_HUNTER: 46,
    CLASS_ROGUE: 45, CLASS_PRIEST: 50, CLASS_DEATH_KNIGHT: 60,
    CLASS_SHAMAN: 52, CLASS_MAGE: 42, CLASS_WARLOCK: 48, CLASS_DRUID: 50,
}

# Mana gain per level for mana classes (simplified)
CLASS_MANA_PER_LEVEL = {
    CLASS_PALADIN: 18, CLASS_HUNTER: 15, CLASS_PRIEST: 5,
    CLASS_SHAMAN: 17, CLASS_MAGE: 22, CLASS_WARLOCK: 20, CLASS_DRUID: 18,
}


# ─── WotLK Melee Crit from Agility (from GtChanceToMeleeCrit.dbc) ───
# {class_id: (base_crit_fraction, ratio_L1, ratio_L80)}
# base is from GtChanceToMeleeCritBase, ratio from GtChanceToMeleeCrit
# C++ formula: (base + agi * ratio) * 100.0f → percentage
GT_MELEE_CRIT = {
    CLASS_WARRIOR:      (0.0000, 0.000523, 0.000080),
    CLASS_PALADIN:      (0.0327, 0.000441, 0.000068),
    CLASS_HUNTER:       (-0.0115, 0.000302, 0.000113),
    CLASS_ROGUE:        (-0.0295, 0.000273, 0.000145),
    CLASS_PRIEST:       (0.0157, 0.000441, 0.000068),
    CLASS_DEATH_KNIGHT: (0.0000, 0.000523, 0.000080),
    CLASS_SHAMAN:       (0.0167, 0.000376, 0.000078),
    CLASS_MAGE:         (0.0335, 0.000508, 0.000078),
    CLASS_WARLOCK:      (0.0183, 0.000441, 0.000068),
    CLASS_DRUID:        (0.0188, 0.000376, 0.000116),
}

# ─── WotLK Spell Crit from Intellect (from GtChanceToSpellCrit.dbc) ─
# {class_id: (base_crit_fraction, ratio_L1, ratio_L80)}
# C++ formula: (base + int * ratio) * 100.0f → percentage
GT_SPELL_CRIT = {
    CLASS_WARRIOR:      (0.0000, 0.000000, 0.000000),
    CLASS_PALADIN:      (0.0328, 0.000342, 0.000068),
    CLASS_HUNTER:       (0.0115, 0.000441, 0.000068),
    CLASS_ROGUE:        (0.0000, 0.000000, 0.000000),
    CLASS_PRIEST:       (0.0124, 0.000660, 0.000060),
    CLASS_DEATH_KNIGHT: (0.0000, 0.000000, 0.000000),
    CLASS_SHAMAN:       (0.0229, 0.000376, 0.000078),
    CLASS_MAGE:         (0.0091, 0.000508, 0.000078),
    CLASS_WARLOCK:      (0.0170, 0.000441, 0.000068),
    CLASS_DRUID:        (0.0188, 0.000376, 0.000078),
}

# ─── WotLK Dodge from Agility (from GetDodgeFromAgility in Player.cpp) ─
# base_dodge[class] + agility * crit_ratio * crit_to_dodge_coeff
# dodge_base values from Player.cpp line 5185
DODGE_BASE = {
    CLASS_WARRIOR: 0.036640, CLASS_PALADIN: 0.034943, CLASS_HUNTER: -0.040873,
    CLASS_ROGUE: 0.020957, CLASS_PRIEST: 0.034178, CLASS_DEATH_KNIGHT: 0.036640,
    CLASS_SHAMAN: 0.021080, CLASS_MAGE: 0.036587, CLASS_WARLOCK: 0.024211,
    CLASS_DRUID: 0.056097,
}

# crit_to_dodge conversion (from Player.cpp line 5200, adjusted by 3.2.0 15% increase)
CRIT_TO_DODGE = {
    CLASS_WARRIOR: 0.85/1.15, CLASS_PALADIN: 1.00/1.15, CLASS_HUNTER: 1.11/1.15,
    CLASS_ROGUE: 2.00/1.15, CLASS_PRIEST: 1.00/1.15, CLASS_DEATH_KNIGHT: 0.85/1.15,
    CLASS_SHAMAN: 1.60/1.15, CLASS_MAGE: 1.00/1.15, CLASS_WARLOCK: 0.97/1.15,
    CLASS_DRUID: 2.00/1.15,
}

# Diminishing returns constants (from StatSystem.cpp line 711)
DR_K = {
    CLASS_WARRIOR: 0.9560, CLASS_PALADIN: 0.9560, CLASS_HUNTER: 0.9880,
    CLASS_ROGUE: 0.9880, CLASS_PRIEST: 0.9830, CLASS_DEATH_KNIGHT: 0.9560,
    CLASS_SHAMAN: 0.9880, CLASS_MAGE: 0.9830, CLASS_WARLOCK: 0.9830,
    CLASS_DRUID: 0.9720,
}

DODGE_CAP = {
    CLASS_WARRIOR: 88.129021, CLASS_PALADIN: 88.129021, CLASS_HUNTER: 145.560408,
    CLASS_ROGUE: 145.560408, CLASS_PRIEST: 150.375940, CLASS_DEATH_KNIGHT: 88.129021,
    CLASS_SHAMAN: 145.560408, CLASS_MAGE: 150.375940, CLASS_WARLOCK: 150.375940,
    CLASS_DRUID: 116.890707,
}

PARRY_CAP = {
    CLASS_WARRIOR: 47.003525, CLASS_PALADIN: 47.003525, CLASS_HUNTER: 145.560408,
    CLASS_ROGUE: 145.560408, CLASS_PRIEST: 0.0, CLASS_DEATH_KNIGHT: 47.003525,
    CLASS_SHAMAN: 145.560408, CLASS_MAGE: 0.0, CLASS_WARLOCK: 0.0,
    CLASS_DRUID: 0.0,
}


# ─── WotLK Spirit->Mana Regen (from GtRegenMPPerSpt.dbc) ────────────
# {class_id: (coeff_L1, coeff_L80)}
# C++ formula: sqrt(int) * spirit * coeff → mana per second (OOC)
GT_REGEN_MP_PER_SPT = {
    CLASS_PALADIN:  (0.0230, 0.0056),
    CLASS_HUNTER:   (0.0230, 0.0045),
    CLASS_PRIEST:   (0.0250, 0.0060),
    CLASS_SHAMAN:   (0.0240, 0.0058),
    CLASS_MAGE:     (0.0250, 0.0060),
    CLASS_WARLOCK:  (0.0240, 0.0058),
    CLASS_DRUID:    (0.0240, 0.0060),
}

# ─── WotLK Spirit->HP Regen (from GtOCTRegenHP + GtRegenHPPerSpt) ───
# base_regen + (spirit_above_50 * spirit_ratio) — simplified
# For most classes it's negligible and handled by OOC flat regen
GT_HP_REGEN_PER_SPT = {
    CLASS_WARRIOR: 0.0, CLASS_PALADIN: 0.0, CLASS_HUNTER: 0.0,
    CLASS_ROGUE: 0.0, CLASS_PRIEST: 0.008, CLASS_DEATH_KNIGHT: 0.0,
    CLASS_SHAMAN: 0.008, CLASS_MAGE: 0.008, CLASS_WARLOCK: 0.008,
    CLASS_DRUID: 0.008,
}


# ─── WotLK Combat Rating Conversions (from GtCombatRatings.dbc) ─────
# Rating needed for 1% at level 80. Scales linearly with level.
# For levels below 80: rating_for_1pct = L80_value * level / 80
CR_DEFENSE_L80 = 4.92       # defense skill rating -> 1 defense skill point
CR_DODGE_L80 = 39.35
CR_PARRY_L80 = 39.35
CR_BLOCK_L80 = 16.39
CR_HIT_MELEE_L80 = 32.79
CR_HIT_RANGED_L80 = 32.79
CR_HIT_SPELL_L80 = 26.23
CR_CRIT_MELEE_L80 = 45.91
CR_CRIT_RANGED_L80 = 45.91
CR_CRIT_SPELL_L80 = 45.91
CR_HASTE_MELEE_L80 = 32.79
CR_HASTE_RANGED_L80 = 32.79
CR_HASTE_SPELL_L80 = 32.79
CR_EXPERTISE_L80 = 8.20     # per 1 expertise (reduces dodge/parry by 0.25%)
CR_ARMOR_PENETRATION_L80 = 13.99
CR_RESILIENCE_L80 = 81.97   # for 1% reduction


# ─── WotLK Spell Power Coefficients (from spell_bonus_data DB) ──────
SP_COEFF_SMITE = 0.7143           # 2.5s / 3.5
SP_COEFF_HEAL = 0.8571            # 3.0s / 3.5
SP_COEFF_MIND_BLAST = 0.4286      # 1.5s / 3.5
SP_COEFF_SW_PAIN_TICK = 0.1833    # per tick (total ~1.1 over 6 ticks)
SP_COEFF_PW_SHIELD = 0.8068       # absorb coefficient
SP_COEFF_RENEW_TICK = 0.1         # per tick (~0.5 total over 5 ticks)
SP_COEFF_HOLY_FIRE = 0.5711       # direct
SP_COEFF_HOLY_FIRE_DOT_TICK = 0.024  # per tick


@dataclass(slots=True)
class EquippedItem:
    """An item equipped in a specific slot."""
    entry: int
    name: str
    inventory_type: int
    score: float
    stats: dict        # {ITEM_MOD_X: value}
    armor: int = 0
    weapon_dps: float = 0.0


# ─── Per-level stat scaling (class-generic WotLK formulas) ───────────

def class_base_stat(class_id: int, stat_index: int, level: int) -> int:
    """Base primary stat at given level for any class.

    stat_index: 0=str, 1=agi, 2=stam, 3=int, 4=spi
    AzerothCore scales stats at ~+1 per level (simplified).
    """
    base = CLASS_BASE_STATS.get(class_id, CLASS_BASE_STATS[CLASS_PRIEST])
    return base[stat_index] + (level - 1)


def player_max_hp(level: int, bonus_stamina: int = 0, bonus_hp: int = 0,
                  class_id: int = CLASS_PRIEST) -> int:
    """HP with WotLK Stamina formula (StatSystem.cpp:GetHealthBonusFromStamina).

    First 20 stamina = 1 HP each, above 20 = 10 HP each.
    """
    base_hp_data = CLASS_BASE_HP_MANA.get(class_id, CLASS_BASE_HP_MANA[CLASS_PRIEST])
    hp_per_lvl = CLASS_HP_PER_LEVEL.get(class_id, 50)
    base_hp = base_hp_data[0] + (level - 1) * hp_per_lvl
    total_stam = class_base_stat(class_id, 2, level) + bonus_stamina
    stam_hp = min(total_stam, 20) + max(total_stam - 20, 0) * 10
    return base_hp + stam_hp + bonus_hp


def player_max_mana(level: int, bonus_intellect: int = 0, bonus_mana: int = 0,
                    class_id: int = CLASS_PRIEST) -> int:
    """Mana with WotLK Intellect formula (StatSystem.cpp:GetManaBonusFromIntellect).

    First 20 int = 1 mana each, above 20 = 15 mana each.
    Returns 0 for non-mana classes (Warrior, Rogue, DK).
    """
    if CLASS_POWER_TYPE.get(class_id, POWER_MANA) != POWER_MANA:
        return 0
    base_mana_data = CLASS_BASE_HP_MANA.get(class_id, CLASS_BASE_HP_MANA[CLASS_PRIEST])
    mana_per_lvl = CLASS_MANA_PER_LEVEL.get(class_id, 5)
    base_mana = base_mana_data[1] + (level - 1) * mana_per_lvl
    total_int = class_base_stat(class_id, 3, level) + bonus_intellect
    int_mana = min(total_int, 20) + max(total_int - 20, 0) * 15
    return base_mana + int_mana + bonus_mana


# ─── GtCombatRatings level scaling (from GtCombatRatings.dbc) ─────────
# All combat ratings share the same relative growth curve per level.
# This table stores the fraction of L80 rating_per_1pct at key levels.
# Derived from: crit L80=45.91, crit L70=22.08, crit L60≈14.0
# Linear interpolation between key points.
_RATING_LEVEL_SCALE = (
    (1, 0.0125), (10, 0.0125),   # flat L1-10 (items don't have ratings here)
    (20, 0.05), (30, 0.10),       # slow growth
    (40, 0.17), (50, 0.25),       # moderate growth
    (60, 0.305), (70, 0.481),     # steeper (L60/L70 from known WotLK data)
    (80, 1.00),                    # reference level
)


def _rating_to_pct(rating: int, l80_value: float, level: int) -> float:
    """Convert combat rating to percentage bonus (WotLK GtCombatRatings scaling).

    Uses the actual non-linear per-level scaling from GtCombatRatings.dbc:
    - Levels 1-10: flat, very low requirement (ratings very effective)
    - Levels 10-60: grows roughly as power curve
    - Levels 60-80: steep growth (ratings much less effective)

    This matches the WoW behavior where 100 crit rating at L1 gives far
    more crit% than the same 100 crit rating at L80.
    """
    if rating <= 0:
        return 0.0
    # Interpolate level scaling factor from lookup table
    scale = _RATING_LEVEL_SCALE
    if level <= scale[0][0]:
        factor = scale[0][1]
    elif level >= scale[-1][0]:
        factor = scale[-1][1]
    else:
        factor = scale[-1][1]  # fallback
        for i in range(len(scale) - 1):
            if scale[i][0] <= level <= scale[i + 1][0]:
                t = (level - scale[i][0]) / (scale[i + 1][0] - scale[i][0])
                factor = scale[i][1] + t * (scale[i + 1][1] - scale[i][1])
                break
    rating_per_pct = max(0.01, l80_value * factor)
    return rating / rating_per_pct


def melee_crit_chance(level: int, total_agility: int, crit_rating: int = 0,
                      class_id: int = CLASS_PRIEST) -> float:
    """Melee crit % from Agility + Crit Rating (GtChanceToMeleeCrit.dbc).

    Formula: (base + agi * ratio) * 100 + rating_bonus
    """
    gt = GT_MELEE_CRIT.get(class_id, GT_MELEE_CRIT[CLASS_PRIEST])
    t = min((level - 1) / 79.0, 1.0)
    agi_ratio = gt[1] * (1 - t) + gt[2] * t
    crit = (gt[0] + total_agility * agi_ratio) * 100.0
    crit += _rating_to_pct(crit_rating, CR_CRIT_MELEE_L80, level)
    return max(crit, 0.0)


def ranged_crit_chance(level: int, total_agility: int, crit_rating: int = 0,
                       class_id: int = CLASS_PRIEST) -> float:
    """Ranged crit % — same Agility formula as melee, different rating."""
    gt = GT_MELEE_CRIT.get(class_id, GT_MELEE_CRIT[CLASS_PRIEST])
    t = min((level - 1) / 79.0, 1.0)
    agi_ratio = gt[1] * (1 - t) + gt[2] * t
    crit = (gt[0] + total_agility * agi_ratio) * 100.0
    crit += _rating_to_pct(crit_rating, CR_CRIT_RANGED_L80, level)
    return max(crit, 0.0)


def spell_crit_chance(level: int, bonus_intellect: int = 0, bonus_crit_rating: int = 0,
                      class_id: int = CLASS_PRIEST) -> float:
    """Spell crit % from Intellect + Crit Rating (GtChanceToSpellCrit.dbc)."""
    gt = GT_SPELL_CRIT.get(class_id, GT_SPELL_CRIT[CLASS_PRIEST])
    total_int = class_base_stat(class_id, 3, level) + bonus_intellect
    t = min((level - 1) / 79.0, 1.0)
    int_ratio = gt[1] * (1 - t) + gt[2] * t
    crit = (gt[0] + total_int * int_ratio) * 100.0
    crit += _rating_to_pct(bonus_crit_rating, CR_CRIT_SPELL_L80, level)
    return max(crit, 0.0)


def melee_haste_pct(level: int, haste_rating: int = 0) -> float:
    """Melee haste % from Haste Rating."""
    return _rating_to_pct(haste_rating, CR_HASTE_MELEE_L80, level)


def ranged_haste_pct(level: int, haste_rating: int = 0) -> float:
    """Ranged haste % from Haste Rating."""
    return _rating_to_pct(haste_rating, CR_HASTE_RANGED_L80, level)


def spell_haste_pct(level: int, bonus_haste_rating: int = 0) -> float:
    """Spell haste % from Haste Rating."""
    return _rating_to_pct(bonus_haste_rating, CR_HASTE_SPELL_L80, level)


def dodge_chance(level: int, total_agility: int, dodge_rating: int = 0,
                 defense_rating: int = 0, class_id: int = CLASS_PRIEST) -> float:
    """Dodge % from Agility + Dodge Rating + Defense (WotLK diminishing returns).

    From Player.cpp:GetDodgeFromAgility + StatSystem.cpp:UpdateDodgePercentage.
    """
    gt = GT_MELEE_CRIT.get(class_id, GT_MELEE_CRIT[CLASS_PRIEST])
    t = min((level - 1) / 79.0, 1.0)
    agi_ratio = gt[1] * (1 - t) + gt[2] * t
    c2d = CRIT_TO_DODGE.get(class_id, CRIT_TO_DODGE[CLASS_PRIEST])
    db = DODGE_BASE.get(class_id, DODGE_BASE[CLASS_PRIEST])

    base_agi = class_base_stat(class_id, 1, level)
    bonus_agi = max(0, total_agility - base_agi)

    nondiminishing = 100.0 * (db + base_agi * agi_ratio * c2d)
    diminishing = 100.0 * bonus_agi * agi_ratio * c2d
    diminishing += _rating_to_pct(dodge_rating, CR_DODGE_L80, level)
    diminishing += _rating_to_pct(defense_rating, CR_DEFENSE_L80, level) * 0.04

    cap = DODGE_CAP.get(class_id, 150.0)
    k = DR_K.get(class_id, 0.983)
    if cap > 0 and diminishing > 0:
        dr_dodge = diminishing * cap / (diminishing + cap * k)
    else:
        dr_dodge = 0.0
    return max(nondiminishing + dr_dodge, 0.0)


def parry_chance(level: int, parry_rating: int = 0, defense_rating: int = 0,
                 class_id: int = CLASS_PRIEST) -> float:
    """Parry % from Parry Rating + Defense (WotLK diminishing returns)."""
    cap = PARRY_CAP.get(class_id, 0.0)
    if cap <= 0:
        return 0.0
    nondiminishing = 5.0  # base 5% for classes that can parry
    diminishing = _rating_to_pct(parry_rating, CR_PARRY_L80, level)
    diminishing += _rating_to_pct(defense_rating, CR_DEFENSE_L80, level) * 0.04
    k = DR_K.get(class_id, 0.983)
    if diminishing > 0:
        dr_parry = diminishing * cap / (diminishing + cap * k)
    else:
        dr_parry = 0.0
    return max(nondiminishing + dr_parry, 0.0)


def block_chance(level: int, block_rating: int = 0, defense_rating: int = 0) -> float:
    """Block % from Block Rating + Defense. 5% base for shield users."""
    base = 5.0
    base += _rating_to_pct(block_rating, CR_BLOCK_L80, level)
    base += _rating_to_pct(defense_rating, CR_DEFENSE_L80, level) * 0.04
    return max(base, 0.0)


def melee_attack_power(level: int, total_str: int, total_agi: int,
                       class_id: int = CLASS_PRIEST) -> int:
    """Melee Attack Power from stats (StatSystem.cpp:UpdateAttackPowerAndDamage).

    Warrior/Paladin/DK: level*3 + str*2 - 20
    Hunter/Shaman/Rogue: level*2 + str + agi - 20
    Mage/Priest/Warlock: str - 10
    Druid (caster): str*2 - 20
    """
    if class_id in (CLASS_WARRIOR, CLASS_PALADIN, CLASS_DEATH_KNIGHT):
        return int(level * 3.0 + total_str * 2.0 - 20.0)
    elif class_id in (CLASS_HUNTER, CLASS_SHAMAN, CLASS_ROGUE):
        return int(level * 2.0 + total_str + total_agi - 20.0)
    elif class_id in (CLASS_MAGE, CLASS_PRIEST, CLASS_WARLOCK):
        return int(total_str - 10.0)
    elif class_id == CLASS_DRUID:
        return int(total_str * 2.0 - 20.0)
    return int(total_str - 10.0)


def ranged_attack_power(level: int, total_str: int, total_agi: int,
                        class_id: int = CLASS_PRIEST) -> int:
    """Ranged Attack Power from stats.

    Hunter: level*2 + agi - 10
    Warrior/Rogue: level + agi - 10
    Others: agi - 10
    """
    if class_id == CLASS_HUNTER:
        return int(level * 2.0 + total_agi - 10.0)
    elif class_id in (CLASS_WARRIOR, CLASS_ROGUE):
        return int(level + total_agi - 10.0)
    return int(total_agi - 10.0)


def expertise_pct(level: int, expertise_rating: int = 0) -> float:
    """Expertise reduces dodge/parry by 0.25% per point.

    Rating -> expertise points via CR_EXPERTISE_L80.
    """
    points = _rating_to_pct(expertise_rating, CR_EXPERTISE_L80, level)
    return points * 0.25  # each expertise point = 0.25% dodge/parry reduction


def armor_penetration_pct(level: int, arp_rating: int = 0) -> float:
    """Armor penetration % from ArP rating."""
    return min(_rating_to_pct(arp_rating, CR_ARMOR_PENETRATION_L80, level), 100.0)


def resilience_pct(level: int, resilience_rating: int = 0) -> float:
    """Resilience reduces crit chance/damage from players."""
    return _rating_to_pct(resilience_rating, CR_RESILIENCE_L80, level)


def hit_chance_melee(level: int, hit_rating: int = 0) -> float:
    """Melee hit % bonus from Hit Rating."""
    return _rating_to_pct(hit_rating, CR_HIT_MELEE_L80, level)


def hit_chance_ranged(level: int, hit_rating: int = 0) -> float:
    """Ranged hit % bonus from Hit Rating."""
    return _rating_to_pct(hit_rating, CR_HIT_RANGED_L80, level)


def hit_chance_spell(level: int, hit_rating: int = 0) -> float:
    """Spell hit % bonus from Hit Rating."""
    return _rating_to_pct(hit_rating, CR_HIT_SPELL_L80, level)


def spirit_mana_regen(level: int, bonus_intellect: int = 0, bonus_spirit: int = 0,
                      class_id: int = CLASS_PRIEST) -> float:
    """Spirit-based mana regen per tick (0.5s) while NOT casting.

    Formula from Player.cpp:OCTRegenMPPerSpirit + UpdateManaRegen:
      power_regen = sqrt(int) * spirit * coeff
    Returns per-tick value (0.5s).
    """
    gt = GT_REGEN_MP_PER_SPT.get(class_id)
    if gt is None:
        return 0.0  # non-mana classes
    total_int = class_base_stat(class_id, 3, level) + bonus_intellect
    total_spirit = class_base_stat(class_id, 4, level) + bonus_spirit
    t = min((level - 1) / 79.0, 1.0)
    coeff = gt[0] * (1 - t) + gt[1] * t
    regen_per_sec = math.sqrt(max(total_int, 1)) * total_spirit * coeff
    return regen_per_sec * 0.5  # per tick


def spirit_hp_regen(level: int, total_spirit: int,
                    class_id: int = CLASS_PRIEST) -> float:
    """Spirit-based HP regen per tick (0.5s) while OOC.

    Very small contribution — most HP regen is flat OOC regen.
    """
    coeff = GT_HP_REGEN_PER_SPT.get(class_id, 0.0)
    if coeff <= 0:
        return 0.0
    extra_spirit = max(0, total_spirit - 50)
    return extra_spirit * coeff * 0.5


def smite_damage(level: int, spell_power: int = 0) -> tuple[int, int]:
    """Smite damage range: base (13-17) + 10 per level + SP*coeff."""
    bonus = (level - 1) * 10
    sp_bonus = int(spell_power * SP_COEFF_SMITE)
    return (13 + bonus + sp_bonus, 17 + bonus + sp_bonus)


def heal_amount(level: int, spell_power: int = 0) -> tuple[int, int]:
    """Lesser Heal range: base (46-56) + 5 per level + SP*coeff."""
    bonus = (level - 1) * 5
    sp_bonus = int(spell_power * SP_COEFF_HEAL)
    return (46 + bonus + sp_bonus, 56 + bonus + sp_bonus)


def mind_blast_damage(level: int, spell_power: int = 0) -> tuple[int, int]:
    """Mind Blast damage range: base (39-43) + 12 per level + SP*coeff."""
    bonus = (level - 1) * 12
    sp_bonus = int(spell_power * SP_COEFF_MIND_BLAST)
    return (39 + bonus + sp_bonus, 43 + bonus + sp_bonus)


def renew_total_heal(level: int, spell_power: int = 0) -> int:
    """Renew total HoT: base 45 + 8 per level + SP*coeff (total over 5 ticks)."""
    return 45 + (level - 1) * 8 + int(spell_power * SP_COEFF_RENEW_TICK * 5)


def holy_fire_damage(level: int, spell_power: int = 0) -> tuple[int, int]:
    """Holy Fire direct damage: base (15-20) + 10 per level + SP*coeff."""
    bonus = (level - 1) * 10
    sp_bonus = int(spell_power * SP_COEFF_HOLY_FIRE)
    return (15 + bonus + sp_bonus, 20 + bonus + sp_bonus)


def holy_fire_dot_total(level: int, spell_power: int = 0) -> int:
    """Holy Fire DoT total: base 12 + 5 per level + SP*coeff (2 ticks)."""
    return 12 + (level - 1) * 5 + int(spell_power * SP_COEFF_HOLY_FIRE_DOT_TICK * 2)


def sw_pain_total(level: int, spell_power: int = 0) -> int:
    """SW:Pain total DoT damage: base 30 + SP*coeff (6 ticks)."""
    return 30 + int(spell_power * SP_COEFF_SW_PAIN_TICK * 6)


def pw_shield_absorb(level: int, spell_power: int = 0) -> int:
    """PW:Shield absorb amount: base 44 + SP*coeff."""
    return 44 + int(spell_power * SP_COEFF_PW_SHIELD)


def inner_fire_values(level: int) -> tuple[int, int]:
    """Inner Fire: (armor, spell_power_bonus). Scales with level."""
    return (10 + (level - 1) * 3, 2 + (level - 1) * 1)


def fortitude_hp_bonus(level: int) -> int:
    """PW:Fortitude HP bonus: base 20 + 8 per level gained."""
    return 20 + (level - 1) * 8


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
    cooldown_ticks: int = 0   # spell-specific cooldown (ticks), 0 = none
    is_hot: bool = False
    hot_heal: int = 0      # total HoT healing
    hot_ticks: int = 0     # total duration in ticks
    hot_interval: int = 6  # ticks between hot ticks (3s = 6 ticks)
    is_buff: bool = False
    buff_duration: int = 0  # ticks


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
    8092: SpellDef(
        id=8092, name="Mind Blast",
        cast_ticks=3, mana_cost=50,  # 1.5s cast
        min_damage=39, max_damage=43,
        spell_range=30.0,
        cooldown_ticks=16,  # 8s = 16 ticks
    ),
    139: SpellDef(
        id=139, name="Renew",
        cast_ticks=0, mana_cost=30,  # instant
        is_hot=True,
        hot_heal=45,       # total heal over 15s
        hot_ticks=30,      # 15s = 30 ticks
        hot_interval=6,    # tick every 3s = 5 HoT ticks
    ),
    14914: SpellDef(
        id=14914, name="Holy Fire",
        cast_ticks=4, mana_cost=40,  # 2.0s cast
        min_damage=15, max_damage=20,
        spell_range=30.0,
        is_dot=True,
        dot_damage=12,     # total 12 over 6s
        dot_ticks=12,      # 6s = 12 ticks
        dot_interval=6,    # tick every 3s = 2 DoT ticks
        cooldown_ticks=20,  # 10s = 20 ticks
    ),
    588: SpellDef(
        id=588, name="Inner Fire",
        cast_ticks=0, mana_cost=20,  # instant
        is_buff=True,
        buff_duration=400,  # 200s ≈ 3.3 min = 400 ticks
    ),
    1243: SpellDef(
        id=1243, name="Power Word: Fortitude",
        cast_ticks=0, mana_cost=30,  # instant
        is_buff=True,
        buff_duration=1200,  # 600s = 10 min = 1200 ticks
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
    299: [  # Diseased Young Wolf — 30 spawns
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
    6: [  # Kobold Vermin — 20 spawns
        (-8783.0, -161.6), (-8774.1, -184.5), (-8794.5, -170.4),
        (-8795.0, -134.2), (-8789.9, -143.3), (-8768.5, -176.4),
        (-8753.0, -160.8), (-8779.8, -195.4), (-8775.9, -148.5),
        (-8785.5, -171.2), (-8765.3, -93.4), (-8771.5, -115.9),
        (-8794.0, -118.5), (-8778.8, -125.7), (-8781.3, -115.6),
        (-8767.0, -117.4), (-8761.0, -127.5), (-8780.0, -108.4),
        (-8772.9, -103.6), (-8749.1, -115.0),
    ],
    69: [  # Diseased Timber Wolf — 16 spawns
        (-8872.6, -58.0), (-8851.4, -84.1), (-8813.4, -179.6),
        (-8804.5, -136.6), (-8747.6, -135.8), (-8781.3, -59.8),
        (-8752.6, -82.0), (-8761.9, -66.3), (-8789.8, -69.8),
        (-8718.8, -148.4), (-8736.3, -72.8), (-8753.2, -33.4),
        (-8748.7, -50.3), (-8737.1, -94.3), (-8766.0, -232.2),
        (-8805.4, -205.5),
    ],
    257: [  # Kobold Worker — 18 spawns
        (-8763.3, -159.3), (-8756.8, -171.4), (-8769.9, -138.1),
        (-8768.6, -113.1), (-8786.9, -105.9), (-8770.1, -117.7),
        (-8767.7, -111.6), (-8752.4, -101.9), (-8721.7, -155.8),
        (-8717.5, -144.6), (-8742.2, -176.2), (-8727.4, -133.8),
        (-8706.3, -129.8), (-8701.4, -118.1), (-8718.7, -98.3),
        (-8713.4, -90.3), (-8707.6, -108.4), (-8725.9, -109.7),
    ],
}


# ─── Inventory Item ──────────────────────────────────────────────────

@dataclass(slots=True)
class InventoryItem:
    """An item stored in the player's inventory."""
    entry: int
    name: str
    quality: int          # 0=Poor(grey), 1=Common, 2=Uncommon, 3=Rare, 4=Epic
    sell_price: int        # copper
    score: float
    inventory_type: int    # 0=non-equip, >0=equipment slot
    stats: dict = None     # {ITEM_MOD_X: value} — None for fallback items
    armor: int = 0
    weapon_dps: float = 0.0


# ─── Vendor NPCs (from AzerothCore DB, Northshire Valley) ────────────

@dataclass(slots=True)
class VendorNPC:
    """A vendor NPC in the simulation world."""
    uid: int
    name: str
    level: int
    x: float
    y: float
    z: float


@dataclass(slots=True)
class QuestNPC:
    """A quest-giver NPC in the simulation world."""
    uid: int
    entry: int
    name: str
    x: float
    y: float
    z: float = 82.0


# Real vendor positions from Northshire Abbey (AzerothCore npc_memory)
VENDOR_DATA = [
    {"name": "Janos Hammerknuckle", "level": 5, "x": -8909.46, "y": -104.163, "z": 82.031},
    {"name": "Godric Rothgar",      "level": 5, "x": -8898.23, "y": -119.838, "z": 82.016},
    {"name": "Dermot Johns",        "level": 5, "x": -8897.71, "y": -115.328, "z": 81.998},
    {"name": "Brother Danil",       "level": 5, "x": -8901.59, "y": -112.716, "z": 82.031},
]


# ─── Player State ─────────────────────────────────────────────────────

INVENTORY_SLOTS = 30  # default bag capacity (no bag logic yet)


@dataclass
class Player:
    class_id: int = CLASS_PRIEST
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
    shield_cooldown: int = 0    # Weakened Soul debuff (ticks)
    # Spell cooldowns: {spell_id: ticks_remaining}
    spell_cooldowns: dict = field(default_factory=dict)
    # HoT (Renew) state
    hot_remaining: int = 0       # ticks until HoT expires
    hot_timer: int = 0           # ticks until next HoT tick
    hot_heal_per_tick: int = 0
    # Buff: Inner Fire
    inner_fire_remaining: int = 0  # ticks
    inner_fire_armor: int = 0
    inner_fire_spellpower: int = 0
    # Buff: PW:Fortitude
    fortitude_remaining: int = 0   # ticks
    fortitude_hp_bonus: int = 0
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
    # Equipment system: slot -> EquippedItem (WotLK slots)
    equipment: dict = field(default_factory=dict)  # inventory_type -> EquippedItem
    equipped_scores: dict = field(default_factory=dict)  # inventory_type -> best score (compat)
    # ─── Gear stats (accumulated from equipped items) ────────────────
    gear_strength: int = 0
    gear_agility: int = 0
    gear_stamina: int = 0
    gear_intellect: int = 0
    gear_spirit: int = 0
    gear_armor: int = 0            # total armor from gear pieces
    gear_bonus_hp: int = 0         # flat HP from ITEM_MOD_HEALTH
    gear_bonus_mana: int = 0       # flat Mana from ITEM_MOD_MANA
    # Offensive ratings from gear
    gear_attack_power: int = 0     # ITEM_MOD_ATTACK_POWER
    gear_ranged_ap: int = 0        # ITEM_MOD_RANGED_ATTACK_POWER
    gear_spell_power: int = 0      # ITEM_MOD_SPELL_POWER + deprecated SP/Heal mods
    gear_hit_rating: int = 0       # combined hit rating (melee+ranged+spell)
    gear_crit_rating: int = 0      # combined crit rating
    gear_haste_rating: int = 0     # combined haste rating
    gear_expertise_rating: int = 0
    gear_armor_pen_rating: int = 0
    gear_spell_pen: int = 0        # flat spell penetration
    # Defensive ratings from gear
    gear_defense_rating: int = 0
    gear_dodge_rating: int = 0
    gear_parry_rating: int = 0
    gear_block_rating: int = 0
    gear_block_value: int = 0      # ITEM_MOD_BLOCK_VALUE
    gear_resilience_rating: int = 0
    # Regen from gear
    gear_mp5: int = 0              # ITEM_MOD_MANA_REGENERATION (flat MP5)
    gear_hp5: int = 0              # ITEM_MOD_HEALTH_REGEN (flat HP5)
    # ─── Derived combat stats (cached, recalculated via recalculate_stats) ─
    # Primary stat totals (base + gear)
    total_strength: int = 0
    total_agility: int = 0
    total_stamina: int = 0
    total_intellect: int = 0
    total_spirit: int = 0
    # Offensive
    total_attack_power: int = 0       # melee AP (from str/agi/level/gear)
    total_ranged_ap: int = 0          # ranged AP
    total_spell_power: int = 0        # gear SP + buffs
    total_melee_crit: float = 0.0     # melee crit % (from Agi + rating)
    total_ranged_crit: float = 0.0    # ranged crit %
    total_spell_crit: float = 0.0     # spell crit % (from Int + rating)
    total_melee_haste: float = 0.0    # melee haste %
    total_ranged_haste: float = 0.0   # ranged haste %
    total_spell_haste: float = 0.0    # spell haste %
    total_hit_melee: float = 0.0      # melee hit %
    total_hit_ranged: float = 0.0     # ranged hit %
    total_hit_spell: float = 0.0      # spell hit %
    total_expertise: float = 0.0      # expertise dodge/parry reduction %
    total_armor_pen: float = 0.0      # armor penetration %
    # Defensive
    total_armor: int = 0              # gear armor + agi*2 + buffs
    total_dodge: float = 0.0          # dodge % (with DR)
    total_parry: float = 0.0          # parry % (with DR)
    total_block: float = 0.0          # block % (shield only)
    total_block_value: int = 0        # block amount (str/2 + gear)
    total_defense: float = 0.0        # bonus defense from rating
    total_resilience: float = 0.0     # resilience %
    # Inventory: actual items stored (for sell copper calculation)
    inventory: list = field(default_factory=list)  # list of InventoryItem
    copper: int = 0                                 # total copper balance
    sell_copper: int = 0                            # copper earned from selling this tick (consume-on-read)
    items_sold: int = 0                             # number of items sold this tick (consume-on-read)
    # Regen tracking
    combat_timer: int = 0       # ticks since last combat action (for OOC regen)
    ooc_regen_accumulator: float = 0.0
    mana_regen_accumulator: float = 0.0
    # Quest tracking (consume-on-read)
    quest_xp_gained: int = 0            # XP from quest turn-ins this tick
    quest_copper_gained: int = 0        # copper from quest turn-ins this tick
    quests_completed_tick: int = 0      # quests completed this tick (consume-on-read)


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
    # DoT tracking (slot 1: SW:Pain)
    dot_remaining: int = 0
    dot_timer: int = 0          # ticks until next dot tick
    dot_damage_per_tick: int = 0
    # DoT tracking (slot 2: Holy Fire)
    dot2_remaining: int = 0
    dot2_timer: int = 0
    dot2_damage_per_tick: int = 0
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
    SELL_RANGE = 6.0          # max vendor interaction range
    MOB_LEASH_RANGE = 60.0    # mob returns home after this distance from spawn
    OOC_DELAY_TICKS = 12      # 6 seconds out of combat before regen starts
    HP_REGEN_PER_TICK = 0.67  # 8 HP per 6 seconds = ~0.67/tick (OOC only)
    MANA_REGEN_PCT_PER_TICK = 0.02  # 2% of max_mana per tick (not casting)
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

    QUEST_NPC_RANGE = 6.0     # max interaction range for quest NPCs

    def __init__(self, num_mobs: int = None, seed: Optional[int] = None,
                 terrain: 'SimTerrain | None' = None, env3d=None,
                 creature_db: 'CreatureDB | None' = None,
                 loot_db: 'LootDB | None' = None,
                 quest_db: 'QuestDB | None' = None,
                 class_id: int = CLASS_PRIEST):
        self.rng = random.Random(seed)
        self.num_mobs = num_mobs  # None = all spawns
        self.terrain = terrain
        self.env3d = env3d        # WoW3DEnvironment for area/zone lookups
        self.creature_db = creature_db
        self.loot_db = loot_db    # LootDB for item drops from CSV loot tables
        self.quest_db = quest_db  # QuestDB for quest definitions and NPCs
        self.class_id = class_id  # player class
        self.map_id = 0           # Eastern Kingdoms
        self.player = Player(class_id=class_id)
        if self.terrain:
            self.player.z = self.terrain.get_height(self.player.x, self.player.y)
        self.mobs: list[Mob] = []
        self.vendors: list[VendorNPC] = []
        self.quest_npcs: list[QuestNPC] = []
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
        self._chunk_vendors: dict[tuple, list[VendorNPC]] = {}
        # Quest state
        self.active_quests: dict = {}     # quest_id -> QuestProgress
        self.completed_quests: set = set()
        self.quests_completed: int = 0    # total quests completed this episode
        self._spawn_vendors()
        self._spawn_quest_npcs()
        if self.creature_db:
            self._update_chunks()
        else:
            self._spawn_mobs()
        self._update_exploration()  # register spawn position
        self.recalculate_stats()    # apply WotLK stat formulas to initial player

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

    def _spawn_vendors(self):
        """Spawn vendor NPCs. Uses creature_db chunks when available, else VENDOR_DATA fallback."""
        self.vendors.clear()
        if self.creature_db:
            return  # vendors loaded via _activate_chunk
        for vdata in VENDOR_DATA:
            z = vdata["z"]
            if self.terrain:
                z = self.terrain.get_height(vdata["x"], vdata["y"])
            self.vendors.append(VendorNPC(
                uid=self._new_uid(),
                name=vdata["name"],
                level=vdata["level"],
                x=vdata["x"], y=vdata["y"], z=z,
            ))

    def _spawn_quest_npcs(self):
        """Spawn quest-giver NPCs from quest_db data."""
        self.quest_npcs.clear()
        if not self.quest_db:
            return
        for npc_data in self.quest_db.npc_data:
            z = npc_data.z
            if self.terrain:
                z = self.terrain.get_height(npc_data.x, npc_data.y)
            self.quest_npcs.append(QuestNPC(
                uid=self._new_uid(),
                entry=npc_data.entry,
                name=npc_data.name,
                x=npc_data.x, y=npc_data.y, z=z,
            ))

    def get_nearest_quest_npc(self, npc_filter=None) -> Optional[QuestNPC]:
        """Return the nearest quest NPC, optionally filtered by entry.

        Args:
            npc_filter: If set, only consider NPCs with this entry.
        """
        best = None
        best_dist = float('inf')
        px, py = self.player.x, self.player.y
        for npc in self.quest_npcs:
            if npc_filter is not None and npc.entry != npc_filter:
                continue
            dx = npc.x - px
            dy = npc.y - py
            d = math.sqrt(dx * dx + dy * dy)
            if d < best_dist:
                best_dist = d
                best = npc
        return best

    def do_quest_interact(self) -> bool:
        """Interact with the nearest quest NPC within range.

        Handles both accepting new quests and turning in completed quests.
        Priority: turn-in first (rewards), then accept new quests.
        Returns True if any interaction occurred.
        """
        if not self.quest_db:
            return False

        p = self.player
        px, py = p.x, p.y
        interacted = False

        for npc in self.quest_npcs:
            dx = npc.x - px
            dy = npc.y - py
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > self.QUEST_NPC_RANGE:
                continue

            # Turn-in completed quests first
            completable = self.quest_db.get_completable_quests(
                npc.entry, self.active_quests)
            for qt in completable:
                self._complete_quest(qt)
                interacted = True

            # Accept available quests
            available = self.quest_db.get_available_quests(
                npc.entry, p.level, self.completed_quests, self.active_quests)
            for qt in available:
                progress = self.quest_db.create_progress(qt.quest_id)
                self.active_quests[qt.quest_id] = progress
                interacted = True

        return interacted

    def _complete_quest(self, qt):
        """Complete a quest: grant rewards, update state."""
        from sim.quest_db import QuestObjectiveType
        p = self.player

        # Grant rewards
        if qt.rewards.xp > 0:
            p.xp_gained += qt.rewards.xp
            p.quest_xp_gained += qt.rewards.xp
            p.xp += qt.rewards.xp
            self._check_level_up()
        if qt.rewards.copper > 0:
            p.copper += qt.rewards.copper
            p.quest_copper_gained += qt.rewards.copper

        # Remove quest items from inventory for COLLECT objectives
        for obj in qt.objectives:
            if obj.obj_type == QuestObjectiveType.COLLECT:
                items_to_remove = obj.count
                self.player.inventory = [
                    item for item in self.player.inventory
                    if item.entry != obj.target or (items_to_remove := items_to_remove - 1) < 0
                ]
                # Recalculate free slots
                p.free_slots = INVENTORY_SLOTS - len(p.inventory)

        # Move to completed
        del self.active_quests[qt.quest_id]
        self.completed_quests.add(qt.quest_id)
        self.quests_completed += 1
        p.quests_completed_tick += 1

    def on_mob_killed(self, mob: Mob):
        """Update quest progress when a mob is killed."""
        if not self.quest_db:
            return
        from sim.quest_db import QuestObjectiveType
        for qid, progress in self.active_quests.items():
            qt = self.quest_db.templates[qid]
            for i, obj in enumerate(qt.objectives):
                if obj.obj_type == QuestObjectiveType.KILL and obj.target == mob.template.entry:
                    if progress.counts[i] < obj.count:
                        progress.counts[i] += 1
            progress.check_complete(qt.objectives)

    def on_mob_looted(self, mob: Mob):
        """Roll quest item drops when a mob is looted."""
        if not self.quest_db:
            return
        from sim.quest_db import QuestObjectiveType
        p = self.player
        for qid, progress in self.active_quests.items():
            qt = self.quest_db.templates[qid]
            for i, obj in enumerate(qt.objectives):
                if (obj.obj_type == QuestObjectiveType.COLLECT
                        and obj.source_creature == mob.template.entry
                        and progress.counts[i] < obj.count):
                    if self.rng.random() < obj.drop_chance:
                        progress.counts[i] += 1
                        # Add quest item to inventory (takes a slot)
                        if p.free_slots > 0:
                            p.free_slots -= 1
                            p.inventory.append(InventoryItem(
                                entry=obj.target,
                                name=f"Quest Item #{obj.target}",
                                quality=1,
                                sell_price=0,
                                score=0.0,
                                inventory_type=0,
                            ))
            progress.check_complete(qt.objectives)

    def _update_quest_exploration(self):
        """Check explore quest objectives against current position."""
        if not self.quest_db:
            return
        from sim.quest_db import QuestObjectiveType
        px, py = self.player.x, self.player.y
        for qid, progress in self.active_quests.items():
            qt = self.quest_db.templates[qid]
            for i, obj in enumerate(qt.objectives):
                if (obj.obj_type == QuestObjectiveType.EXPLORE
                        and progress.counts[i] < obj.count):
                    dx = obj.target_x - px
                    dy = obj.target_y - py
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist <= obj.radius:
                        progress.counts[i] = obj.count
            progress.check_complete(qt.objectives)

    def get_nearest_vendor(self) -> Optional[VendorNPC]:
        """Return the nearest vendor NPC, or None if none exist."""
        best = None
        best_dist = float('inf')
        px, py = self.player.x, self.player.y
        for v in self.vendors:
            dx = v.x - px
            dy = v.y - py
            d = math.sqrt(dx * dx + dy * dy)
            if d < best_dist:
                best_dist = d
                best = v
        return best

    def reset(self) -> None:
        """Reset player and mobs to initial state."""
        self.player = Player(class_id=self.class_id)
        if self.terrain:
            self.player.z = self.terrain.get_height(self.player.x, self.player.y)
        self.target = None
        self.tick_count = 0
        self.damage_dealt = 0
        self.kills = 0
        self._next_uid = 1
        self.vendors.clear()
        self.visited_areas.clear()
        self.visited_zones.clear()
        self.visited_maps.clear()
        self._new_areas = 0
        self._new_zones = 0
        self._new_maps = 0
        # Clear terrain height cache to bound memory across episodes
        if self.terrain:
            self.terrain.clear_height_cache()
        # Reset chunk state
        self._player_chunk = None
        self._active_chunks.clear()
        self._chunk_mobs.clear()
        self._chunk_vendors.clear()
        self.mobs.clear()
        self._spawn_vendors()
        self._spawn_quest_npcs()
        # Reset quest state
        self.active_quests.clear()
        self.completed_quests.clear()
        self.quests_completed = 0
        if self.creature_db:
            self._update_chunks()
        else:
            self._spawn_mobs()
        self._update_exploration()
        self.recalculate_stats()

    # ─── Equipment & Stat System ────────────────────────────────────

    def recalculate_gear_stats(self):
        """Sum all stats from equipped items into player gear_* fields."""
        p = self.player
        # Reset all gear stats
        p.gear_strength = 0
        p.gear_agility = 0
        p.gear_stamina = 0
        p.gear_intellect = 0
        p.gear_spirit = 0
        p.gear_armor = 0
        p.gear_bonus_hp = 0
        p.gear_bonus_mana = 0
        p.gear_attack_power = 0
        p.gear_ranged_ap = 0
        p.gear_spell_power = 0
        p.gear_hit_rating = 0
        p.gear_crit_rating = 0
        p.gear_haste_rating = 0
        p.gear_expertise_rating = 0
        p.gear_armor_pen_rating = 0
        p.gear_spell_pen = 0
        p.gear_defense_rating = 0
        p.gear_dodge_rating = 0
        p.gear_parry_rating = 0
        p.gear_block_rating = 0
        p.gear_block_value = 0
        p.gear_resilience_rating = 0
        p.gear_mp5 = 0
        p.gear_hp5 = 0

        for item in p.equipment.values():
            p.gear_armor += item.armor
            for st, sv in item.stats.items():
                if st == ITEM_MOD_STRENGTH:
                    p.gear_strength += sv
                elif st == ITEM_MOD_AGILITY:
                    p.gear_agility += sv
                elif st == ITEM_MOD_STAMINA:
                    p.gear_stamina += sv
                elif st == ITEM_MOD_INTELLECT:
                    p.gear_intellect += sv
                elif st == ITEM_MOD_SPIRIT:
                    p.gear_spirit += sv
                elif st == ITEM_MOD_HEALTH:
                    p.gear_bonus_hp += sv
                elif st == ITEM_MOD_MANA:
                    p.gear_bonus_mana += sv
                # Offensive
                elif st == ITEM_MOD_ATTACK_POWER:
                    p.gear_attack_power += sv
                elif st == ITEM_MOD_RANGED_ATTACK_POWER:
                    p.gear_ranged_ap += sv
                elif st == ITEM_MOD_SPELL_POWER:
                    p.gear_spell_power += sv
                elif st == ITEM_MOD_SPELL_DAMAGE_DONE:
                    p.gear_spell_power += sv
                elif st == ITEM_MOD_SPELL_HEALING_DONE:
                    p.gear_spell_power += sv
                # Hit (combined and per-type)
                elif st == ITEM_MOD_HIT_RATING:
                    p.gear_hit_rating += sv
                elif st == ITEM_MOD_HIT_MELEE_RATING:
                    p.gear_hit_rating += sv
                elif st == ITEM_MOD_HIT_RANGED_RATING:
                    p.gear_hit_rating += sv
                elif st == ITEM_MOD_HIT_SPELL_RATING:
                    p.gear_hit_rating += sv
                # Crit (combined and per-type)
                elif st == ITEM_MOD_CRIT_RATING:
                    p.gear_crit_rating += sv
                elif st == ITEM_MOD_CRIT_MELEE_RATING:
                    p.gear_crit_rating += sv
                elif st == ITEM_MOD_CRIT_RANGED_RATING:
                    p.gear_crit_rating += sv
                elif st == ITEM_MOD_CRIT_SPELL_RATING:
                    p.gear_crit_rating += sv
                # Haste (combined and per-type)
                elif st == ITEM_MOD_HASTE_RATING:
                    p.gear_haste_rating += sv
                elif st == ITEM_MOD_HASTE_MELEE_RATING:
                    p.gear_haste_rating += sv
                elif st == ITEM_MOD_HASTE_RANGED_RATING:
                    p.gear_haste_rating += sv
                elif st == ITEM_MOD_HASTE_SPELL_RATING:
                    p.gear_haste_rating += sv
                # Defensive ratings
                elif st == ITEM_MOD_DEFENSE_SKILL_RATING:
                    p.gear_defense_rating += sv
                elif st == ITEM_MOD_DODGE_RATING:
                    p.gear_dodge_rating += sv
                elif st == ITEM_MOD_PARRY_RATING:
                    p.gear_parry_rating += sv
                elif st == ITEM_MOD_BLOCK_RATING:
                    p.gear_block_rating += sv
                elif st == ITEM_MOD_BLOCK_VALUE:
                    p.gear_block_value += sv
                elif st == ITEM_MOD_RESILIENCE_RATING:
                    p.gear_resilience_rating += sv
                # Other
                elif st == ITEM_MOD_EXPERTISE_RATING:
                    p.gear_expertise_rating += sv
                elif st == ITEM_MOD_ARMOR_PENETRATION_RATING:
                    p.gear_armor_pen_rating += sv
                elif st == ITEM_MOD_SPELL_PENETRATION:
                    p.gear_spell_pen += sv
                elif st == ITEM_MOD_MANA_REGENERATION:
                    p.gear_mp5 += sv
                elif st == ITEM_MOD_HEALTH_REGEN:
                    p.gear_hp5 += sv

    def recalculate_stats(self):
        """Recalculate all derived stats from gear + level + buffs.

        Call after equip, level-up, or buff change. Uses exact WotLK formulas
        from AzerothCore C++ source (StatSystem.cpp, Player.cpp).
        """
        p = self.player
        cls = p.class_id
        self.recalculate_gear_stats()

        # ─── Primary stat totals (base + gear) ──────────────────────
        p.total_strength = class_base_stat(cls, 0, p.level) + p.gear_strength
        p.total_agility = class_base_stat(cls, 1, p.level) + p.gear_agility
        p.total_stamina = class_base_stat(cls, 2, p.level) + p.gear_stamina
        p.total_intellect = class_base_stat(cls, 3, p.level) + p.gear_intellect
        p.total_spirit = class_base_stat(cls, 4, p.level) + p.gear_spirit

        # ─── Max HP (preserve ratio) ────────────────────────────────
        old_max_hp = max(p.max_hp, 1)
        fort_bonus = p.fortitude_hp_bonus if p.fortitude_remaining > 0 else 0
        p.max_hp = player_max_hp(p.level, p.gear_stamina, p.gear_bonus_hp, cls) + fort_bonus
        hp_ratio = p.hp / old_max_hp
        p.hp = max(1, int(hp_ratio * p.max_hp))

        # ─── Max Mana (preserve ratio) ──────────────────────────────
        old_max_mana = max(p.max_mana, 1)
        p.max_mana = player_max_mana(p.level, p.gear_intellect, p.gear_bonus_mana, cls)
        if p.max_mana > 0:
            mana_ratio = p.mana / old_max_mana
            p.mana = max(0, int(mana_ratio * p.max_mana))

        # ─── Armor (gear + agi*2 + Inner Fire) ──────────────────────
        p.total_armor = p.gear_armor + p.total_agility * 2
        if p.inner_fire_remaining > 0:
            p.total_armor += p.inner_fire_armor

        # ─── Attack Power (melee + ranged) ───────────────────────────
        p.total_attack_power = melee_attack_power(
            p.level, p.total_strength, p.total_agility, cls) + p.gear_attack_power
        p.total_ranged_ap = ranged_attack_power(
            p.level, p.total_strength, p.total_agility, cls) + p.gear_ranged_ap

        # ─── Spell Power (gear + Inner Fire buff) ───────────────────
        inner_fire_sp = p.inner_fire_spellpower if p.inner_fire_remaining > 0 else 0
        p.total_spell_power = p.gear_spell_power + inner_fire_sp

        # ─── Crit (melee, ranged, spell) ─────────────────────────────
        p.total_melee_crit = melee_crit_chance(
            p.level, p.total_agility, p.gear_crit_rating, cls)
        p.total_ranged_crit = ranged_crit_chance(
            p.level, p.total_agility, p.gear_crit_rating, cls)
        p.total_spell_crit = spell_crit_chance(
            p.level, p.gear_intellect, p.gear_crit_rating, cls)

        # ─── Haste (melee, ranged, spell) ────────────────────────────
        p.total_melee_haste = melee_haste_pct(p.level, p.gear_haste_rating)
        p.total_ranged_haste = ranged_haste_pct(p.level, p.gear_haste_rating)
        p.total_spell_haste = spell_haste_pct(p.level, p.gear_haste_rating)

        # ─── Hit (melee, ranged, spell) ──────────────────────────────
        p.total_hit_melee = hit_chance_melee(p.level, p.gear_hit_rating)
        p.total_hit_ranged = hit_chance_ranged(p.level, p.gear_hit_rating)
        p.total_hit_spell = hit_chance_spell(p.level, p.gear_hit_rating)

        # ─── Dodge (with diminishing returns) ────────────────────────
        p.total_dodge = dodge_chance(
            p.level, p.total_agility, p.gear_dodge_rating,
            p.gear_defense_rating, cls)

        # ─── Parry (with diminishing returns) ────────────────────────
        p.total_parry = parry_chance(
            p.level, p.gear_parry_rating, p.gear_defense_rating, cls)

        # ─── Block ───────────────────────────────────────────────────
        # Only classes with shields can block
        has_shield = 14 in p.equipment  # inventory_type 14 = Shield
        if has_shield:
            p.total_block = block_chance(
                p.level, p.gear_block_rating, p.gear_defense_rating)
            # Block value: str/2 - 10 + gear (StatSystem.cpp:GetShieldBlockValue)
            p.total_block_value = max(0, int(p.total_strength * 0.5 - 10) + p.gear_block_value)
        else:
            p.total_block = 0.0
            p.total_block_value = 0

        # ─── Defense ─────────────────────────────────────────────────
        p.total_defense = _rating_to_pct(
            p.gear_defense_rating, CR_DEFENSE_L80, p.level)

        # ─── Expertise ───────────────────────────────────────────────
        p.total_expertise = expertise_pct(p.level, p.gear_expertise_rating)

        # ─── Armor Penetration ───────────────────────────────────────
        p.total_armor_pen = armor_penetration_pct(p.level, p.gear_armor_pen_rating)

        # ─── Resilience ──────────────────────────────────────────────
        p.total_resilience = resilience_pct(p.level, p.gear_resilience_rating)

        # ─── Backwards compat aliases ────────────────────────────────
        # (old code uses total_haste_pct — keep it as spell haste)
        p.total_haste_pct = p.total_spell_haste

    def try_equip_item(self, item_data) -> bool:
        """Try to equip an item if it's better than current (score-based).

        item_data can be a LootDB ItemData or an InventoryItem with stats.
        Returns True if equipped (upgrade detected).
        """
        inv_type = item_data.inventory_type
        if inv_type <= 0:
            return False

        current = self.player.equipment.get(inv_type)
        current_score = current.score if current else 0.0
        new_score = item_data.score

        if new_score <= current_score:
            return False

        # Equip the new item
        stats = getattr(item_data, 'stats', None) or {}
        armor = getattr(item_data, 'armor', 0)
        weapon_dps = getattr(item_data, 'weapon_dps', 0.0)

        self.player.equipment[inv_type] = EquippedItem(
            entry=item_data.entry,
            name=item_data.name,
            inventory_type=inv_type,
            score=new_score,
            stats=stats,
            armor=armor,
            weapon_dps=weapon_dps,
        )
        self.player.equipped_scores[inv_type] = new_score
        self.recalculate_stats()
        return True

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
            self._chunk_vendors.pop(key, None)

        # Activate new chunks
        for key in needed - self._active_chunks:
            self._activate_chunk(key)

        self._active_chunks = needed

        # Rebuild mobs list from active chunks (mob objects persist)
        self.mobs = []
        for key in self._active_chunks:
            self.mobs.extend(self._chunk_mobs.get(key, []))

        # Rebuild vendors list from active chunks
        self.vendors = []
        for key in self._active_chunks:
            self.vendors.extend(self._chunk_vendors.get(key, []))

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

        # Spawn vendors from this chunk
        vendor_spawns = self.creature_db.vendor_index.get(chunk_key, [])
        chunk_vendors: list[VendorNPC] = []
        for sp in vendor_spawns:
            tmpl = self.creature_db.templates.get(sp.entry)
            if tmpl is None:
                continue
            z = self.terrain.get_height(sp.x, sp.y) if self.terrain else sp.z
            chunk_vendors.append(VendorNPC(
                uid=self._new_uid(),
                name=tmpl.name,
                level=tmpl.min_level,
                x=sp.x, y=sp.y, z=z,
            ))
        self._chunk_vendors[chunk_key] = chunk_vendors

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

    def do_move_to(self, tx: float, ty: float) -> bool:
        """Move player toward target coordinates by MOVE_SPEED units.

        Used by vendor navigation — moves directly toward (tx, ty) and
        updates orientation to face the target. Returns False if already
        at the target or movement is blocked.
        """
        if self.player.is_casting:
            return False
        p = self.player
        dx = tx - p.x
        dy = ty - p.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.5:
            return False  # already there

        move = min(self.MOVE_SPEED, dist)
        new_x = p.x + (dx / dist) * move
        new_y = p.y + (dy / dist) * move

        if self.terrain:
            self.terrain.ensure_loaded(new_x, new_y)
            new_z = self.terrain.get_height(new_x, new_y)
            if not self.terrain.check_walkable(p.x, p.y, p.z, new_x, new_y, new_z):
                return False
            p.z = new_z

        p.x = new_x
        p.y = new_y
        # Face toward target
        p.orientation = math.atan2(dy, dx)
        return True

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

    def do_cast_mind_blast(self) -> bool:
        """Start casting Mind Blast. Returns True if cast started."""
        return self._start_cast(8092)

    def do_cast_renew(self) -> bool:
        """Cast Renew (instant HoT)."""
        return self._start_cast(139)

    def do_cast_holy_fire(self) -> bool:
        """Start casting Holy Fire. Returns True if cast started."""
        return self._start_cast(14914)

    def do_cast_inner_fire(self) -> bool:
        """Cast Inner Fire (instant self-buff)."""
        return self._start_cast(588)

    def do_cast_fortitude(self) -> bool:
        """Cast Power Word: Fortitude (instant self-buff)."""
        return self._start_cast(1243)

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
                    self.player.inventory.append(InventoryItem(
                        entry=result.item.entry,
                        name=result.item.name,
                        quality=result.item.quality,
                        sell_price=result.item.sell_price,
                        score=result.item.score,
                        inventory_type=result.item.inventory_type,
                        stats=result.item.stats,
                        armor=result.item.armor,
                        weapon_dps=result.item.weapon_dps,
                    ))
                    # Auto-equip if it's an upgrade
                    if result.item.inventory_type > 0:
                        if self.try_equip_item(result.item):
                            self.player.equipped_upgrade = True
                else:
                    self.player.loot_failed.append(quality)
        else:
            # Fallback: random loot (no loot_db loaded)
            if self.rng.random() < self.LOOT_CHANCE:
                score = self.rng.randint(*self.ITEM_SCORE_RANGE)
                sell_price = score * 2  # rough copper value
                quality = 1  # assume Common for fallback items
                if self.player.free_slots > 0:
                    self.player.loot_score += score
                    self.player.free_slots -= 1
                    self.player.loot_items.append(quality)
                    self.player.inventory.append(InventoryItem(
                        entry=0, name="Loot",
                        quality=quality,
                        sell_price=sell_price,
                        score=score,
                        inventory_type=0,
                    ))
                    if self.rng.random() < self.UPGRADE_CHANCE:
                        self.player.equipped_upgrade = True
                else:
                    self.player.loot_failed.append(quality)
        # Quest collect objective tracking
        self.on_mob_looted(best)
        return True

    def do_sell(self) -> bool:
        """Sell all inventory items at the nearest vendor within SELL_RANGE.

        Requires proximity to a vendor NPC. Converts inventory items to copper
        based on their sell_price. Returns True if items were sold.
        """
        if self.player.free_slots >= INVENTORY_SLOTS:
            return False
        # Must be near a vendor
        vendor = self.get_nearest_vendor()
        if vendor is None:
            return False
        dist = self._dist(self.player.x, self.player.y, vendor.x, vendor.y)
        if dist > self.SELL_RANGE:
            return False
        # Calculate copper from inventory sell prices
        num_items = len(self.player.inventory)
        copper = sum(item.sell_price for item in self.player.inventory)
        self.player.copper += copper
        self.player.sell_copper += copper
        self.player.items_sold += num_items
        self.player.inventory.clear()
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

        # Spell-specific cooldown check
        if self.player.spell_cooldowns.get(spell_id, 0) > 0:
            return False

        # Range check for offensive spells (with target requirement)
        if spell_id in (585, 589, 8092, 14914):
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

        # Shield: check if already shielded or Weakened Soul active
        if spell_id == 17 and (self.player.shield_remaining > 0
                               or self.player.shield_cooldown > 0):
            return False

        # Renew: block if HoT already active
        if spell_id == 139 and self.player.hot_remaining > 0:
            return False

        # Inner Fire: block if already active
        if spell_id == 588 and self.player.inner_fire_remaining > 0:
            return False

        # Fortitude: block if already active
        if spell_id == 1243 and self.player.fortitude_remaining > 0:
            return False

        # Spend mana
        self.player.mana -= spell.mana_cost

        # GCD
        self.player.gcd_remaining = spell.gcd_ticks

        # Spell-specific cooldown
        if spell.cooldown_ticks > 0:
            self.player.spell_cooldowns[spell_id] = spell.cooldown_ticks

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
        """Apply spell effect when cast completes. Uses total_spell_power for scaling."""
        spell = SPELLS[spell_id]
        sp = self.player.total_spell_power
        crit_pct = self.player.total_spell_crit

        if spell_id == 585:  # Smite — level-scaled + SP
            if self.target and self.target.alive:
                min_dmg, max_dmg = smite_damage(self.player.level, sp)
                dmg = self.rng.randint(min_dmg, max_dmg)
                if self.rng.random() * 100 < crit_pct:
                    dmg = int(dmg * 1.5)  # spell crit = 150%
                self._damage_mob(self.target, dmg)

        elif spell_id == 2050:  # Lesser Heal — level-scaled + SP
            min_h, max_h = heal_amount(self.player.level, sp)
            heal = self.rng.randint(min_h, max_h)
            if self.rng.random() * 100 < crit_pct:
                heal = int(heal * 1.5)  # healing crit = 150%
            self.player.hp = min(self.player.max_hp, self.player.hp + heal)

        elif spell_id == 589:  # SW:Pain — SP-scaled DoT
            if self.target and self.target.alive:
                total_dmg = sw_pain_total(self.player.level, sp)
                total_ticks = spell.dot_ticks // spell.dot_interval
                dmg_per_tick = total_dmg // max(1, total_ticks)
                self.target.dot_remaining = spell.dot_ticks
                self.target.dot_timer = spell.dot_interval
                self.target.dot_damage_per_tick = dmg_per_tick

        elif spell_id == 17:  # PW:Shield — SP-scaled absorb
            absorb = pw_shield_absorb(self.player.level, sp)
            self.player.shield_absorb = absorb
            self.player.shield_remaining = spell.shield_duration
            self.player.shield_cooldown = 30  # Weakened Soul: 15s = 30 ticks

        elif spell_id == 8092:  # Mind Blast — level-scaled + SP
            if self.target and self.target.alive:
                min_dmg, max_dmg = mind_blast_damage(self.player.level, sp)
                dmg = self.rng.randint(min_dmg, max_dmg)
                if self.rng.random() * 100 < crit_pct:
                    dmg = int(dmg * 1.5)
                self._damage_mob(self.target, dmg)

        elif spell_id == 139:  # Renew — SP-scaled HoT
            total = renew_total_heal(self.player.level, sp)
            total_ticks = spell.hot_ticks // spell.hot_interval  # 5
            heal_per = total // max(1, total_ticks)
            self.player.hot_remaining = spell.hot_ticks
            self.player.hot_timer = spell.hot_interval
            self.player.hot_heal_per_tick = heal_per

        elif spell_id == 14914:  # Holy Fire — direct + SP-scaled DoT (slot 2)
            if self.target and self.target.alive:
                min_dmg, max_dmg = holy_fire_damage(self.player.level, sp)
                dmg = self.rng.randint(min_dmg, max_dmg)
                if self.rng.random() * 100 < crit_pct:
                    dmg = int(dmg * 1.5)
                self._damage_mob(self.target, dmg)
                # DoT component on slot 2
                dot_total = holy_fire_dot_total(self.player.level, sp)
                dot_ticks_count = spell.dot_ticks // spell.dot_interval  # 2
                dot_per = dot_total // max(1, dot_ticks_count)
                self.target.dot2_remaining = spell.dot_ticks
                self.target.dot2_timer = spell.dot_interval
                self.target.dot2_damage_per_tick = dot_per

        elif spell_id == 588:  # Inner Fire — armor + spellpower buff
            armor, sp_buff = inner_fire_values(self.player.level)
            self.player.inner_fire_remaining = spell.buff_duration
            self.player.inner_fire_armor = armor
            self.player.inner_fire_spellpower = sp_buff
            self.recalculate_stats()  # SP changed

        elif spell_id == 1243:  # PW:Fortitude — max HP buff
            bonus = fortitude_hp_bonus(self.player.level)
            self.player.fortitude_remaining = spell.buff_duration
            self.player.fortitude_hp_bonus = bonus
            self.recalculate_stats()  # HP changed
            self.player.hp = min(self.player.hp + bonus, self.player.max_hp)

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
            # Quest kill objective tracking
            self.on_mob_killed(mob)
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
        self.recalculate_stats()
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
        self._update_quest_exploration()

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
                    # Far non-combat mob: only process DoTs then skip
                    if mob.dot_remaining > 0:
                        mob.dot_remaining -= 1
                        mob.dot_timer -= 1
                        if mob.dot_timer <= 0:
                            self._damage_mob(mob, mob.dot_damage_per_tick)
                            mob.dot_timer = 6
                    if mob.dot2_remaining > 0:
                        mob.dot2_remaining -= 1
                        mob.dot2_timer -= 1
                        if mob.dot2_timer <= 0:
                            self._damage_mob(mob, mob.dot2_damage_per_tick)
                            mob.dot2_timer = 6
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

            # DoT processing (slot 1: SW:Pain)
            if mob.dot_remaining > 0:
                mob.dot_remaining -= 1
                mob.dot_timer -= 1
                if mob.dot_timer <= 0:
                    self._damage_mob(mob, mob.dot_damage_per_tick)
                    mob.dot_timer = 6
            # DoT processing (slot 2: Holy Fire)
            if mob.dot2_remaining > 0:
                mob.dot2_remaining -= 1
                mob.dot2_timer -= 1
                if mob.dot2_timer <= 0:
                    self._damage_mob(mob, mob.dot2_damage_per_tick)
                    mob.dot2_timer = 6

        # --- Shield decay ---
        if p.shield_remaining > 0:
            p.shield_remaining -= 1
            if p.shield_remaining <= 0:
                p.shield_absorb = 0
        if p.shield_cooldown > 0:
            p.shield_cooldown -= 1

        # --- Spell cooldowns ---
        for sid in list(p.spell_cooldowns):
            p.spell_cooldowns[sid] -= 1
            if p.spell_cooldowns[sid] <= 0:
                del p.spell_cooldowns[sid]

        # --- HoT (Renew) ---
        if p.hot_remaining > 0:
            p.hot_remaining -= 1
            p.hot_timer -= 1
            if p.hot_timer <= 0:
                p.hp = min(p.max_hp, p.hp + p.hot_heal_per_tick)
                p.hot_timer = 6
            if p.hot_remaining <= 0:
                p.hot_heal_per_tick = 0

        # --- Buff: Inner Fire ---
        if p.inner_fire_remaining > 0:
            p.inner_fire_remaining -= 1
            if p.inner_fire_remaining <= 0:
                p.inner_fire_armor = 0
                p.inner_fire_spellpower = 0

        # --- Buff: PW:Fortitude ---
        if p.fortitude_remaining > 0:
            p.fortitude_remaining -= 1
            if p.fortitude_remaining <= 0:
                # Remove HP bonus
                p.max_hp -= p.fortitude_hp_bonus
                p.hp = min(p.hp, p.max_hp)
                p.fortitude_hp_bonus = 0

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

        # Mana regen (WotLK 5-second rule):
        #   OOC / not casting: Spirit-based regen + MP5 from gear
        #   While casting: only MP5 from gear (flat)
        mp5_per_tick = p.gear_mp5 / 5.0 * 0.5  # MP5 -> per tick (0.5s)
        if not p.is_casting:
            spi_regen = spirit_mana_regen(
                p.level, p.gear_intellect, p.gear_spirit, p.class_id)
            p.mana_regen_accumulator += spi_regen + mp5_per_tick
        else:
            p.mana_regen_accumulator += mp5_per_tick
        # Fallback minimum: 2% of max_mana per tick if spirit regen is too low
        min_regen = p.max_mana * self.MANA_REGEN_PCT_PER_TICK
        if not p.is_casting and p.mana_regen_accumulator < min_regen:
            p.mana_regen_accumulator = min_regen
        if p.mana_regen_accumulator >= 1.0:
            regen = int(p.mana_regen_accumulator)
            p.mana = min(p.max_mana, p.mana + regen)
            p.mana_regen_accumulator -= regen

    def _damage_player(self, damage: int):
        """Apply damage to player, considering armor mitigation and shield.

        Armor mitigation uses the WotLK formula from Unit::CalcArmorReducedDamage
        (Unit.cpp:2067):
          eff_level = attacker_level + 4.5*(attacker_level-59) if >59
          dr = 0.1 * armor / (8.5 * eff_level + 40)
          mitigation = dr / (1 + dr), capped at 75%
        """
        p = self.player
        # total_armor is pre-computed in recalculate_stats (gear + agi*2 + inner fire)
        if p.total_armor > 0:
            mob_level = 1  # approximate attacker level
            if self.target and self.target.alive:
                mob_level = self.target.level
            else:
                for m in self.mobs:
                    if m.alive and m.target_player:
                        mob_level = m.level
                        break
            eff_level = mob_level + (4.5 * (mob_level - 59) if mob_level > 59 else 0)
            dr = 0.1 * p.total_armor / (8.5 * eff_level + 40)
            mitigation = dr / (1 + dr)
            mitigation = min(mitigation, 0.75)
            damage = max(1, int(damage * (1 - mitigation)))
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
        """Get list of nearby mobs and vendors (within scan range)."""
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
                    "vendor": 0,
                })
        # Include quest NPCs
        for qnpc in self.quest_npcs:
            dx = qnpc.x - px
            dy = qnpc.y - py
            dsq = dx * dx + dy * dy
            if dsq <= r_sq:
                d = _sqrt(dsq)
                result.append({
                    "uid": qnpc.uid,
                    "name": qnpc.name,
                    "level": 10,
                    "hp": 100,
                    "max_hp": 100,
                    "alive": True,
                    "x": qnpc.x,
                    "y": qnpc.y,
                    "z": qnpc.z,
                    "dist": d,
                    "target_player": False,
                    "looted": False,
                    "attackable": 0,
                    "vendor": 0,
                    "questgiver": 1,
                    "entry": qnpc.entry,
                })
        # Include vendor NPCs
        for v in self.vendors:
            dx = v.x - px
            dy = v.y - py
            dsq = dx * dx + dy * dy
            if dsq <= r_sq:
                d = _sqrt(dsq)
                result.append({
                    "uid": v.uid,
                    "name": v.name,
                    "level": v.level,
                    "hp": 100,
                    "max_hp": 100,
                    "alive": True,
                    "x": v.x,
                    "y": v.y,
                    "z": v.z,
                    "dist": d,
                    "target_player": False,
                    "looted": False,
                    "attackable": 0,
                    "vendor": 1,
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
            "has_holy_fire": self.target.dot2_remaining > 0,
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
            "has_renew": "true" if p.hot_remaining > 0 else "false",
            "has_inner_fire": "true" if p.inner_fire_remaining > 0 else "false",
            "has_fortitude": "true" if p.fortitude_remaining > 0 else "false",
            "mind_blast_ready": "true" if p.spell_cooldowns.get(8092, 0) <= 0 else "false",
            "holy_fire_ready": "true" if p.spell_cooldowns.get(14914, 0) <= 0 else "false",
            "target_has_sw_pain": "true" if t_info.get("has_sw_pain") else "false",
            "target_has_holy_fire": "true" if t_info.get("has_holy_fire") else "false",
            "nearby_mobs": [
                {
                    "guid": str(m["uid"]),
                    "name": m["name"],
                    "level": m["level"],
                    "attackable": m["attackable"],
                    "vendor": m.get("vendor", 0),
                    "questgiver": m.get("questgiver", 0),
                    "target": "1" if m["target_player"] else "0",
                    "hp": m["hp"],
                    "x": m["x"],
                    "y": m["y"],
                }
                for m in nearby
            ],
            # Primary stats (sim-only, total = base + gear)
            "total_strength": p.total_strength,
            "total_agility": p.total_agility,
            "total_stamina": p.total_stamina,
            "total_intellect": p.total_intellect,
            "total_spirit": p.total_spirit,
            # Offensive stats
            "attack_power": p.total_attack_power,
            "ranged_ap": p.total_ranged_ap,
            "spell_power": p.total_spell_power,
            "melee_crit": p.total_melee_crit,
            "ranged_crit": p.total_ranged_crit,
            "spell_crit": p.total_spell_crit,
            "melee_haste": p.total_melee_haste,
            "ranged_haste": p.total_ranged_haste,
            "spell_haste": p.total_spell_haste,
            "hit_melee": p.total_hit_melee,
            "hit_ranged": p.total_hit_ranged,
            "hit_spell": p.total_hit_spell,
            "expertise": p.total_expertise,
            "armor_pen": p.total_armor_pen,
            # Defensive stats
            "total_armor": p.total_armor,
            "dodge": p.total_dodge,
            "parry": p.total_parry,
            "block": p.total_block,
            "block_value": p.total_block_value,
            "defense": p.total_defense,
            "resilience": p.total_resilience,
            # Gear totals (for backwards compat)
            "gear_armor": p.gear_armor,
            "gear_stamina": p.gear_stamina,
            "gear_intellect": p.gear_intellect,
            "gear_spirit": p.gear_spirit,
            # Quest state (sim-only, not present in live TCP stream)
            "quest_active": len(self.active_quests) > 0,
            "quest_progress": self._get_quest_progress_ratio(),
            "quests_completed_total": self.quests_completed,
        }

    def _get_quest_progress_ratio(self) -> float:
        """Get aggregate quest completion progress as 0-1 ratio."""
        if not self.active_quests or not self.quest_db:
            return 0.0
        total_needed = 0
        total_done = 0
        for qid, prog in self.active_quests.items():
            qt = self.quest_db.templates[qid]
            for i, obj in enumerate(qt.objectives):
                total_needed += obj.count
                total_done += min(prog.counts[i], obj.count)
        return total_done / max(1, total_needed)

    def get_best_quest_npc(self) -> tuple:
        """Find the most relevant quest NPC for the bot to interact with.

        Returns (npc, npc_type) where npc_type is:
          'turn_in' — NPC can accept a completed quest
          'accept'  — NPC can give a new quest
          None      — no relevant quest NPC
        """
        if not self.quest_db:
            return None, None

        px, py = self.player.x, self.player.y
        best_npc = None
        best_type = None
        best_dist = float('inf')

        for npc in self.quest_npcs:
            dx = npc.x - px
            dy = npc.y - py
            dist = math.sqrt(dx * dx + dy * dy)

            # Turn-in NPCs have highest priority
            completable = self.quest_db.get_completable_quests(
                npc.entry, self.active_quests)
            if completable and dist < best_dist:
                best_npc = npc
                best_type = 'turn_in'
                best_dist = dist
                continue

            # Accept NPCs have second priority
            available = self.quest_db.get_available_quests(
                npc.entry, self.player.level, self.completed_quests,
                self.active_quests)
            if available and (best_type != 'turn_in') and dist < best_dist:
                best_npc = npc
                best_type = 'accept'
                best_dist = dist

        return best_npc, best_type

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
            "sell_copper": p.sell_copper,
            "items_sold": p.items_sold,
            "new_areas": self._new_areas,
            "new_zones": self._new_zones,
            "new_maps": self._new_maps,
            "quest_xp": p.quest_xp_gained,
            "quest_copper": p.quest_copper_gained,
            "quests_completed": p.quests_completed_tick,
        }
        p.xp_gained = 0
        p.loot_copper = 0
        p.loot_score = 0
        p.equipped_upgrade = False
        p.leveled_up = False
        p.levels_gained = 0
        p.loot_items.clear()
        p.loot_failed.clear()
        p.sell_copper = 0
        p.items_sold = 0
        p.quest_xp_gained = 0
        p.quest_copper_gained = 0
        p.quests_completed_tick = 0
        self._new_areas = 0
        self._new_zones = 0
        self._new_maps = 0
        return events
