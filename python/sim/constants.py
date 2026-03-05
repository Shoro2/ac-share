"""
WoW WotLK 3.3.5 Constants and DBC Tables.

All game constants, class definitions, stat weights, equipment slots,
base stats, combat rating tables, and spell power coefficients.

Primary stat tables (base stats, HP/Mana per level) and GameTable values
(combat ratings, crit from stat, spirit regen) are loaded from DBC/CSV files
at import time when available. Hardcoded fallback values are used when the
data files are not found.

Data files expected in <repo>/data/:
  - player_class_stats.csv  (per-class per-level base stats from AzerothCore DB)
  - dbc/gtCombatRatings.dbc (combat rating conversion tables)
  - dbc/gtChanceToMeleeCrit.dbc, dbc/gtChanceToMeleeCritBase.dbc
  - dbc/gtChanceToSpellCrit.dbc, dbc/gtChanceToSpellCritBase.dbc
  - dbc/gtRegenMPPerSpt.dbc, dbc/gtRegenHPPerSpt.dbc
  - dbc/gtOCTRegenHP.dbc, dbc/gtOCTRegenMP.dbc
"""

import os

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


# ─── Class-Specific Stat Weights for Item Scoring ───────────────────
# Per-class weights based on WotLK 3.3.5 SimulationCraft stat priorities.
# Each class uses its dominant DPS/heal spec's priority order.
# Weights are applied in class_aware_score() to replace the flat
# TotalStats*2 component of GetItemScore.
#
# Source: 3.3.5 SimCraft stat priorities per class/spec:
#   Warrior (Arms/Fury):  ArP > STR > Crit > Expertise(26) > Hit(8%)
#   Paladin (Retribution): STR > Hit(8%) > Expertise(26) > Crit > AGI > Haste
#   Hunter (MM/Surv):     AGI > ArP > Hit(8%) > Crit > INT
#   Rogue (Assassination): Hit(12%) > Expertise(26) > AGI > AP > Haste > Crit
#   Priest (Shadow):      Hit(17%) > SP > Haste > Crit
#   DK (Unholy/Frost):    STR > Hit(8%) > Crit > Haste > ArP
#   Shaman (Enhancement): Hit(17%) > Expertise(26) > Haste > AP > AGI > Crit
#   Mage (Fire):          SP > Haste > Crit > Hit(17%)
#   Warlock (Affli/Destro): SP > Haste > Hit(17%) > Crit
#   Druid (Feral Cat):    ArP > AGI > STR > Crit > Haste

# Helper: common "useless for this archetype" stats at low weights
_PHYS_IGNORE = {
    ITEM_MOD_INTELLECT: 0.1, ITEM_MOD_SPIRIT: 0.0, ITEM_MOD_SPELL_POWER: 0.0,
    ITEM_MOD_SPELL_HEALING_DONE: 0.0, ITEM_MOD_SPELL_DAMAGE_DONE: 0.0,
    ITEM_MOD_MANA_REGENERATION: 0.0, ITEM_MOD_SPELL_PENETRATION: 0.0,
    ITEM_MOD_CRIT_SPELL_RATING: 0.1, ITEM_MOD_HASTE_SPELL_RATING: 0.1,
    ITEM_MOD_HIT_SPELL_RATING: 0.1,
}
_CAST_IGNORE = {
    ITEM_MOD_STRENGTH: 0.1, ITEM_MOD_AGILITY: 0.1,
    ITEM_MOD_ATTACK_POWER: 0.0, ITEM_MOD_RANGED_ATTACK_POWER: 0.0,
    ITEM_MOD_EXPERTISE_RATING: 0.0, ITEM_MOD_ARMOR_PENETRATION_RATING: 0.0,
    ITEM_MOD_HIT_MELEE_RATING: 0.1, ITEM_MOD_CRIT_MELEE_RATING: 0.1,
    ITEM_MOD_HASTE_MELEE_RATING: 0.1, ITEM_MOD_HIT_RANGED_RATING: 0.0,
    ITEM_MOD_CRIT_RANGED_RATING: 0.0, ITEM_MOD_HASTE_RANGED_RATING: 0.0,
    ITEM_MOD_DODGE_RATING: 0.2, ITEM_MOD_PARRY_RATING: 0.0,
    ITEM_MOD_BLOCK_RATING: 0.0, ITEM_MOD_DEFENSE_SKILL_RATING: 0.2,
    ITEM_MOD_BLOCK_VALUE: 0.0,
}

# ── Warrior (Arms/Fury): ArP > STR > Crit > Expertise > Hit ─────────
_WARRIOR_WEIGHTS = {
    ITEM_MOD_ARMOR_PENETRATION_RATING: 3.5,  # king stat at high gear
    ITEM_MOD_STRENGTH: 3.0,
    ITEM_MOD_CRIT_RATING: 2.5, ITEM_MOD_CRIT_MELEE_RATING: 2.5,
    ITEM_MOD_EXPERTISE_RATING: 2.5,          # mandatory to 26 cap
    ITEM_MOD_HIT_RATING: 2.5, ITEM_MOD_HIT_MELEE_RATING: 2.5,  # 8% cap
    ITEM_MOD_ATTACK_POWER: 2.0,
    ITEM_MOD_HASTE_RATING: 1.5, ITEM_MOD_HASTE_MELEE_RATING: 1.5,
    ITEM_MOD_AGILITY: 1.0, ITEM_MOD_STAMINA: 1.0,
    ITEM_MOD_HEALTH: 0.5, ITEM_MOD_MANA: 0.0,
    ITEM_MOD_DODGE_RATING: 0.5, ITEM_MOD_PARRY_RATING: 0.5,
    ITEM_MOD_BLOCK_RATING: 0.3, ITEM_MOD_BLOCK_VALUE: 0.3,
    ITEM_MOD_DEFENSE_SKILL_RATING: 0.3,
    ITEM_MOD_RESILIENCE_RATING: 0.3, ITEM_MOD_HEALTH_REGEN: 0.3,
    ITEM_MOD_RANGED_ATTACK_POWER: 0.1,
    ITEM_MOD_HIT_RANGED_RATING: 0.1, ITEM_MOD_CRIT_RANGED_RATING: 0.1,
    ITEM_MOD_HASTE_RANGED_RATING: 0.1,
    **_PHYS_IGNORE,
}

# ── Paladin (Retribution): STR > Hit > Expertise > Crit > AGI > Haste
_PALADIN_WEIGHTS = {
    ITEM_MOD_STRENGTH: 3.5,
    ITEM_MOD_HIT_RATING: 3.0, ITEM_MOD_HIT_MELEE_RATING: 3.0,  # 8% cap
    ITEM_MOD_EXPERTISE_RATING: 3.0,          # 26 cap
    ITEM_MOD_CRIT_RATING: 2.5, ITEM_MOD_CRIT_MELEE_RATING: 2.5,
    ITEM_MOD_AGILITY: 1.5,
    ITEM_MOD_HASTE_RATING: 1.5, ITEM_MOD_HASTE_MELEE_RATING: 1.5,
    ITEM_MOD_ATTACK_POWER: 2.0,
    ITEM_MOD_SPELL_POWER: 1.0,               # Ret scales partially with SP
    ITEM_MOD_STAMINA: 1.0, ITEM_MOD_INTELLECT: 0.5,
    ITEM_MOD_ARMOR_PENETRATION_RATING: 0.5,
    ITEM_MOD_HEALTH: 0.5, ITEM_MOD_MANA: 0.3,
    ITEM_MOD_MANA_REGENERATION: 0.3, ITEM_MOD_HEALTH_REGEN: 0.3,
    ITEM_MOD_DODGE_RATING: 0.3, ITEM_MOD_PARRY_RATING: 0.3,
    ITEM_MOD_BLOCK_RATING: 0.3, ITEM_MOD_BLOCK_VALUE: 0.3,
    ITEM_MOD_DEFENSE_SKILL_RATING: 0.3,
    ITEM_MOD_RESILIENCE_RATING: 0.3,
    ITEM_MOD_SPIRIT: 0.1,
    ITEM_MOD_SPELL_HEALING_DONE: 0.3, ITEM_MOD_SPELL_DAMAGE_DONE: 0.3,
    ITEM_MOD_SPELL_PENETRATION: 0.1,
    ITEM_MOD_CRIT_SPELL_RATING: 0.5, ITEM_MOD_HIT_SPELL_RATING: 0.5,
    ITEM_MOD_HASTE_SPELL_RATING: 0.5,
    ITEM_MOD_RANGED_ATTACK_POWER: 0.0,
    ITEM_MOD_HIT_RANGED_RATING: 0.0, ITEM_MOD_CRIT_RANGED_RATING: 0.0,
    ITEM_MOD_HASTE_RANGED_RATING: 0.0,
}

# ── Hunter (MM/Survival): AGI > ArP > Hit(8%) > Crit > INT ──────────
_HUNTER_WEIGHTS = {
    ITEM_MOD_AGILITY: 3.5,
    ITEM_MOD_ARMOR_PENETRATION_RATING: 3.0,
    ITEM_MOD_HIT_RATING: 2.8, ITEM_MOD_HIT_RANGED_RATING: 2.8,  # 8% cap
    ITEM_MOD_CRIT_RATING: 2.5, ITEM_MOD_CRIT_RANGED_RATING: 2.5,
    ITEM_MOD_RANGED_ATTACK_POWER: 2.5, ITEM_MOD_ATTACK_POWER: 2.0,
    ITEM_MOD_HASTE_RATING: 2.0, ITEM_MOD_HASTE_RANGED_RATING: 2.0,
    ITEM_MOD_INTELLECT: 1.5,
    ITEM_MOD_STAMINA: 1.0, ITEM_MOD_HEALTH: 0.5,
    ITEM_MOD_MANA: 0.3, ITEM_MOD_MANA_REGENERATION: 0.3,
    ITEM_MOD_STRENGTH: 0.3,
    ITEM_MOD_EXPERTISE_RATING: 0.3,           # irrelevant for ranged
    ITEM_MOD_RESILIENCE_RATING: 0.3, ITEM_MOD_HEALTH_REGEN: 0.3,
    ITEM_MOD_SPIRIT: 0.1,
    ITEM_MOD_DODGE_RATING: 0.3, ITEM_MOD_PARRY_RATING: 0.1,
    ITEM_MOD_BLOCK_RATING: 0.0, ITEM_MOD_BLOCK_VALUE: 0.0,
    ITEM_MOD_DEFENSE_SKILL_RATING: 0.1,
    ITEM_MOD_SPELL_POWER: 0.0, ITEM_MOD_SPELL_HEALING_DONE: 0.0,
    ITEM_MOD_SPELL_DAMAGE_DONE: 0.0, ITEM_MOD_SPELL_PENETRATION: 0.0,
    ITEM_MOD_CRIT_MELEE_RATING: 0.3, ITEM_MOD_HIT_MELEE_RATING: 0.3,
    ITEM_MOD_HASTE_MELEE_RATING: 0.3,
    ITEM_MOD_CRIT_SPELL_RATING: 0.1, ITEM_MOD_HIT_SPELL_RATING: 0.1,
    ITEM_MOD_HASTE_SPELL_RATING: 0.1,
}

# ── Rogue (Assassination): Hit(12%) > Expertise(26) > AGI > AP > Haste > Crit
_ROGUE_WEIGHTS = {
    ITEM_MOD_HIT_RATING: 3.5, ITEM_MOD_HIT_MELEE_RATING: 3.5,  # 12% for poisons
    ITEM_MOD_EXPERTISE_RATING: 3.5,          # 26 cap mandatory
    ITEM_MOD_AGILITY: 3.0,
    ITEM_MOD_ATTACK_POWER: 2.5,
    ITEM_MOD_HASTE_RATING: 2.0, ITEM_MOD_HASTE_MELEE_RATING: 2.0,
    ITEM_MOD_CRIT_RATING: 1.8, ITEM_MOD_CRIT_MELEE_RATING: 1.8,
    ITEM_MOD_ARMOR_PENETRATION_RATING: 2.5,  # Combat values ArP highly
    ITEM_MOD_STRENGTH: 1.0, ITEM_MOD_STAMINA: 1.0,
    ITEM_MOD_HEALTH: 0.5, ITEM_MOD_MANA: 0.0,
    ITEM_MOD_DODGE_RATING: 0.5, ITEM_MOD_PARRY_RATING: 0.3,
    ITEM_MOD_RESILIENCE_RATING: 0.3, ITEM_MOD_HEALTH_REGEN: 0.3,
    ITEM_MOD_BLOCK_RATING: 0.0, ITEM_MOD_BLOCK_VALUE: 0.0,
    ITEM_MOD_DEFENSE_SKILL_RATING: 0.1,
    ITEM_MOD_RANGED_ATTACK_POWER: 0.1,
    ITEM_MOD_HIT_RANGED_RATING: 0.1, ITEM_MOD_CRIT_RANGED_RATING: 0.1,
    ITEM_MOD_HASTE_RANGED_RATING: 0.1,
    **_PHYS_IGNORE,
}

# ── Priest (Shadow): Hit(17%) > SP > Haste > Crit ───────────────────
_PRIEST_WEIGHTS = {
    ITEM_MOD_HIT_RATING: 3.5, ITEM_MOD_HIT_SPELL_RATING: 3.5,  # 17% cap priority #1
    ITEM_MOD_SPELL_POWER: 3.0,
    ITEM_MOD_HASTE_RATING: 2.8, ITEM_MOD_HASTE_SPELL_RATING: 2.8,
    ITEM_MOD_CRIT_RATING: 2.0, ITEM_MOD_CRIT_SPELL_RATING: 2.0,
    ITEM_MOD_INTELLECT: 2.0,
    ITEM_MOD_SPIRIT: 1.5,                    # shadow tap + regen
    ITEM_MOD_SPELL_DAMAGE_DONE: 2.5,
    ITEM_MOD_SPELL_HEALING_DONE: 1.0,        # less value for shadow
    ITEM_MOD_STAMINA: 1.0, ITEM_MOD_MANA: 0.5, ITEM_MOD_HEALTH: 0.5,
    ITEM_MOD_MANA_REGENERATION: 1.0,
    ITEM_MOD_SPELL_PENETRATION: 1.5,         # useful in PvE shadow
    ITEM_MOD_RESILIENCE_RATING: 0.3, ITEM_MOD_HEALTH_REGEN: 0.3,
    **_CAST_IGNORE,
}

# ── Death Knight (Unholy/Frost): STR > Hit(8%) > Crit > Haste > ArP ─
_DK_WEIGHTS = {
    ITEM_MOD_STRENGTH: 3.5,
    ITEM_MOD_HIT_RATING: 3.0, ITEM_MOD_HIT_MELEE_RATING: 3.0,  # 8% cap
    ITEM_MOD_CRIT_RATING: 2.5, ITEM_MOD_CRIT_MELEE_RATING: 2.5,
    ITEM_MOD_EXPERTISE_RATING: 2.5,          # 26 cap
    ITEM_MOD_HASTE_RATING: 2.0, ITEM_MOD_HASTE_MELEE_RATING: 2.0,
    ITEM_MOD_ARMOR_PENETRATION_RATING: 1.8,  # lower than Warrior for UH/Frost
    ITEM_MOD_ATTACK_POWER: 2.0,
    ITEM_MOD_AGILITY: 0.5, ITEM_MOD_STAMINA: 1.0,
    ITEM_MOD_HEALTH: 0.5, ITEM_MOD_MANA: 0.0,
    ITEM_MOD_DODGE_RATING: 0.5, ITEM_MOD_PARRY_RATING: 0.5,
    ITEM_MOD_BLOCK_RATING: 0.0, ITEM_MOD_BLOCK_VALUE: 0.0,
    ITEM_MOD_DEFENSE_SKILL_RATING: 0.3,
    ITEM_MOD_RESILIENCE_RATING: 0.3, ITEM_MOD_HEALTH_REGEN: 0.3,
    ITEM_MOD_RANGED_ATTACK_POWER: 0.0,
    ITEM_MOD_HIT_RANGED_RATING: 0.0, ITEM_MOD_CRIT_RANGED_RATING: 0.0,
    ITEM_MOD_HASTE_RANGED_RATING: 0.0,
    **_PHYS_IGNORE,
}

# ── Shaman (Enhancement): Hit(17%) > Expertise(26) > Haste > AP > AGI > Crit
_SHAMAN_WEIGHTS = {
    ITEM_MOD_HIT_RATING: 3.5, ITEM_MOD_HIT_MELEE_RATING: 3.5,   # 17% spell cap!
    ITEM_MOD_HIT_SPELL_RATING: 3.5,          # spells use spell hit
    ITEM_MOD_EXPERTISE_RATING: 3.0,          # 26 cap
    ITEM_MOD_HASTE_RATING: 2.5, ITEM_MOD_HASTE_MELEE_RATING: 2.5,
    ITEM_MOD_ATTACK_POWER: 2.5,
    ITEM_MOD_AGILITY: 2.0,
    ITEM_MOD_CRIT_RATING: 1.8, ITEM_MOD_CRIT_MELEE_RATING: 1.8,
    ITEM_MOD_STRENGTH: 1.0, ITEM_MOD_STAMINA: 1.0,
    ITEM_MOD_INTELLECT: 1.0,                 # some mana benefit
    ITEM_MOD_SPELL_POWER: 1.0,               # enhance scales partially
    ITEM_MOD_ARMOR_PENETRATION_RATING: 1.0,
    ITEM_MOD_HEALTH: 0.5, ITEM_MOD_MANA: 0.3,
    ITEM_MOD_MANA_REGENERATION: 0.3, ITEM_MOD_HEALTH_REGEN: 0.3,
    ITEM_MOD_RESILIENCE_RATING: 0.3,
    ITEM_MOD_DODGE_RATING: 0.3, ITEM_MOD_PARRY_RATING: 0.3,
    ITEM_MOD_BLOCK_RATING: 0.3, ITEM_MOD_BLOCK_VALUE: 0.3,
    ITEM_MOD_DEFENSE_SKILL_RATING: 0.3,
    ITEM_MOD_SPIRIT: 0.1,
    ITEM_MOD_CRIT_SPELL_RATING: 1.0, ITEM_MOD_HASTE_SPELL_RATING: 1.0,
    ITEM_MOD_SPELL_HEALING_DONE: 0.3, ITEM_MOD_SPELL_DAMAGE_DONE: 0.5,
    ITEM_MOD_SPELL_PENETRATION: 0.1,
    ITEM_MOD_RANGED_ATTACK_POWER: 0.0,
    ITEM_MOD_HIT_RANGED_RATING: 0.0, ITEM_MOD_CRIT_RANGED_RATING: 0.0,
    ITEM_MOD_HASTE_RANGED_RATING: 0.0,
}

# ── Mage (Fire): SP > Haste > Crit > Hit(17%) ───────────────────────
_MAGE_WEIGHTS = {
    ITEM_MOD_SPELL_POWER: 3.5,
    ITEM_MOD_HASTE_RATING: 3.0, ITEM_MOD_HASTE_SPELL_RATING: 3.0,
    ITEM_MOD_CRIT_RATING: 2.5, ITEM_MOD_CRIT_SPELL_RATING: 2.5,
    ITEM_MOD_HIT_RATING: 2.5, ITEM_MOD_HIT_SPELL_RATING: 2.5,  # 17% cap (11% w/ talents)
    ITEM_MOD_INTELLECT: 2.0,
    ITEM_MOD_SPELL_DAMAGE_DONE: 3.0,
    ITEM_MOD_SPIRIT: 0.5,                    # Molten Armor converts to crit
    ITEM_MOD_STAMINA: 1.0, ITEM_MOD_MANA: 0.5, ITEM_MOD_HEALTH: 0.5,
    ITEM_MOD_MANA_REGENERATION: 0.5,
    ITEM_MOD_SPELL_HEALING_DONE: 0.3,
    ITEM_MOD_SPELL_PENETRATION: 1.0,
    ITEM_MOD_RESILIENCE_RATING: 0.3, ITEM_MOD_HEALTH_REGEN: 0.3,
    **_CAST_IGNORE,
}

# ── Warlock (Affliction/Destro): SP > Haste > Hit(17%) > Crit ───────
_WARLOCK_WEIGHTS = {
    ITEM_MOD_SPELL_POWER: 3.5,
    ITEM_MOD_HASTE_RATING: 3.0, ITEM_MOD_HASTE_SPELL_RATING: 3.0,
    ITEM_MOD_HIT_RATING: 2.8, ITEM_MOD_HIT_SPELL_RATING: 2.8,  # 17% cap
    ITEM_MOD_CRIT_RATING: 2.0, ITEM_MOD_CRIT_SPELL_RATING: 2.0,
    ITEM_MOD_INTELLECT: 2.0,
    ITEM_MOD_SPELL_DAMAGE_DONE: 3.0,
    ITEM_MOD_SPIRIT: 1.0,                    # Fel Armor converts to SP
    ITEM_MOD_STAMINA: 1.0, ITEM_MOD_MANA: 0.5, ITEM_MOD_HEALTH: 0.5,
    ITEM_MOD_MANA_REGENERATION: 0.5,
    ITEM_MOD_SPELL_HEALING_DONE: 0.3,
    ITEM_MOD_SPELL_PENETRATION: 1.5,         # important for Affliction
    ITEM_MOD_RESILIENCE_RATING: 0.3, ITEM_MOD_HEALTH_REGEN: 0.3,
    **_CAST_IGNORE,
}

# ── Druid (Feral Cat): ArP > AGI > STR > Crit > Haste ──────────────
_DRUID_WEIGHTS = {
    ITEM_MOD_ARMOR_PENETRATION_RATING: 3.5,  # king stat for feral
    ITEM_MOD_AGILITY: 3.0,
    ITEM_MOD_STRENGTH: 2.0,
    ITEM_MOD_CRIT_RATING: 2.5, ITEM_MOD_CRIT_MELEE_RATING: 2.5,
    ITEM_MOD_HASTE_RATING: 1.5, ITEM_MOD_HASTE_MELEE_RATING: 1.5,
    ITEM_MOD_ATTACK_POWER: 2.0,
    ITEM_MOD_HIT_RATING: 2.0, ITEM_MOD_HIT_MELEE_RATING: 2.0,
    ITEM_MOD_EXPERTISE_RATING: 2.5,          # dodge cap important
    ITEM_MOD_STAMINA: 1.0, ITEM_MOD_HEALTH: 0.5,
    ITEM_MOD_DODGE_RATING: 0.5,
    ITEM_MOD_RESILIENCE_RATING: 0.3, ITEM_MOD_HEALTH_REGEN: 0.3,
    ITEM_MOD_INTELLECT: 0.1, ITEM_MOD_SPIRIT: 0.1,
    ITEM_MOD_MANA: 0.1, ITEM_MOD_MANA_REGENERATION: 0.1,
    ITEM_MOD_SPELL_POWER: 0.1, ITEM_MOD_SPELL_HEALING_DONE: 0.1,
    ITEM_MOD_SPELL_DAMAGE_DONE: 0.1, ITEM_MOD_SPELL_PENETRATION: 0.0,
    ITEM_MOD_PARRY_RATING: 0.0,              # druids can't parry
    ITEM_MOD_BLOCK_RATING: 0.0, ITEM_MOD_BLOCK_VALUE: 0.0,
    ITEM_MOD_DEFENSE_SKILL_RATING: 0.3,
    ITEM_MOD_RANGED_ATTACK_POWER: 0.0,
    ITEM_MOD_HIT_RANGED_RATING: 0.0, ITEM_MOD_CRIT_RANGED_RATING: 0.0,
    ITEM_MOD_HASTE_RANGED_RATING: 0.0,
    ITEM_MOD_CRIT_SPELL_RATING: 0.1, ITEM_MOD_HIT_SPELL_RATING: 0.1,
    ITEM_MOD_HASTE_SPELL_RATING: 0.1,
}

# Map class_id -> stat weights
CLASS_STAT_WEIGHTS = {
    CLASS_WARRIOR:      _WARRIOR_WEIGHTS,
    CLASS_PALADIN:      _PALADIN_WEIGHTS,
    CLASS_HUNTER:       _HUNTER_WEIGHTS,
    CLASS_ROGUE:        _ROGUE_WEIGHTS,
    CLASS_PRIEST:       _PRIEST_WEIGHTS,
    CLASS_DEATH_KNIGHT: _DK_WEIGHTS,
    CLASS_SHAMAN:       _SHAMAN_WEIGHTS,
    CLASS_MAGE:         _MAGE_WEIGHTS,
    CLASS_WARLOCK:      _WARLOCK_WEIGHTS,
    CLASS_DRUID:        _DRUID_WEIGHTS,
}

_DEFAULT_STAT_WEIGHT = 1.0  # fallback for unmapped stat types


def class_aware_score(item_stats: dict, item_quality: int, item_level: int,
                      armor: int, weapon_dps: float, class_id: int) -> float:
    """Compute a class-aware item score using stat weights.

    Replaces the flat TotalStats*2 with weighted stat values.
    Base components (Quality, ItemLevel, Armor, WeaponDPS) remain unchanged.

    Args:
        item_stats: Dict of {ITEM_MOD_*: value} from the item.
        item_quality: WoW quality (0-4).
        item_level: Item level.
        armor: Armor value.
        weapon_dps: Average weapon DPS.
        class_id: Player class constant (CLASS_PRIEST, etc.).

    Returns:
        Weighted item score (float).
    """
    weights = CLASS_STAT_WEIGHTS.get(class_id, {})
    weighted_stats = 0.0
    for stat_type, stat_value in item_stats.items():
        w = weights.get(stat_type, _DEFAULT_STAT_WEIGHT)
        weighted_stats += abs(stat_value) * w
    base = (item_quality * 10) + item_level + armor + weapon_dps
    return base + weighted_stats


# ─── WoW Equipment Slots (Player.h EQUIPMENT_SLOT_*) ────────────────
EQUIPMENT_SLOT_HEAD = 0
EQUIPMENT_SLOT_NECK = 1
EQUIPMENT_SLOT_SHOULDERS = 2
EQUIPMENT_SLOT_BODY = 3        # Shirt
EQUIPMENT_SLOT_CHEST = 4
EQUIPMENT_SLOT_WAIST = 5
EQUIPMENT_SLOT_LEGS = 6
EQUIPMENT_SLOT_FEET = 7
EQUIPMENT_SLOT_WRISTS = 8
EQUIPMENT_SLOT_HANDS = 9
EQUIPMENT_SLOT_FINGER1 = 10
EQUIPMENT_SLOT_FINGER2 = 11
EQUIPMENT_SLOT_TRINKET1 = 12
EQUIPMENT_SLOT_TRINKET2 = 13
EQUIPMENT_SLOT_BACK = 14
EQUIPMENT_SLOT_MAINHAND = 15
EQUIPMENT_SLOT_OFFHAND = 16
EQUIPMENT_SLOT_RANGED = 17
EQUIPMENT_SLOT_TABARD = 18
EQUIPMENT_SLOT_END = 19

# ─── WoW Bag Slots (Player.h INVENTORY_SLOT_BAG_*) ──────────────────
# 4 equippable bag slots (in addition to the default 16-slot backpack)
BAG_SLOT_START = 19   # first bag slot (after equipment)
BAG_SLOT_END = 23     # exclusive end (4 bag slots: 19, 20, 21, 22)
NUM_BAG_SLOTS = BAG_SLOT_END - BAG_SLOT_START  # 4
DEFAULT_BACKPACK_SLOTS = 16   # default WoW backpack, not removable
INVTYPE_BAG = 18              # WoW InventoryType for bags

EQUIPMENT_SLOT_NAMES = {
    EQUIPMENT_SLOT_HEAD: "Head",
    EQUIPMENT_SLOT_NECK: "Neck",
    EQUIPMENT_SLOT_SHOULDERS: "Shoulders",
    EQUIPMENT_SLOT_BODY: "Shirt",
    EQUIPMENT_SLOT_CHEST: "Chest",
    EQUIPMENT_SLOT_WAIST: "Waist",
    EQUIPMENT_SLOT_LEGS: "Legs",
    EQUIPMENT_SLOT_FEET: "Feet",
    EQUIPMENT_SLOT_WRISTS: "Wrists",
    EQUIPMENT_SLOT_HANDS: "Hands",
    EQUIPMENT_SLOT_FINGER1: "Finger 1",
    EQUIPMENT_SLOT_FINGER2: "Finger 2",
    EQUIPMENT_SLOT_TRINKET1: "Trinket 1",
    EQUIPMENT_SLOT_TRINKET2: "Trinket 2",
    EQUIPMENT_SLOT_BACK: "Back",
    EQUIPMENT_SLOT_MAINHAND: "Main Hand",
    EQUIPMENT_SLOT_OFFHAND: "Off Hand",
    EQUIPMENT_SLOT_RANGED: "Ranged",
    EQUIPMENT_SLOT_TABARD: "Tabard",
}

# Mapping: WoW InventoryType -> valid equipment slot(s)
# Dual entries (rings, trinkets) allow filling either slot.
INVTYPE_TO_SLOTS = {
    1:  [EQUIPMENT_SLOT_HEAD],
    2:  [EQUIPMENT_SLOT_NECK],
    3:  [EQUIPMENT_SLOT_SHOULDERS],
    4:  [EQUIPMENT_SLOT_BODY],
    5:  [EQUIPMENT_SLOT_CHEST],
    6:  [EQUIPMENT_SLOT_WAIST],
    7:  [EQUIPMENT_SLOT_LEGS],
    8:  [EQUIPMENT_SLOT_FEET],
    9:  [EQUIPMENT_SLOT_WRISTS],
    10: [EQUIPMENT_SLOT_HANDS],
    11: [EQUIPMENT_SLOT_FINGER1, EQUIPMENT_SLOT_FINGER2],   # Ring
    12: [EQUIPMENT_SLOT_TRINKET1, EQUIPMENT_SLOT_TRINKET2], # Trinket
    13: [EQUIPMENT_SLOT_MAINHAND],   # One-Hand
    14: [EQUIPMENT_SLOT_OFFHAND],    # Shield
    15: [EQUIPMENT_SLOT_RANGED],     # Ranged (Bow/Gun)
    16: [EQUIPMENT_SLOT_BACK],       # Cloak
    17: [EQUIPMENT_SLOT_MAINHAND],   # Two-Hand (clears offhand)
    19: [EQUIPMENT_SLOT_TABARD],
    20: [EQUIPMENT_SLOT_CHEST],      # Robe = Chest
    21: [EQUIPMENT_SLOT_MAINHAND],   # Main Hand only
    22: [EQUIPMENT_SLOT_OFFHAND],    # Off Hand (misc)
    23: [EQUIPMENT_SLOT_OFFHAND],    # Holdable (off-hand frill)
    25: [EQUIPMENT_SLOT_RANGED],     # Thrown
    26: [EQUIPMENT_SLOT_RANGED],     # Ranged Right (Wand)
    28: [EQUIPMENT_SLOT_RANGED],     # Relic
}

INVTYPE_TWO_HAND = 17  # Two-hand weapons clear offhand on equip


# ─── WotLK Base Stats per Class (from PlayerClassLevelInfo at L1) ────
# {class_id: (strength, agility, stamina, intellect, spirit)}
# Fallback values — overridden by player_class_stats.csv when available.
CLASS_BASE_STATS = {
    CLASS_WARRIOR:      (23, 20, 22, 20, 20),
    CLASS_PALADIN:      (22, 20, 22, 20, 21),
    CLASS_HUNTER:       (20, 23, 22, 20, 21),
    CLASS_ROGUE:        (21, 23, 21, 20, 20),
    CLASS_PRIEST:       (20, 20, 20, 22, 23),
    CLASS_DEATH_KNIGHT: (24, 16, 23, 11, 18),
    CLASS_SHAMAN:       (21, 20, 21, 21, 22),
    CLASS_MAGE:         (20, 20, 20, 24, 22),
    CLASS_WARLOCK:      (20, 20, 21, 22, 23),
    CLASS_DRUID:        (21, 20, 20, 22, 23),
}

# Base HP / Mana at level 1 per class (from PlayerClassLevelInfo)
# (base_hp, base_mana) — the "create" values before stamina/intellect bonus.
# Fallback values — overridden by player_class_stats.csv when available.
CLASS_BASE_HP_MANA = {
    CLASS_WARRIOR:      (20, 0),
    CLASS_PALADIN:      (28, 60),
    CLASS_HUNTER:       (36, 65),
    CLASS_ROGUE:        (25, 0),
    CLASS_PRIEST:       (52, 73),
    CLASS_DEATH_KNIGHT: (130, 0),
    CLASS_SHAMAN:       (37, 55),
    CLASS_MAGE:         (32, 100),
    CLASS_WARLOCK:      (23, 90),
    CLASS_DRUID:        (34, 60),
}

# Per-class per-level stat table loaded from CSV (populated at module init).
# {(class_id, level): (base_hp, base_mana, str, agi, sta, int, spi)}
# When loaded, class_base_stat() and player_max_hp/mana use this directly.
PLAYER_CLASS_LEVEL_STATS = None

# HP gain per level — fallback only (unused when CSV is loaded)
CLASS_HP_PER_LEVEL = {
    CLASS_WARRIOR: 60, CLASS_PALADIN: 55, CLASS_HUNTER: 46,
    CLASS_ROGUE: 45, CLASS_PRIEST: 50, CLASS_DEATH_KNIGHT: 60,
    CLASS_SHAMAN: 52, CLASS_MAGE: 42, CLASS_WARLOCK: 48, CLASS_DRUID: 50,
}

# Mana gain per level — fallback only (unused when CSV is loaded)
CLASS_MANA_PER_LEVEL = {
    CLASS_PALADIN: 18, CLASS_HUNTER: 15, CLASS_PRIEST: 5,
    CLASS_SHAMAN: 17, CLASS_MAGE: 22, CLASS_WARLOCK: 20, CLASS_DRUID: 18,
}


# ─── WotLK Melee Crit from Agility (from GtChanceToMeleeCrit.dbc) ───
# {class_id: (base_crit_fraction, ratio_L1, ratio_L80)}
# base is from GtChanceToMeleeCritBase, ratio from GtChanceToMeleeCrit
# C++ formula: (base + agi * ratio) * 100.0f → percentage
# Fallback values — corrected from DBC. Overridden by per-level DBC tables.
GT_MELEE_CRIT = {
    CLASS_WARRIOR:      (0.031891, 0.002587, 0.000160),
    CLASS_PALADIN:      (0.032685, 0.002164, 0.000192),
    CLASS_HUNTER:       (-0.015320, 0.002840, 0.000120),
    CLASS_ROGUE:        (-0.002950, 0.004476, 0.000120),
    CLASS_PRIEST:       (0.031765, 0.000912, 0.000192),
    CLASS_DEATH_KNIGHT: (0.031891, 0.002587, 0.000160),
    CLASS_SHAMAN:       (0.029220, 0.001039, 0.000120),
    CLASS_MAGE:         (0.034540, 0.000773, 0.000196),
    CLASS_WARLOCK:      (0.026220, 0.001189, 0.000198),
    CLASS_DRUID:        (0.074755, 0.001262, 0.000120),
}

# ─── WotLK Spell Crit from Intellect (from GtChanceToSpellCrit.dbc) ─
# {class_id: (base_crit_fraction, ratio_L1, ratio_L80)}
# C++ formula: (base + int * ratio) * 100.0f → percentage
# Fallback values — corrected from DBC. Overridden by per-level DBC tables.
GT_SPELL_CRIT = {
    CLASS_WARRIOR:      (0.0000, 0.000000, 0.000000),
    CLASS_PALADIN:      (0.033355, 0.000832, 0.000060),
    CLASS_HUNTER:       (0.036020, 0.000699, 0.000060),
    CLASS_ROGUE:        (0.0000, 0.000000, 0.000000),
    CLASS_PRIEST:       (0.012375, 0.001710, 0.000060),
    CLASS_DEATH_KNIGHT: (0.0000, 0.000000, 0.000000),
    CLASS_SHAMAN:       (0.022010, 0.001333, 0.000060),
    CLASS_MAGE:         (0.009075, 0.001637, 0.000060),
    CLASS_WARLOCK:      (0.017000, 0.001500, 0.000060),
    CLASS_DRUID:        (0.018515, 0.000000, 0.000000),
}

# Per-level DBC lookup tables (populated at module init if DBC files found)
GT_MELEE_CRIT_TABLE = None      # {(class_id, level): crit_per_agi}
GT_MELEE_CRIT_BASE_TABLE = None # {class_id: base_crit_fraction}
GT_SPELL_CRIT_TABLE = None      # {(class_id, level): crit_per_int}
GT_SPELL_CRIT_BASE_TABLE = None # {class_id: base_crit_fraction}
GT_COMBAT_RATINGS = None        # {(cr_type, level): rating_per_pct}
GT_REGEN_MP_PER_SPT_TABLE = None  # {(class_id, level): coeff}
GT_REGEN_HP_PER_SPT_TABLE = None  # {(class_id, level): coeff}

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
# {class_id: (coeff_L1, coeff_L80)} — fallback when DBC not loaded.
# Corrected: all mana classes share the same DBC values (0.062937 at L1).
# C++ formula: sqrt(int) * spirit * coeff → mana per second (OOC)
GT_REGEN_MP_PER_SPT = {
    CLASS_PALADIN:  (0.062937, 0.003345),
    CLASS_HUNTER:   (0.062937, 0.003345),
    CLASS_PRIEST:   (0.062937, 0.003345),
    CLASS_SHAMAN:   (0.062937, 0.003345),
    CLASS_MAGE:     (0.062937, 0.003345),
    CLASS_WARLOCK:  (0.062937, 0.003345),
    CLASS_DRUID:    (0.062937, 0.003345),
}

# ─── WotLK Spirit->HP Regen (from GtRegenHPPerSpt.dbc) ──────────────
# {class_id: (coeff_L1, coeff_L80)} — fallback when DBC not loaded.
GT_HP_REGEN_PER_SPT = {
    CLASS_WARRIOR: (1.500000, 0.500000),
    CLASS_PALADIN: (0.375000, 0.125000),
    CLASS_HUNTER:  (0.375000, 0.125000),
    CLASS_ROGUE:   (1.000000, 0.333333),
    CLASS_PRIEST:  (0.125000, 0.041667),
    CLASS_DEATH_KNIGHT: (1.500000, 0.500000),
    CLASS_SHAMAN:  (0.214286, 0.071429),
    CLASS_MAGE:    (0.125000, 0.041667),
    CLASS_WARLOCK: (0.136364, 0.045455),
    CLASS_DRUID:   (0.0, 0.0),
}


# ─── WotLK Combat Rating Conversions (from GtCombatRatings.dbc) ─────
# Rating needed for 1% at level 80. Used as fallback when DBC not loaded.
# Corrected from actual DBC values (Dodge/Parry/Resilience/ArP were wrong).
CR_DEFENSE_L80 = 4.918498
CR_DODGE_L80 = 45.250187
CR_PARRY_L80 = 45.250187
CR_BLOCK_L80 = 16.394995
CR_HIT_MELEE_L80 = 32.789989
CR_HIT_RANGED_L80 = 32.789989
CR_HIT_SPELL_L80 = 26.231993
CR_CRIT_MELEE_L80 = 45.905987
CR_CRIT_RANGED_L80 = 45.905987
CR_CRIT_SPELL_L80 = 45.905987
CR_HASTE_MELEE_L80 = 32.789989
CR_HASTE_RANGED_L80 = 32.789989
CR_HASTE_SPELL_L80 = 32.789989
CR_EXPERTISE_L80 = 8.197496
CR_ARMOR_PENETRATION_L80 = 15.395300
CR_RESILIENCE_L80 = 94.271225

# Combat rating type enum (from SharedDefines.h, used with GT_COMBAT_RATINGS)
from sim.dbc_loader import (  # noqa: E402
    CR_DEFENSE_SKILL as CR_TYPE_DEFENSE,
    CR_DODGE as CR_TYPE_DODGE,
    CR_PARRY as CR_TYPE_PARRY,
    CR_BLOCK as CR_TYPE_BLOCK,
    CR_HIT_MELEE as CR_TYPE_HIT_MELEE,
    CR_HIT_RANGED as CR_TYPE_HIT_RANGED,
    CR_HIT_SPELL as CR_TYPE_HIT_SPELL,
    CR_CRIT_MELEE as CR_TYPE_CRIT_MELEE,
    CR_CRIT_RANGED as CR_TYPE_CRIT_RANGED,
    CR_CRIT_SPELL as CR_TYPE_CRIT_SPELL,
    CR_RESILIENCE as CR_TYPE_RESILIENCE,
    CR_HASTE_MELEE as CR_TYPE_HASTE_MELEE,
    CR_HASTE_RANGED as CR_TYPE_HASTE_RANGED,
    CR_HASTE_SPELL as CR_TYPE_HASTE_SPELL,
    CR_EXPERTISE as CR_TYPE_EXPERTISE,
    CR_ARMOR_PENETRATION as CR_TYPE_ARMOR_PEN,
)


# ─── WotLK Spell Power Coefficients (from spell_bonus_data DB) ──────
# Per-rank SP coefficients from AzerothCore spell_bonus_data table.
# Maps spell_id -> (direct_bonus, dot_bonus_per_tick).
SP_COEFFICIENTS = {
    # Smite (R1-R8): coefficient scales up with cast time
    585: (0.1230, 0.0),    # R1 1.5s cast
    591: (0.2710, 0.0),    # R2 2.0s cast
    598: (0.5540, 0.0),    # R3 2.5s cast
    984: (0.7140, 0.0),    # R4 2.5s
    1004: (0.7140, 0.0),   # R5
    6060: (0.7140, 0.0),   # R6
    10933: (0.7140, 0.0),  # R7
    10934: (0.7140, 0.0),  # R8
    # Lesser Heal (R1-R3)
    2050: (0.2310, 0.0),   # R1 1.5s
    2052: (0.4310, 0.0),   # R2 2.0s
    2053: (0.7550, 0.0),   # R3 2.5s
    # Heal (R1-R4)
    2054: (1.3700, 0.0),   # R1 3.0s
    2055: (1.3700, 0.0),   # R2
    6063: (1.3700, 0.0),   # R3
    6064: (1.3700, 0.0),   # R4
    # Greater Heal (R1-R5)
    2060: (1.6110, 0.0),   # R1 3.0s
    10963: (1.6110, 0.0),  # R2
    10964: (1.6110, 0.0),  # R3
    10965: (1.6110, 0.0),  # R4
    25314: (1.6110, 0.0),  # R5
    # Flash Heal (R1-R7)
    2061: (0.8057, 0.0),   # R1 1.5s
    9472: (0.8057, 0.0),   # R2
    9473: (0.8057, 0.0),   # R3
    9474: (0.8057, 0.0),   # R4
    10915: (0.8057, 0.0),  # R5
    10916: (0.8057, 0.0),  # R6
    10917: (0.8057, 0.0),  # R7
    # Shadow Word: Pain (R1-R8)
    589: (0.0, 0.1833),    # R1
    594: (0.0, 0.1833),    # R2
    970: (0.0, 0.1833),    # R3
    992: (0.0, 0.1833),    # R4
    2767: (0.0, 0.1833),   # R5
    10892: (0.0, 0.1833),  # R6
    10893: (0.0, 0.1833),  # R7
    10894: (0.0, 0.1833),  # R8
    # Power Word: Shield (R1-R10) — uses DBC BonusMult, no spell_bonus_data entry
    17: (0.8068, 0.0),     # R1
    592: (0.8068, 0.0),    # R2
    600: (0.8068, 0.0),    # R3
    3747: (0.8068, 0.0),   # R4
    6065: (0.8068, 0.0),   # R5
    6066: (0.8068, 0.0),   # R6
    10898: (0.8068, 0.0),  # R7
    10899: (0.8068, 0.0),  # R8
    10900: (0.8068, 0.0),  # R9
    10901: (0.8068, 0.0),  # R10
    # Mind Blast (R1-R9)
    8092: (0.2680, 0.0),   # R1
    8102: (0.3640, 0.0),   # R2
    8103: (0.4286, 0.0),   # R3
    8104: (0.4286, 0.0),   # R4
    8105: (0.4286, 0.0),   # R5
    8106: (0.4286, 0.0),   # R6
    10945: (0.4286, 0.0),  # R7
    10946: (0.4286, 0.0),  # R8
    10947: (0.4286, 0.0),  # R9
    # Renew (R1-R10)
    139: (0.0, 0.2070),    # R1
    6074: (0.0, 0.2910),   # R2
    6075: (0.0, 0.3760),   # R3
    6076: (0.0, 0.3760),   # R4
    6077: (0.0, 0.3760),   # R5
    6078: (0.0, 0.3760),   # R6
    10927: (0.0, 0.3760),  # R7
    10928: (0.0, 0.3760),  # R8
    10929: (0.0, 0.3760),  # R9
    25315: (0.0, 0.3760),  # R10
    # Holy Fire (R1-R8) — direct + DoT
    14914: (0.5710, 0.0529),  # R1
    15262: (0.5710, 0.0529),  # R2
    15263: (0.5710, 0.0529),  # R3
    15264: (0.5710, 0.0529),  # R4
    15265: (0.5710, 0.0529),  # R5
    15266: (0.5710, 0.0529),  # R6
    15267: (0.5710, 0.0529),  # R7
    15261: (0.5710, 0.0529),  # R8
    # Inner Fire (R1-R6) — no SP coefficient (buff, not damage/heal)
    588: (0.0, 0.0),
    7128: (0.0, 0.0),
    602: (0.0, 0.0),
    1006: (0.0, 0.0),
    10951: (0.0, 0.0),
    10952: (0.0, 0.0),
    # PW:Fortitude (R1-R6) — no SP coefficient
    1243: (0.0, 0.0),
    1244: (0.0, 0.0),
    1245: (0.0, 0.0),
    2791: (0.0, 0.0),
    10937: (0.0, 0.0),
    10938: (0.0, 0.0),
}

# Legacy SP_COEFF_* aliases for backward compatibility
SP_COEFF_SMITE = 0.1230
SP_COEFF_HEAL = 0.2310
SP_COEFF_MIND_BLAST = 0.2680
SP_COEFF_SW_PAIN_TICK = 0.1833
SP_COEFF_PW_SHIELD = 0.8068
SP_COEFF_RENEW_TICK = 0.2070
SP_COEFF_HOLY_FIRE = 0.5710
SP_COEFF_HOLY_FIRE_DOT_TICK = 0.0529

def get_sp_coeff(spell_id):
    """Get (direct_bonus, dot_bonus_per_tick) for a spell ID."""
    return SP_COEFFICIENTS.get(spell_id, (0.0, 0.0))

# ─── Spell Families and Ranks ──────────────────────────────────────
# Family ID = first (R1) spell ID. Ranks sorted by level requirement.
FAMILY_SMITE = 585
FAMILY_HEAL = 2050       # Lesser Heal → Heal → Greater Heal (unified healing line)
FAMILY_FLASH_HEAL = 2061
FAMILY_SW_PAIN = 589
FAMILY_PW_SHIELD = 17
FAMILY_MIND_BLAST = 8092
FAMILY_RENEW = 139
FAMILY_HOLY_FIRE = 14914
FAMILY_INNER_FIRE = 588
FAMILY_FORTITUDE = 1243

# Each entry: (train_level, spell_id)
SPELL_RANKS = {
    FAMILY_SMITE: [
        (1, 585), (6, 591), (14, 598), (22, 984), (30, 1004),
        (38, 6060), (46, 10933), (54, 10934),
    ],
    FAMILY_HEAL: [  # Lesser Heal → Heal → Greater Heal
        (1, 2050), (4, 2052), (10, 2053),           # Lesser Heal R1-R3
        (16, 2054), (22, 2055), (28, 6063), (34, 6064),  # Heal R1-R4
        (40, 2060), (46, 10963), (52, 10964), (58, 10965), (60, 25314),  # Greater Heal R1-R5
    ],
    FAMILY_FLASH_HEAL: [
        (20, 2061), (26, 9472), (32, 9473), (38, 9474),
        (44, 10915), (50, 10916), (56, 10917),
    ],
    FAMILY_SW_PAIN: [
        (4, 589), (10, 594), (18, 970), (26, 992), (34, 2767),
        (42, 10892), (50, 10893), (58, 10894),
    ],
    FAMILY_PW_SHIELD: [
        (6, 17), (12, 592), (18, 600), (24, 3747), (30, 6065),
        (36, 6066), (42, 10898), (48, 10899), (54, 10900), (60, 10901),
    ],
    FAMILY_MIND_BLAST: [
        (10, 8092), (16, 8102), (22, 8103), (28, 8104), (34, 8105),
        (40, 8106), (46, 10945), (52, 10946), (58, 10947),
    ],
    FAMILY_RENEW: [
        (8, 139), (14, 6074), (20, 6075), (26, 6076), (32, 6077),
        (38, 6078), (44, 10927), (50, 10928), (56, 10929), (60, 25315),
    ],
    FAMILY_HOLY_FIRE: [
        (20, 14914), (24, 15262), (30, 15263), (36, 15264),
        (42, 15265), (48, 15266), (54, 15267), (60, 15261),
    ],
    FAMILY_INNER_FIRE: [
        (12, 588), (20, 7128), (30, 602), (40, 1006),
        (50, 10951), (60, 10952),
    ],
    FAMILY_FORTITUDE: [
        (1, 1243), (12, 1244), (24, 1245), (36, 2791),
        (48, 10937), (60, 10938),
    ],
}

# Set of all spell IDs across all ranks
ALL_RANKED_SPELL_IDS = set()
for _ranks in SPELL_RANKS.values():
    for _lvl, _sid in _ranks:
        ALL_RANKED_SPELL_IDS.add(_sid)

def get_best_rank(family_id, player_level):
    """Return the highest-rank spell_id available at the given player level.

    Returns None if the player hasn't learned any rank yet.
    """
    ranks = SPELL_RANKS.get(family_id)
    if not ranks:
        return None
    best = None
    for lvl_req, spell_id in ranks:
        if player_level >= lvl_req:
            best = spell_id
    return best

# Spell level requirements from trainer_spell table (all ranks)
SPELL_LEVEL_REQ = {}
for _ranks in SPELL_RANKS.values():
    for _lvl, _sid in _ranks:
        SPELL_LEVEL_REQ[_sid] = _lvl

# Spell mana cost as % of class base mana (from Spell.dbc ManaCostPercentage)
SPELL_MANA_PCT = {
    # Smite: R1=9%, R2=12%, R3+=15%
    585: 9, 591: 12, 598: 15, 984: 15, 1004: 15,
    6060: 15, 10933: 15, 10934: 15,
    # Lesser Heal: R1=16%, R2=21%, R3=27%
    2050: 16, 2052: 21, 2053: 27,
    # Heal: all 32%
    2054: 32, 2055: 32, 6063: 32, 6064: 32,
    # Greater Heal: all 32%
    2060: 32, 10963: 32, 10964: 32, 10965: 32, 25314: 32,
    # Flash Heal: all 18%
    2061: 18, 9472: 18, 9473: 18, 9474: 18,
    10915: 18, 10916: 18, 10917: 18,
    # SW:Pain: all 22%
    589: 22, 594: 22, 970: 22, 992: 22, 2767: 22,
    10892: 22, 10893: 22, 10894: 22,
    # PW:Shield: all 23%
    17: 23, 592: 23, 600: 23, 3747: 23, 6065: 23,
    6066: 23, 10898: 23, 10899: 23, 10900: 23, 10901: 23,
    # Mind Blast: all 17%
    8092: 17, 8102: 17, 8103: 17, 8104: 17, 8105: 17,
    8106: 17, 10945: 17, 10946: 17, 10947: 17,
    # Renew: all 17%
    139: 17, 6074: 17, 6075: 17, 6076: 17, 6077: 17,
    6078: 17, 10927: 17, 10928: 17, 10929: 17, 25315: 17,
    # Holy Fire: all 11%
    14914: 11, 15262: 11, 15263: 11, 15264: 11,
    15265: 11, 15266: 11, 15267: 11, 15261: 11,
    # Inner Fire: all 14%
    588: 14, 7128: 14, 602: 14, 1006: 14, 10951: 14, 10952: 14,
    # PW:Fortitude: all 27%
    1243: 27, 1244: 27, 1245: 27, 2791: 27, 10937: 27, 10938: 27,
}


# ─── Load DBC/CSV data at import time ────────────────────────────────
# Auto-discovers data/ directory relative to this module's location.
# If data files are found, the module-level lookup tables are populated
# and formulas.py will use them for exact per-level values.

def _init_dbc_tables():
    """Try to load DBC/CSV tables from repo data/ directory."""
    global PLAYER_CLASS_LEVEL_STATS
    global GT_COMBAT_RATINGS, GT_MELEE_CRIT_TABLE, GT_MELEE_CRIT_BASE_TABLE
    global GT_SPELL_CRIT_TABLE, GT_SPELL_CRIT_BASE_TABLE
    global GT_REGEN_MP_PER_SPT_TABLE, GT_REGEN_HP_PER_SPT_TABLE
    global CLASS_BASE_STATS, CLASS_BASE_HP_MANA

    from sim.dbc_loader import load_all_dbc_tables

    # Look for data/ relative to this file: python/sim/constants.py -> ../../data
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'data')
    data_dir = os.path.normpath(data_dir)

    if not os.path.isdir(data_dir):
        return

    tables = load_all_dbc_tables(data_dir)

    # Player class-level stat table (per-class per-level base stats)
    if tables['player_class_stats']:
        PLAYER_CLASS_LEVEL_STATS = tables['player_class_stats']
        # Update CLASS_BASE_STATS and CLASS_BASE_HP_MANA from the loaded L1 data
        for cid in [CLASS_WARRIOR, CLASS_PALADIN, CLASS_HUNTER, CLASS_ROGUE,
                    CLASS_PRIEST, CLASS_DEATH_KNIGHT, CLASS_SHAMAN, CLASS_MAGE,
                    CLASS_WARLOCK, CLASS_DRUID]:
            key = (cid, 1)
            if key in PLAYER_CLASS_LEVEL_STATS:
                hp, mana, s, a, st, i, sp = PLAYER_CLASS_LEVEL_STATS[key]
                CLASS_BASE_STATS[cid] = (s, a, st, i, sp)
                CLASS_BASE_HP_MANA[cid] = (hp, mana)

    # GameTable DBC tables
    if tables['gt_combat_ratings']:
        GT_COMBAT_RATINGS = tables['gt_combat_ratings']
    if tables['gt_melee_crit']:
        GT_MELEE_CRIT_TABLE = tables['gt_melee_crit']
    if tables['gt_melee_crit_base']:
        GT_MELEE_CRIT_BASE_TABLE = tables['gt_melee_crit_base']
    if tables['gt_spell_crit']:
        GT_SPELL_CRIT_TABLE = tables['gt_spell_crit']
    if tables['gt_spell_crit_base']:
        GT_SPELL_CRIT_BASE_TABLE = tables['gt_spell_crit_base']
    if tables['gt_regen_mp_per_spt']:
        GT_REGEN_MP_PER_SPT_TABLE = tables['gt_regen_mp_per_spt']
    if tables['gt_regen_hp_per_spt']:
        GT_REGEN_HP_PER_SPT_TABLE = tables['gt_regen_hp_per_spt']


_init_dbc_tables()

