"""
WoW WotLK 3.3.5 Constants and DBC Tables.

All game constants, class definitions, stat weights, equipment slots,
base stats, combat rating tables, and spell power coefficients.
Derived from AzerothCore C++ source (SharedDefines.h, Player.h,
StatSystem.cpp, GtCombatRatings.dbc, spell_bonus_data).
"""

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

