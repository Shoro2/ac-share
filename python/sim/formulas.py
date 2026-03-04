"""
WoW WotLK 3.3.5 Stat Calculation Formulas.

Pure functions for computing derived stats: HP/Mana from primary stats,
combat rating conversions, spell damage/heal calculations, Spirit regen,
and attack power formulas. All formulas derived from AzerothCore C++ source.

When DBC/CSV data files are loaded (via constants.py at import time), these
functions use exact per-level lookup tables. Otherwise they fall back to
corrected hardcoded approximations.
"""

import math

import sim.constants as _c
from sim.constants import (
    CLASS_PRIEST, CLASS_WARRIOR, CLASS_PALADIN, CLASS_HUNTER, CLASS_ROGUE,
    CLASS_DEATH_KNIGHT, CLASS_SHAMAN, CLASS_MAGE, CLASS_WARLOCK, CLASS_DRUID,
    CLASS_BASE_STATS, CLASS_BASE_HP_MANA, CLASS_HP_PER_LEVEL, CLASS_MANA_PER_LEVEL,
    CLASS_POWER_TYPE, POWER_MANA,
    GT_MELEE_CRIT, GT_SPELL_CRIT, DODGE_BASE, CRIT_TO_DODGE, DR_K,
    DODGE_CAP, PARRY_CAP, GT_REGEN_MP_PER_SPT, GT_HP_REGEN_PER_SPT,
    CR_DEFENSE_L80, CR_DODGE_L80, CR_PARRY_L80, CR_BLOCK_L80,
    CR_HIT_MELEE_L80, CR_HIT_RANGED_L80, CR_HIT_SPELL_L80,
    CR_CRIT_MELEE_L80, CR_CRIT_RANGED_L80, CR_CRIT_SPELL_L80,
    CR_HASTE_MELEE_L80, CR_HASTE_RANGED_L80, CR_HASTE_SPELL_L80,
    CR_EXPERTISE_L80, CR_ARMOR_PENETRATION_L80, CR_RESILIENCE_L80,
    SP_COEFF_SMITE, SP_COEFF_HEAL, SP_COEFF_MIND_BLAST,
    SP_COEFF_SW_PAIN_TICK, SP_COEFF_PW_SHIELD, SP_COEFF_RENEW_TICK,
    SP_COEFF_HOLY_FIRE, SP_COEFF_HOLY_FIRE_DOT_TICK,
)
from sim.dbc_loader import (
    CR_DEFENSE_SKILL as _CR_DEFENSE,
    CR_DODGE as _CR_DODGE,
    CR_PARRY as _CR_PARRY,
    CR_BLOCK as _CR_BLOCK,
    CR_HIT_MELEE as _CR_HIT_MELEE,
    CR_HIT_RANGED as _CR_HIT_RANGED,
    CR_HIT_SPELL as _CR_HIT_SPELL,
    CR_CRIT_MELEE as _CR_CRIT_MELEE,
    CR_CRIT_RANGED as _CR_CRIT_RANGED,
    CR_CRIT_SPELL as _CR_CRIT_SPELL,
    CR_RESILIENCE as _CR_RESILIENCE,
    CR_HASTE_MELEE as _CR_HASTE_MELEE,
    CR_HASTE_RANGED as _CR_HASTE_RANGED,
    CR_HASTE_SPELL as _CR_HASTE_SPELL,
    CR_EXPERTISE as _CR_EXPERTISE,
    CR_ARMOR_PENETRATION as _CR_ARMOR_PEN,
)


# ─── Per-level stat scaling ──────────────────────────────────────────

def class_base_stat(class_id: int, stat_index: int, level: int) -> int:
    """Base primary stat at given level for any class.

    stat_index: 0=str, 1=agi, 2=stam, 3=int, 4=spi
    Uses exact per-level values from player_class_stats.csv when loaded,
    otherwise falls back to base + (level-1) approximation.
    """
    tbl = _c.PLAYER_CLASS_LEVEL_STATS
    if tbl is not None:
        key = (class_id, min(level, 80))
        if key in tbl:
            # tuple: (base_hp, base_mana, str, agi, sta, int, spi)
            return tbl[key][stat_index + 2]
    # Fallback
    base = CLASS_BASE_STATS.get(class_id, CLASS_BASE_STATS[CLASS_PRIEST])
    return base[stat_index] + (level - 1)


def player_max_hp(level: int, bonus_stamina: int = 0, bonus_hp: int = 0,
                  class_id: int = CLASS_PRIEST) -> int:
    """HP with WotLK Stamina formula (StatSystem.cpp:GetHealthBonusFromStamina).

    base_hp from PlayerClassLevelInfo (CSV), then:
    First 20 stamina = 1 HP each, above 20 = 10 HP each.
    bonus_stamina is gear stamina only (base stamina from level table).
    """
    tbl = _c.PLAYER_CLASS_LEVEL_STATS
    if tbl is not None:
        key = (class_id, min(level, 80))
        if key in tbl:
            base_hp = tbl[key][0]  # BaseHP from CSV
            total_stam = tbl[key][4] + bonus_stamina  # CSV sta + gear sta
            stam_hp = min(total_stam, 20) + max(total_stam - 20, 0) * 10
            return base_hp + stam_hp + bonus_hp
    # Fallback: old formula
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
    tbl = _c.PLAYER_CLASS_LEVEL_STATS
    if tbl is not None:
        key = (class_id, min(level, 80))
        if key in tbl:
            base_mana = tbl[key][1]  # BaseMana from CSV
            total_int = tbl[key][5] + bonus_intellect  # CSV int + gear int
            int_mana = min(total_int, 20) + max(total_int - 20, 0) * 15
            return base_mana + int_mana + bonus_mana
    # Fallback
    base_mana_data = CLASS_BASE_HP_MANA.get(class_id, CLASS_BASE_HP_MANA[CLASS_PRIEST])
    mana_per_lvl = CLASS_MANA_PER_LEVEL.get(class_id, 5)
    base_mana = base_mana_data[1] + (level - 1) * mana_per_lvl
    total_int = class_base_stat(class_id, 3, level) + bonus_intellect
    int_mana = min(total_int, 20) + max(total_int - 20, 0) * 15
    return base_mana + int_mana + bonus_mana


# ─── GtCombatRatings (from GtCombatRatings.dbc) ──────────────────────

# Fallback level scaling curve (used only when DBC not loaded)
_RATING_LEVEL_SCALE = (
    (1, 0.0125), (10, 0.0125),
    (20, 0.05), (30, 0.10),
    (40, 0.17), (50, 0.25),
    (60, 0.305), (70, 0.481),
    (80, 1.00),
)


def _rating_to_pct(rating, l80_value_or_cr_type, level):
    """Convert combat rating to percentage bonus.

    When GT_COMBAT_RATINGS is loaded from DBC, l80_value_or_cr_type can be
    the CR_TYPE_* enum for exact per-level lookup. Otherwise uses the L80
    anchor value with interpolated scaling curve.
    """
    if rating <= 0:
        return 0.0

    # Try DBC lookup first
    gt = _c.GT_COMBAT_RATINGS
    if gt is not None:
        # Determine CR type — if it's a float, map it to the enum
        cr_type = l80_value_or_cr_type
        if isinstance(cr_type, float):
            cr_type = _l80_to_cr_type(cr_type)
        if cr_type is not None:
            lvl = min(max(level, 1), 100)
            val = gt.get((cr_type, lvl))
            if val is not None and val > 0:
                return rating / val

    # Fallback: interpolated scaling from L80 anchor
    l80_value = l80_value_or_cr_type
    if not isinstance(l80_value, (int, float)):
        return 0.0
    scale = _RATING_LEVEL_SCALE
    if level <= scale[0][0]:
        factor = scale[0][1]
    elif level >= scale[-1][0]:
        factor = scale[-1][1]
    else:
        factor = scale[-1][1]
        for i in range(len(scale) - 1):
            if scale[i][0] <= level <= scale[i + 1][0]:
                t = (level - scale[i][0]) / (scale[i + 1][0] - scale[i][0])
                factor = scale[i][1] + t * (scale[i + 1][1] - scale[i][1])
                break
    rating_per_pct = max(0.01, l80_value * factor)
    return rating / rating_per_pct


# Map L80 anchor floats to CR type enums (for backward compat with old callers)
_L80_TO_CR = None

def _l80_to_cr_type(l80_value):
    """Map an L80 anchor value to CR type enum."""
    global _L80_TO_CR
    if _L80_TO_CR is None:
        _L80_TO_CR = {
            CR_DEFENSE_L80: _CR_DEFENSE,
            CR_DODGE_L80: _CR_DODGE,
            CR_PARRY_L80: _CR_PARRY,
            CR_BLOCK_L80: _CR_BLOCK,
            CR_HIT_MELEE_L80: _CR_HIT_MELEE,
            CR_HIT_RANGED_L80: _CR_HIT_RANGED,
            CR_HIT_SPELL_L80: _CR_HIT_SPELL,
            CR_CRIT_MELEE_L80: _CR_CRIT_MELEE,
            CR_CRIT_RANGED_L80: _CR_CRIT_RANGED,
            CR_CRIT_SPELL_L80: _CR_CRIT_SPELL,
            CR_HASTE_MELEE_L80: _CR_HASTE_MELEE,
            CR_HASTE_RANGED_L80: _CR_HASTE_RANGED,
            CR_HASTE_SPELL_L80: _CR_HASTE_SPELL,
            CR_EXPERTISE_L80: _CR_EXPERTISE,
            CR_ARMOR_PENETRATION_L80: _CR_ARMOR_PEN,
            CR_RESILIENCE_L80: _CR_RESILIENCE,
        }
    return _L80_TO_CR.get(l80_value)


# ─── Crit from stats ─────────────────────────────────────────────────

def _get_melee_crit_from_agi(class_id, level):
    """Get melee crit-per-agility ratio from DBC or fallback."""
    tbl = _c.GT_MELEE_CRIT_TABLE
    if tbl is not None:
        val = tbl.get((class_id, min(level, 80)))
        if val is not None:
            return val
    # Fallback: lerp L1..L80
    gt = GT_MELEE_CRIT.get(class_id, GT_MELEE_CRIT[CLASS_PRIEST])
    t = min((level - 1) / 79.0, 1.0)
    return gt[1] * (1 - t) + gt[2] * t


def _get_melee_crit_base(class_id):
    """Get base melee crit fraction from DBC or fallback."""
    tbl = _c.GT_MELEE_CRIT_BASE_TABLE
    if tbl is not None:
        val = tbl.get(class_id)
        if val is not None:
            return val
    gt = GT_MELEE_CRIT.get(class_id, GT_MELEE_CRIT[CLASS_PRIEST])
    return gt[0]


def _get_spell_crit_from_int(class_id, level):
    """Get spell crit-per-intellect ratio from DBC or fallback."""
    tbl = _c.GT_SPELL_CRIT_TABLE
    if tbl is not None:
        val = tbl.get((class_id, min(level, 80)))
        if val is not None:
            return val
    gt = GT_SPELL_CRIT.get(class_id, GT_SPELL_CRIT[CLASS_PRIEST])
    t = min((level - 1) / 79.0, 1.0)
    return gt[1] * (1 - t) + gt[2] * t


def _get_spell_crit_base(class_id):
    """Get base spell crit fraction from DBC or fallback."""
    tbl = _c.GT_SPELL_CRIT_BASE_TABLE
    if tbl is not None:
        val = tbl.get(class_id)
        if val is not None:
            return val
    gt = GT_SPELL_CRIT.get(class_id, GT_SPELL_CRIT[CLASS_PRIEST])
    return gt[0]


def melee_crit_chance(level: int, total_agility: int, crit_rating: int = 0,
                      class_id: int = CLASS_PRIEST) -> float:
    """Melee crit % from Agility + Crit Rating (GtChanceToMeleeCrit.dbc).

    Formula: (base + agi * ratio) * 100 + rating_bonus
    """
    base = _get_melee_crit_base(class_id)
    agi_ratio = _get_melee_crit_from_agi(class_id, level)
    crit = (base + total_agility * agi_ratio) * 100.0
    crit += _rating_to_pct(crit_rating, CR_CRIT_MELEE_L80, level)
    return max(crit, 0.0)


def ranged_crit_chance(level: int, total_agility: int, crit_rating: int = 0,
                       class_id: int = CLASS_PRIEST) -> float:
    """Ranged crit % — same Agility formula as melee, different rating."""
    base = _get_melee_crit_base(class_id)
    agi_ratio = _get_melee_crit_from_agi(class_id, level)
    crit = (base + total_agility * agi_ratio) * 100.0
    crit += _rating_to_pct(crit_rating, CR_CRIT_RANGED_L80, level)
    return max(crit, 0.0)


def spell_crit_chance(level: int, bonus_intellect: int = 0, bonus_crit_rating: int = 0,
                      class_id: int = CLASS_PRIEST) -> float:
    """Spell crit % from Intellect + Crit Rating (GtChanceToSpellCrit.dbc)."""
    base = _get_spell_crit_base(class_id)
    total_int = class_base_stat(class_id, 3, level) + bonus_intellect
    int_ratio = _get_spell_crit_from_int(class_id, level)
    crit = (base + total_int * int_ratio) * 100.0
    crit += _rating_to_pct(bonus_crit_rating, CR_CRIT_SPELL_L80, level)
    return max(crit, 0.0)


# ─── Haste ────────────────────────────────────────────────────────────

def melee_haste_pct(level: int, haste_rating: int = 0) -> float:
    """Melee haste % from Haste Rating."""
    return _rating_to_pct(haste_rating, CR_HASTE_MELEE_L80, level)


def ranged_haste_pct(level: int, haste_rating: int = 0) -> float:
    """Ranged haste % from Haste Rating."""
    return _rating_to_pct(haste_rating, CR_HASTE_RANGED_L80, level)


def spell_haste_pct(level: int, bonus_haste_rating: int = 0) -> float:
    """Spell haste % from Haste Rating."""
    return _rating_to_pct(bonus_haste_rating, CR_HASTE_SPELL_L80, level)


# ─── Dodge / Parry / Block ───────────────────────────────────────────

def dodge_chance(level: int, total_agility: int, dodge_rating: int = 0,
                 defense_rating: int = 0, class_id: int = CLASS_PRIEST) -> float:
    """Dodge % from Agility + Dodge Rating + Defense (WotLK diminishing returns).

    From Player.cpp:GetDodgeFromAgility + StatSystem.cpp:UpdateDodgePercentage.
    """
    agi_ratio = _get_melee_crit_from_agi(class_id, level)
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


# ─── Attack Power ─────────────────────────────────────────────────────

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


# ─── Other combat stats ──────────────────────────────────────────────

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


# ─── Spirit Regen ─────────────────────────────────────────────────────

def spirit_mana_regen(level: int, bonus_intellect: int = 0, bonus_spirit: int = 0,
                      class_id: int = CLASS_PRIEST) -> float:
    """Spirit-based mana regen per tick (0.5s) while NOT casting.

    Formula from Player.cpp:OCTRegenMPPerSpirit + UpdateManaRegen:
      power_regen = sqrt(int) * spirit * coeff
    Returns per-tick value (0.5s).
    """
    # Get coefficient from DBC table or fallback
    tbl = _c.GT_REGEN_MP_PER_SPT_TABLE
    if tbl is not None:
        coeff = tbl.get((class_id, min(level, 80)))
        if coeff is None or coeff <= 0:
            return 0.0
    else:
        gt = GT_REGEN_MP_PER_SPT.get(class_id)
        if gt is None:
            return 0.0
        t = min((level - 1) / 79.0, 1.0)
        coeff = gt[0] * (1 - t) + gt[1] * t

    total_int = class_base_stat(class_id, 3, level) + bonus_intellect
    total_spirit = class_base_stat(class_id, 4, level) + bonus_spirit
    regen_per_sec = math.sqrt(max(total_int, 1)) * total_spirit * coeff
    return regen_per_sec * 0.5  # per tick


def spirit_hp_regen(level: int, total_spirit: int,
                    class_id: int = CLASS_PRIEST) -> float:
    """Spirit-based HP regen per tick (0.5s) while OOC.

    Uses GtRegenHPPerSpt.dbc coefficient when available.
    """
    tbl = _c.GT_REGEN_HP_PER_SPT_TABLE
    if tbl is not None:
        coeff = tbl.get((class_id, min(level, 80)), 0.0)
    else:
        gt = GT_HP_REGEN_PER_SPT.get(class_id, (0.0, 0.0))
        if isinstance(gt, (int, float)):
            coeff = gt
        else:
            t = min((level - 1) / 79.0, 1.0)
            coeff = gt[0] * (1 - t) + gt[1] * t
    if coeff <= 0:
        return 0.0
    return total_spirit * coeff * 0.5


# ─── Spell Damage / Heal Formulas ────────────────────────────────────

def base_mana_for_level(level: int, class_id: int = CLASS_PRIEST) -> int:
    """Get class BaseMana at given level from player_class_stats.csv.

    Used for computing % mana costs (Spell.dbc ManaCostPercentage).
    """
    tbl = _c.PLAYER_CLASS_LEVEL_STATS
    if tbl is not None:
        key = (class_id, min(level, 80))
        if key in tbl:
            return tbl[key][1]  # BaseMana column
    # Fallback: approximate from CLASS_BASE_HP_MANA
    base_mana_data = _c.CLASS_BASE_HP_MANA.get(class_id, _c.CLASS_BASE_HP_MANA[CLASS_PRIEST])
    mana_per_lvl = _c.CLASS_MANA_PER_LEVEL.get(class_id, 5)
    return base_mana_data[1] + (level - 1) * mana_per_lvl


def spell_mana_cost(spell_id: int, level: int,
                    class_id: int = CLASS_PRIEST) -> int:
    """Compute actual mana cost = BaseMana * ManaCostPct / 100.

    Uses SPELL_MANA_PCT from Spell.dbc. Falls back to SpellDef.mana_cost
    when pct is 0.
    """
    pct = _c.SPELL_MANA_PCT.get(spell_id, 0)
    if pct > 0:
        return int(base_mana_for_level(level, class_id) * pct / 100)
    # Fallback: use flat cost from SpellDef
    from sim.models import SPELLS
    spell = SPELLS.get(spell_id)
    return spell.mana_cost if spell else 0


def smite_damage(level: int, spell_power: int = 0) -> tuple[int, int]:
    """Smite damage range. DBC: BasePoints=12, DieSides=5, RealPointsPerLevel=0.5.

    Damage = (12 + floor(0.5 * (level-1)) + 1) to (12 + floor(0.5 * (level-1)) + 5)
           + SP * SP_COEFF_SMITE. Capped at MaxLevel=6 for per-level scaling.
    """
    rpl_levels = min(level, 6) - 1  # RealPointsPerLevel capped at MaxLevel
    bonus = int(0.5 * rpl_levels)
    sp_bonus = int(spell_power * SP_COEFF_SMITE)
    base_lo = 12 + bonus + 1  # BasePoints + floor(RPL) + 1
    base_hi = 12 + bonus + 5  # BasePoints + floor(RPL) + DieSides
    return (base_lo + sp_bonus, base_hi + sp_bonus)


def heal_amount(level: int, spell_power: int = 0) -> tuple[int, int]:
    """Lesser Heal range. DBC: BasePoints=45, DieSides=11, RPL=0.9, MaxLevel=3."""
    rpl_levels = min(level, 3) - 1
    bonus = int(0.9 * rpl_levels)
    sp_bonus = int(spell_power * SP_COEFF_HEAL)
    return (45 + bonus + 1 + sp_bonus, 45 + bonus + 11 + sp_bonus)


def mind_blast_damage(level: int, spell_power: int = 0) -> tuple[int, int]:
    """Mind Blast damage range. DBC: BasePoints=38, DieSides=5, RPL=0.6, MaxLevel=15."""
    rpl_levels = min(level, 15) - 1
    bonus = int(0.6 * rpl_levels)
    sp_bonus = int(spell_power * SP_COEFF_MIND_BLAST)
    return (38 + bonus + 1 + sp_bonus, 38 + bonus + 5 + sp_bonus)


def renew_total_heal(level: int, spell_power: int = 0) -> int:
    """Renew total HoT. DBC: 9/tick x5 ticks = 45 base (RPL=0, no level scaling).

    SP bonus = SP_COEFF_RENEW_TICK * 5 ticks.
    """
    return 45 + int(spell_power * SP_COEFF_RENEW_TICK * 5)


def holy_fire_damage(level: int, spell_power: int = 0) -> tuple[int, int]:
    """Holy Fire direct damage. DBC: BasePoints=101, DieSides=27, RPL=1.5, MaxLevel=24."""
    rpl_levels = min(level, 24) - 1
    bonus = int(1.5 * rpl_levels)
    sp_bonus = int(spell_power * SP_COEFF_HOLY_FIRE)
    return (101 + bonus + 1 + sp_bonus, 101 + bonus + 27 + sp_bonus)


def holy_fire_dot_total(level: int, spell_power: int = 0) -> int:
    """Holy Fire DoT total. DBC: 3/tick x7 ticks = 21 base.

    SP bonus = SP_COEFF_HOLY_FIRE_DOT_TICK * 7 ticks.
    """
    return 21 + int(spell_power * SP_COEFF_HOLY_FIRE_DOT_TICK * 7)


def sw_pain_total(level: int, spell_power: int = 0) -> int:
    """SW:Pain total DoT damage. DBC: 5/tick x6 ticks = 30 base.

    SP bonus = SP_COEFF_SW_PAIN_TICK * 6 ticks.
    """
    return 30 + int(spell_power * SP_COEFF_SW_PAIN_TICK * 6)


def pw_shield_absorb(level: int, spell_power: int = 0) -> int:
    """PW:Shield absorb. DBC: BasePoints=43, DieSides=1, RPL=0.8, MaxLevel=11.

    Absorb = 44 + floor(0.8 * (min(level,11)-1)) + SP*coeff.
    """
    rpl_levels = min(level, 11) - 1
    bonus = int(0.8 * rpl_levels)
    return 44 + bonus + int(spell_power * SP_COEFF_PW_SHIELD)


def inner_fire_values(level: int) -> tuple[int, int]:
    """Inner Fire R1 from DBC: +315 Armor, NO spell power bonus.

    DBC: BasePoints=314, DieSides=1, AuraName=22 (MOD_RESISTANCE), MiscValue=1 (Armor).
    R1 has no spell power component (SP bonus starts at higher ranks).
    Returns (armor_bonus, spell_power_bonus).
    """
    return (315, 0)


def fortitude_hp_bonus(level: int) -> int:
    """Legacy: flat HP bonus. Kept for backward compat but no longer used.

    PW:Fortitude now gives Stamina via fortitude_stamina_bonus().
    """
    return 0


def fortitude_stamina_bonus(level: int) -> int:
    """PW:Fortitude R1 Stamina bonus from DBC.

    DBC: BasePoints=2, DieSides=1, AuraName=29 (MOD_STAT), MiscValue=2 (STAMINA).
    +3 Stamina at all levels (Rank 1 has no per-level scaling, RPL=0).
    """
    return 3


# ─── Combat Resolution Formulas (WotLK attack table) ─────────────────

# Melee attack outcomes (single-roll table, WotLK)
MELEE_MISS = 'miss'
MELEE_DODGE = 'dodge'
MELEE_PARRY = 'parry'
MELEE_BLOCK = 'block'       # partial mitigation (damage reduced by block_value)
MELEE_CRIT = 'crit'
MELEE_NORMAL = 'normal'
MELEE_CRUSHING = 'crushing'  # 150% damage, mob 4+ levels above player

# Spell outcomes
SPELL_MISS = 'miss'
SPELL_HIT = 'hit'
SPELL_CRIT = 'crit'


def spell_miss_chance(player_level: int, mob_level: int,
                      hit_bonus_pct: float = 0.0) -> float:
    """Spell miss % based on level difference (WotLK SpellMgr.cpp).

    WotLK base spell miss against same-level mob: 4%.
    Each level of mob above player: +1% (up to +2 diff).
    At +3 level diff: big jump to 17% (boss penalty).
    Hit rating reduces miss chance (cannot go below 1%).
    """
    diff = mob_level - player_level
    if diff <= 0:
        base_miss = max(4.0 + diff, 1.0)  # easier mobs = less miss, floor 1%
    elif diff == 1:
        base_miss = 5.0
    elif diff == 2:
        base_miss = 6.0
    elif diff == 3:
        base_miss = 17.0  # boss-level jump
    else:
        base_miss = 17.0 + (diff - 3) * 11.0  # beyond +3: very punishing
    return max(base_miss - hit_bonus_pct, 1.0)


def mob_melee_miss_chance(attacker_level: int, defender_level: int,
                          defender_defense_bonus: float = 0.0) -> float:
    """Mob melee miss chance against player (Unit.cpp:RollMeleeOutcomeAgainst).

    WotLK: base 5% miss for equal level.
    Defense skill above attacker weapon skill adds 0.04% miss per point.
    Weapon skill = level * 5, defense skill = level * 5 + bonus.
    """
    attacker_skill = attacker_level * 5
    defender_skill = defender_level * 5 + defender_defense_bonus
    skill_diff = defender_skill - attacker_skill
    miss = 5.0 + skill_diff * 0.04
    return max(miss, 0.0)


def mob_melee_crit_chance(attacker_level: int, defender_level: int,
                          defender_defense_bonus: float = 0.0,
                          defender_resilience_pct: float = 0.0) -> float:
    """Mob melee crit chance against player (Unit.cpp:RollMeleeOutcomeAgainst).

    WotLK: base 5% crit for mobs.
    Each point of attacker weapon skill above defender defense skill: +0.04% crit.
    Defense bonus and Resilience reduce crit.
    """
    attacker_skill = attacker_level * 5
    defender_skill = defender_level * 5 + defender_defense_bonus
    skill_diff = attacker_skill - defender_skill
    crit = 5.0 + skill_diff * 0.04
    crit -= defender_resilience_pct  # resilience reduces incoming crit
    return max(crit, 0.0)


def mob_crushing_chance(attacker_level: int, defender_level: int) -> float:
    """Crushing blow chance: mob must be 4+ levels above player.

    WotLK: 2% per missing defense skill point above 15-point gap.
    Weapon skill = level * 5.
    """
    skill_diff = (attacker_level * 5) - (defender_level * 5)
    if skill_diff < 15:
        return 0.0
    return max(0.0, (skill_diff - 15) * 2.0)


def resolve_mob_melee_attack(attacker_level: int, defender_level: int,
                             defender_dodge: float, defender_parry: float,
                             defender_block: float,
                             defender_defense_bonus: float = 0.0,
                             defender_resilience_pct: float = 0.0,
                             roll: float = 0.0) -> str:
    """Single-roll melee attack table for mob attacking player.

    WotLK uses a single roll against cumulative thresholds:
      miss -> dodge -> parry -> block -> crit -> crushing -> normal

    Args:
        roll: 0-100 uniform random value (for deterministic testing)

    Returns one of: MELEE_MISS, MELEE_DODGE, MELEE_PARRY, MELEE_BLOCK,
                    MELEE_CRIT, MELEE_CRUSHING, MELEE_NORMAL
    """
    miss = mob_melee_miss_chance(attacker_level, defender_level,
                                defender_defense_bonus)
    crit = mob_melee_crit_chance(attacker_level, defender_level,
                                defender_defense_bonus,
                                defender_resilience_pct)
    crushing = mob_crushing_chance(attacker_level, defender_level)

    # Cumulative thresholds on a single 0-100 roll
    threshold = 0.0

    threshold += miss
    if roll < threshold:
        return MELEE_MISS

    threshold += defender_dodge
    if roll < threshold:
        return MELEE_DODGE

    threshold += defender_parry
    if roll < threshold:
        return MELEE_PARRY

    # Block: checked before crit in WotLK (block can't prevent crit,
    # but in the single-roll table block is checked before crit)
    threshold += defender_block
    if roll < threshold:
        return MELEE_BLOCK

    threshold += crit
    if roll < threshold:
        return MELEE_CRIT

    threshold += crushing
    if roll < threshold:
        return MELEE_CRUSHING

    return MELEE_NORMAL


def resolve_spell_hit(player_level: int, mob_level: int,
                      hit_bonus_pct: float, spell_crit_pct: float,
                      roll_hit: float, roll_crit: float) -> str:
    """Resolve spell outcome: miss, hit, or crit.

    Two separate rolls (spells use two-roll unlike melee single-roll):
      1. Hit roll: if roll < miss_chance -> SPELL_MISS
      2. Crit roll: if roll < crit_chance -> SPELL_CRIT, else SPELL_HIT

    Args:
        roll_hit: 0-100 for miss check
        roll_crit: 0-100 for crit check
    """
    miss = spell_miss_chance(player_level, mob_level, hit_bonus_pct)
    if roll_hit < miss:
        return SPELL_MISS
    if roll_crit < spell_crit_pct:
        return SPELL_CRIT
    return SPELL_HIT
