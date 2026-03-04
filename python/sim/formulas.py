"""
WoW WotLK 3.3.5 Stat Calculation Formulas.

Pure functions for computing derived stats: HP/Mana from primary stats,
combat rating conversions, spell damage/heal calculations, Spirit regen,
and attack power formulas. All formulas derived from AzerothCore C++ source.
"""

import math

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


