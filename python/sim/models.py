"""
WoW Simulation Data Models.

Data classes for the combat simulation: items, spells, mobs, vendors,
quest NPCs, player state, and mob instances. Also includes hardcoded
game data (spell definitions, mob templates, spawn positions, vendor data).
"""

from dataclasses import dataclass, field

from sim.constants import (
    CLASS_PRIEST, DEFAULT_BACKPACK_SLOTS,
    SPELL_LEVEL_REQ, SPELL_MANA_PCT,
    FAMILY_SMITE, FAMILY_HEAL, FAMILY_FLASH_HEAL, FAMILY_SW_PAIN,
    FAMILY_PW_SHIELD, FAMILY_MIND_BLAST, FAMILY_RENEW, FAMILY_HOLY_FIRE,
    FAMILY_INNER_FIRE, FAMILY_FORTITUDE,
    FAMILY_DEVOURING_PLAGUE, FAMILY_PSYCHIC_SCREAM, FAMILY_SHADOW_PROTECTION,
    FAMILY_DIVINE_SPIRIT, FAMILY_FEAR_WARD, FAMILY_HOLY_NOVA, FAMILY_DISPEL_MAGIC,
    FAMILY_MIND_FLAY, FAMILY_VAMPIRIC_TOUCH, FAMILY_DISPERSION,
)


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


@dataclass(slots=True)
class EquippedBag:
    """A bag equipped in one of the 4 bag slots."""
    entry: int
    name: str
    container_slots: int   # number of inventory slots this bag provides
    quality: int = 0
    sell_price: int = 0



# ─── Spell Definitions ───────────────────────────────────────────────

@dataclass
class SpellDef:
    id: int
    name: str
    cast_ticks: int       # cast time in ticks (1 tick = 0.5s)
    mana_cost: int        # legacy flat cost (used if mana_pct == 0)
    spell_family: int = 0 # family ID (first rank's spell_id), e.g. FAMILY_SMITE=585
    level_req: int = 1    # min player level to learn/cast (from trainer_spell)
    mana_pct: int = 0     # % of class base mana (from Spell.dbc ManaCostPercentage)
    # DBC base values for generic damage/heal calculation
    base_points: int = 0  # DBC BasePoints (damage = base_points+1 to base_points+die_sides)
    die_sides: int = 0    # DBC DieSides (random range)
    rpl: float = 0.0      # DBC RealPointsPerLevel (per-level scaling)
    max_level: int = 0    # DBC MaxLevel cap for RPL scaling (0 = use spell level)
    # Legacy direct values (computed from DBC for backward compat)
    min_damage: int = 0
    max_damage: int = 0
    min_heal: int = 0
    max_heal: int = 0
    spell_range: float = 0.0
    is_dot: bool = False
    dot_damage: int = 0   # total DoT damage (base, no SP)
    dot_per_tick: int = 0 # DBC damage per tick (base)
    dot_ticks: int = 0     # total duration in ticks
    dot_interval: int = 6  # ticks between dot ticks (3s = 6 ticks)
    is_shield: bool = False
    shield_base: int = 0  # DBC BasePoints for absorb
    shield_die: int = 0   # DBC DieSides for absorb
    shield_rpl: float = 0.0  # DBC RPL for absorb
    shield_max_level: int = 0
    shield_absorb: int = 0   # total absorb at base level (base_points + die_sides)
    shield_duration: int = 0  # ticks
    gcd_ticks: int = 3    # 1.5s = 3 ticks
    cooldown_ticks: int = 0   # spell-specific cooldown (ticks), 0 = none
    is_hot: bool = False
    hot_heal: int = 0      # total HoT healing (base, no SP)
    hot_per_tick: int = 0  # DBC heal per tick (base)
    hot_ticks: int = 0     # total duration in ticks
    hot_interval: int = 6  # ticks between hot ticks (3s = 6 ticks)
    is_buff: bool = False
    buff_duration: int = 0  # ticks
    buff_value: int = 0    # generic buff value (armor for Inner Fire, stamina for Fort)


# All values verified from Spell.dbc binary (WotLK 3.3.5 build 12340),
# trainer_spell.csv (level requirements), and spell_bonus_data.csv (SP coefficients).
# Mana costs use ManaCostPercentage from DBC — actual cost = BaseMana * pct / 100.
#
# DBC damage formula: min = BasePoints + 1 + floor(RPL * (min(level, MaxLevel) - SpellLevel))
#                     max = BasePoints + DieSides + floor(RPL * ...)
# For DoTs/HoTs: dot_per_tick is the base amount per tick from DBC.
#
# All ranks up to level 60 for each spell family.

def _sd(sid, name, family, cast_ticks, mana_cost=0, spell_range=0.0,
        bp=0, ds=0, rpl=0.0, ml=0,
        is_dot=False, dot_per_tick=0, dot_dur_ticks=0, dot_interval=6,
        is_shield=False, s_bp=0, s_ds=0, s_rpl=0.0, s_ml=0, s_dur=0,
        is_hot=False, hot_per_tick=0, hot_dur_ticks=0, hot_interval=6,
        is_buff=False, buff_dur=0, buff_val=0,
        cd=0):
    """Helper to build SpellDef with DBC values."""
    lvl = SPELL_LEVEL_REQ.get(sid, 1)
    pct = SPELL_MANA_PCT.get(sid, 0)
    n_dot_ticks = dot_dur_ticks // dot_interval if dot_interval > 0 and dot_dur_ticks > 0 else 0
    n_hot_ticks = hot_dur_ticks // hot_interval if hot_interval > 0 and hot_dur_ticks > 0 else 0
    return SpellDef(
        id=sid, name=name, cast_ticks=cast_ticks, mana_cost=mana_cost,
        spell_family=family, level_req=lvl, mana_pct=pct,
        base_points=bp, die_sides=ds, rpl=rpl, max_level=ml,
        min_damage=bp + 1 if ds > 0 and not is_hot else 0,
        max_damage=bp + ds if ds > 0 and not is_hot else 0,
        min_heal=bp + 1 if ds > 0 and name in ('Lesser Heal', 'Heal', 'Greater Heal', 'Flash Heal') else 0,
        max_heal=bp + ds if ds > 0 and name in ('Lesser Heal', 'Heal', 'Greater Heal', 'Flash Heal') else 0,
        spell_range=spell_range,
        is_dot=is_dot, dot_damage=dot_per_tick * n_dot_ticks, dot_per_tick=dot_per_tick,
        dot_ticks=dot_dur_ticks, dot_interval=dot_interval,
        is_shield=is_shield,
        shield_base=s_bp, shield_die=s_ds, shield_rpl=s_rpl, shield_max_level=s_ml,
        shield_absorb=s_bp + s_ds if s_ds > 0 else 0,
        shield_duration=s_dur,
        gcd_ticks=3, cooldown_ticks=cd,
        is_hot=is_hot, hot_heal=hot_per_tick * n_hot_ticks, hot_per_tick=hot_per_tick,
        hot_ticks=hot_dur_ticks, hot_interval=hot_interval,
        is_buff=is_buff, buff_duration=buff_dur, buff_value=buff_val,
    )

SPELLS = {
    # ── Smite (R1-R8) ── cast 1.5s/2.0s/2.5s, range 30
    585:   _sd(585, "Smite", FAMILY_SMITE, 3, spell_range=30.0, bp=12, ds=5, rpl=0.5, ml=101),
    591:   _sd(591, "Smite", FAMILY_SMITE, 4, spell_range=30.0, bp=24, ds=7, rpl=0.6, ml=101),
    598:   _sd(598, "Smite", FAMILY_SMITE, 5, spell_range=30.0, bp=53, ds=9, rpl=0.9, ml=101),
    984:   _sd(984, "Smite", FAMILY_SMITE, 5, spell_range=30.0, bp=90, ds=15, rpl=1.3, ml=101),
    1004:  _sd(1004, "Smite", FAMILY_SMITE, 5, spell_range=30.0, bp=149, ds=21, rpl=1.6, ml=101),
    6060:  _sd(6060, "Smite", FAMILY_SMITE, 5, spell_range=30.0, bp=211, ds=29, rpl=2.0, ml=101),
    10933: _sd(10933, "Smite", FAMILY_SMITE, 5, spell_range=30.0, bp=286, ds=37, rpl=2.3, ml=101),
    10934: _sd(10934, "Smite", FAMILY_SMITE, 5, spell_range=30.0, bp=370, ds=45, rpl=2.7, ml=101),

    # ── Lesser Heal (R1-R3) ── cast 1.5s/2.0s/2.5s, self-cast
    2050:  _sd(2050, "Lesser Heal", FAMILY_HEAL, 3, bp=45, ds=11, rpl=0.9, ml=101),
    2052:  _sd(2052, "Lesser Heal", FAMILY_HEAL, 4, bp=70, ds=15, rpl=1.1, ml=101),
    2053:  _sd(2053, "Lesser Heal", FAMILY_HEAL, 5, bp=134, ds=23, rpl=1.6, ml=101),

    # ── Heal (R1-R4) ── cast 3.0s, self-cast
    2054:  _sd(2054, "Heal", FAMILY_HEAL, 6, bp=294, ds=47, rpl=2.4, ml=101),
    2055:  _sd(2055, "Heal", FAMILY_HEAL, 6, bp=428, ds=63, rpl=3.2, ml=101),
    6063:  _sd(6063, "Heal", FAMILY_HEAL, 6, bp=565, ds=77, rpl=4.0, ml=101),
    6064:  _sd(6064, "Heal", FAMILY_HEAL, 6, bp=711, ds=93, rpl=4.5, ml=101),

    # ── Greater Heal (R1-R5) ── cast 3.0s, self-cast
    2060:  _sd(2060, "Greater Heal", FAMILY_HEAL, 6, bp=898, ds=115, rpl=5.1, ml=101),
    10963: _sd(10963, "Greater Heal", FAMILY_HEAL, 6, bp=1148, ds=141, rpl=5.8, ml=101),
    10964: _sd(10964, "Greater Heal", FAMILY_HEAL, 6, bp=1436, ds=173, rpl=6.6, ml=101),
    10965: _sd(10965, "Greater Heal", FAMILY_HEAL, 6, bp=1797, ds=209, rpl=7.5, ml=101),
    25314: _sd(25314, "Greater Heal", FAMILY_HEAL, 6, bp=1965, ds=229, rpl=8.1, ml=101),

    # ── Flash Heal (R1-R7) ── cast 1.5s, self-cast
    2061:  _sd(2061, "Flash Heal", FAMILY_FLASH_HEAL, 3, bp=192, ds=45, rpl=1.9, ml=101),
    9472:  _sd(9472, "Flash Heal", FAMILY_FLASH_HEAL, 3, bp=257, ds=57, rpl=2.2, ml=101),
    9473:  _sd(9473, "Flash Heal", FAMILY_FLASH_HEAL, 3, bp=326, ds=67, rpl=2.5, ml=101),
    9474:  _sd(9474, "Flash Heal", FAMILY_FLASH_HEAL, 3, bp=399, ds=79, rpl=2.8, ml=101),
    10915: _sd(10915, "Flash Heal", FAMILY_FLASH_HEAL, 3, bp=517, ds=99, rpl=3.3, ml=101),
    10916: _sd(10916, "Flash Heal", FAMILY_FLASH_HEAL, 3, bp=643, ds=121, rpl=3.7, ml=101),
    10917: _sd(10917, "Flash Heal", FAMILY_FLASH_HEAL, 3, bp=811, ds=147, rpl=4.2, ml=101),

    # ── Shadow Word: Pain (R1-R8) ── instant, range 30, 18s duration, 3s ticks
    589:   _sd(589, "Shadow Word: Pain", FAMILY_SW_PAIN, 0, spell_range=30.0,
              is_dot=True, dot_per_tick=5, dot_dur_ticks=36, dot_interval=6),
    594:   _sd(594, "Shadow Word: Pain", FAMILY_SW_PAIN, 0, spell_range=30.0,
              is_dot=True, dot_per_tick=10, dot_dur_ticks=36, dot_interval=6),
    970:   _sd(970, "Shadow Word: Pain", FAMILY_SW_PAIN, 0, spell_range=30.0,
              is_dot=True, dot_per_tick=20, dot_dur_ticks=36, dot_interval=6),
    992:   _sd(992, "Shadow Word: Pain", FAMILY_SW_PAIN, 0, spell_range=30.0,
              is_dot=True, dot_per_tick=35, dot_dur_ticks=36, dot_interval=6),
    2767:  _sd(2767, "Shadow Word: Pain", FAMILY_SW_PAIN, 0, spell_range=30.0,
              is_dot=True, dot_per_tick=55, dot_dur_ticks=36, dot_interval=6),
    10892: _sd(10892, "Shadow Word: Pain", FAMILY_SW_PAIN, 0, spell_range=30.0,
              is_dot=True, dot_per_tick=77, dot_dur_ticks=36, dot_interval=6),
    10893: _sd(10893, "Shadow Word: Pain", FAMILY_SW_PAIN, 0, spell_range=30.0,
              is_dot=True, dot_per_tick=101, dot_dur_ticks=36, dot_interval=6),
    10894: _sd(10894, "Shadow Word: Pain", FAMILY_SW_PAIN, 0, spell_range=30.0,
              is_dot=True, dot_per_tick=128, dot_dur_ticks=36, dot_interval=6),

    # ── Power Word: Shield (R1-R10) ── instant, 30s duration
    17:    _sd(17, "Power Word: Shield", FAMILY_PW_SHIELD, 0,
              is_shield=True, s_bp=43, s_ds=1, s_rpl=0.8, s_ml=101, s_dur=60),
    592:   _sd(592, "Power Word: Shield", FAMILY_PW_SHIELD, 0,
              is_shield=True, s_bp=87, s_ds=1, s_rpl=1.2, s_ml=101, s_dur=60),
    600:   _sd(600, "Power Word: Shield", FAMILY_PW_SHIELD, 0,
              is_shield=True, s_bp=157, s_ds=1, s_rpl=1.6, s_ml=101, s_dur=60),
    3747:  _sd(3747, "Power Word: Shield", FAMILY_PW_SHIELD, 0,
              is_shield=True, s_bp=233, s_ds=1, s_rpl=2.0, s_ml=101, s_dur=60),
    6065:  _sd(6065, "Power Word: Shield", FAMILY_PW_SHIELD, 0,
              is_shield=True, s_bp=300, s_ds=1, s_rpl=2.3, s_ml=101, s_dur=60),
    6066:  _sd(6066, "Power Word: Shield", FAMILY_PW_SHIELD, 0,
              is_shield=True, s_bp=380, s_ds=1, s_rpl=2.6, s_ml=101, s_dur=60),
    10898: _sd(10898, "Power Word: Shield", FAMILY_PW_SHIELD, 0,
              is_shield=True, s_bp=483, s_ds=1, s_rpl=3.0, s_ml=101, s_dur=60),
    10899: _sd(10899, "Power Word: Shield", FAMILY_PW_SHIELD, 0,
              is_shield=True, s_bp=604, s_ds=1, s_rpl=3.4, s_ml=101, s_dur=60),
    10900: _sd(10900, "Power Word: Shield", FAMILY_PW_SHIELD, 0,
              is_shield=True, s_bp=762, s_ds=1, s_rpl=3.9, s_ml=101, s_dur=60),
    10901: _sd(10901, "Power Word: Shield", FAMILY_PW_SHIELD, 0,
              is_shield=True, s_bp=941, s_ds=1, s_rpl=4.3, s_ml=101, s_dur=60),

    # ── Mind Blast (R1-R9) ── cast 1.5s, range 30, 8s CD
    8092:  _sd(8092, "Mind Blast", FAMILY_MIND_BLAST, 3, spell_range=30.0, bp=38, ds=5, rpl=0.6, ml=101, cd=16),
    8102:  _sd(8102, "Mind Blast", FAMILY_MIND_BLAST, 3, spell_range=30.0, bp=71, ds=7, rpl=0.9, ml=101, cd=16),
    8103:  _sd(8103, "Mind Blast", FAMILY_MIND_BLAST, 3, spell_range=30.0, bp=111, ds=9, rpl=1.1, ml=101, cd=16),
    8104:  _sd(8104, "Mind Blast", FAMILY_MIND_BLAST, 3, spell_range=30.0, bp=166, ds=11, rpl=1.4, ml=101, cd=16),
    8105:  _sd(8105, "Mind Blast", FAMILY_MIND_BLAST, 3, spell_range=30.0, bp=216, ds=15, rpl=1.6, ml=101, cd=16),
    8106:  _sd(8106, "Mind Blast", FAMILY_MIND_BLAST, 3, spell_range=30.0, bp=278, ds=19, rpl=1.9, ml=101, cd=16),
    10945: _sd(10945, "Mind Blast", FAMILY_MIND_BLAST, 3, spell_range=30.0, bp=345, ds=21, rpl=2.1, ml=101, cd=16),
    10946: _sd(10946, "Mind Blast", FAMILY_MIND_BLAST, 3, spell_range=30.0, bp=424, ds=25, rpl=2.4, ml=101, cd=16),
    10947: _sd(10947, "Mind Blast", FAMILY_MIND_BLAST, 3, spell_range=30.0, bp=502, ds=29, rpl=2.6, ml=101, cd=16),

    # ── Renew (R1-R10) ── instant, 15s duration, 3s ticks (5 ticks)
    139:   _sd(139, "Renew", FAMILY_RENEW, 0,
              is_hot=True, hot_per_tick=9, hot_dur_ticks=30, hot_interval=6),
    6074:  _sd(6074, "Renew", FAMILY_RENEW, 0,
              is_hot=True, hot_per_tick=20, hot_dur_ticks=30, hot_interval=6),
    6075:  _sd(6075, "Renew", FAMILY_RENEW, 0,
              is_hot=True, hot_per_tick=35, hot_dur_ticks=30, hot_interval=6),
    6076:  _sd(6076, "Renew", FAMILY_RENEW, 0,
              is_hot=True, hot_per_tick=49, hot_dur_ticks=30, hot_interval=6),
    6077:  _sd(6077, "Renew", FAMILY_RENEW, 0,
              is_hot=True, hot_per_tick=63, hot_dur_ticks=30, hot_interval=6),
    6078:  _sd(6078, "Renew", FAMILY_RENEW, 0,
              is_hot=True, hot_per_tick=80, hot_dur_ticks=30, hot_interval=6),
    10927: _sd(10927, "Renew", FAMILY_RENEW, 0,
              is_hot=True, hot_per_tick=102, hot_dur_ticks=30, hot_interval=6),
    10928: _sd(10928, "Renew", FAMILY_RENEW, 0,
              is_hot=True, hot_per_tick=130, hot_dur_ticks=30, hot_interval=6),
    10929: _sd(10929, "Renew", FAMILY_RENEW, 0,
              is_hot=True, hot_per_tick=162, hot_dur_ticks=30, hot_interval=6),
    25315: _sd(25315, "Renew", FAMILY_RENEW, 0,
              is_hot=True, hot_per_tick=194, hot_dur_ticks=30, hot_interval=6),

    # ── Holy Fire (R1-R8) ── cast 2.0s, range 30, 10s CD, 7s DoT (1s ticks)
    14914: _sd(14914, "Holy Fire", FAMILY_HOLY_FIRE, 4, spell_range=30.0,
              bp=101, ds=27, rpl=1.5, ml=101,
              is_dot=True, dot_per_tick=3, dot_dur_ticks=14, dot_interval=2, cd=20),
    15262: _sd(15262, "Holy Fire", FAMILY_HOLY_FIRE, 4, spell_range=30.0,
              bp=136, ds=37, rpl=1.7, ml=101,
              is_dot=True, dot_per_tick=4, dot_dur_ticks=14, dot_interval=2, cd=20),
    15263: _sd(15263, "Holy Fire", FAMILY_HOLY_FIRE, 4, spell_range=30.0,
              bp=199, ds=53, rpl=2.0, ml=101,
              is_dot=True, dot_per_tick=6, dot_dur_ticks=14, dot_interval=2, cd=20),
    15264: _sd(15264, "Holy Fire", FAMILY_HOLY_FIRE, 4, spell_range=30.0,
              bp=266, ds=73, rpl=2.2, ml=101,
              is_dot=True, dot_per_tick=8, dot_dur_ticks=14, dot_interval=2, cd=20),
    15265: _sd(15265, "Holy Fire", FAMILY_HOLY_FIRE, 4, spell_range=30.0,
              bp=347, ds=93, rpl=2.5, ml=101,
              is_dot=True, dot_per_tick=10, dot_dur_ticks=14, dot_interval=2, cd=20),
    15266: _sd(15266, "Holy Fire", FAMILY_HOLY_FIRE, 4, spell_range=30.0,
              bp=429, ds=117, rpl=2.9, ml=101,
              is_dot=True, dot_per_tick=13, dot_dur_ticks=14, dot_interval=2, cd=20),
    15267: _sd(15267, "Holy Fire", FAMILY_HOLY_FIRE, 4, spell_range=30.0,
              bp=528, ds=143, rpl=3.2, ml=101,
              is_dot=True, dot_per_tick=16, dot_dur_ticks=14, dot_interval=2, cd=20),
    15261: _sd(15261, "Holy Fire", FAMILY_HOLY_FIRE, 4, spell_range=30.0,
              bp=638, ds=173, rpl=3.4, ml=101,
              is_dot=True, dot_per_tick=18, dot_dur_ticks=14, dot_interval=2, cd=20),

    # ── Inner Fire (R1-R6) ── instant, 30min buff, armor bonus
    588:   _sd(588, "Inner Fire", FAMILY_INNER_FIRE, 0,
              is_buff=True, buff_dur=3600, buff_val=315),
    7128:  _sd(7128, "Inner Fire", FAMILY_INNER_FIRE, 0,
              is_buff=True, buff_dur=3600, buff_val=495),
    602:   _sd(602, "Inner Fire", FAMILY_INNER_FIRE, 0,
              is_buff=True, buff_dur=3600, buff_val=720),
    1006:  _sd(1006, "Inner Fire", FAMILY_INNER_FIRE, 0,
              is_buff=True, buff_dur=3600, buff_val=945),
    10951: _sd(10951, "Inner Fire", FAMILY_INNER_FIRE, 0,
              is_buff=True, buff_dur=3600, buff_val=1170),
    10952: _sd(10952, "Inner Fire", FAMILY_INNER_FIRE, 0,
              is_buff=True, buff_dur=3600, buff_val=1395),

    # ── Power Word: Fortitude (R1-R6) ── instant, 30min buff, stamina bonus
    1243:  _sd(1243, "Power Word: Fortitude", FAMILY_FORTITUDE, 0,
              is_buff=True, buff_dur=3600, buff_val=3),
    1244:  _sd(1244, "Power Word: Fortitude", FAMILY_FORTITUDE, 0,
              is_buff=True, buff_dur=3600, buff_val=8),
    1245:  _sd(1245, "Power Word: Fortitude", FAMILY_FORTITUDE, 0,
              is_buff=True, buff_dur=3600, buff_val=20),
    2791:  _sd(2791, "Power Word: Fortitude", FAMILY_FORTITUDE, 0,
              is_buff=True, buff_dur=3600, buff_val=32),
    10937: _sd(10937, "Power Word: Fortitude", FAMILY_FORTITUDE, 0,
              is_buff=True, buff_dur=3600, buff_val=43),
    10938: _sd(10938, "Power Word: Fortitude", FAMILY_FORTITUDE, 0,
              is_buff=True, buff_dur=3600, buff_val=54),

    # ── Devouring Plague (R1-R6) ── instant, range 30, 24s DoT, 3s ticks, heals caster
    # DBC: Shadow damage DoT, heals caster for 100% of damage dealt
    2944:  _sd(2944, "Devouring Plague", FAMILY_DEVOURING_PLAGUE, 0, spell_range=30.0,
              is_dot=True, dot_per_tick=19, dot_dur_ticks=48, dot_interval=6),
    19276: _sd(19276, "Devouring Plague", FAMILY_DEVOURING_PLAGUE, 0, spell_range=30.0,
              is_dot=True, dot_per_tick=30, dot_dur_ticks=48, dot_interval=6),
    19277: _sd(19277, "Devouring Plague", FAMILY_DEVOURING_PLAGUE, 0, spell_range=30.0,
              is_dot=True, dot_per_tick=44, dot_dur_ticks=48, dot_interval=6),
    19278: _sd(19278, "Devouring Plague", FAMILY_DEVOURING_PLAGUE, 0, spell_range=30.0,
              is_dot=True, dot_per_tick=61, dot_dur_ticks=48, dot_interval=6),
    19279: _sd(19279, "Devouring Plague", FAMILY_DEVOURING_PLAGUE, 0, spell_range=30.0,
              is_dot=True, dot_per_tick=81, dot_dur_ticks=48, dot_interval=6),
    19280: _sd(19280, "Devouring Plague", FAMILY_DEVOURING_PLAGUE, 0, spell_range=30.0,
              is_dot=True, dot_per_tick=103, dot_dur_ticks=48, dot_interval=6),

    # ── Psychic Scream (R1-R4) ── instant, AoE fear 8 yards, 30s CD
    # DBC: Causes up to 5 nearby enemies to flee for X seconds
    # Fear duration scales: R1=8s(16t), R2=8s, R3=8s, R4=8s
    8122:  _sd(8122, "Psychic Scream", FAMILY_PSYCHIC_SCREAM, 0, spell_range=8.0,
              cd=60, bp=0, ds=0),
    8124:  _sd(8124, "Psychic Scream", FAMILY_PSYCHIC_SCREAM, 0, spell_range=8.0,
              cd=60, bp=0, ds=0),
    10888: _sd(10888, "Psychic Scream", FAMILY_PSYCHIC_SCREAM, 0, spell_range=8.0,
              cd=60, bp=0, ds=0),
    10890: _sd(10890, "Psychic Scream", FAMILY_PSYCHIC_SCREAM, 0, spell_range=8.0,
              cd=60, bp=0, ds=0),

    # ── Shadow Protection (R1-R3) ── instant, 10min buff, shadow resistance
    976:   _sd(976, "Shadow Protection", FAMILY_SHADOW_PROTECTION, 0,
              is_buff=True, buff_dur=1200, buff_val=30),
    10957: _sd(10957, "Shadow Protection", FAMILY_SHADOW_PROTECTION, 0,
              is_buff=True, buff_dur=1200, buff_val=45),
    10958: _sd(10958, "Shadow Protection", FAMILY_SHADOW_PROTECTION, 0,
              is_buff=True, buff_dur=1200, buff_val=60),

    # ── Divine Spirit (R1-R4) ── instant, 30min buff, +Spirit
    14752: _sd(14752, "Divine Spirit", FAMILY_DIVINE_SPIRIT, 0,
              is_buff=True, buff_dur=3600, buff_val=17),
    14818: _sd(14818, "Divine Spirit", FAMILY_DIVINE_SPIRIT, 0,
              is_buff=True, buff_dur=3600, buff_val=23),
    14819: _sd(14819, "Divine Spirit", FAMILY_DIVINE_SPIRIT, 0,
              is_buff=True, buff_dur=3600, buff_val=33),
    27681: _sd(27681, "Divine Spirit", FAMILY_DIVINE_SPIRIT, 0,
              is_buff=True, buff_dur=3600, buff_val=40),

    # ── Fear Ward (R1) ── instant, 3min buff, absorbs one fear effect
    6346:  _sd(6346, "Fear Ward", FAMILY_FEAR_WARD, 0,
              is_buff=True, buff_dur=360, buff_val=1),

    # ── Holy Nova (R1-R6) ── instant, PBAoE 10yd, damage + self-heal
    # DBC: Causes Holy damage to all enemies within 10 yards and heals
    # all friendly targets (self in sim). No target required.
    15237: _sd(15237, "Holy Nova", FAMILY_HOLY_NOVA, 0, spell_range=10.0,
              bp=28, ds=5, rpl=0.5, ml=101),
    15430: _sd(15430, "Holy Nova", FAMILY_HOLY_NOVA, 0, spell_range=10.0,
              bp=52, ds=7, rpl=0.7, ml=101),
    15431: _sd(15431, "Holy Nova", FAMILY_HOLY_NOVA, 0, spell_range=10.0,
              bp=85, ds=11, rpl=0.9, ml=101),
    27799: _sd(27799, "Holy Nova", FAMILY_HOLY_NOVA, 0, spell_range=10.0,
              bp=127, ds=15, rpl=1.2, ml=101),
    27800: _sd(27800, "Holy Nova", FAMILY_HOLY_NOVA, 0, spell_range=10.0,
              bp=173, ds=19, rpl=1.5, ml=101),
    27801: _sd(27801, "Holy Nova", FAMILY_HOLY_NOVA, 0, spell_range=10.0,
              bp=234, ds=25, rpl=1.8, ml=101),

    # ── Dispel Magic (R1-R2) ── instant, range 30, removes 1/2 buffs or debuffs
    527:   _sd(527, "Dispel Magic", FAMILY_DISPEL_MAGIC, 0, spell_range=30.0),
    988:   _sd(988, "Dispel Magic", FAMILY_DISPEL_MAGIC, 0, spell_range=30.0),

    # ── Mind Flay (R1-R7) ── channeled 3s (6 ticks), range 30, 3 damage ticks
    # DBC: Shadow channeled spell, 3 ticks over 3 seconds. Talent-granted (Mind Flay 1/1).
    # dot_per_tick = per-tick base damage, dot_dur_ticks = 6 (3s channel), dot_interval = 2 (1s)
    15407: _sd(15407, "Mind Flay", FAMILY_MIND_FLAY, 6, spell_range=30.0,
              is_dot=True, dot_per_tick=26, dot_dur_ticks=6, dot_interval=2),
    17311: _sd(17311, "Mind Flay", FAMILY_MIND_FLAY, 6, spell_range=30.0,
              is_dot=True, dot_per_tick=43, dot_dur_ticks=6, dot_interval=2),
    17312: _sd(17312, "Mind Flay", FAMILY_MIND_FLAY, 6, spell_range=30.0,
              is_dot=True, dot_per_tick=61, dot_dur_ticks=6, dot_interval=2),
    17313: _sd(17313, "Mind Flay", FAMILY_MIND_FLAY, 6, spell_range=30.0,
              is_dot=True, dot_per_tick=84, dot_dur_ticks=6, dot_interval=2),
    17314: _sd(17314, "Mind Flay", FAMILY_MIND_FLAY, 6, spell_range=30.0,
              is_dot=True, dot_per_tick=109, dot_dur_ticks=6, dot_interval=2),
    18807: _sd(18807, "Mind Flay", FAMILY_MIND_FLAY, 6, spell_range=30.0,
              is_dot=True, dot_per_tick=136, dot_dur_ticks=6, dot_interval=2),
    25387: _sd(25387, "Mind Flay", FAMILY_MIND_FLAY, 6, spell_range=30.0,
              is_dot=True, dot_per_tick=165, dot_dur_ticks=6, dot_interval=2),

    # ── Vampiric Touch (R1-R3) ── cast 1.5s, range 30, 15s DoT (3s ticks, 5 ticks)
    # DBC: Shadow DoT, replenishes mana to party. Talent-granted (Vampiric Touch 1/1).
    34914: _sd(34914, "Vampiric Touch", FAMILY_VAMPIRIC_TOUCH, 3, spell_range=30.0,
              is_dot=True, dot_per_tick=65, dot_dur_ticks=30, dot_interval=6),
    34916: _sd(34916, "Vampiric Touch", FAMILY_VAMPIRIC_TOUCH, 3, spell_range=30.0,
              is_dot=True, dot_per_tick=100, dot_dur_ticks=30, dot_interval=6),
    34917: _sd(34917, "Vampiric Touch", FAMILY_VAMPIRIC_TOUCH, 3, spell_range=30.0,
              is_dot=True, dot_per_tick=145, dot_dur_ticks=30, dot_interval=6),

    # ── Dispersion (R1) ── instant, 6s duration, 3min CD
    # Talent-granted. -90% damage taken, +6% mana per second for 6 seconds.
    47585: _sd(47585, "Dispersion", FAMILY_DISPERSION, 0,
              is_buff=True, buff_dur=12, buff_val=0, cd=360),  # 3min CD = 360 ticks
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

INVENTORY_SLOTS = DEFAULT_BACKPACK_SLOTS  # starting capacity (16 slots, just the default backpack)


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
    # Buff: PW:Fortitude (DBC: +Stamina, AuraName=29 MOD_STAT, MiscValue=2)
    fortitude_remaining: int = 0   # ticks
    fortitude_hp_bonus: int = 0    # legacy (unused, kept for compat)
    fortitude_stamina_bonus: int = 0  # DBC: +3 Stamina (Rank 1)
    # Buff: Shadow Protection (+shadow resistance)
    shadow_prot_remaining: int = 0
    shadow_prot_value: int = 0     # shadow resistance amount
    # Buff: Divine Spirit (+Spirit)
    divine_spirit_remaining: int = 0
    divine_spirit_bonus: int = 0   # Spirit bonus amount
    # Buff: Fear Ward (absorbs one fear)
    fear_ward_remaining: int = 0
    # Accumulated rewards (consumed on read like real server)
    xp_gained: int = 0
    loot_copper: int = 0
    loot_score: int = 0
    equipped_upgrade: float = 0.0  # score improvement (0.0 = no upgrade)
    leveled_up: bool = False    # set True on level-up, consumed on read
    levels_gained: int = 0      # how many levels gained this tick (consumed on read)
    # Quality of items successfully looted this tick (consume-on-read)
    loot_items: list = field(default_factory=list)
    # Quality of items that couldn't be picked up — inventory full (consume-on-read)
    loot_failed: list = field(default_factory=list)
    # Equipment system: EQUIPMENT_SLOT_* -> EquippedItem
    equipment: dict = field(default_factory=dict)  # slot -> EquippedItem
    equipped_scores: dict = field(default_factory=dict)  # slot -> best score (compat)
    # Bag system: BAG_SLOT (19-22) -> EquippedBag
    bags: dict = field(default_factory=dict)  # bag_slot -> EquippedBag
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
    # Combat event counters (consume-on-read, for reward/obs signals)
    dodges: int = 0           # mob attacks dodged this tick
    parries: int = 0          # mob attacks parried this tick
    blocks: int = 0           # mob attacks blocked this tick
    mob_misses: int = 0       # mob attacks that missed this tick
    mob_crits: int = 0        # mob attacks that crit this tick
    mob_crushings: int = 0    # crushing blows received this tick
    spell_misses: int = 0     # player spells that missed this tick
    spell_crits: int = 0      # player spells that crit this tick
    # Quest tracking (consume-on-read)
    quest_xp_gained: int = 0            # XP from quest turn-ins this tick
    quest_copper_gained: int = 0        # copper from quest turn-ins this tick
    quests_completed_tick: int = 0      # quests completed this tick (consume-on-read)
    # Eat/Drink state
    is_eating: bool = False             # True while eating/drinking (regen 5% HP+Mana/s)
    # ─── Talent System ─────────────────────────────────────────────────
    talent_points: dict = field(default_factory=dict)  # talent_name -> current points
    # Shadowform state (talent-granted persistent buff)
    shadowform_active: bool = False
    # Spirit Tap proc (after kill)
    spirit_tap_remaining: int = 0       # ticks remaining on Spirit Tap proc
    # Shadow Weaving stacks on current target (0-5)
    # (tracked per mob via mob.shadow_weaving_stacks)
    # Dispersion state
    dispersion_remaining: int = 0       # ticks remaining (12 = 6s)
    # Mind Flay channel state (using existing cast system)
    channel_remaining: int = 0          # ticks remaining in channel
    channel_spell_id: int = 0           # spell being channeled
    channel_tick_timer: int = 0         # ticks until next channel damage tick
    channel_target_uid: int = 0         # UID of channel target (for VT DoT slot)

    @property
    def total_bag_slots(self) -> int:
        """Total inventory capacity: default backpack + all equipped bag slots."""
        return DEFAULT_BACKPACK_SLOTS + sum(
            bag.container_slots for bag in self.bags.values())

    def recalculate_free_slots(self):
        """Recompute free_slots from total capacity minus inventory count."""
        self.free_slots = self.total_bag_slots - len(self.inventory)


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
    # DoT tracking (slot 3: Devouring Plague — also heals caster)
    dot3_remaining: int = 0
    dot3_timer: int = 0
    dot3_damage_per_tick: int = 0
    dot3_heals_caster: bool = False  # Devouring Plague heals for damage dealt
    # Fear state (Psychic Scream)
    feared: bool = False
    fear_remaining: int = 0  # ticks until fear ends
    fear_dx: float = 0.0     # fear flee direction X
    fear_dy: float = 0.0     # fear flee direction Y
    looted: bool = False
    spawn_x: float = 0.0
    spawn_y: float = 0.0
    spawn_z: float = 82.0
    # Shadow Weaving debuff stacks (from talent, max 5)
    shadow_weaving_stacks: int = 0
    shadow_weaving_timer: int = 0     # ticks until stacks expire
    # Vampiric Touch DoT (slot 4)
    dot4_remaining: int = 0
    dot4_timer: int = 0
    dot4_damage_per_tick: int = 0
    # Misery debuff (+spell hit, from talent)
    misery_stacks: int = 0            # 0 or 1 (debuff active)


