"""
WotLK 3.3.5 Priest Talent System — Shadow Priest Leveling Build (13/0/58).

Defines talent trees, talent effects, and the predefined leveling build order.
The bot auto-assigns 1 talent point per level starting at level 10,
following the Shadow Priest leveling template from WarcraftTavern.

Build: 13 Discipline / 0 Holy / 58 Shadow
Key milestones:
  - Level 20: Mind Flay (talent-granted spell)
  - Level 30: Vampiric Embrace (self-heal from Shadow damage)
  - Level 40: Shadowform (+15% Shadow dmg, -15% physical dmg taken)
  - Level 50: Vampiric Touch (talent-granted DoT spell)
  - Level 60: Dispersion (defensive cooldown)
  - Level 71+: Discipline talents (Meditation for combat mana regen)
"""

# ─── Talent Definitions ──────────────────────────────────────────────
# Each talent: (tree, tier, max_points, description)
# Effects are implemented in combat_sim.py via get_talent_points()

TALENT_DEFS = {
    # === SHADOW TREE ===
    "spirit_tap": {
        "tree": "shadow", "tier": 1, "max": 3,
        "effect": "After killing a target that yields XP: 33/66/100% chance "
                  "to gain 100% bonus Spirit for 15s",
    },
    "improved_spirit_tap": {
        "tree": "shadow", "tier": 1, "max": 2,
        "effect": "+5/10% total Spirit (passive)",
    },
    "darkness": {
        "tree": "shadow", "tier": 1, "max": 5,
        "effect": "+2/4/6/8/10% Shadow spell damage",
    },
    "shadow_focus": {
        "tree": "shadow", "tier": 2, "max": 3,
        "effect": "+1/2/3% Shadow spell hit chance",
    },
    "improved_sw_pain": {
        "tree": "shadow", "tier": 2, "max": 2,
        "effect": "+3/6% Shadow Word: Pain damage",
    },
    "improved_mind_blast": {
        "tree": "shadow", "tier": 3, "max": 5,
        "effect": "-0.5/1.0/1.5/2.0/2.5s Mind Blast cooldown",
    },
    "mind_flay": {
        "tree": "shadow", "tier": 3, "max": 1,
        "effect": "Unlocks Mind Flay spell (channeled Shadow damage)",
    },
    "shadow_weaving": {
        "tree": "shadow", "tier": 4, "max": 3,
        "effect": "Shadow spells have 33/66/100% chance to apply Shadow Weaving: "
                  "target takes +2% Shadow damage per stack (max 5 stacks, 10% total)",
    },
    "shadow_reach": {
        "tree": "shadow", "tier": 4, "max": 2,
        "effect": "+10/20% range of Shadow damage spells",
    },
    "vampiric_embrace": {
        "tree": "shadow", "tier": 5, "max": 1,
        "effect": "Heals you for 15% of single-target Shadow spell damage",
    },
    "improved_vampiric_embrace": {
        "tree": "shadow", "tier": 5, "max": 2,
        "effect": "Increases VE healing by 33/67% (to 20/25% of Shadow damage)",
    },
    "focused_mind": {
        "tree": "shadow", "tier": 5, "max": 3,
        "effect": "-5/10/15% mana cost of Mind Blast, Mind Flay, Mind Sear",
    },
    "mind_melt": {
        "tree": "shadow", "tier": 6, "max": 2,
        "effect": "+3/6% crit chance on Mind Blast and Mind Flay",
    },
    "improved_devouring_plague": {
        "tree": "shadow", "tier": 6, "max": 3,
        "effect": "Devouring Plague instant damage: 10/20/30% of total DoT damage",
    },
    "shadowform": {
        "tree": "shadow", "tier": 7, "max": 1,
        "effect": "+15% Shadow damage, -15% physical damage taken. "
                  "Periodic Shadow spells can crit.",
    },
    "shadow_power": {
        "tree": "shadow", "tier": 7, "max": 5,
        "effect": "+20/40/60/80/100% crit damage bonus on Mind Blast and Mind Flay "
                  "(from 50% extra to 100% extra)",
    },
    "improved_shadowform": {
        "tree": "shadow", "tier": 8, "max": 2,
        "effect": "While in Shadowform: 50/100% reduced spell pushback on Shadow spells",
    },
    "misery": {
        "tree": "shadow", "tier": 8, "max": 3,
        "effect": "SW:Pain, Mind Flay, Vampiric Touch increase target's "
                  "chance to be hit by spells by 1/2/3%",
    },
    "vampiric_touch": {
        "tree": "shadow", "tier": 9, "max": 1,
        "effect": "Unlocks Vampiric Touch spell (Shadow DoT, mana return)",
    },
    "pain_and_suffering": {
        "tree": "shadow", "tier": 9, "max": 3,
        "effect": "Mind Flay has 33/66/100% chance to refresh SW:Pain on target",
    },
    "twisted_faith": {
        "tree": "shadow", "tier": 9, "max": 5,
        "effect": "+4/8/12/16/20% of Spirit as Spell Power. "
                  "+2/4/6/8/10% Mind Blast and Mind Flay damage if target has SW:Pain.",
    },
    "dispersion": {
        "tree": "shadow", "tier": 11, "max": 1,
        "effect": "6s: -90% damage taken, +6% mana/s. 3min CD.",
    },

    # === DISCIPLINE TREE ===
    "twin_disciplines": {
        "tree": "discipline", "tier": 1, "max": 5,
        "effect": "+1/2/3/4/5% damage and healing of instant spells",
    },
    "improved_inner_fire": {
        "tree": "discipline", "tier": 2, "max": 3,
        "effect": "+15/30/45% armor from Inner Fire",
    },
    "improved_pw_fortitude": {
        "tree": "discipline", "tier": 2, "max": 2,
        "effect": "+15/30% Stamina bonus from PW:Fortitude",
    },
    "meditation": {
        "tree": "discipline", "tier": 3, "max": 3,
        "effect": "+17/33/50% of Spirit-based mana regen continues while casting",
    },
}


# ─── Shadow Priest Leveling Build Order ──────────────────────────────
# One entry per level (10-80 = 71 talent points).
# Each entry is the talent_name receiving 1 point at that level.
#
# Matches the WarcraftTavern WotLK Shadow Priest leveling guide (13/0/58).
# Key milestones: Mind Flay@20, VE@30, Shadowform@40, VT@50, Dispersion@60.

SHADOW_PRIEST_BUILD = [
    # Levels 10-12: Spirit Tap 3/3 (Shadow T1)
    "spirit_tap",               # 10
    "spirit_tap",               # 11
    "spirit_tap",               # 12
    # Levels 13-14: Improved Spirit Tap 2/2 (Shadow T1)
    "improved_spirit_tap",      # 13
    "improved_spirit_tap",      # 14
    # Levels 15-19: Darkness 5/5 (Shadow T1)
    "darkness",                 # 15
    "darkness",                 # 16
    "darkness",                 # 17
    "darkness",                 # 18
    "darkness",                 # 19
    # Level 20: Mind Flay 1/1 (Shadow T3 — unlocks the spell)
    "mind_flay",                # 20
    # Levels 21-25: Improved Mind Blast 5/5 (Shadow T3)
    "improved_mind_blast",      # 21
    "improved_mind_blast",      # 22
    "improved_mind_blast",      # 23
    "improved_mind_blast",      # 24
    "improved_mind_blast",      # 25
    # Levels 26-28: Shadow Weaving 3/3 (Shadow T4)
    "shadow_weaving",           # 26
    "shadow_weaving",           # 27
    "shadow_weaving",           # 28
    # Level 29: Shadow Reach 1/2 (Shadow T4)
    "shadow_reach",             # 29
    # Level 30: Vampiric Embrace 1/1 (Shadow T5 — self-heal)
    "vampiric_embrace",         # 30
    # Level 31: Shadow Reach 2/2
    "shadow_reach",             # 31
    # Levels 32-33: Improved Vampiric Embrace 2/2
    "improved_vampiric_embrace",  # 32
    "improved_vampiric_embrace",  # 33
    # Levels 34-36: Focused Mind 3/3
    "focused_mind",             # 34
    "focused_mind",             # 35
    "focused_mind",             # 36
    # Levels 37-39: Shadow Focus 3/3 (Shadow T2)
    "shadow_focus",             # 37
    "shadow_focus",             # 38
    "shadow_focus",             # 39
    # Level 40: Shadowform 1/1 (Shadow T7)
    "shadowform",               # 40
    # Levels 41-42: Improved SW:Pain 2/2 (Shadow T2)
    "improved_sw_pain",         # 41
    "improved_sw_pain",         # 42
    # Levels 43-44: Mind Melt 2/2 (Shadow T6)
    "mind_melt",                # 43
    "mind_melt",                # 44
    # Levels 45-46: Improved Shadowform 2/2 (Shadow T8)
    "improved_shadowform",      # 45
    "improved_shadowform",      # 46
    # Levels 47-49: Misery 3/3 (Shadow T8)
    "misery",                   # 47
    "misery",                   # 48
    "misery",                   # 49
    # Level 50: Vampiric Touch 1/1 (Shadow T9 — unlocks the spell)
    "vampiric_touch",           # 50
    # Levels 51-53: Pain and Suffering 3/3 (Shadow T9)
    "pain_and_suffering",       # 51
    "pain_and_suffering",       # 52
    "pain_and_suffering",       # 53
    # Levels 54-56: Improved Devouring Plague 3/3 (Shadow T6)
    "improved_devouring_plague",  # 54
    "improved_devouring_plague",  # 55
    "improved_devouring_plague",  # 56
    # Levels 57-59: Twisted Faith 3/5 (Shadow T9)
    "twisted_faith",            # 57
    "twisted_faith",            # 58
    "twisted_faith",            # 59
    # Level 60: Dispersion 1/1 (Shadow T11)
    "dispersion",               # 60
    # Levels 61-62: Twisted Faith 5/5
    "twisted_faith",            # 61
    "twisted_faith",            # 62
    # Levels 63-67: Shadow Power 5/5 (Shadow T7)
    "shadow_power",             # 63
    "shadow_power",             # 64
    "shadow_power",             # 65
    "shadow_power",             # 66
    "shadow_power",             # 67
    # === Discipline Tree (levels 68-80) ===
    # Levels 68-72: Twin Disciplines 5/5 (Disc T1)
    "twin_disciplines",         # 68
    "twin_disciplines",         # 69
    "twin_disciplines",         # 70
    "twin_disciplines",         # 71
    "twin_disciplines",         # 72
    # Levels 73-75: Improved Inner Fire 3/3 (Disc T2)
    "improved_inner_fire",      # 73
    "improved_inner_fire",      # 74
    "improved_inner_fire",      # 75
    # Levels 76-77: Improved PW:Fortitude 2/2 (Disc T2)
    "improved_pw_fortitude",    # 76
    "improved_pw_fortitude",    # 77
    # Levels 78-80: Meditation 3/3 (Disc T3)
    "meditation",               # 78
    "meditation",               # 79
    "meditation",               # 80
]

assert len(SHADOW_PRIEST_BUILD) == 71, \
    f"Build has {len(SHADOW_PRIEST_BUILD)} points, expected 71 (levels 10-80)"

# Verify point totals match 13/0/58
_shadow_pts = sum(1 for t in SHADOW_PRIEST_BUILD if TALENT_DEFS[t]["tree"] == "shadow")
_disc_pts = sum(1 for t in SHADOW_PRIEST_BUILD if TALENT_DEFS[t]["tree"] == "discipline")
assert _shadow_pts == 58, f"Shadow points: {_shadow_pts}, expected 58"
assert _disc_pts == 13, f"Discipline points: {_disc_pts}, expected 13"


def get_talent_for_level(level: int) -> str | None:
    """Return the talent name to assign at the given level, or None if < 10."""
    if level < 10 or level > 80:
        return None
    idx = level - 10
    if idx >= len(SHADOW_PRIEST_BUILD):
        return None
    return SHADOW_PRIEST_BUILD[idx]
