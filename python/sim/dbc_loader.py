"""
Loader for WotLK 3.3.5 DBC GameTable files and player stat CSVs.

Parses binary WDBC GameTable files (gtCombatRatings, gtChanceToMeleeCrit, etc.)
and the player_class_stats.csv / player_race_stats.csv exports from AzerothCore.
All tables are loaded into plain dicts for O(1) lookup by (class_id, level).
"""

import csv
import os
import struct

# WoW class IDs → DBC GameTable row index (0-based, 11 slots)
# Class IDs 1-9 map to indices 0-8, class 11 (Druid) maps to index 10
_CLASS_IDS =   [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
_DBC_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]

# GtCombatRatings type indices (from SharedDefines.h CR_ enum)
CR_WEAPON_SKILL = 0
CR_DEFENSE_SKILL = 1
CR_DODGE = 2
CR_PARRY = 3
CR_BLOCK = 4
CR_HIT_MELEE = 5
CR_HIT_RANGED = 6
CR_HIT_SPELL = 7
CR_CRIT_MELEE = 8
CR_CRIT_RANGED = 9
CR_CRIT_SPELL = 10
CR_RESILIENCE = 14       # CR_CRIT_TAKEN_MELEE (same value for ranged/spell)
CR_HASTE_MELEE = 17
CR_HASTE_RANGED = 18
CR_HASTE_SPELL = 19
CR_EXPERTISE = 23
CR_ARMOR_PENETRATION = 24


def _parse_gt_dbc(path):
    """Parse a GameTable DBC file (WDBC format, single float per record).

    Returns list of float values, or None if file not found.
    """
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'WDBC':
            return None
        nrecs, nfields, recsize, strsize = struct.unpack('<IIII', f.read(16))
        records = []
        for _ in range(nrecs):
            data = f.read(recsize)
            records.append(struct.unpack('<f', data)[0])
    return records


def load_gt_combat_ratings(dbc_dir):
    """Load gtCombatRatings.dbc → {(cr_type, level): rating_per_pct}

    The DBC has 32 rating types × 100 levels (3200 records).
    Index = type * 100 + (level - 1).
    Each value is the rating needed for 1% of that stat at that level.
    """
    recs = _parse_gt_dbc(os.path.join(dbc_dir, 'gtCombatRatings.dbc'))
    if recs is None:
        return None
    result = {}
    for rt in range(32):
        for lvl in range(1, 101):
            idx = rt * 100 + (lvl - 1)
            if idx < len(recs):
                result[(rt, lvl)] = recs[idx]
    return result


def load_gt_per_class(dbc_dir, filename):
    """Load a per-class GameTable DBC (11 classes × 100 levels).

    Returns {(class_id, level): float_value} or None if file not found.
    """
    recs = _parse_gt_dbc(os.path.join(dbc_dir, filename))
    if recs is None:
        return None
    result = {}
    for cid, dbc_idx in zip(_CLASS_IDS, _DBC_INDICES):
        for lvl in range(1, 101):
            idx = dbc_idx * 100 + (lvl - 1)
            if idx < len(recs):
                result[(cid, lvl)] = recs[idx]
    return result


def load_gt_per_class_base(dbc_dir, filename):
    """Load a per-class base DBC (11 records, one per class).

    Returns {class_id: float_value} or None if file not found.
    """
    recs = _parse_gt_dbc(os.path.join(dbc_dir, filename))
    if recs is None:
        return None
    result = {}
    for cid, dbc_idx in zip(_CLASS_IDS, _DBC_INDICES):
        if dbc_idx < len(recs):
            result[cid] = recs[dbc_idx]
    return result


def load_player_class_stats(csv_path):
    """Load player_class_stats.csv from AzerothCore DB export.

    Returns {(class_id, level): (base_hp, base_mana, str, agi, sta, int, spi)}
    or None if file not found.

    The CSV is semicolon-delimited with double-quote quoting.
    """
    if not os.path.isfile(csv_path):
        return None
    result = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=';', quotechar='"')
        next(reader)  # skip header
        for row in reader:
            if len(row) < 9:
                continue
            try:
                cls = int(row[0])
                lvl = int(row[1])
                base_hp = int(row[2])
                base_mana = int(row[3])
                str_val = int(row[4])
                agi_val = int(row[5])
                sta_val = int(row[6])
                int_val = int(row[7])
                spi_val = int(row[8])
                result[(cls, lvl)] = (base_hp, base_mana, str_val, agi_val,
                                      sta_val, int_val, spi_val)
            except (ValueError, IndexError):
                continue
    return result if result else None


def load_all_dbc_tables(data_dir):
    """Load all DBC/CSV tables from the data directory.

    Args:
        data_dir: Path to the data/ directory containing player_class_stats.csv
                  and a dbc/ subdirectory with the GT DBC files.

    Returns:
        Dict with loaded tables (value is None for any table that failed to load):
        {
            'player_class_stats': {(class_id, level): (hp, mana, str, agi, sta, int, spi)},
            'gt_combat_ratings': {(cr_type, level): value},
            'gt_melee_crit': {(class_id, level): value},
            'gt_melee_crit_base': {class_id: value},
            'gt_spell_crit': {(class_id, level): value},
            'gt_spell_crit_base': {class_id: value},
            'gt_regen_mp_per_spt': {(class_id, level): value},
            'gt_regen_hp_per_spt': {(class_id, level): value},
            'gt_oct_regen_hp': {(class_id, level): value},
            'gt_oct_regen_mp': {(class_id, level): value},
        }
    """
    dbc_dir = os.path.join(data_dir, 'dbc')

    return {
        'player_class_stats': load_player_class_stats(
            os.path.join(data_dir, 'player_class_stats.csv')),
        'gt_combat_ratings': load_gt_combat_ratings(dbc_dir),
        'gt_melee_crit': load_gt_per_class(dbc_dir, 'gtChanceToMeleeCrit.dbc'),
        'gt_melee_crit_base': load_gt_per_class_base(dbc_dir, 'gtChanceToMeleeCritBase.dbc'),
        'gt_spell_crit': load_gt_per_class(dbc_dir, 'gtChanceToSpellCrit.dbc'),
        'gt_spell_crit_base': load_gt_per_class_base(dbc_dir, 'gtChanceToSpellCritBase.dbc'),
        'gt_regen_mp_per_spt': load_gt_per_class(dbc_dir, 'gtRegenMPPerSpt.dbc'),
        'gt_regen_hp_per_spt': load_gt_per_class(dbc_dir, 'gtRegenHPPerSpt.dbc'),
        'gt_oct_regen_hp': load_gt_per_class(dbc_dir, 'gtOCTRegenHP.dbc'),
        'gt_oct_regen_mp': load_gt_per_class(dbc_dir, 'gtOCTRegenMP.dbc'),
    }
