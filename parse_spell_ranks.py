#!/usr/bin/env python3
"""
Parse Priest spell ranks from AzerothCore data files.

Reads:
  - data/trainer_spell.csv       (priest trainer spell list with level requirements)
  - data/dbc/Spell.dbc           (binary WDBC - spell values, names, cast times, etc.)
  - data/dbc/SpellCastTimes.dbc  (binary WDBC - cast time lookup)
  - data/dbc/SpellDuration.dbc   (binary WDBC - duration lookup)
  - data/spell_bonus_data.csv    (SP coefficients)

Outputs a formatted table of all ranks for each priest spell family up to level 60.

Confirmed WotLK 3.3.5a (build 12340) Spell.dbc field layout (234 uint32 fields):
  [0]       SpellID
  [1]       Category
  [4]       Attributes
  [28]      CastingTimeIndex
  [35]      MaxLevel (RPL scaling cap)
  [37]      SpellLevel
  [40]      DurationIndex
  [41]      PowerType (0=mana, 1=rage, 2=focus, 3=energy)
  [42]      ManaCost (flat, stored *10 for rage)
  [71-73]   Effect[0-2] (2=damage, 10=heal, 6=apply_aura)
  [74-76]   EffectDieSides[0-2]
  [77-79]   EffectRealPointsPerLevel[0-2] (float32)
  [80-82]   EffectBasePoints[0-2] (int32)
  [95-97]   EffectApplyAuraName[0-2] (3=periodic_damage, 8=periodic_heal)
  [98-100]  EffectAmplitude[0-2] (tick interval in ms)
  [131]     SpellIconID
  [136-152] SpellName (17 locale fields, [136]=enUS/first locale)
  [153-169] Rank (17 locale fields, [153]=enUS/first locale)
  [204]     ManaCostPercentage (% of base mana)
"""

import csv
import struct
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DBC_DIR = os.path.join(DATA_DIR, "dbc")

# ── Spell families ────────────────────────────────────────────────────────
SPELL_FAMILIES = {
    585:   "Smite",
    2050:  "Lesser Heal",
    2054:  "Heal",
    2060:  "Greater Heal",
    2061:  "Flash Heal",
    589:   "Shadow Word: Pain",
    17:    "Power Word: Shield",
    8092:  "Mind Blast",
    139:   "Renew",
    14914: "Holy Fire",
    588:   "Inner Fire",
    1243:  "Power Word: Fortitude",
}

DISPLAY_GROUPS = [
    ("Smite",                [585]),
    ("Healing Line (Lesser Heal / Heal / Greater Heal / Flash Heal)",
                             [2050, 2054, 2060, 2061]),
    ("Shadow Word: Pain",    [589]),
    ("Power Word: Shield",   [17]),
    ("Mind Blast",           [8092]),
    ("Renew",                [139]),
    ("Holy Fire",            [14914]),
    ("Inner Fire",           [588]),
    ("Power Word: Fortitude",[1243]),
]


def read_dbc(filepath):
    """Read a WDBC file. Returns (nFields, records, string_block)."""
    with open(filepath, "rb") as f:
        header = f.read(20)
        magic, n_records, n_fields, record_size, string_block_size = struct.unpack("<4sIIII", header)
        assert magic == b"WDBC", f"Bad magic in {filepath}: {magic}"
        records = []
        for _ in range(n_records):
            raw = f.read(record_size)
            fields = struct.unpack(f"<{n_fields}I", raw)
            records.append(fields)
        string_block = f.read(string_block_size)
    return n_fields, records, string_block


def read_string(string_block, offset):
    if offset == 0 or offset >= len(string_block):
        return ""
    end = string_block.index(b"\x00", offset)
    return string_block[offset:end].decode("utf-8", errors="replace")


def u2f(u):
    """uint32 -> float32"""
    return struct.unpack("<f", struct.pack("<I", u))[0]


def u2i(u):
    """uint32 -> int32"""
    return struct.unpack("<i", struct.pack("<I", u))[0]


def load_cast_times():
    _, records, _ = read_dbc(os.path.join(DBC_DIR, "SpellCastTimes.dbc"))
    return {r[0]: r[1] for r in records}  # ID -> CastTime ms


def load_durations():
    _, records, _ = read_dbc(os.path.join(DBC_DIR, "SpellDuration.dbc"))
    return {r[0]: u2i(r[1]) for r in records}  # ID -> Duration ms


def load_spell_dbc():
    """Returns dict: SpellID -> spell info dict."""
    n_fields, records, string_block = read_dbc(os.path.join(DBC_DIR, "Spell.dbc"))

    spells = {}
    for r in records:
        spell_id = r[0]

        effects = []
        for i in range(3):
            effects.append({
                "type":        r[71 + i],
                "die_sides":   u2i(r[74 + i]),
                "rpl":         u2f(r[77 + i]),
                "base_points": u2i(r[80 + i]),
                "aura":        r[95 + i],
                "amplitude":   r[98 + i],
            })

        spells[spell_id] = {
            "id":              spell_id,
            "name":            read_string(string_block, r[136]),
            "rank_str":        read_string(string_block, r[153]),
            "category":        r[1],
            "cast_time_idx":   r[28],
            "max_level":       r[35],
            "spell_level":     r[37],
            "duration_idx":    r[40],
            "power_type":      r[41],
            "mana_cost":       r[42],
            "mana_cost_pct":   r[204],
            "effects":         effects,
        }

    return spells


def load_trainer_spells():
    path = os.path.join(DATA_DIR, "trainer_spell.csv")
    spells = []
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter=";", quotechar='"')
        for row in reader:
            if int(row["TrainerId"]) == 11:
                spells.append({
                    "spell_id": int(row["SpellId"]),
                    "req_level": int(row["ReqLevel"]),
                    "req_ability1": int(row["ReqAbility1"]),
                })
    return spells


def load_sp_coefficients():
    path = os.path.join(DATA_DIR, "spell_bonus_data.csv")
    result = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter=";", quotechar='"')
        for row in reader:
            entry = int(row["entry"])
            result[entry] = {
                "direct_bonus": float(row["direct_bonus"]),
                "dot_bonus":    float(row["dot_bonus"]),
                "ap_bonus":     float(row["ap_bonus"]),
                "ap_dot_bonus": float(row["ap_dot_bonus"]),
                "comment":      row.get("comments", ""),
            }
    return result


def build_spell_chains(trainer_spells, base_ids):
    by_id = {s["spell_id"]: s for s in trainer_spells}
    children = {}
    for s in trainer_spells:
        req = s["req_ability1"]
        if req not in children:
            children[req] = []
        children[req].append(s["spell_id"])

    result = {}
    for base_id in base_ids:
        chain = []
        if base_id in by_id:
            chain.append((base_id, by_id[base_id]["req_level"]))
        else:
            chain.append((base_id, 1))

        current = base_id
        while current in children:
            next_ids = children[current]
            if not next_ids:
                break
            next_id = next_ids[0]
            if next_id in by_id:
                chain.append((next_id, by_id[next_id]["req_level"]))
            current = next_id

        for rank_idx, (sid, rlvl) in enumerate(chain, 1):
            result[sid] = {"req_level": rlvl, "rank": rank_idx, "base": base_id}

    return result


def format_duration(dur_ms):
    if dur_ms == 0:
        return ""
    if dur_ms == -1:
        return "Perm"
    secs = dur_ms / 1000
    if secs >= 3600:
        return f"{secs/3600:.0f}h"
    if secs >= 60:
        return f"{secs/60:.0f}m"
    return f"{secs:.0f}s"


def format_cast_time(ct_ms):
    if ct_ms == 0:
        return "Inst"
    return f"{ct_ms/1000:.1f}s"


def main():
    print("Loading data files...")
    trainer_spells = load_trainer_spells()
    print(f"  Trainer spells (priest): {len(trainer_spells)}")

    spell_db = load_spell_dbc()
    print(f"  Spell.dbc entries: {len(spell_db)}")

    cast_times = load_cast_times()
    print(f"  SpellCastTimes.dbc entries: {len(cast_times)}")

    durations = load_durations()
    print(f"  SpellDuration.dbc entries: {len(durations)}")

    sp_coeffs = load_sp_coefficients()
    print(f"  spell_bonus_data entries: {len(sp_coeffs)}")

    for group_name, base_ids in DISPLAY_GROUPS:
        chain_info = build_spell_chains(trainer_spells, base_ids)

        rows = []
        for sid, info in chain_info.items():
            if info["req_level"] > 60:
                continue
            if sid not in spell_db:
                continue

            sp = spell_db[sid]
            ct_ms = cast_times.get(sp["cast_time_idx"], 0)
            dur_ms = durations.get(sp["duration_idx"], 0)

            # Build effect descriptions
            eff_parts = []
            for i, eff in enumerate(sp["effects"]):
                if eff["type"] == 0 and eff["aura"] == 0:
                    continue

                bp = eff["base_points"]
                ds = eff["die_sides"]
                rpl = eff["rpl"]

                # Damage formula: min = bp+1, max = bp+ds (base_dice=1 implicit)
                val_min = bp + 1
                val_max = bp + ds if ds > 0 else bp + 1

                type_map = {2: "DMG", 10: "HEAL", 6: "AURA", 3: "DUMMY", 36: "LEARN"}
                aura_map = {3: "per_dmg", 8: "per_heal", 22: "mod_resist", 69: "absorb",
                            29: "mod_stat", 42: "proc_trigger", 4: "dummy"}

                etype = type_map.get(eff["type"], f"E{eff['type']}")
                eaura = ""
                if eff["type"] == 6:
                    eaura = "/" + aura_map.get(eff["aura"], f"A{eff['aura']}")

                if val_min == val_max:
                    val_str = str(val_min)
                else:
                    val_str = f"{val_min}-{val_max}"

                rpl_str = f" +{rpl:.1f}/lvl" if rpl != 0 else ""
                amp_str = f" @{eff['amplitude']/1000:.0f}s" if eff["amplitude"] > 0 else ""

                eff_parts.append(f"Eff{i+1}:{etype}{eaura}={val_str}{rpl_str}{amp_str}")

            # SP coefficient
            sp_coeff = sp_coeffs.get(sid)
            sp_str = ""
            if sp_coeff:
                parts = []
                if sp_coeff["direct_bonus"] != 0:
                    parts.append(f"d={sp_coeff['direct_bonus']:.4f}")
                if sp_coeff["dot_bonus"] != 0:
                    parts.append(f"dot={sp_coeff['dot_bonus']:.4f}")
                sp_str = " ".join(parts)

            base_name = SPELL_FAMILIES.get(info["base"], "?")

            rows.append({
                "spell_id": sid,
                "name": sp["name"],
                "rank_str": sp["rank_str"],
                "base_name": base_name,
                "rank": info["rank"],
                "req_level": info["req_level"],
                "spell_level": sp["spell_level"],
                "mana_cost_pct": sp["mana_cost_pct"],
                "mana_cost": sp["mana_cost"],
                "cast_time": ct_ms,
                "duration": dur_ms,
                "max_level": sp["max_level"],
                "effects": "  ".join(eff_parts),
                "sp_coeff": sp_str,
            })

        family_order = {bid: idx for idx, bid in enumerate(base_ids)}
        rows.sort(key=lambda r: (family_order.get(
            next((bid for bid in base_ids if SPELL_FAMILIES.get(bid) == r["base_name"]), 0), 99
        ), r["rank"]))

        if not rows:
            continue

        print(f"\n{'='*160}")
        print(f"  {group_name}")
        print(f"{'='*160}")
        header = (f"{'ID':>6} {'Name':<20} {'Rank':<8} {'Rk#':>3} {'TrainLv':>7} {'SpLv':>4} "
                  f"{'Mana%':>5} {'Cast':>6} {'Dur':>6} {'MaxLv':>5}  "
                  f"{'Effects':<65} {'SP Coeff'}")
        print(header)
        print("-" * 160)

        for r in rows:
            ct_str = format_cast_time(r["cast_time"])
            dur_str = format_duration(r["duration"])
            mana_str = f"{r['mana_cost_pct']}%" if r["mana_cost_pct"] > 0 else str(r["mana_cost"])
            maxlv_str = str(r["max_level"]) if r["max_level"] > 0 else ""

            print(f"{r['spell_id']:>6} {r['name']:<20} {r['rank_str']:<8} {r['rank']:>3} "
                  f"{r['req_level']:>7} {r['spell_level']:>4} "
                  f"{mana_str:>5} {ct_str:>6} {dur_str:>6} {maxlv_str:>5}  "
                  f"{r['effects']:<65} {r['sp_coeff']}")

    # ── SP Coefficients Summary ──
    print(f"\n{'='*100}")
    print(f"  SP Coefficients from spell_bonus_data.csv (all Priest spells)")
    print(f"{'='*100}")
    print(f"{'ID':>6} {'direct':>8} {'dot':>8}  {'Comment'}")
    print(f"{'-'*6} {'-'*8} {'-'*8}  {'-'*50}")

    all_family_ids = set()
    for _, base_ids in DISPLAY_GROUPS:
        chain_info = build_spell_chains(trainer_spells, base_ids)
        all_family_ids.update(chain_info.keys())

    for sid in sorted(sp_coeffs.keys()):
        c = sp_coeffs[sid]
        if sid in all_family_ids:
            print(f"{sid:>6} {c['direct_bonus']:>8.4f} {c['dot_bonus']:>8.4f}  {c['comment']}")


if __name__ == "__main__":
    main()
