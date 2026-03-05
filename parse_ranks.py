#!/usr/bin/env python3
"""Parse all Priest spell ranks from DBC + trainer_spell data."""
import struct, csv, os

DATA = "/home/user/ac-share/data"
DBC = os.path.join(DATA, "dbc")

# --- Parse Spell.dbc ---
def parse_spell_dbc():
    path = os.path.join(DBC, "Spell.dbc")
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"WDBC"
        n_records, n_fields, rec_size, str_size = struct.unpack("<4I", f.read(16))
        data_start = 20
        str_block_start = data_start + n_records * rec_size

        # Read string block
        f.seek(str_block_start)
        str_block = f.read(str_size)

        spells = {}
        for i in range(n_records):
            f.seek(data_start + i * rec_size)
            raw = f.read(rec_size)
            fields = struct.unpack(f"<{n_fields}I", raw)

            spell_id = fields[0]
            # Name offset is at position 134
            name_off = fields[134]
            if name_off < len(str_block):
                end = str_block.index(b'\x00', name_off)
                name = str_block[name_off:end].decode('utf-8', errors='replace')
            else:
                name = ""

            spells[spell_id] = {
                'id': spell_id,
                'name': name,
                'category': fields[4],
                'cast_time_idx': fields[30],
                'duration_idx': fields[37],
                'mana_pct': fields[131],
                'eff1_type': fields[71],
                'eff2_type': fields[72],
                'eff3_type': fields[73],
                'eff1_base': struct.unpack("<i", struct.pack("<I", fields[80]))[0],
                'eff2_base': struct.unpack("<i", struct.pack("<I", fields[81]))[0],
                'eff3_base': struct.unpack("<i", struct.pack("<I", fields[82]))[0],
                'eff1_die': fields[83],
                'eff2_die': fields[84],
                'eff3_die': fields[85],
                'eff1_rpl': struct.unpack("<f", struct.pack("<I", fields[89]))[0],
                'eff2_rpl': struct.unpack("<f", struct.pack("<I", fields[90]))[0],
                'eff3_rpl': struct.unpack("<f", struct.pack("<I", fields[91]))[0],
                'eff1_maxlvl': fields[92],
                'eff2_maxlvl': fields[93],
                'eff3_maxlvl': fields[94],
                'eff1_aura': fields[95],
                'eff2_aura': fields[96],
                'eff3_aura': fields[97],
                'eff1_misc': struct.unpack("<i", struct.pack("<I", fields[101]))[0],
                'eff2_misc': struct.unpack("<i", struct.pack("<I", fields[102]))[0],
                'eff3_misc': struct.unpack("<i", struct.pack("<I", fields[103]))[0],
                'eff1_bonus_mult': struct.unpack("<f", struct.pack("<I", fields[176]))[0],
                'eff2_bonus_mult': struct.unpack("<f", struct.pack("<I", fields[177]))[0],
                'eff3_bonus_mult': struct.unpack("<f", struct.pack("<I", fields[178]))[0],
                'spell_range_idx': fields[33],
            }
        return spells

# --- Parse cast times ---
def parse_cast_times():
    path = os.path.join(DBC, "SpellCastTimes.dbc")
    with open(path, "rb") as f:
        f.read(4)  # WDBC
        n_rec, n_fields, rec_size, _ = struct.unpack("<4I", f.read(16))
        result = {}
        for i in range(n_rec):
            raw = f.read(rec_size)
            fields = struct.unpack(f"<{n_fields}I", raw)
            ct_id = fields[0]
            ct_ms = struct.unpack("<i", struct.pack("<I", fields[1]))[0]
            result[ct_id] = ct_ms
        return result

# --- Parse durations ---
def parse_durations():
    path = os.path.join(DBC, "SpellDuration.dbc")
    with open(path, "rb") as f:
        f.read(4)
        n_rec, n_fields, rec_size, _ = struct.unpack("<4I", f.read(16))
        result = {}
        for i in range(n_rec):
            raw = f.read(rec_size)
            fields = struct.unpack(f"<{n_fields}I", raw)
            dur_id = fields[0]
            dur_ms = struct.unpack("<i", struct.pack("<I", fields[1]))[0]
            result[dur_id] = dur_ms
        return result

# --- Parse SpellRange.dbc ---
def parse_ranges():
    path = os.path.join(DBC, "SpellRange.dbc")
    with open(path, "rb") as f:
        f.read(4)
        n_rec, n_fields, rec_size, str_size = struct.unpack("<4I", f.read(16))
        result = {}
        for i in range(n_rec):
            raw = f.read(rec_size)
            fields = struct.unpack(f"<{n_fields}I", raw)
            range_id = fields[0]
            min_range = struct.unpack("<f", struct.pack("<I", fields[1]))[0]
            max_range = struct.unpack("<f", struct.pack("<I", fields[3]))[0]
            result[range_id] = max_range
        return result

# --- Parse trainer_spell.csv ---
def parse_trainer_spell():
    path = os.path.join(DATA, "trainer_spell.csv")
    result = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter=';', quotechar='"')
        for row in reader:
            spell_id = int(row['SpellId'])
            req_level = int(row['ReqLevel'])
            trainer_id = int(row['TrainerId'])
            result[spell_id] = {'req_level': req_level, 'trainer_id': trainer_id}
    return result

# --- Parse spell_bonus_data.csv ---
def parse_spell_bonus():
    path = os.path.join(DATA, "spell_bonus_data.csv")
    result = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter=';', quotechar='"')
        for row in reader:
            entry = int(row['entry'])
            result[entry] = {
                'direct_bonus': float(row['direct_bonus']),
                'dot_bonus': float(row['dot_bonus']),
                'ap_bonus': float(row['ap_bonus']),
                'ap_dot_bonus': float(row['ap_dot_bonus']),
            }
    return result

# --- Main ---
# Priest spell names to look for (all ranks)
PRIEST_SPELL_NAMES = [
    'Smite', 'Lesser Heal', 'Heal', 'Greater Heal', 'Flash Heal',
    'Shadow Word: Pain', 'Power Word: Shield', 'Power Word: Fortitude',
    'Mind Blast', 'Renew', 'Holy Fire', 'Inner Fire',
    'Shadow Word: Death', 'Psychic Scream', 'Fade',
    'Mind Flay', 'Mana Burn', 'Holy Nova', 'Prayer of Healing',
    'Devouring Plague', 'Vampiric Touch', 'Vampiric Embrace',
    'Dispel Magic', 'Abolish Disease', 'Cure Disease',
    'Fear Ward', 'Prayer of Fortitude', 'Shadow Protection',
    'Prayer of Shadow Protection', 'Divine Spirit', 'Prayer of Spirit',
]

print("Parsing DBC files...")
spells_dbc = parse_spell_dbc()
cast_times = parse_cast_times()
durations = parse_durations()
ranges = parse_ranges()
trainer = parse_trainer_spell()
bonus = parse_spell_bonus()

print(f"Spell.dbc: {len(spells_dbc)} spells")
print(f"Trainer: {len(trainer)} entries")
print(f"SP Bonus: {len(bonus)} entries")

# Find all priest spells by name
priest_spells = {}
for sid, s in spells_dbc.items():
    name = s['name']
    if name in PRIEST_SPELL_NAMES:
        if name not in priest_spells:
            priest_spells[name] = []
        priest_spells[name].append(s)

# Focus on our 9 spell families + healing line
FOCUS = [
    'Smite', 'Lesser Heal', 'Heal', 'Greater Heal', 'Flash Heal',
    'Shadow Word: Pain', 'Power Word: Shield', 'Power Word: Fortitude',
    'Mind Blast', 'Renew', 'Holy Fire', 'Inner Fire',
]

for spell_name in FOCUS:
    entries = priest_spells.get(spell_name, [])
    if not entries:
        continue

    # Filter to only spells that are in trainer data (learnable)
    learnable = []
    for s in entries:
        sid = s['id']
        if sid in trainer:
            t = trainer[sid]
            if t['req_level'] <= 60:  # up to level 60
                learnable.append((t['req_level'], s))

    if not learnable:
        # Try finding by spell name match without trainer filter
        print(f"\n=== {spell_name} === (no trainer entries found)")
        # Show all with mana_pct > 0 as likely player spells
        for s in sorted(entries, key=lambda x: x['id']):
            if s['mana_pct'] > 0:
                ct = cast_times.get(s['cast_time_idx'], 0)
                dur = durations.get(s['duration_idx'], 0)
                rng = ranges.get(s['spell_range_idx'], 0)
                sp = bonus.get(s['id'], {})
                print(f"  ID={s['id']:5d}  Mana={s['mana_pct']:2d}%  "
                      f"Cast={ct}ms  Dur={dur}ms  Range={rng:.0f}  "
                      f"E1: base={s['eff1_base']}+{s['eff1_die']}d  RPL={s['eff1_rpl']:.1f} maxLvl={s['eff1_maxlvl']}  "
                      f"E2: base={s['eff2_base']}+{s['eff2_die']}d  "
                      f"SP={sp}")
        continue

    learnable.sort(key=lambda x: x[0])

    print(f"\n=== {spell_name} === ({len(learnable)} ranks)")
    print(f"  {'Rank':>4} {'ID':>6} {'Lvl':>3} {'Mana%':>5} {'Cast':>6} {'Dur':>8} {'Range':>5}  "
          f"{'E1base':>6} {'E1die':>5} {'E1RPL':>6} {'E1max':>5}  "
          f"{'E2base':>6} {'E2die':>5}  "
          f"{'SP_direct':>9} {'SP_dot':>6}  "
          f"{'E1aura':>6} {'E1misc':>6}")

    for rank, (level, s) in enumerate(learnable, 1):
        ct = cast_times.get(s['cast_time_idx'], 0)
        dur = durations.get(s['duration_idx'], 0)
        rng = ranges.get(s['spell_range_idx'], 0)
        sp = bonus.get(s['id'], {})
        sp_d = sp.get('direct_bonus', -1)
        sp_dot = sp.get('dot_bonus', -1)

        print(f"  R{rank:>3} {s['id']:>6} L{level:>2}  {s['mana_pct']:>4}% {ct:>5}ms {dur:>7}ms {rng:>5.0f}  "
              f"{s['eff1_base']:>6} {s['eff1_die']:>5} {s['eff1_rpl']:>6.2f} {s['eff1_maxlvl']:>5}  "
              f"{s['eff2_base']:>6} {s['eff2_die']:>5}  "
              f"{sp_d:>9.4f} {sp_dot:>6.4f}  "
              f"{s['eff1_aura']:>6} {s['eff1_misc']:>6}")

print("\n\n=== TRAINER IDs for Priest ===")
# Find which trainerIDs teach priest spells
priest_trainer_ids = set()
for sid, s in spells_dbc.items():
    if s['name'] in PRIEST_SPELL_NAMES and sid in trainer:
        priest_trainer_ids.add(trainer[sid]['trainer_id'])
print(f"Trainer IDs: {sorted(priest_trainer_ids)}")
