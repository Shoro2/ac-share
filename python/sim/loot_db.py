"""
Loads loot table data from AzerothCore CSV exports for realistic item drops.

Requires CSV exports (semicolon-delimited, double-quote enclosed):
  - creature_loot_template.csv  (Entry;Item;Reference;Chance;QuestRequired;LootMode;GroupId;MinCount;MaxCount;Comment)
  - item_template.csv           (entry;name;class;subclass;Quality;SellPrice;InventoryType;ItemLevel;armor;dmg_min1;dmg_max1;delay;stat_type1..10;stat_value1..10;...)
  - reference_loot_template.csv (optional, same schema as creature_loot_template)

Implements the full AzerothCore loot group system:
  - Group 0: each entry rolls independently (chance %)
  - Group N>0: exactly one entry wins per group (weighted selection)
  - chance=0 in grouped entries: equal share of remaining probability
  - References: resolved recursively from reference_loot_template

Usage:
    loot_db = LootDB("/path/to/data")
    if loot_db.loaded:
        results = loot_db.roll_loot(creature_loot_id, rng)
        # results = [LootResult(item=ItemData(...), count=1), ...]
"""

import csv
import os
import random
from dataclasses import dataclass


@dataclass(slots=True)
class ItemData:
    """Precomputed item data from item_template."""
    entry: int
    name: str
    quality: int           # 0=Poor(grey), 1=Common, 2=Uncommon, 3=Rare, 4=Epic
    sell_price: int        # copper
    inventory_type: int    # 0=non-equip, 1=Head, 2=Neck, ..., 14=Shield, etc.
    item_level: int
    item_class: int        # 0=Consumable, 2=Weapon, 4=Armor, ...
    item_subclass: int
    score: float           # precomputed GetItemScore
    # WotLK 3.3.5 item stats — individual stat types and values
    stats: dict            # {stat_type_int: value}, e.g. {7: 5, 5: 3} = +5 Stam, +3 Int
    armor: int             # armor value for armor items
    weapon_dps: float      # (dmg_min+dmg_max)/2 / speed for weapons


@dataclass(slots=True)
class LootEntry:
    """One row from creature_loot_template / reference_loot_template."""
    item: int
    reference: int
    chance: float
    quest_required: int
    loot_mode: int
    group_id: int
    min_count: int
    max_count: int


@dataclass(slots=True)
class LootResult:
    """One item produced by rolling a loot table."""
    item: ItemData
    count: int


class LootDB:
    """Loads AzerothCore loot tables and item data for realistic item drops.

    Auto-discovers CSV files in the given directory. Missing files are silently
    skipped — check `loaded` property before using `roll_loot`.
    """

    MAX_REFERENCE_DEPTH = 5  # prevent infinite loops from circular refs

    def __init__(self, data_dir: str, quiet: bool = False):
        self.items: dict[int, ItemData] = {}
        self.creature_loot: dict[int, list[LootEntry]] = {}
        self.reference_loot: dict[int, list[LootEntry]] = {}
        self._quiet = quiet

        item_path = os.path.join(data_dir, 'item_template.csv')
        creature_loot_path = os.path.join(data_dir, 'creature_loot_template.csv')
        ref_loot_path = os.path.join(data_dir, 'reference_loot_template.csv')

        if os.path.exists(item_path):
            self._load_items(item_path)
        if os.path.exists(creature_loot_path):
            self._load_loot_table(creature_loot_path, self.creature_loot)
        if os.path.exists(ref_loot_path):
            self._load_loot_table(ref_loot_path, self.reference_loot)

        if not quiet:
            total_entries = sum(len(v) for v in self.creature_loot.values())
            print(f"  [LootDB] {len(self.items)} items, "
                  f"{len(self.creature_loot)} creature loot tables "
                  f"({total_entries} entries), "
                  f"{len(self.reference_loot)} reference tables")

    @property
    def loaded(self) -> bool:
        """True if both items and at least one loot table are loaded."""
        return len(self.items) > 0 and len(self.creature_loot) > 0

    # ─── CSV Loading ──────────────────────────────────────────────

    def _load_items(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';', quotechar='"')
            for row in reader:
                entry = int(row['entry'])
                score = self._compute_score(row)
                stats = self._parse_stats(row)
                armor = int(row.get('armor', 0))
                dmg_min = float(row.get('dmg_min1', 0))
                dmg_max = float(row.get('dmg_max1', 0))
                delay = int(row.get('delay', 1000))
                weapon_dps = 0.0
                if dmg_max > 0 and delay > 0:
                    weapon_dps = (dmg_min + dmg_max) / 2.0 / (delay / 1000.0)
                self.items[entry] = ItemData(
                    entry=entry,
                    name=row['name'],
                    quality=int(row.get('Quality', 0)),
                    sell_price=int(row.get('SellPrice', 0)),
                    inventory_type=int(row.get('InventoryType', 0)),
                    item_level=int(row.get('ItemLevel', 0)),
                    item_class=int(row.get('class', 0)),
                    item_subclass=int(row.get('subclass', 0)),
                    score=score,
                    stats=stats,
                    armor=armor,
                    weapon_dps=weapon_dps,
                )

    @staticmethod
    def _parse_stats(row: dict) -> dict:
        """Parse individual stat_type/stat_value pairs from item_template.

        Returns {stat_type: value} dict. WotLK ITEM_MOD enum:
          0=Mana, 1=Health, 3=Agility, 4=Strength, 5=Intellect, 6=Spirit,
          7=Stamina, 12=DefenseRating, 31=HitRating, 32=CritRating,
          35=Resilience, 36=HasteRating, 38=AttackPower, 43=MP5,
          45=SpellPower, 46=HP5, 47=SpellPenetration, etc.
        """
        stats = {}
        for i in range(1, 11):
            st = int(row.get(f'stat_type{i}', 0))
            sv = int(row.get(f'stat_value{i}', 0))
            if sv != 0:
                stats[st] = stats.get(st, 0) + sv
        return stats

    @staticmethod
    def _compute_score(row: dict) -> float:
        """Compute item score matching C++ GetItemScore formula.

        score = (Quality * 10) + ItemLevel + Armor + WeaponDPS + (TotalStats * 2)
        """
        quality = int(row.get('Quality', 0))
        item_level = int(row.get('ItemLevel', 0))
        armor = int(row.get('armor', 0))

        # Weapon DPS
        dps = 0.0
        dmg_min = float(row.get('dmg_min1', 0))
        dmg_max = float(row.get('dmg_max1', 0))
        delay = int(row.get('delay', 1000))
        if dmg_max > 0 and delay > 0:
            dps = (dmg_min + dmg_max) / 2.0 / (delay / 1000.0)

        # Sum all stat values (stat_value1 through stat_value10)
        total_stats = 0
        for i in range(1, 11):
            val = int(row.get(f'stat_value{i}', 0))
            total_stats += abs(val)

        return (quality * 10) + item_level + armor + dps + (total_stats * 2)

    def _load_loot_table(self, path: str, target: dict):
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';', quotechar='"')
            for row in reader:
                entry_id = int(row['Entry'])
                le = LootEntry(
                    item=int(row['Item']),
                    reference=int(row.get('Reference', 0)),
                    chance=float(row.get('Chance', 0)),
                    quest_required=int(row.get('QuestRequired', 0)),
                    loot_mode=int(row.get('LootMode', 1)),
                    group_id=int(row.get('GroupId', 0)),
                    min_count=int(row.get('MinCount', 1)),
                    max_count=int(row.get('MaxCount', 1)),
                )
                if entry_id not in target:
                    target[entry_id] = []
                target[entry_id].append(le)

    # ─── Loot Rolling ─────────────────────────────────────────────

    def roll_loot(self, loot_id: int, rng: random.Random) -> list[LootResult]:
        """Roll the loot table for a creature.

        Implements AzerothCore group logic:
        - Group 0: each entry rolls independently (chance %)
        - Group N>0: exactly one entry wins per group (weighted selection)
        - References: resolved recursively from reference_loot_template
        """
        entries = self.creature_loot.get(loot_id)
        if not entries:
            return []
        return self._process_entries(entries, rng, depth=0)

    def _process_entries(self, entries: list[LootEntry],
                         rng: random.Random, depth: int) -> list[LootResult]:
        if depth > self.MAX_REFERENCE_DEPTH:
            return []

        results: list[LootResult] = []

        # Separate by group, skip quest-required items
        groups: dict[int, list[LootEntry]] = {}
        for e in entries:
            if e.quest_required:
                continue
            groups.setdefault(e.group_id, []).append(e)

        # Group 0: independent rolls (each entry has its own chance)
        for entry in groups.get(0, []):
            if entry.chance <= 0:
                # chance=0 in group 0 = guaranteed drop
                self._resolve_entry(entry, rng, results, depth)
            elif rng.random() * 100.0 < entry.chance:
                self._resolve_entry(entry, rng, results, depth)

        # Groups 1+: one winner per group
        for gid, group_entries in sorted(groups.items()):
            if gid == 0:
                continue

            # Build effective chances
            nonzero = [(e, e.chance) for e in group_entries if e.chance > 0]
            zero_chance = [e for e in group_entries if e.chance <= 0]

            nonzero_total = sum(c for _, c in nonzero)

            # Entries with chance=0 share remaining probability equally
            if zero_chance:
                remaining = max(0.0, 100.0 - nonzero_total)
                equal_share = remaining / len(zero_chance) if remaining > 0 else 0.0
                effective = nonzero + [(e, equal_share) for e in zero_chance]
            else:
                effective = nonzero

            total_chance = sum(c for _, c in effective)
            if total_chance <= 0:
                continue

            # If total < 100, there's a chance nothing drops from this group
            roll = rng.random() * 100.0
            if roll >= total_chance:
                continue

            # Pick one winner within the group
            pick = rng.random() * total_chance
            cumulative = 0.0
            for entry, chance in effective:
                cumulative += chance
                if pick < cumulative:
                    self._resolve_entry(entry, rng, results, depth)
                    break

        return results

    def _resolve_entry(self, entry: LootEntry, rng: random.Random,
                       results: list[LootResult], depth: int):
        """Resolve a loot entry — either a direct item or a reference."""
        if entry.reference != 0:
            # Reference: process the referenced template
            # MaxCount on a reference = how many times to process it
            ref_entries = self.reference_loot.get(entry.reference, [])
            if ref_entries:
                times = max(1, entry.max_count)
                for _ in range(times):
                    sub = self._process_entries(ref_entries, rng, depth + 1)
                    results.extend(sub)
        else:
            item = self.items.get(entry.item)
            if item:
                count = rng.randint(entry.min_count,
                                    max(entry.min_count, entry.max_count))
                results.append(LootResult(item=item, count=count))

    # ─── Utilities ────────────────────────────────────────────────

    def get_item(self, entry: int) -> ItemData | None:
        """Look up an item by entry ID."""
        return self.items.get(entry)

    def has_loot_table(self, loot_id: int) -> bool:
        """Check if a creature loot table exists for the given ID."""
        return loot_id in self.creature_loot
