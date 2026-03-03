"""
Quest system for the WoW combat simulation.

Loads quest definitions from AzerothCore CSV exports:
  - quest_template.csv             (quest definitions, objectives, rewards)
  - quest_template_addon.csv       (chain info: PrevQuestID, NextQuestID)
  - creature_queststarter[r].csv   (NPC -> quest giver mapping)
  - creature_questender.csv        (NPC -> quest ender mapping)
  - creature_template.csv          (NPC names — reused from CreatureDB)
  - creature.csv                   (NPC spawn positions — reused from CreatureDB)

Falls back to hardcoded Northshire quests when CSVs are not available.

Quest types supported:
  - KILL: Kill N creatures of a specific entry (RequiredNpcOrGo > 0)
  - COLLECT: Collect N quest items (RequiredItemId, source from loot tables)
  - EXPLORE: Visit a location (hardcoded only — no CSV equivalent)

Usage:
    quest_db = QuestDB("data/")          # load from CSV
    quest_db = QuestDB()                  # hardcoded fallback
    available = quest_db.get_available_quests(npc_entry=823, player_level=1,
                                              completed=set(), active={})
"""

import csv
import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ─── QuestXP.dbc Approximation ─────────────────────────────────────────
# Without the real DBC file, we approximate quest XP from quest level +
# difficulty index using anchor-based interpolation.
#
# Difficulty indices (from QuestXP.dbc columns):
#   0 = always 0
#   1 = trivial ("speak with X", nearby)
#   2 = easy delivery quests
#   3 = moderate travel quests
#   4 = standard quests (low kill count)
#   5 = standard kill quests
#   6 = extended kill quests (high count)
#   7 = elite / group quests
#   8 = chain-end / hard group quests
#   9 = always 0

_QUEST_XP_LEVELS = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
_QUEST_XP_BASE_D5 = [100, 250, 400, 550, 900, 1350, 1800, 2300, 2850,
                     3900, 5100, 6300, 9500, 16800]
_QUEST_XP_DIFF_MULT = [0.0, 0.40, 0.52, 0.64, 0.78, 1.00,
                       1.16, 1.34, 1.56, 0.0]


def _interpolate(x: int, xs: list, ys: list) -> int:
    """Linear interpolation between anchor points."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return max(1, int(ys[i] + t * (ys[i + 1] - ys[i])))
    return ys[-1]


def _estimate_quest_xp(quest_level: int, difficulty: int) -> int:
    """Approximate QuestXP.dbc lookup.

    Returns base XP for a quest at the given level and difficulty index.
    Uses anchor-based interpolation matching creature_db.py pattern.
    """
    if difficulty <= 0 or difficulty >= 9 or quest_level <= 0:
        return 0
    base = _interpolate(quest_level, _QUEST_XP_LEVELS, _QUEST_XP_BASE_D5)
    return max(1, int(base * _QUEST_XP_DIFF_MULT[difficulty]))


class QuestObjectiveType(IntEnum):
    KILL = 0       # Kill N creatures of entry X
    COLLECT = 1    # Collect N quest items (drop from creatures when quest active)
    EXPLORE = 2    # Visit location (x, y) within radius


@dataclass(slots=True)
class QuestObjective:
    """Single quest objective (one quest can have multiple)."""
    obj_type: QuestObjectiveType
    target: int              # creature_entry (KILL), item_id (COLLECT), or 0 (EXPLORE)
    count: int               # required count
    # COLLECT: which creatures drop this item and at what rate
    source_creature: int = 0  # creature entry that drops the item
    drop_chance: float = 0.5  # probability per kill
    # EXPLORE: target location
    target_x: float = 0.0
    target_y: float = 0.0
    radius: float = 25.0
    description: str = ""


@dataclass(slots=True)
class QuestReward:
    """Rewards granted on quest turn-in."""
    xp: int = 0
    copper: int = 0


@dataclass(slots=True)
class QuestTemplate:
    """Definition of a quest — loaded from CSV or hardcoded data."""
    quest_id: int
    title: str
    min_level: int = 1
    quest_level: int = 1
    giver_entry: int = 0      # NPC entry who gives the quest
    ender_entry: int = 0      # NPC entry who completes the quest
    objectives: list = field(default_factory=list)   # list[QuestObjective]
    rewards: QuestReward = field(default_factory=QuestReward)
    next_quest: int = 0       # chain: next quest ID (0 = none)
    prev_quest: int = 0       # chain: required previous quest (0 = none)


@dataclass
class QuestProgress:
    """Tracks progress for one active quest."""
    quest_id: int
    counts: list              # progress per objective (parallel to QuestTemplate.objectives)
    completed: bool = False   # True when all objectives met

    def check_complete(self, objectives: list) -> bool:
        """Update completed flag based on current counts vs required."""
        self.completed = all(
            c >= obj.count for c, obj in zip(self.counts, objectives)
        )
        return self.completed


@dataclass(slots=True)
class QuestNPCData:
    """Static quest NPC spawn data."""
    entry: int
    name: str
    x: float
    y: float
    z: float = 82.0


# ─── Hardcoded Northshire Valley Quest NPCs ─────────────────────────────
QUEST_NPC_DATA = [
    QuestNPCData(entry=823,  name="Deputy Willem",    x=-8949.0, y=-152.0, z=82.0),
    QuestNPCData(entry=197,  name="Marshal McBride",  x=-8914.0, y=-133.0, z=82.0),
    QuestNPCData(entry=6774, name="Brother Neals",    x=-8899.0, y=-170.0, z=82.0),
]

# ─── Hardcoded Northshire Valley Quests ──────────────────────────────────
# Fallback when CSVs are not available. Based on real WoW quests.
#   299  = Young Wolf (L1)
#   6    = Kobold Vermin (L1-2)
#   40   = Kobold Worker (L2-3)

QUEST_TEMPLATES = {
    # ── Quest chain 1: Deputy Willem's wolf quests ──
    33: QuestTemplate(
        quest_id=33,
        title="Wolves Across the Border",
        min_level=1, quest_level=2,
        giver_entry=823, ender_entry=823,  # Deputy Willem
        objectives=[
            QuestObjective(
                obj_type=QuestObjectiveType.KILL,
                target=299, count=10,
                description="Kill 10 Young Wolves",
            ),
        ],
        rewards=QuestReward(xp=250, copper=50),
        next_quest=7,
    ),

    # ── Quest chain 2: Marshal McBride's kobold quests ──
    7: QuestTemplate(
        quest_id=7,
        title="Kobold Camp Cleanup",
        min_level=1, quest_level=3,
        giver_entry=197, ender_entry=197,  # Marshal McBride
        objectives=[
            QuestObjective(
                obj_type=QuestObjectiveType.KILL,
                target=6, count=10,
                description="Kill 10 Kobold Vermin",
            ),
        ],
        rewards=QuestReward(xp=450, copper=100),
        prev_quest=33,
        next_quest=15,
    ),

    15: QuestTemplate(
        quest_id=15,
        title="Investigate Echo Ridge",
        min_level=2, quest_level=4,
        giver_entry=197, ender_entry=197,  # Marshal McBride
        objectives=[
            QuestObjective(
                obj_type=QuestObjectiveType.KILL,
                target=40, count=10,
                description="Kill 10 Kobold Workers",
            ),
        ],
        rewards=QuestReward(xp=675, copper=200),
        prev_quest=7,
    ),

    # ── Standalone: Brother Neals exploration quest ──
    100001: QuestTemplate(
        quest_id=100001,
        title="Scout the Vineyards",
        min_level=1, quest_level=2,
        giver_entry=6774, ender_entry=6774,  # Brother Neals
        objectives=[
            QuestObjective(
                obj_type=QuestObjectiveType.EXPLORE,
                target=0, count=1,
                target_x=-8860.0, target_y=-60.0, radius=30.0,
                description="Scout the vineyard area",
            ),
        ],
        rewards=QuestReward(xp=170, copper=35),
    ),

    # ── Standalone: Deputy Willem collect quest ──
    100002: QuestTemplate(
        quest_id=100002,
        title="Diseased Wolf Pelts",
        min_level=1, quest_level=2,
        giver_entry=823, ender_entry=823,  # Deputy Willem
        objectives=[
            QuestObjective(
                obj_type=QuestObjectiveType.COLLECT,
                target=100001, count=6,  # quest item ID (virtual)
                source_creature=299,     # Young Wolf (entry 299)
                drop_chance=0.5,
                description="Collect 6 Diseased Wolf Pelts",
            ),
        ],
        rewards=QuestReward(xp=360, copper=75),
    ),
}


class QuestDB:
    """Quest database — loads from AzerothCore CSV exports with hardcoded fallback.

    Follows the CreatureDB/LootDB pattern:
    - Without data_dir: uses hardcoded Northshire quests (5 quests, 3 NPCs)
    - With data_dir: loads from CSV, keeps hardcoded custom quests (ID >= 100000)
    - Graceful degradation: missing CSVs are silently skipped
    """

    def __init__(self, data_dir: str = None, quiet: bool = False):
        self.templates: dict[int, QuestTemplate] = dict(QUEST_TEMPLATES)
        self.npc_data: list[QuestNPCData] = list(QUEST_NPC_DATA)
        self._csv_loaded = False

        if data_dir:
            self._load_csv(data_dir, quiet)

        # Build lookup maps: npc_entry -> [quest_ids]
        self.giver_map: dict[int, list[int]] = {}
        self.ender_map: dict[int, list[int]] = {}
        self._build_maps()

        if not quiet:
            src = "CSV" if self._csv_loaded else "hardcoded"
            print(f"  [QuestDB] {len(self.templates)} quests, "
                  f"{len(self.npc_data)} quest NPCs ({src})")

    @property
    def loaded(self) -> bool:
        """True if quest data was loaded from CSV (vs hardcoded only)."""
        return self._csv_loaded

    def _build_maps(self):
        """Build giver/ender lookup maps from templates."""
        self.giver_map.clear()
        self.ender_map.clear()
        for qt in self.templates.values():
            if qt.giver_entry:
                self.giver_map.setdefault(qt.giver_entry, []).append(qt.quest_id)
            if qt.ender_entry:
                self.ender_map.setdefault(qt.ender_entry, []).append(qt.quest_id)

    # ─── CSV Loading ──────────────────────────────────────────────

    def _load_csv(self, data_dir: str, quiet: bool):
        """Load quest data from AzerothCore CSV exports.

        Reads quest_template.csv + related CSVs. Replaces hardcoded quests
        (ID < 100000) with CSV data, keeps custom quests (ID >= 100000).
        """
        qt_path = os.path.join(data_dir, 'quest_template.csv')
        if not os.path.isfile(qt_path):
            return

        # 1. Load quest templates
        csv_templates = self._load_quest_templates(qt_path)
        if not csv_templates:
            return

        # 2. Load chain info (PrevQuestID, NextQuestID)
        addon_path = os.path.join(data_dir, 'quest_template_addon.csv')
        if os.path.isfile(addon_path):
            self._load_addon(addon_path, csv_templates)

        # 3. Load quest giver NPCs (handle typo: creature_queststarterr.csv)
        giver_path = os.path.join(data_dir, 'creature_queststarter.csv')
        if not os.path.isfile(giver_path):
            giver_path = os.path.join(data_dir, 'creature_queststarterr.csv')
        if os.path.isfile(giver_path):
            self._load_quest_relations(giver_path, csv_templates, 'giver')

        # 4. Load quest ender NPCs
        ender_path = os.path.join(data_dir, 'creature_questender.csv')
        if os.path.isfile(ender_path):
            self._load_quest_relations(ender_path, csv_templates, 'ender')

        # 5. Replace hardcoded templates — keep custom quests (ID >= 100000)
        self.templates = {qid: qt for qid, qt in self.templates.items()
                          if qid >= 100000}
        self.templates.update(csv_templates)

        # 6. Build quest NPC spawn data from creature CSVs
        self._build_npc_data_from_csv(data_dir)

        self._csv_loaded = True
        if not quiet:
            with_obj = sum(1 for qt in csv_templates.values() if qt.objectives)
            print(f"  [QuestDB] Loaded {len(csv_templates)} quests from CSV "
                  f"({with_obj} with objectives)")

    def _load_quest_templates(self, path: str) -> dict:
        """Parse quest_template.csv into QuestTemplate objects."""
        templates = {}
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';', quotechar='"')
            for row in reader:
                quest_id = int(row['ID'])
                quest_level = int(row.get('QuestLevel', 0))
                min_level = int(row.get('MinLevel', 1))
                title = row.get('LogTitle', f'Quest {quest_id}')

                # Parse objectives from RequiredNpcOrGo + RequiredItemId
                objectives = self._parse_objectives(row)

                # XP reward (approximate QuestXP.dbc lookup)
                xp_difficulty = int(row.get('RewardXPDifficulty', 0))
                xp = _estimate_quest_xp(max(1, quest_level), xp_difficulty)

                # Money reward (negative = cost to complete)
                copper = max(0, int(row.get('RewardMoney', 0)))

                # RewardNextQuest (chain at turn-in)
                next_quest = int(row.get('RewardNextQuest', 0))

                templates[quest_id] = QuestTemplate(
                    quest_id=quest_id,
                    title=title,
                    min_level=max(1, min_level),
                    quest_level=max(1, quest_level),
                    objectives=objectives,
                    rewards=QuestReward(xp=xp, copper=copper),
                    next_quest=next_quest,
                )
        return templates

    @staticmethod
    def _parse_objectives(row: dict) -> list:
        """Parse KILL and COLLECT objectives from a quest_template CSV row.

        KILL: RequiredNpcOrGo1-4 > 0 with RequiredNpcOrGoCount1-4
        COLLECT: RequiredItemId1-6 with RequiredItemCount1-6
              (source_creature is heuristic: first RequiredNpcOrGo entry)
        """
        objectives = []

        # KILL objectives (RequiredNpcOrGo > 0 = creature entry)
        kill_creatures = []
        for i in range(1, 5):
            npc_or_go = int(row.get(f'RequiredNpcOrGo{i}', 0))
            count = int(row.get(f'RequiredNpcOrGoCount{i}', 0))
            if npc_or_go > 0 and count > 0:
                kill_creatures.append(npc_or_go)
                desc = row.get(f'ObjectiveText{i}', '') or ''
                objectives.append(QuestObjective(
                    obj_type=QuestObjectiveType.KILL,
                    target=npc_or_go,
                    count=count,
                    description=desc,
                ))

        # COLLECT objectives (RequiredItemId > 0 = item entry)
        for i in range(1, 7):
            item_id = int(row.get(f'RequiredItemId{i}', 0))
            count = int(row.get(f'RequiredItemCount{i}', 0))
            if item_id > 0 and count > 0:
                # Heuristic: use first kill creature as item source
                source = kill_creatures[0] if kill_creatures else 0
                obj_idx = min(i, 4)
                desc = row.get(f'ObjectiveText{obj_idx}', '') or ''
                objectives.append(QuestObjective(
                    obj_type=QuestObjectiveType.COLLECT,
                    target=item_id,
                    count=count,
                    source_creature=source,
                    drop_chance=0.5,
                    description=desc,
                ))

        return objectives

    def _load_addon(self, path: str, templates: dict):
        """Parse quest_template_addon.csv for PrevQuestID and NextQuestID."""
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';', quotechar='"')
            for row in reader:
                quest_id = int(row['ID'])
                if quest_id not in templates:
                    continue
                qt = templates[quest_id]
                prev = int(row.get('PrevQuestID', 0))
                next_q = int(row.get('NextQuestID', 0))
                # PrevQuestID > 0: must be completed first
                # PrevQuestID < 0: must be active (not supported yet)
                if prev > 0:
                    qt.prev_quest = prev
                # NextQuestID supplements RewardNextQuest
                if next_q > 0 and qt.next_quest == 0:
                    qt.next_quest = next_q

    def _load_quest_relations(self, path: str, templates: dict, kind: str):
        """Parse creature_queststarter/questender CSV for NPC-quest mappings."""
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';', quotechar='"')
            for row in reader:
                quest_id = int(row['quest'])
                npc_entry = int(row['id'])
                if quest_id not in templates:
                    continue
                qt = templates[quest_id]
                # First NPC wins (most quests have a single giver/ender)
                if kind == 'giver' and qt.giver_entry == 0:
                    qt.giver_entry = npc_entry
                elif kind == 'ender' and qt.ender_entry == 0:
                    qt.ender_entry = npc_entry

    def _build_npc_data_from_csv(self, data_dir: str):
        """Build QuestNPCData list from creature CSVs for all quest givers/enders."""
        # Collect all unique NPC entries
        npc_entries = set()
        for qt in self.templates.values():
            if qt.giver_entry > 0:
                npc_entries.add(qt.giver_entry)
            if qt.ender_entry > 0:
                npc_entries.add(qt.ender_entry)

        if not npc_entries:
            return

        # Skip entries we already have from hardcoded data
        existing = {npc.entry for npc in self.npc_data}
        needed = npc_entries - existing
        if not needed:
            return

        # Load NPC names from creature_template.csv
        npc_names: dict[int, str] = {}
        tmpl_path = os.path.join(data_dir, 'creature_template.csv')
        if os.path.isfile(tmpl_path):
            with open(tmpl_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';', quotechar='"')
                for row in reader:
                    entry = int(row['entry'])
                    if entry in needed:
                        npc_names[entry] = row['name']

        # Load first spawn position per NPC from creature.csv
        npc_positions: dict[int, tuple] = {}
        spawn_path = os.path.join(data_dir, 'creature.csv')
        if os.path.isfile(spawn_path):
            with open(spawn_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';', quotechar='"')
                for row in reader:
                    entry = int(row['id1'])
                    if entry in needed and entry not in npc_positions:
                        npc_positions[entry] = (
                            float(row['position_x']),
                            float(row['position_y']),
                            float(row['position_z']),
                        )

        # Create QuestNPCData for each found NPC
        for entry in sorted(needed):
            pos = npc_positions.get(entry)
            if pos is None:
                continue
            name = npc_names.get(entry, f'NPC {entry}')
            self.npc_data.append(QuestNPCData(
                entry=entry,
                name=name,
                x=pos[0], y=pos[1], z=pos[2],
            ))

    # ─── Public API ──────────────────────────────────────────────

    def get_available_quests(self, npc_entry: int, player_level: int,
                            completed: set, active: dict) -> list[QuestTemplate]:
        """Get quests this NPC can offer the player right now.

        Args:
            npc_entry: NPC creature entry
            player_level: Player's current level
            completed: Set of completed quest IDs
            active: Dict of quest_id -> QuestProgress for active quests
        """
        available = []
        for qid in self.giver_map.get(npc_entry, []):
            qt = self.templates[qid]
            if qid in completed or qid in active:
                continue
            if player_level < qt.min_level:
                continue
            if qt.prev_quest and qt.prev_quest not in completed:
                continue
            available.append(qt)
        return available

    def get_completable_quests(self, npc_entry: int,
                               active: dict) -> list[QuestTemplate]:
        """Get quests this NPC can complete that are ready for turn-in.

        Args:
            npc_entry: NPC creature entry
            active: Dict of quest_id -> QuestProgress for active quests
        """
        completable = []
        for qid in self.ender_map.get(npc_entry, []):
            if qid in active and active[qid].completed:
                completable.append(self.templates[qid])
        return completable

    def create_progress(self, quest_id: int) -> QuestProgress:
        """Create a fresh QuestProgress for the given quest."""
        qt = self.templates[quest_id]
        return QuestProgress(
            quest_id=quest_id,
            counts=[0] * len(qt.objectives),
        )
