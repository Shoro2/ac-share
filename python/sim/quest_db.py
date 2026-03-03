"""
Quest system for the WoW combat simulation.

Provides quest definitions, objective tracking, and quest NPC data for
Northshire Valley. Follows the CreatureDB/LootDB pattern: hardcoded starter
quests with extension points for CSV loading from AzerothCore DB exports.

Quest types supported:
  - KILL: Kill N creatures of a specific entry
  - COLLECT: Collect N quest items (drops from specific creatures)
  - EXPLORE: Visit a location (x, y within radius)

Usage:
    quest_db = QuestDB()
    available = quest_db.get_available_quests(npc_entry=823, player_level=1,
                                              completed=set(), active={})
    # available = [QuestTemplate(quest_id=33, title="Wolves Across the Border", ...)]
"""

import csv
import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


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
    """Definition of a quest — loaded from hardcoded data or CSV."""
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


# ─── Northshire Valley Quest NPCs ─────────────────────────────────────

QUEST_NPC_DATA = [
    QuestNPCData(entry=823,  name="Deputy Willem",    x=-8949.0, y=-152.0, z=82.0),
    QuestNPCData(entry=197,  name="Marshal McBride",  x=-8914.0, y=-133.0, z=82.0),
    QuestNPCData(entry=6774, name="Brother Neals",    x=-8899.0, y=-170.0, z=82.0),
]


# ─── Northshire Valley Quests ──────────────────────────────────────────
# Based on real WoW quests, using the mob entries already in combat_sim.py:
#   299  = Young Wolf (L1)
#   1984 = Diseased Young Wolf (L1-2)
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
                source_creature=299,     # Diseased Young Wolf (entry 299)
                drop_chance=0.5,
                description="Collect 6 Diseased Wolf Pelts",
            ),
        ],
        rewards=QuestReward(xp=360, copper=75),
    ),
}


class QuestDB:
    """
    Quest database — hardcoded Northshire quests with optional CSV extension.

    Follows the CreatureDB/LootDB pattern:
    - Constructor loads hardcoded data
    - Optional data_dir enables CSV loading
    - Graceful degradation (always has at least the hardcoded quests)
    """

    def __init__(self, data_dir: str = None, quiet: bool = False):
        self.templates: dict[int, QuestTemplate] = dict(QUEST_TEMPLATES)
        self.npc_data: list[QuestNPCData] = list(QUEST_NPC_DATA)

        # Build lookup maps: npc_entry -> [quest_ids]
        self.giver_map: dict[int, list[int]] = {}
        self.ender_map: dict[int, list[int]] = {}
        self._build_maps()

        if data_dir:
            self._load_csv(data_dir, quiet)

        if not quiet:
            print(f"  QuestDB: {len(self.templates)} quests, "
                  f"{len(self.npc_data)} quest NPCs")

    def _build_maps(self):
        """Build giver/ender lookup maps from templates."""
        self.giver_map.clear()
        self.ender_map.clear()
        for qt in self.templates.values():
            self.giver_map.setdefault(qt.giver_entry, []).append(qt.quest_id)
            self.ender_map.setdefault(qt.ender_entry, []).append(qt.quest_id)

    def _load_csv(self, data_dir: str, quiet: bool):
        """Load quest_template.csv — placeholder for AzerothCore CSV exports."""
        qt_path = os.path.join(data_dir, "quest_template.csv")
        if not os.path.isfile(qt_path):
            return
        # TODO: CSV loading similar to creature_db.py
        # For now, only hardcoded quests are used.

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
