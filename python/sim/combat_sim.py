"""
WoW Combat Simulation Engine — Pure Python, no server needed.

Simulates WoW 3.3.5 WotLK characters with full stat system.
All formulas derived from AzerothCore C++ source (StatSystem.cpp,
Player.cpp, Unit.cpp, DBC game tables).

Supports all 10 classes (stat framework), leveling 1–80.
Currently only Priest has spell implementations.

Tick-based: 1 tick = 0.5 seconds (matches WoWEnv decision interval).
"""

import math
import os
import random
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sim.terrain import SimTerrain
    from sim.creature_db import CreatureDB
    from sim.loot_db import LootDB
    from sim.quest_db import QuestDB

# Re-export all public names from submodules so that existing imports
# like "from sim.combat_sim import XP_TABLE" continue to work unchanged.

from sim.constants import *  # noqa: F401,F403 — re-export all constants
from sim.formulas import *   # noqa: F401,F403 — re-export all formulas
from sim.models import *     # noqa: F401,F403 — re-export all models

# Explicit imports used by CombatSimulation class below
from sim.constants import (
    XP_TABLE, MAX_LEVEL, base_xp_gain,
    CLASS_PRIEST, CLASS_WARRIOR, CLASS_PALADIN, CLASS_HUNTER, CLASS_ROGUE,
    CLASS_DEATH_KNIGHT, CLASS_SHAMAN, CLASS_MAGE, CLASS_WARLOCK, CLASS_DRUID,
    CLASS_NAMES, CLASS_POWER_TYPE, POWER_MANA,
    ITEM_MOD_MANA, ITEM_MOD_HEALTH, ITEM_MOD_AGILITY, ITEM_MOD_STRENGTH,
    ITEM_MOD_INTELLECT, ITEM_MOD_SPIRIT, ITEM_MOD_STAMINA,
    ITEM_MOD_DEFENSE_SKILL_RATING, ITEM_MOD_DODGE_RATING, ITEM_MOD_PARRY_RATING,
    ITEM_MOD_BLOCK_RATING, ITEM_MOD_HIT_MELEE_RATING, ITEM_MOD_HIT_RANGED_RATING,
    ITEM_MOD_HIT_SPELL_RATING, ITEM_MOD_CRIT_MELEE_RATING, ITEM_MOD_CRIT_RANGED_RATING,
    ITEM_MOD_CRIT_SPELL_RATING, ITEM_MOD_HASTE_MELEE_RATING, ITEM_MOD_HASTE_RANGED_RATING,
    ITEM_MOD_HASTE_SPELL_RATING, ITEM_MOD_HIT_RATING, ITEM_MOD_CRIT_RATING,
    ITEM_MOD_RESILIENCE_RATING, ITEM_MOD_HASTE_RATING, ITEM_MOD_EXPERTISE_RATING,
    ITEM_MOD_ATTACK_POWER, ITEM_MOD_RANGED_ATTACK_POWER, ITEM_MOD_SPELL_HEALING_DONE,
    ITEM_MOD_SPELL_DAMAGE_DONE, ITEM_MOD_MANA_REGENERATION, ITEM_MOD_ARMOR_PENETRATION_RATING,
    ITEM_MOD_SPELL_POWER, ITEM_MOD_HEALTH_REGEN, ITEM_MOD_BLOCK_VALUE,
    class_aware_score, CLASS_STAT_WEIGHTS,
    EQUIPMENT_SLOT_HEAD, EQUIPMENT_SLOT_NECK, EQUIPMENT_SLOT_SHOULDERS,
    EQUIPMENT_SLOT_BODY, EQUIPMENT_SLOT_CHEST, EQUIPMENT_SLOT_WAIST,
    EQUIPMENT_SLOT_LEGS, EQUIPMENT_SLOT_FEET, EQUIPMENT_SLOT_WRISTS,
    EQUIPMENT_SLOT_HANDS, EQUIPMENT_SLOT_FINGER1, EQUIPMENT_SLOT_FINGER2,
    EQUIPMENT_SLOT_TRINKET1, EQUIPMENT_SLOT_TRINKET2, EQUIPMENT_SLOT_BACK,
    EQUIPMENT_SLOT_MAINHAND, EQUIPMENT_SLOT_OFFHAND, EQUIPMENT_SLOT_RANGED,
    EQUIPMENT_SLOT_TABARD, EQUIPMENT_SLOT_END, EQUIPMENT_SLOT_NAMES,
    INVTYPE_TO_SLOTS, INVTYPE_TWO_HAND, INVTYPE_BAG,
    BAG_SLOT_START, BAG_SLOT_END,
    CLASS_BASE_STATS, CLASS_BASE_HP_MANA,
    SP_COEFF_SMITE, SP_COEFF_HEAL, SP_COEFF_MIND_BLAST,
    SP_COEFF_SW_PAIN_TICK, SP_COEFF_PW_SHIELD, SP_COEFF_RENEW_TICK,
    SP_COEFF_HOLY_FIRE, SP_COEFF_HOLY_FIRE_DOT_TICK,
    SP_COEFF_MIND_FLAY_TICK, SP_COEFF_VAMPIRIC_TOUCH_TICK,
    SPELL_LEVEL_REQ, SPELL_MANA_PCT,
    get_best_rank, FAMILY_SMITE, FAMILY_HEAL, FAMILY_FLASH_HEAL,
    FAMILY_SW_PAIN, FAMILY_PW_SHIELD, FAMILY_MIND_BLAST,
    FAMILY_RENEW, FAMILY_HOLY_FIRE, FAMILY_INNER_FIRE, FAMILY_FORTITUDE,
    FAMILY_DEVOURING_PLAGUE, FAMILY_PSYCHIC_SCREAM, FAMILY_SHADOW_PROTECTION,
    FAMILY_DIVINE_SPIRIT, FAMILY_FEAR_WARD, FAMILY_HOLY_NOVA, FAMILY_DISPEL_MAGIC,
    FAMILY_MIND_FLAY, FAMILY_VAMPIRIC_TOUCH, FAMILY_DISPERSION,
    SP_COEFF_HOLY_NOVA_HEAL,
    ALL_RANKED_SPELL_IDS,
)

from sim.formulas import (
    _rating_to_pct,
    class_base_stat, player_max_hp, player_max_mana,
    base_mana_for_level, spell_mana_cost,
    smite_damage, heal_amount, mind_blast_damage,
    renew_total_heal, holy_fire_damage, holy_fire_dot_total,
    sw_pain_total, pw_shield_absorb, inner_fire_values,
    fortitude_hp_bonus, fortitude_stamina_bonus,
    spell_direct_value, spell_dot_per_tick, spell_hot_per_tick,
    spell_shield_absorb, spell_buff_value,
    spell_crit_chance, melee_crit_chance, ranged_crit_chance,
    melee_haste_pct, ranged_haste_pct, spell_haste_pct,
    dodge_chance, parry_chance, block_chance,
    melee_attack_power, ranged_attack_power,
    expertise_pct, armor_penetration_pct, resilience_pct,
    hit_chance_melee, hit_chance_ranged, hit_chance_spell,
    spirit_mana_regen,
    # Combat resolution
    resolve_mob_melee_attack, resolve_spell_hit,
    MELEE_MISS, MELEE_DODGE, MELEE_PARRY, MELEE_BLOCK,
    MELEE_CRIT, MELEE_NORMAL, MELEE_CRUSHING,
    SPELL_MISS, SPELL_HIT, SPELL_CRIT,
)

from sim.models import (
    EquippedItem, EquippedBag, SpellDef, SPELLS,
    MobTemplate,
    InventoryItem, VendorNPC, QuestNPC, VENDOR_DATA,
    INVENTORY_SLOTS, Player, Mob,
)

from sim.talent_data import get_talent_for_level, TALENT_DEFS


# ─── Combat Simulation ───────────────────────────────────────────────

class CombatSimulation:
    """
    Pure Python combat simulation for Level 1 Priest in Northshire.

    Tick-based: 1 tick = 0.5 seconds.
    No TCP, no server, no real-time — runs as fast as Python allows.
    """

    TICK_DURATION = 0.5       # seconds per tick
    MOVE_SPEED = 3.0          # units per move_forward action
    TURN_AMOUNT = 0.5         # radians per turn action
    SCAN_RANGE = 50.0         # mob visibility range
    TARGET_RANGE = 30.0       # max targeting range
    LOOT_RANGE = 10.0         # max looting range
    SELL_RANGE = 6.0          # max vendor interaction range
    MOB_LEASH_RANGE = 60.0    # mob returns home after this distance from spawn
    OOC_DELAY_TICKS = 12      # 6 seconds out of combat before regen starts
    HP_REGEN_PER_TICK = 0.67  # 8 HP per 6 seconds = ~0.67/tick (OOC only)
    MANA_REGEN_PCT_PER_TICK = 0.02  # 2% of max_mana per tick (not casting)
    RESPAWN_TICKS = 120       # 60 seconds = 120 ticks
    MOB_SPEED = 1.0           # units per tick when chasing
    LOOT_CHANCE = 0.7         # probability of getting loot
    ITEM_SCORE_RANGE = (5, 25)
    UPGRADE_CHANCE = 0.15     # chance that looted item is an upgrade
    # Exploration grid sizes (for area/zone discovery tracking)
    AREA_CELL_SIZE = 50.0     # ~50x50 units per area cell
    ZONE_CELL_SIZE = 200.0    # ~200x200 units per zone cell
    # Chunk management (for creature_db mode)
    CHUNK_SIZE = 100.0        # world-units per chunk (must match creature_db.CHUNK_SIZE)
    CHUNK_RADIUS = 2          # activate 5×5 = 25 chunks around player

    QUEST_NPC_RANGE = 6.0     # max interaction range for quest NPCs

    def __init__(self, num_mobs: int = None, seed: Optional[int] = None,
                 terrain: 'SimTerrain | None' = None, env3d=None,
                 creature_db: 'CreatureDB | None' = None,
                 loot_db: 'LootDB | None' = None,
                 quest_db: 'QuestDB | None' = None,
                 class_id: int = CLASS_PRIEST):
        self.rng = random.Random(seed)
        self.num_mobs = num_mobs  # None = all spawns
        self.terrain = terrain
        self.env3d = env3d        # WoW3DEnvironment for area/zone lookups
        # Auto-discover data/ directory for DB fallback loading
        _data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            '..', 'data')
        _data_dir = os.path.normpath(_data_dir)
        # Auto-load creature_db from data/ directory if not provided
        if creature_db is None:
            from sim.creature_db import CreatureDB
            if os.path.isfile(os.path.join(_data_dir, 'creature_template.csv')):
                creature_db = CreatureDB(_data_dir, quiet=True)
        self.creature_db = creature_db
        # Auto-load loot_db from data/ directory if not provided
        if loot_db is None:
            from sim.loot_db import LootDB
            _loot_db = LootDB(_data_dir, quiet=True)
            if _loot_db.loaded:
                loot_db = _loot_db
        self.loot_db = loot_db    # LootDB for item drops from CSV loot tables
        self.quest_db = quest_db  # QuestDB for quest definitions and NPCs
        self.class_id = class_id  # player class
        self.map_id = 0           # Eastern Kingdoms
        self.player = Player(class_id=class_id)
        if self.terrain:
            self.player.z = self.terrain.get_height(self.player.x, self.player.y)
        self.mobs: list[Mob] = []
        self.vendors: list[VendorNPC] = []
        self.quest_npcs: list[QuestNPC] = []
        self.target: Optional[Mob] = None
        self.tick_count: int = 0
        self.damage_dealt: int = 0
        self.kills: int = 0
        self._next_uid = 1
        # Exploration tracking (uses real WoW area IDs if env3d available, grid fallback otherwise)
        self.visited_areas: set = set()   # set of area_id (or (x,y) cells as fallback)
        self.visited_zones: set = set()   # set of zone_id (or (x,y) cells as fallback)
        self.visited_maps: set = set()    # set of map_id
        self._new_areas: int = 0          # consumed on read
        self._new_zones: int = 0          # consumed on read
        self._new_maps: int = 0           # consumed on read
        # Chunk management (creature_db mode)
        self._player_chunk: Optional[tuple] = None  # (map, cx, cy)
        self._active_chunks: set[tuple] = set()
        self._chunk_mobs: dict[tuple, list[Mob]] = {}
        self._chunk_vendors: dict[tuple, list[VendorNPC]] = {}
        # Quest state
        self.active_quests: dict = {}     # quest_id -> QuestProgress
        self.completed_quests: set = set()
        self.quests_completed: int = 0    # total quests completed this episode
        self._spawn_vendors()
        self._spawn_quest_npcs()
        self._update_chunks()
        self._update_exploration()  # register spawn position
        self.recalculate_stats()    # apply WotLK stat formulas to initial player

    def _new_uid(self) -> int:
        uid = self._next_uid
        self._next_uid += 1
        return uid

    def _spawn_vendors(self):
        """Spawn vendor NPCs. Uses creature_db chunks when available, else VENDOR_DATA fallback."""
        self.vendors.clear()
        if self.creature_db:
            return  # vendors loaded via _activate_chunk
        for vdata in VENDOR_DATA:
            z = vdata["z"]
            if self.terrain:
                z = self.terrain.get_height(vdata["x"], vdata["y"])
            self.vendors.append(VendorNPC(
                uid=self._new_uid(),
                name=vdata["name"],
                level=vdata["level"],
                x=vdata["x"], y=vdata["y"], z=z,
            ))

    def _spawn_quest_npcs(self):
        """Spawn quest-giver NPCs from quest_db data."""
        self.quest_npcs.clear()
        if not self.quest_db:
            return
        for npc_data in self.quest_db.npc_data:
            z = npc_data.z
            if self.terrain:
                z = self.terrain.get_height(npc_data.x, npc_data.y)
            self.quest_npcs.append(QuestNPC(
                uid=self._new_uid(),
                entry=npc_data.entry,
                name=npc_data.name,
                x=npc_data.x, y=npc_data.y, z=z,
            ))

    def get_nearest_quest_npc(self, npc_filter=None) -> Optional[QuestNPC]:
        """Return the nearest quest NPC, optionally filtered by entry.

        Args:
            npc_filter: If set, only consider NPCs with this entry.
        """
        best = None
        best_dist = float('inf')
        px, py = self.player.x, self.player.y
        for npc in self.quest_npcs:
            if npc_filter is not None and npc.entry != npc_filter:
                continue
            dx = npc.x - px
            dy = npc.y - py
            d = math.sqrt(dx * dx + dy * dy)
            if d < best_dist:
                best_dist = d
                best = npc
        return best

    def do_quest_interact(self) -> bool:
        """Interact with the nearest quest NPC within range.

        Handles both accepting new quests and turning in completed quests.
        Priority: turn-in first (rewards), then accept new quests.
        Returns True if any interaction occurred.
        """
        if not self.quest_db:
            return False

        p = self.player
        px, py = p.x, p.y
        interacted = False

        for npc in self.quest_npcs:
            dx = npc.x - px
            dy = npc.y - py
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > self.QUEST_NPC_RANGE:
                continue

            # Turn-in completed quests first
            completable = self.quest_db.get_completable_quests(
                npc.entry, self.active_quests)
            for qt in completable:
                self._complete_quest(qt)
                interacted = True

            # Accept available quests
            available = self.quest_db.get_available_quests(
                npc.entry, p.level, self.completed_quests, self.active_quests)
            for qt in available:
                progress = self.quest_db.create_progress(qt.quest_id)
                self.active_quests[qt.quest_id] = progress
                interacted = True

        return interacted

    def _complete_quest(self, qt):
        """Complete a quest: grant rewards, update state."""
        from sim.quest_db import QuestObjectiveType
        p = self.player

        # Grant rewards
        if qt.rewards.xp > 0:
            p.xp_gained += qt.rewards.xp
            p.quest_xp_gained += qt.rewards.xp
            p.xp += qt.rewards.xp
            self._check_level_up()
        if qt.rewards.copper > 0:
            p.copper += qt.rewards.copper
            p.quest_copper_gained += qt.rewards.copper

        # Remove quest items from inventory for COLLECT objectives
        for obj in qt.objectives:
            if obj.obj_type == QuestObjectiveType.COLLECT:
                items_to_remove = obj.count
                self.player.inventory = [
                    item for item in self.player.inventory
                    if item.entry != obj.target or (items_to_remove := items_to_remove - 1) < 0
                ]
                # Recalculate free slots
                p.recalculate_free_slots()

        # Move to completed
        del self.active_quests[qt.quest_id]
        self.completed_quests.add(qt.quest_id)
        self.quests_completed += 1
        p.quests_completed_tick += 1

    def on_mob_killed(self, mob: Mob):
        """Update quest progress when a mob is killed."""
        if not self.quest_db:
            return
        from sim.quest_db import QuestObjectiveType
        for qid, progress in self.active_quests.items():
            qt = self.quest_db.templates[qid]
            for i, obj in enumerate(qt.objectives):
                if obj.obj_type == QuestObjectiveType.KILL and obj.target == mob.template.entry:
                    if progress.counts[i] < obj.count:
                        progress.counts[i] += 1
            progress.check_complete(qt.objectives)

    def on_mob_looted(self, mob: Mob):
        """Roll quest item drops when a mob is looted."""
        if not self.quest_db:
            return
        from sim.quest_db import QuestObjectiveType
        p = self.player
        for qid, progress in self.active_quests.items():
            qt = self.quest_db.templates[qid]
            for i, obj in enumerate(qt.objectives):
                if (obj.obj_type == QuestObjectiveType.COLLECT
                        and obj.source_creature == mob.template.entry
                        and progress.counts[i] < obj.count):
                    if self.rng.random() < obj.drop_chance:
                        progress.counts[i] += 1
                        # Add quest item to inventory (takes a slot)
                        if p.free_slots > 0:
                            p.free_slots -= 1
                            p.inventory.append(InventoryItem(
                                entry=obj.target,
                                name=f"Quest Item #{obj.target}",
                                quality=1,
                                sell_price=0,
                                score=0.0,
                                inventory_type=0,
                            ))
            progress.check_complete(qt.objectives)

    def _update_quest_exploration(self):
        """Check explore quest objectives against current position."""
        if not self.quest_db:
            return
        from sim.quest_db import QuestObjectiveType
        px, py = self.player.x, self.player.y
        for qid, progress in self.active_quests.items():
            qt = self.quest_db.templates[qid]
            for i, obj in enumerate(qt.objectives):
                if (obj.obj_type == QuestObjectiveType.EXPLORE
                        and progress.counts[i] < obj.count):
                    dx = obj.target_x - px
                    dy = obj.target_y - py
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist <= obj.radius:
                        progress.counts[i] = obj.count
            progress.check_complete(qt.objectives)

    def get_nearest_vendor(self) -> Optional[VendorNPC]:
        """Return the nearest vendor NPC, or None if none exist."""
        best = None
        best_dist = float('inf')
        px, py = self.player.x, self.player.y
        for v in self.vendors:
            dx = v.x - px
            dy = v.y - py
            d = math.sqrt(dx * dx + dy * dy)
            if d < best_dist:
                best_dist = d
                best = v
        return best

    def reset(self) -> None:
        """Reset player and mobs to initial state."""
        self.player = Player(class_id=self.class_id)
        if self.terrain:
            self.player.z = self.terrain.get_height(self.player.x, self.player.y)
        self.target = None
        self.tick_count = 0
        self.damage_dealt = 0
        self.kills = 0
        self._next_uid = 1
        self.vendors.clear()
        self.visited_areas.clear()
        self.visited_zones.clear()
        self.visited_maps.clear()
        self._new_areas = 0
        self._new_zones = 0
        self._new_maps = 0
        # Clear terrain height cache to bound memory across episodes
        if self.terrain:
            self.terrain.clear_height_cache()
        # Reset chunk state
        self._player_chunk = None
        self._active_chunks.clear()
        self._chunk_mobs.clear()
        self._chunk_vendors.clear()
        self.mobs.clear()
        self._spawn_vendors()
        self._spawn_quest_npcs()
        # Reset quest state
        self.active_quests.clear()
        self.completed_quests.clear()
        self.quests_completed = 0
        self._update_chunks()
        self._update_exploration()
        self.recalculate_stats()

    # ─── Equipment & Stat System ────────────────────────────────────

    def recalculate_gear_stats(self):
        """Sum all stats from equipped items into player gear_* fields."""
        p = self.player
        # Reset all gear stats
        p.gear_strength = 0
        p.gear_agility = 0
        p.gear_stamina = 0
        p.gear_intellect = 0
        p.gear_spirit = 0
        p.gear_armor = 0
        p.gear_bonus_hp = 0
        p.gear_bonus_mana = 0
        p.gear_attack_power = 0
        p.gear_ranged_ap = 0
        p.gear_spell_power = 0
        p.gear_hit_rating = 0
        p.gear_crit_rating = 0
        p.gear_haste_rating = 0
        p.gear_expertise_rating = 0
        p.gear_armor_pen_rating = 0
        p.gear_spell_pen = 0
        p.gear_defense_rating = 0
        p.gear_dodge_rating = 0
        p.gear_parry_rating = 0
        p.gear_block_rating = 0
        p.gear_block_value = 0
        p.gear_resilience_rating = 0
        p.gear_mp5 = 0
        p.gear_hp5 = 0

        for item in p.equipment.values():
            p.gear_armor += item.armor
            for st, sv in item.stats.items():
                if st == ITEM_MOD_STRENGTH:
                    p.gear_strength += sv
                elif st == ITEM_MOD_AGILITY:
                    p.gear_agility += sv
                elif st == ITEM_MOD_STAMINA:
                    p.gear_stamina += sv
                elif st == ITEM_MOD_INTELLECT:
                    p.gear_intellect += sv
                elif st == ITEM_MOD_SPIRIT:
                    p.gear_spirit += sv
                elif st == ITEM_MOD_HEALTH:
                    p.gear_bonus_hp += sv
                elif st == ITEM_MOD_MANA:
                    p.gear_bonus_mana += sv
                # Offensive
                elif st == ITEM_MOD_ATTACK_POWER:
                    p.gear_attack_power += sv
                elif st == ITEM_MOD_RANGED_ATTACK_POWER:
                    p.gear_ranged_ap += sv
                elif st == ITEM_MOD_SPELL_POWER:
                    p.gear_spell_power += sv
                elif st == ITEM_MOD_SPELL_DAMAGE_DONE:
                    p.gear_spell_power += sv
                elif st == ITEM_MOD_SPELL_HEALING_DONE:
                    p.gear_spell_power += sv
                # Hit (combined and per-type)
                elif st == ITEM_MOD_HIT_RATING:
                    p.gear_hit_rating += sv
                elif st == ITEM_MOD_HIT_MELEE_RATING:
                    p.gear_hit_rating += sv
                elif st == ITEM_MOD_HIT_RANGED_RATING:
                    p.gear_hit_rating += sv
                elif st == ITEM_MOD_HIT_SPELL_RATING:
                    p.gear_hit_rating += sv
                # Crit (combined and per-type)
                elif st == ITEM_MOD_CRIT_RATING:
                    p.gear_crit_rating += sv
                elif st == ITEM_MOD_CRIT_MELEE_RATING:
                    p.gear_crit_rating += sv
                elif st == ITEM_MOD_CRIT_RANGED_RATING:
                    p.gear_crit_rating += sv
                elif st == ITEM_MOD_CRIT_SPELL_RATING:
                    p.gear_crit_rating += sv
                # Haste (combined and per-type)
                elif st == ITEM_MOD_HASTE_RATING:
                    p.gear_haste_rating += sv
                elif st == ITEM_MOD_HASTE_MELEE_RATING:
                    p.gear_haste_rating += sv
                elif st == ITEM_MOD_HASTE_RANGED_RATING:
                    p.gear_haste_rating += sv
                elif st == ITEM_MOD_HASTE_SPELL_RATING:
                    p.gear_haste_rating += sv
                # Defensive ratings
                elif st == ITEM_MOD_DEFENSE_SKILL_RATING:
                    p.gear_defense_rating += sv
                elif st == ITEM_MOD_DODGE_RATING:
                    p.gear_dodge_rating += sv
                elif st == ITEM_MOD_PARRY_RATING:
                    p.gear_parry_rating += sv
                elif st == ITEM_MOD_BLOCK_RATING:
                    p.gear_block_rating += sv
                elif st == ITEM_MOD_BLOCK_VALUE:
                    p.gear_block_value += sv
                elif st == ITEM_MOD_RESILIENCE_RATING:
                    p.gear_resilience_rating += sv
                # Other
                elif st == ITEM_MOD_EXPERTISE_RATING:
                    p.gear_expertise_rating += sv
                elif st == ITEM_MOD_ARMOR_PENETRATION_RATING:
                    p.gear_armor_pen_rating += sv
                elif st == ITEM_MOD_SPELL_PENETRATION:
                    p.gear_spell_pen += sv
                elif st == ITEM_MOD_MANA_REGENERATION:
                    p.gear_mp5 += sv
                elif st == ITEM_MOD_HEALTH_REGEN:
                    p.gear_hp5 += sv

    def recalculate_stats(self):
        """Recalculate all derived stats from gear + level + buffs.

        Call after equip, level-up, or buff change. Uses exact WotLK formulas
        from AzerothCore C++ source (StatSystem.cpp, Player.cpp).
        """
        p = self.player
        cls = p.class_id
        self.recalculate_gear_stats()

        # ─── Primary stat totals (base + gear + buffs + talents) ──────
        p.total_strength = class_base_stat(cls, 0, p.level) + p.gear_strength
        p.total_agility = class_base_stat(cls, 1, p.level) + p.gear_agility
        # PW:Fortitude adds Stamina (DBC: AuraName=29 MOD_STAT, MiscValue=2)
        fort_sta = p.fortitude_stamina_bonus if p.fortitude_remaining > 0 else 0
        # Improved PW:Fortitude: +15/30% Stamina bonus from Fortitude
        imp_fort_pts = self._get_talent_points("improved_pw_fortitude")
        if imp_fort_pts > 0 and fort_sta > 0:
            fort_sta = int(fort_sta * (1.0 + 0.15 * imp_fort_pts))
        p.total_stamina = class_base_stat(cls, 2, p.level) + p.gear_stamina + fort_sta
        p.total_intellect = class_base_stat(cls, 3, p.level) + p.gear_intellect
        # Divine Spirit adds Spirit bonus
        ds_spi = p.divine_spirit_bonus if p.divine_spirit_remaining > 0 else 0
        base_spirit = class_base_stat(cls, 4, p.level) + p.gear_spirit + ds_spi
        # Improved Spirit Tap: +5/10% total Spirit (passive)
        imp_st_pts = self._get_talent_points("improved_spirit_tap")
        if imp_st_pts > 0:
            base_spirit = int(base_spirit * (1.0 + 0.05 * imp_st_pts))
        # Spirit Tap proc: +100% Spirit for 15s after kill (at 3/3)
        if p.spirit_tap_remaining > 0:
            st_pts = self._get_talent_points("spirit_tap")
            # 3/3 = 100% bonus Spirit
            base_spirit = int(base_spirit * (1.0 + (st_pts / 3.0)))
        p.total_spirit = base_spirit

        # ─── Max HP (preserve ratio) ────────────────────────────────
        old_max_hp = max(p.max_hp, 1)
        p.max_hp = player_max_hp(p.level, p.gear_stamina + fort_sta,
                                 p.gear_bonus_hp, cls)
        hp_ratio = p.hp / old_max_hp
        p.hp = max(1, int(hp_ratio * p.max_hp))

        # ─── Max Mana (preserve ratio) ──────────────────────────────
        old_max_mana = max(p.max_mana, 1)
        p.max_mana = player_max_mana(p.level, p.gear_intellect, p.gear_bonus_mana, cls)
        if p.max_mana > 0:
            mana_ratio = p.mana / old_max_mana
            p.mana = max(0, int(mana_ratio * p.max_mana))

        # ─── Armor (gear + agi*2 + Inner Fire + Improved Inner Fire talent) ─
        p.total_armor = p.gear_armor + p.total_agility * 2
        if p.inner_fire_remaining > 0:
            if_armor = p.inner_fire_armor
            # Improved Inner Fire: +15/30/45% armor from Inner Fire
            iif_pts = self._get_talent_points("improved_inner_fire")
            if iif_pts > 0:
                if_armor = int(if_armor * (1.0 + 0.15 * iif_pts))
            p.total_armor += if_armor

        # ─── Attack Power (melee + ranged) ───────────────────────────
        p.total_attack_power = melee_attack_power(
            p.level, p.total_strength, p.total_agility, cls) + p.gear_attack_power
        p.total_ranged_ap = ranged_attack_power(
            p.level, p.total_strength, p.total_agility, cls) + p.gear_ranged_ap

        # ─── Spell Power (gear + Inner Fire buff + talents) ──────────
        inner_fire_sp = p.inner_fire_spellpower if p.inner_fire_remaining > 0 else 0
        p.total_spell_power = p.gear_spell_power + inner_fire_sp
        # Twisted Faith: +4/8/12/16/20% of Spirit as Spell Power
        tf_pts = self._get_talent_points("twisted_faith")
        if tf_pts > 0:
            p.total_spell_power += int(p.total_spirit * 0.04 * tf_pts)

        # ─── Crit (melee, ranged, spell + Mind Melt for MB/MF) ────────
        p.total_melee_crit = melee_crit_chance(
            p.level, p.total_agility, p.gear_crit_rating, cls)
        p.total_ranged_crit = ranged_crit_chance(
            p.level, p.total_agility, p.gear_crit_rating, cls)
        p.total_spell_crit = spell_crit_chance(
            p.level, p.gear_intellect, p.gear_crit_rating, cls)
        # Mind Melt: +3/6% crit on Mind Blast and Mind Flay (applied per-spell, not here)

        # ─── Haste (melee, ranged, spell) ────────────────────────────
        p.total_melee_haste = melee_haste_pct(p.level, p.gear_haste_rating)
        p.total_ranged_haste = ranged_haste_pct(p.level, p.gear_haste_rating)
        p.total_spell_haste = spell_haste_pct(p.level, p.gear_haste_rating)

        # ─── Hit (melee, ranged, spell + Shadow Focus talent) ─────────
        p.total_hit_melee = hit_chance_melee(p.level, p.gear_hit_rating)
        p.total_hit_ranged = hit_chance_ranged(p.level, p.gear_hit_rating)
        p.total_hit_spell = hit_chance_spell(p.level, p.gear_hit_rating)
        # Shadow Focus: +1/2/3% Shadow spell hit chance
        sf_pts = self._get_talent_points("shadow_focus")
        if sf_pts > 0:
            p.total_hit_spell += sf_pts

        # ─── Dodge (with diminishing returns) ────────────────────────
        p.total_dodge = dodge_chance(
            p.level, p.total_agility, p.gear_dodge_rating,
            p.gear_defense_rating, cls)

        # ─── Parry (with diminishing returns) ────────────────────────
        p.total_parry = parry_chance(
            p.level, p.gear_parry_rating, p.gear_defense_rating, cls)

        # ─── Block ───────────────────────────────────────────────────
        # Only classes with shields can block
        offhand = p.equipment.get(EQUIPMENT_SLOT_OFFHAND)
        has_shield = offhand is not None and offhand.inventory_type == 14
        if has_shield:
            p.total_block = block_chance(
                p.level, p.gear_block_rating, p.gear_defense_rating)
            # Block value: str/2 - 10 + gear (StatSystem.cpp:GetShieldBlockValue)
            p.total_block_value = max(0, int(p.total_strength * 0.5 - 10) + p.gear_block_value)
        else:
            p.total_block = 0.0
            p.total_block_value = 0

        # ─── Defense ─────────────────────────────────────────────────
        p.total_defense = _rating_to_pct(
            p.gear_defense_rating, CR_DEFENSE_L80, p.level)

        # ─── Expertise ───────────────────────────────────────────────
        p.total_expertise = expertise_pct(p.level, p.gear_expertise_rating)

        # ─── Armor Penetration ───────────────────────────────────────
        p.total_armor_pen = armor_penetration_pct(p.level, p.gear_armor_pen_rating)

        # ─── Resilience ──────────────────────────────────────────────
        p.total_resilience = resilience_pct(p.level, p.gear_resilience_rating)

        # ─── Backwards compat aliases ────────────────────────────────
        # (old code uses total_haste_pct — keep it as spell haste)
        p.total_haste_pct = p.total_spell_haste

    def _find_equip_slot(self, inv_type: int) -> int | None:
        """Find the best equipment slot for an item by inventory_type.

        For dual-slot items (rings, trinkets): fills empty slot first,
        then targets the slot with the lower-score item.
        Returns None if the item can't be equipped.
        """
        slots = INVTYPE_TO_SLOTS.get(inv_type)
        if not slots:
            return None
        if len(slots) == 1:
            return slots[0]
        # Dual-slot: prefer empty slot
        for slot in slots:
            if slot not in self.player.equipment:
                return slot
        # Both occupied: return slot with lower score
        return min(slots, key=lambda s: self.player.equipment[s].score)

    @staticmethod
    def _make_equipped_item(item_data) -> 'EquippedItem':
        """Create an EquippedItem from any item data object (ItemData, InventoryItem)."""
        stats = getattr(item_data, 'stats', None) or {}
        armor = getattr(item_data, 'armor', 0)
        weapon_dps = getattr(item_data, 'weapon_dps', 0.0)
        return EquippedItem(
            entry=item_data.entry,
            name=item_data.name,
            inventory_type=item_data.inventory_type,
            score=item_data.score,
            stats=stats,
            armor=armor,
            weapon_dps=weapon_dps,
        )

    def _equipped_to_inventory_item(self, item: 'EquippedItem') -> 'InventoryItem':
        """Convert an EquippedItem to an InventoryItem for inventory storage."""
        return InventoryItem(
            entry=item.entry,
            name=item.name,
            quality=0,  # quality not stored on EquippedItem
            sell_price=0,
            score=item.score,
            inventory_type=item.inventory_type,
            stats=item.stats,
            armor=item.armor,
            weapon_dps=item.weapon_dps,
        )

    def equip_item(self, item_data, slot: int = None) -> tuple:
        """Equip an item in the given slot (or auto-select slot).

        Unlike try_equip_item(), this always equips regardless of score.
        The displaced item (if any) is returned to inventory.
        Blocked while in combat (WoW behaviour).

        Args:
            item_data: ItemData, InventoryItem, or any object with entry/name/
                       inventory_type/score/stats/armor/weapon_dps attributes.
            slot: Target EQUIPMENT_SLOT_* constant, or None for auto-select.

        Returns:
            (success: bool, old_item: EquippedItem | None)
        """
        if self.player.in_combat:
            return (False, None)

        inv_type = item_data.inventory_type
        if inv_type <= 0:
            return (False, None)

        if slot is None:
            slot = self._find_equip_slot(inv_type)
        if slot is None:
            return (False, None)

        p = self.player
        old_item = p.equipment.pop(slot, None)

        # Two-hand weapon: clear offhand
        displaced_offhand = None
        if inv_type == INVTYPE_TWO_HAND:
            displaced_offhand = p.equipment.pop(EQUIPMENT_SLOT_OFFHAND, None)

        # Return old items to inventory
        for displaced in (old_item, displaced_offhand):
            if displaced is not None and p.free_slots > 0:
                p.free_slots -= 1
                p.inventory.append(self._equipped_to_inventory_item(displaced))

        # Equip new item
        p.equipment[slot] = self._make_equipped_item(item_data)
        p.equipped_scores[slot] = item_data.score
        self.recalculate_stats()
        return (True, old_item)

    def unequip_item(self, slot: int) -> 'EquippedItem | None':
        """Remove item from equipment slot, recalculate stats.

        The removed item is added to the player's inventory (if space).
        Blocked while in combat (WoW behaviour).

        Args:
            slot: EQUIPMENT_SLOT_* constant.

        Returns:
            The removed EquippedItem, or None if slot was empty/in combat.
        """
        p = self.player
        if p.in_combat:
            return None
        item = p.equipment.pop(slot, None)
        if item is None:
            return None

        p.equipped_scores.pop(slot, None)

        # Return to inventory
        if p.free_slots > 0:
            p.free_slots -= 1
            p.inventory.append(self._equipped_to_inventory_item(item))

        self.recalculate_stats()
        return item

    def try_equip_item(self, item_data) -> bool:
        """Try to equip an item if it's better than current (class-aware scoring).

        item_data can be a LootDB ItemData or an InventoryItem with stats.
        Uses proper WoW equipment slots via INVTYPE_TO_SLOTS mapping.
        Uses class-specific stat weights to determine if the item is an upgrade.
        Returns True if equipped (upgrade detected).
        """
        inv_type = item_data.inventory_type
        if inv_type <= 0:
            return False

        slot = self._find_equip_slot(inv_type)
        if slot is None:
            return False

        cid = self.player.class_id
        item_stats = getattr(item_data, 'stats', None) or {}
        new_score = class_aware_score(
            item_stats, item_data.quality if hasattr(item_data, 'quality') else 0,
            item_data.item_level if hasattr(item_data, 'item_level') else 0,
            getattr(item_data, 'armor', 0),
            getattr(item_data, 'weapon_dps', 0.0), cid)

        current = self.player.equipment.get(slot)
        if current:
            cur_stats = getattr(current, 'stats', None) or {}
            cur_quality = getattr(current, 'quality', 0)
            cur_ilvl = getattr(current, 'item_level', 0)
            current_score = class_aware_score(
                cur_stats, cur_quality, cur_ilvl,
                getattr(current, 'armor', 0),
                getattr(current, 'weapon_dps', 0.0), cid)
        else:
            current_score = 0.0

        if new_score <= current_score:
            return False

        score_diff = new_score - current_score
        # Equip via equip_item (handles offhand clearing, inventory return, stat recalc)
        success, _ = self.equip_item(item_data, slot)
        if success:
            # Accumulate score improvement for reward scaling
            self.player.equipped_upgrade += score_diff
        return success

    # ─── Bag System ─────────────────────────────────────────────────

    def _find_bag_slot(self, container_slots: int) -> int | None:
        """Find best bag slot: empty first, then smallest existing bag.

        Returns bag slot index (BAG_SLOT_START..BAG_SLOT_END-1) or None.
        Only replaces if new bag is strictly larger than the smallest equipped.
        """
        p = self.player
        # Prefer empty slot
        for slot in range(BAG_SLOT_START, BAG_SLOT_END):
            if slot not in p.bags:
                return slot
        # All slots full — find the smallest bag
        smallest_slot = min(p.bags, key=lambda s: p.bags[s].container_slots)
        if container_slots > p.bags[smallest_slot].container_slots:
            return smallest_slot
        return None

    def equip_bag(self, item_data, slot: int = None) -> bool:
        """Equip a bag into a bag slot.

        If replacing a smaller bag, the old bag goes to inventory.
        Not allowed during combat. Returns True if equipped.
        """
        p = self.player
        if p.in_combat:
            return False

        container_slots = getattr(item_data, 'container_slots', 0)
        if container_slots <= 0:
            return False

        if slot is None:
            slot = self._find_bag_slot(container_slots)
        if slot is None:
            return False

        old_bag = p.bags.pop(slot, None)

        # Equip new bag
        p.bags[slot] = EquippedBag(
            entry=item_data.entry,
            name=item_data.name,
            container_slots=container_slots,
            quality=getattr(item_data, 'quality', 0),
            sell_price=getattr(item_data, 'sell_price', 0),
        )

        # Return old bag to inventory if there was one and space exists
        if old_bag is not None:
            # Recalculate capacity first (new bag already equipped)
            p.recalculate_free_slots()
            if p.free_slots > 0:
                p.inventory.append(InventoryItem(
                    entry=old_bag.entry,
                    name=old_bag.name,
                    quality=old_bag.quality,
                    sell_price=old_bag.sell_price,
                    score=0.0,
                    inventory_type=INVTYPE_BAG,
                ))
                p.recalculate_free_slots()
        else:
            p.recalculate_free_slots()
        return True

    def try_equip_bag(self, item_data) -> bool:
        """Try to equip a bag if it would increase total inventory capacity.

        Only equips normal bags (bag_family == 0). Auto-selects the best slot.
        Returns True if bag was equipped.
        """
        bag_family = getattr(item_data, 'bag_family', 0)
        if bag_family != 0:
            return False
        container_slots = getattr(item_data, 'container_slots', 0)
        if container_slots <= 0:
            return False

        slot = self._find_bag_slot(container_slots)
        if slot is None:
            return False

        return self.equip_bag(item_data, slot)

    def _update_exploration(self):
        """Track area/zone/map discovery based on player position.

        Uses real WoW area IDs from AreaTable.dbc if env3d is available,
        falls back to grid-based cells otherwise (also for tiles not pre-loaded).
        """
        p = self.player

        if self.env3d and self.env3d.area_table:
            area_id = self.env3d.get_area_id(self.map_id, p.x, p.y)
            zone_id = self.env3d.get_zone_id(self.map_id, p.x, p.y)

            if area_id > 0:
                # Real WoW area/zone from pre-loaded tiles
                if area_id not in self.visited_areas:
                    self.visited_areas.add(area_id)
                    self._new_areas += 1
                if zone_id > 0 and zone_id not in self.visited_zones:
                    self.visited_zones.add(zone_id)
                    self._new_zones += 1
            else:
                # Outside pre-loaded tiles — grid-based fallback
                area_key = (int(p.x // self.AREA_CELL_SIZE), int(p.y // self.AREA_CELL_SIZE))
                zone_key = (int(p.x // self.ZONE_CELL_SIZE), int(p.y // self.ZONE_CELL_SIZE))
                if area_key not in self.visited_areas:
                    self.visited_areas.add(area_key)
                    self._new_areas += 1
                if zone_key not in self.visited_zones:
                    self.visited_zones.add(zone_key)
                    self._new_zones += 1

            if self.map_id not in self.visited_maps:
                self.visited_maps.add(self.map_id)
                self._new_maps += 1
        else:
            # Grid-based fallback (no 3D data)
            area_key = (int(p.x // self.AREA_CELL_SIZE), int(p.y // self.AREA_CELL_SIZE))
            zone_key = (int(p.x // self.ZONE_CELL_SIZE), int(p.y // self.ZONE_CELL_SIZE))

            if area_key not in self.visited_areas:
                self.visited_areas.add(area_key)
                self._new_areas += 1
            if zone_key not in self.visited_zones:
                self.visited_zones.add(zone_key)
                self._new_zones += 1

    # ─── Chunk Management (creature_db mode) ───────────────────────

    def _update_chunks(self):
        """Activate/deactivate chunks based on player position.

        Checks if the player moved to a new chunk and updates the active
        chunk set accordingly. Requires creature_db to be loaded.
        """

        p = self.player
        cs = self.CHUNK_SIZE
        cx = int(p.x // cs) if p.x >= 0 else int(p.x // cs) - 1
        cy = int(p.y // cs) if p.y >= 0 else int(p.y // cs) - 1
        current_chunk = (self.map_id, cx, cy)

        if current_chunk == self._player_chunk:
            return  # player hasn't moved to a new chunk
        self._player_chunk = current_chunk

        # Determine which chunks should be active
        r = self.CHUNK_RADIUS
        needed: set[tuple] = set()
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                needed.add((self.map_id, cx + dx, cy + dy))

        # Deactivate old chunks
        for key in self._active_chunks - needed:
            for mob in self._chunk_mobs.pop(key, []):
                if self.target is mob:
                    self.target = None
            self._chunk_vendors.pop(key, None)

        # Activate new chunks
        for key in needed - self._active_chunks:
            self._activate_chunk(key)

        self._active_chunks = needed

        # Rebuild mobs list from active chunks (mob objects persist)
        self.mobs = []
        for key in self._active_chunks:
            self.mobs.extend(self._chunk_mobs.get(key, []))

        # Rebuild vendors list from active chunks
        self.vendors = []
        for key in self._active_chunks:
            self.vendors.extend(self._chunk_vendors.get(key, []))

    def _activate_chunk(self, chunk_key: tuple):
        """Spawn mobs for a newly activated chunk from creature_db."""
        db = self.creature_db
        spawns = db.spatial_index.get(chunk_key, [])
        chunk_mobs: list[Mob] = []

        for sp in spawns:
            tmpl = db.templates.get(sp.entry)
            if tmpl is None:
                continue

            level = self.rng.randint(tmpl.min_level, tmpl.max_level)
            stats = db.get_mob_stats(tmpl, level)

            mob_template = MobTemplate(
                entry=tmpl.entry,
                name=tmpl.name,
                min_level=tmpl.min_level,
                max_level=tmpl.max_level,
                base_hp=stats['hp'],
                min_damage=stats['min_damage'],
                max_damage=stats['max_damage'],
                attack_speed=tmpl.attack_speed_ticks,
                detect_range=tmpl.detection_range,
                min_gold=tmpl.min_gold,
                max_gold=tmpl.max_gold,
                xp_reward=stats['xp'],
                loot_id=tmpl.lootid,
            )

            z = self.terrain.get_height(sp.x, sp.y) if self.terrain else sp.z
            mob = Mob(
                uid=self._new_uid(),
                template=mob_template,
                hp=stats['hp'],
                max_hp=stats['hp'],
                level=level,
                x=sp.x, y=sp.y, z=z,
                spawn_x=sp.x, spawn_y=sp.y, spawn_z=z,
            )
            chunk_mobs.append(mob)

        self._chunk_mobs[chunk_key] = chunk_mobs

        # Spawn vendors from this chunk
        vendor_spawns = self.creature_db.vendor_index.get(chunk_key, [])
        chunk_vendors: list[VendorNPC] = []
        for sp in vendor_spawns:
            tmpl = self.creature_db.templates.get(sp.entry)
            if tmpl is None:
                continue
            z = self.terrain.get_height(sp.x, sp.y) if self.terrain else sp.z
            chunk_vendors.append(VendorNPC(
                uid=self._new_uid(),
                name=tmpl.name,
                level=tmpl.min_level,
                x=sp.x, y=sp.y, z=z,
            ))
        self._chunk_vendors[chunk_key] = chunk_vendors

    def _dist(self, x1: float, y1: float, x2: float, y2: float) -> float:
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx * dx + dy * dy)

    def _dist_to_mob(self, mob: Mob) -> float:
        return self._dist(self.player.x, self.player.y, mob.x, mob.y)

    def _angle_to(self, tx: float, ty: float) -> float:
        """Relative angle from player orientation to target. [-pi, pi]"""
        dx = tx - self.player.x
        dy = ty - self.player.y
        abs_angle = math.atan2(dy, dx)
        rel = abs_angle - self.player.orientation
        while rel > math.pi:
            rel -= 2 * math.pi
        while rel < -math.pi:
            rel += 2 * math.pi
        return rel

    # ─── Actions ──────────────────────────────────────────────────

    def do_noop(self):
        pass

    def do_move_forward(self):
        """Move 3 units in current orientation direction."""
        if self.player.is_casting:
            return
        p = self.player
        if p.is_eating:
            p.is_eating = False
        new_x = p.x + math.cos(p.orientation) * self.MOVE_SPEED
        new_y = p.y + math.sin(p.orientation) * self.MOVE_SPEED

        if self.terrain:
            self.terrain.ensure_loaded(new_x, new_y)
            new_z = self.terrain.get_height(new_x, new_y)
            if not self.terrain.check_walkable(p.x, p.y, p.z, new_x, new_y, new_z):
                return  # blocked by terrain (AC IsWalkableClimb / slope)
            p.z = new_z

        p.x = new_x
        p.y = new_y

    def do_turn_left(self):
        if self.player.is_casting:
            return
        if self.player.is_eating:
            self.player.is_eating = False
        self.player.orientation += self.TURN_AMOUNT
        if self.player.orientation > math.pi:
            self.player.orientation -= 2 * math.pi

    def do_turn_right(self):
        if self.player.is_casting:
            return
        if self.player.is_eating:
            self.player.is_eating = False
        self.player.orientation -= self.TURN_AMOUNT
        if self.player.orientation < -math.pi:
            self.player.orientation += 2 * math.pi

    def do_move_to(self, tx: float, ty: float) -> bool:
        """Move player toward target coordinates by MOVE_SPEED units.

        Used by vendor navigation — moves directly toward (tx, ty) and
        updates orientation to face the target. Returns False if already
        at the target or movement is blocked.
        """
        if self.player.is_casting:
            return False
        p = self.player
        if p.is_eating:
            p.is_eating = False
        dx = tx - p.x
        dy = ty - p.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.5:
            return False  # already there

        move = min(self.MOVE_SPEED, dist)
        new_x = p.x + (dx / dist) * move
        new_y = p.y + (dy / dist) * move

        if self.terrain:
            self.terrain.ensure_loaded(new_x, new_y)
            new_z = self.terrain.get_height(new_x, new_y)
            if not self.terrain.check_walkable(p.x, p.y, p.z, new_x, new_y, new_z):
                return False
            p.z = new_z

        p.x = new_x
        p.y = new_y
        # Face toward target
        p.orientation = math.atan2(dy, dx)
        return True

    def do_target_nearest(self):
        """Target the nearest alive, attackable mob in range."""
        if self.player.is_casting:
            return
        best = None
        best_dist = self.TARGET_RANGE
        for mob in self.mobs:
            if not mob.alive:
                continue
            d = self._dist_to_mob(mob)
            if d < best_dist:
                best_dist = d
                best = mob
        self.target = best

    def do_cast_smite(self) -> bool:
        """Start casting Smite (best rank for level)."""
        sid = get_best_rank(FAMILY_SMITE, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_heal(self) -> bool:
        """Start casting heal (Lesser Heal → Heal → Greater Heal, best rank)."""
        sid = get_best_rank(FAMILY_HEAL, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_flash_heal(self) -> bool:
        """Start casting Flash Heal (best rank for level)."""
        sid = get_best_rank(FAMILY_FLASH_HEAL, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_sw_pain(self) -> bool:
        """Cast Shadow Word: Pain (best rank for level)."""
        sid = get_best_rank(FAMILY_SW_PAIN, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_pw_shield(self) -> bool:
        """Cast Power Word: Shield (best rank for level)."""
        sid = get_best_rank(FAMILY_PW_SHIELD, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_mind_blast(self) -> bool:
        """Start casting Mind Blast (best rank for level)."""
        sid = get_best_rank(FAMILY_MIND_BLAST, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_renew(self) -> bool:
        """Cast Renew (best rank for level)."""
        sid = get_best_rank(FAMILY_RENEW, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_holy_fire(self) -> bool:
        """Start casting Holy Fire (best rank for level)."""
        sid = get_best_rank(FAMILY_HOLY_FIRE, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_inner_fire(self) -> bool:
        """Cast Inner Fire (best rank for level)."""
        sid = get_best_rank(FAMILY_INNER_FIRE, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_fortitude(self) -> bool:
        """Cast Power Word: Fortitude (best rank for level)."""
        sid = get_best_rank(FAMILY_FORTITUDE, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_devouring_plague(self) -> bool:
        """Cast Devouring Plague (best rank for level)."""
        sid = get_best_rank(FAMILY_DEVOURING_PLAGUE, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_psychic_scream(self) -> bool:
        """Cast Psychic Scream (best rank for level)."""
        sid = get_best_rank(FAMILY_PSYCHIC_SCREAM, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_shadow_protection(self) -> bool:
        """Cast Shadow Protection (best rank for level)."""
        sid = get_best_rank(FAMILY_SHADOW_PROTECTION, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_divine_spirit(self) -> bool:
        """Cast Divine Spirit (best rank for level)."""
        sid = get_best_rank(FAMILY_DIVINE_SPIRIT, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_fear_ward(self) -> bool:
        """Cast Fear Ward (best rank for level)."""
        sid = get_best_rank(FAMILY_FEAR_WARD, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_holy_nova(self) -> bool:
        """Cast Holy Nova (best rank for level). PBAoE, no target needed."""
        sid = get_best_rank(FAMILY_HOLY_NOVA, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_dispel_magic(self) -> bool:
        """Cast Dispel Magic (best rank for level)."""
        sid = get_best_rank(FAMILY_DISPEL_MAGIC, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_mind_flay(self) -> bool:
        """Cast Mind Flay (channeled, talent-granted). Requires mind_flay talent."""
        if self._get_talent_points("mind_flay") < 1:
            return False
        sid = get_best_rank(FAMILY_MIND_FLAY, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_vampiric_touch(self) -> bool:
        """Cast Vampiric Touch (talent-granted DoT). Requires vampiric_touch talent."""
        if self._get_talent_points("vampiric_touch") < 1:
            return False
        sid = get_best_rank(FAMILY_VAMPIRIC_TOUCH, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_cast_dispersion(self) -> bool:
        """Cast Dispersion (talent-granted defensive CD). Requires dispersion talent."""
        if self._get_talent_points("dispersion") < 1:
            return False
        sid = get_best_rank(FAMILY_DISPERSION, self.player.level)
        return self._start_cast(sid) if sid else False

    def do_toggle_shadowform(self) -> bool:
        """Toggle Shadowform on/off. Requires shadowform talent."""
        p = self.player
        if self._get_talent_points("shadowform") < 1:
            return False
        if p.is_casting or p.gcd_remaining > 0:
            return False
        if p.shadowform_active:
            p.shadowform_active = False
        else:
            p.shadowform_active = True
        p.gcd_remaining = 3  # GCD
        self.recalculate_stats()
        return True

    def do_eat_drink(self) -> bool:
        """Start eating/drinking. Regenerates 5% HP and Mana per second.

        Only works out of combat. Interrupted by movement, taking damage,
        entering combat, or reaching full HP and Mana.
        """
        p = self.player
        if p.in_combat or p.is_casting or p.is_eating:
            return False
        # Don't start if already full
        if p.hp >= p.max_hp and p.mana >= p.max_mana:
            return False
        p.is_eating = True
        return True

    def _interrupt_eating(self):
        """Cancel the eat/drink state."""
        self.player.is_eating = False

    def do_loot(self) -> bool:
        """Loot nearest dead mob within range.

        Uses loot tables from LootDB when loaded (creature_loot_template +
        item_template CSVs). Falls back to the original random loot system
        when no loot data is available.

        Each item requires 1 free inventory slot. Items that don't fit are
        tracked in player.loot_failed (quality list) for penalty signals.
        Gold never requires inventory space.
        """
        best = None
        best_dist = self.LOOT_RANGE
        for mob in self.mobs:
            if mob.alive or mob.looted:
                continue
            d = self._dist_to_mob(mob)
            if d < best_dist:
                best_dist = d
                best = mob
        if best is None:
            return False
        best.looted = True

        # Gold (always from creature_template min/max gold — separate from item loot)
        gold = self.rng.randint(best.template.min_gold,
                                max(best.template.min_gold, best.template.max_gold))
        self.player.loot_copper += gold

        # Item loot: use loot tables if available, else random fallback
        loot_id = best.template.loot_id
        if self.loot_db and self.loot_db.loaded and loot_id > 0:
            results = self.loot_db.roll_loot(loot_id, self.rng)
            for result in results:
                quality = result.item.quality
                # Bag items: try to equip directly (normal bags only)
                if (result.item.inventory_type == INVTYPE_BAG
                        and getattr(result.item, 'container_slots', 0) > 0
                        and getattr(result.item, 'bag_family', 0) == 0):
                    if self.try_equip_bag(result.item):
                        self.player.loot_items.append(quality)
                        continue
                if self.player.free_slots > 0:
                    self.player.loot_score += int(result.item.score * result.count)
                    self.player.free_slots -= 1
                    self.player.loot_items.append(quality)
                    self.player.inventory.append(InventoryItem(
                        entry=result.item.entry,
                        name=result.item.name,
                        quality=result.item.quality,
                        sell_price=result.item.sell_price,
                        score=result.item.score,
                        inventory_type=result.item.inventory_type,
                        stats=result.item.stats,
                        armor=result.item.armor,
                        weapon_dps=result.item.weapon_dps,
                    ))
                    # Auto-equip if it's an upgrade (score diff tracked by try_equip_item)
                    if result.item.inventory_type > 0:
                        self.try_equip_item(result.item)
                else:
                    self.player.loot_failed.append(quality)
        else:
            # Fallback: random loot (no loot_db loaded)
            if self.rng.random() < self.LOOT_CHANCE:
                score = self.rng.randint(*self.ITEM_SCORE_RANGE)
                sell_price = score * 2  # rough copper value
                quality = 1  # assume Common for fallback items
                if self.player.free_slots > 0:
                    self.player.loot_score += score
                    self.player.free_slots -= 1
                    self.player.loot_items.append(quality)
                    self.player.inventory.append(InventoryItem(
                        entry=0, name="Loot",
                        quality=quality,
                        sell_price=sell_price,
                        score=score,
                        inventory_type=0,
                    ))
                    if self.rng.random() < self.UPGRADE_CHANCE:
                        self.player.equipped_upgrade += score  # fallback: use raw score as diff
                else:
                    self.player.loot_failed.append(quality)
        # Quest collect objective tracking
        self.on_mob_looted(best)
        return True

    def do_sell(self) -> bool:
        """Sell all inventory items at the nearest vendor within SELL_RANGE.

        Requires proximity to a vendor NPC. Converts inventory items to copper
        based on their sell_price. Returns True if items were sold.
        """
        p = self.player
        if len(p.inventory) == 0:
            return False
        # Must be near a vendor
        vendor = self.get_nearest_vendor()
        if vendor is None:
            return False
        dist = self._dist(p.x, p.y, vendor.x, vendor.y)
        if dist > self.SELL_RANGE:
            return False
        # Calculate copper from inventory sell prices
        num_items = len(p.inventory)
        copper = sum(item.sell_price for item in p.inventory)
        p.copper += copper
        p.sell_copper += copper
        p.items_sold += num_items
        p.inventory.clear()
        p.recalculate_free_slots()
        return True

    # Offensive spell families (need alive target + range)
    _OFFENSIVE_FAMILIES = {FAMILY_SMITE, FAMILY_SW_PAIN, FAMILY_MIND_BLAST, FAMILY_HOLY_FIRE,
                           FAMILY_DEVOURING_PLAGUE, FAMILY_MIND_FLAY, FAMILY_VAMPIRIC_TOUCH}
    # AoE families: no target needed, hits mobs in range
    _AOE_FAMILIES = {FAMILY_HOLY_NOVA, FAMILY_PSYCHIC_SCREAM}

    def _start_cast(self, spell_id: int) -> bool:
        """Attempt to start casting a spell."""
        if self.player.is_casting:
            return False
        if self.player.gcd_remaining > 0:
            return False

        spell = SPELLS.get(spell_id)
        if spell is None:
            return False

        # Level gate: spell not yet learned
        if self.player.level < spell.level_req:
            return False

        # Mana cost: % of BaseMana (from Spell.dbc ManaCostPercentage)
        mana_cost = spell_mana_cost(spell_id, self.player.level,
                                    self.player.class_id)
        if self.player.mana < mana_cost:
            return False

        # Spell-specific cooldown check (use family for shared CDs like Mind Blast)
        family = spell.spell_family
        if spell.cooldown_ticks > 0:
            # Check CD on family (all ranks share cooldown)
            if self.player.spell_cooldowns.get(family, 0) > 0:
                return False

        # Range check for offensive spells (with target requirement)
        if family in self._OFFENSIVE_FAMILIES:
            if self.target is None or not self.target.alive:
                return False
            if self._dist_to_mob(self.target) > spell.spell_range:
                return False
            # LOS check
            if self.terrain:
                if not self.terrain.check_los(
                    self.player.x, self.player.y, self.player.z,
                    self.target.x, self.target.y, self.target.z
                ):
                    return False

        # Shield: check if already shielded or Weakened Soul active
        if family == FAMILY_PW_SHIELD and (self.player.shield_remaining > 0
                                           or self.player.shield_cooldown > 0):
            return False

        # Renew: block if HoT already active
        if family == FAMILY_RENEW and self.player.hot_remaining > 0:
            return False

        # Inner Fire: block if already active
        if family == FAMILY_INNER_FIRE and self.player.inner_fire_remaining > 0:
            return False

        # Fortitude: block if already active
        if family == FAMILY_FORTITUDE and self.player.fortitude_remaining > 0:
            return False

        # Shadow Protection: block if already active
        if family == FAMILY_SHADOW_PROTECTION and self.player.shadow_prot_remaining > 0:
            return False

        # Divine Spirit: block if already active
        if family == FAMILY_DIVINE_SPIRIT and self.player.divine_spirit_remaining > 0:
            return False

        # Fear Ward: block if already active
        if family == FAMILY_FEAR_WARD and self.player.fear_ward_remaining > 0:
            return False

        # Devouring Plague: block if already active on target (one per target)
        if family == FAMILY_DEVOURING_PLAGUE:
            if self.target and self.target.dot3_remaining > 0:
                return False

        # Vampiric Touch: block if already active on target (slot 4)
        if family == FAMILY_VAMPIRIC_TOUCH:
            if self.target and self.target.dot4_remaining > 0:
                return False

        # Dispersion: block if already active
        if family == FAMILY_DISPERSION:
            if self.player.dispersion_remaining > 0:
                return False

        # Mind Flay: block if already channeling
        if family == FAMILY_MIND_FLAY:
            if self.player.channel_remaining > 0:
                return False

        # Spend mana (% of BaseMana)
        # Focused Mind: -5/10/15% mana cost of Mind Blast, Mind Flay, Mind Sear
        actual_cost = mana_cost
        if family in (FAMILY_MIND_BLAST, FAMILY_MIND_FLAY):
            fm_pts = self._get_talent_points("focused_mind")
            if fm_pts > 0:
                actual_cost = int(actual_cost * (1.0 - 0.05 * fm_pts))
        self.player.mana -= actual_cost

        # GCD
        self.player.gcd_remaining = spell.gcd_ticks

        # Spell-specific cooldown (keyed by family so all ranks share)
        if spell.cooldown_ticks > 0:
            cd_ticks = spell.cooldown_ticks
            # Improved Mind Blast: -0.5s per rank = -1 tick per rank
            if family == FAMILY_MIND_BLAST:
                imb_pts = self._get_talent_points("improved_mind_blast")
                if imb_pts > 0:
                    cd_ticks = max(0, cd_ticks - imb_pts)
            self.player.spell_cooldowns[family] = cd_ticks

        if family == FAMILY_MIND_FLAY:
            # Mind Flay: channeled spell — set up channel state
            self.player.is_casting = True
            self.player.cast_remaining = spell.dot_ticks  # channel duration = dot_dur_ticks
            self.player.cast_spell_id = spell_id
            self.player.channel_remaining = spell.dot_ticks
            self.player.channel_spell_id = spell_id
            self.player.channel_tick_timer = spell.dot_interval  # first tick after 1s
            self.player.channel_target_uid = self.target.uid if self.target else 0
            # Apply initial hit check for Mind Flay
            if self.target and self.target.alive:
                outcome = self._resolve_offensive_spell(self.target.level)
                if outcome == SPELL_MISS:
                    # Cancel channel on miss
                    self.player.is_casting = False
                    self.player.cast_remaining = 0
                    self.player.channel_remaining = 0
                    return True  # mana still spent
        elif spell.cast_ticks > 0:
            # Regular cast time spell
            self.player.is_casting = True
            self.player.cast_remaining = spell.cast_ticks
            self.player.cast_spell_id = spell_id
        else:
            # Instant cast — apply immediately
            self._apply_spell(spell_id)

        return True

    def _resolve_offensive_spell(self, mob_level: int) -> str:
        """Resolve hit/miss/crit for an offensive spell against a mob.

        Uses two-roll system: first miss check, then crit check.
        Level difference increases miss chance; hit rating reduces it.
        """
        p = self.player
        outcome = resolve_spell_hit(
            player_level=p.level,
            mob_level=mob_level,
            hit_bonus_pct=p.total_hit_spell,
            spell_crit_pct=p.total_spell_crit,
            roll_hit=self.rng.random() * 100.0,
            roll_crit=self.rng.random() * 100.0,
        )
        if outcome == SPELL_MISS:
            p.spell_misses += 1
        elif outcome == SPELL_CRIT:
            p.spell_crits += 1
        return outcome

    def _apply_spell(self, spell_id: int):
        """Apply spell effect when cast completes. Generic for any rank.

        Uses spell_family to determine spell type, then applies effects
        using the rank-specific SpellDef values + SP coefficients.
        """
        spell = SPELLS[spell_id]
        sp = self.player.total_spell_power
        family = spell.spell_family

        if family == FAMILY_SMITE:
            # Direct Holy damage offensive spell
            if self.target and self.target.alive:
                outcome = self._resolve_offensive_spell(self.target.level)
                if outcome == SPELL_MISS:
                    return
                min_dmg, max_dmg = spell_direct_value(spell_id, sp)
                dmg = self.rng.randint(min_dmg, max_dmg)
                if outcome == SPELL_CRIT:
                    dmg = int(dmg * 1.5)
                self._damage_mob(self.target, dmg)

        elif family == FAMILY_HEAL or family == FAMILY_FLASH_HEAL:
            # Direct heal (self-cast, no miss)
            min_h, max_h = spell_direct_value(spell_id, sp)
            heal = self.rng.randint(min_h, max_h)
            if self.rng.random() * 100 < self.player.total_spell_crit:
                heal = int(heal * 1.5)
                self.player.spell_crits += 1
            self.player.hp = min(self.player.max_hp, self.player.hp + heal)

        elif family == FAMILY_SW_PAIN:
            # Shadow DoT (miss check on application)
            if self.target and self.target.alive:
                outcome = self._resolve_offensive_spell(self.target.level)
                if outcome == SPELL_MISS:
                    return
                dmg_per_tick = spell_dot_per_tick(spell_id, sp)
                # Apply Shadow damage modifiers to per-tick value
                mod = self._shadow_damage_mod(self.target)
                iswp_pts = self._get_talent_points("improved_sw_pain")
                if iswp_pts > 0:
                    mod *= 1.0 + 0.03 * iswp_pts
                dmg_per_tick = int(dmg_per_tick * mod)
                self.target.dot_remaining = spell.dot_ticks
                self.target.dot_timer = spell.dot_interval
                self.target.dot_damage_per_tick = dmg_per_tick
                # Misery: SW:Pain applies spell hit debuff
                self._apply_misery(self.target)
                self._apply_shadow_weaving(self.target)

        elif family == FAMILY_PW_SHIELD:
            # Absorb shield (no miss on friendly)
            absorb = spell_shield_absorb(spell_id, sp)
            self.player.shield_absorb = absorb
            self.player.shield_remaining = spell.shield_duration
            self.player.shield_cooldown = 30  # Weakened Soul: 15s = 30 ticks

        elif family == FAMILY_MIND_BLAST:
            # Shadow direct damage with cooldown
            if self.target and self.target.alive:
                outcome = self._resolve_offensive_spell(self.target.level)
                if outcome == SPELL_MISS:
                    return
                min_dmg, max_dmg = spell_direct_value(spell_id, sp)
                dmg = self.rng.randint(min_dmg, max_dmg)
                # Mind Melt: +3/6% crit on Mind Blast
                mm_pts = self._get_talent_points("mind_melt")
                extra_crit = mm_pts * 3.0 if mm_pts > 0 else 0.0
                if outcome == SPELL_CRIT or (outcome == SPELL_HIT and extra_crit > 0
                        and self.rng.random() * 100 < extra_crit):
                    # Shadow Power: +20/40/60/80/100% crit damage bonus
                    sp_pts = self._get_talent_points("shadow_power")
                    crit_mult = 1.5 + 0.1 * sp_pts
                    dmg = int(dmg * crit_mult)
                    if outcome != SPELL_CRIT:
                        self.player.spell_crits += 1
                self._damage_mob(self.target, dmg, is_shadow=True,
                                 spell_family=FAMILY_MIND_BLAST)

        elif family == FAMILY_RENEW:
            # HoT (no miss on friendly)
            heal_per = spell_hot_per_tick(spell_id, sp)
            self.player.hot_remaining = spell.hot_ticks
            self.player.hot_timer = spell.hot_interval
            self.player.hot_heal_per_tick = heal_per

        elif family == FAMILY_HOLY_FIRE:
            # Direct damage + DoT (slot 2)
            if self.target and self.target.alive:
                outcome = self._resolve_offensive_spell(self.target.level)
                if outcome == SPELL_MISS:
                    return
                min_dmg, max_dmg = spell_direct_value(spell_id, sp)
                dmg = self.rng.randint(min_dmg, max_dmg)
                if outcome == SPELL_CRIT:
                    dmg = int(dmg * 1.5)
                self._damage_mob(self.target, dmg)
                # DoT component on slot 2
                dot_per = spell_dot_per_tick(spell_id, sp)
                self.target.dot2_remaining = spell.dot_ticks
                self.target.dot2_timer = spell.dot_interval
                self.target.dot2_damage_per_tick = dot_per

        elif family == FAMILY_INNER_FIRE:
            # Armor buff (rank determines armor amount)
            armor = spell_buff_value(spell_id)
            self.player.inner_fire_remaining = spell.buff_duration
            self.player.inner_fire_armor = armor
            self.player.inner_fire_spellpower = 0  # SP bonus starts at rank 7 (level 70+)
            self.recalculate_stats()

        elif family == FAMILY_FORTITUDE:
            # Stamina buff (rank determines stamina amount)
            sta_bonus = spell_buff_value(spell_id)
            self.player.fortitude_remaining = spell.buff_duration
            self.player.fortitude_stamina_bonus = sta_bonus
            old_hp = self.player.hp
            self.recalculate_stats()
            hp_gain = self.player.max_hp - old_hp
            if hp_gain > 0:
                self.player.hp = min(self.player.hp + hp_gain, self.player.max_hp)

        elif family == FAMILY_DEVOURING_PLAGUE:
            # Shadow DoT (slot 3) that heals caster for damage dealt
            if self.target and self.target.alive:
                outcome = self._resolve_offensive_spell(self.target.level)
                if outcome == SPELL_MISS:
                    return
                dmg_per_tick = spell_dot_per_tick(spell_id, sp)
                # Apply Shadow damage modifiers to per-tick
                mod = self._shadow_damage_mod(self.target)
                dmg_per_tick = int(dmg_per_tick * mod)
                self.target.dot3_remaining = spell.dot_ticks
                self.target.dot3_timer = spell.dot_interval
                self.target.dot3_damage_per_tick = dmg_per_tick
                self.target.dot3_heals_caster = True
                # Improved Devouring Plague: instant damage = 10/20/30% of total DoT
                idp_pts = self._get_talent_points("improved_devouring_plague")
                if idp_pts > 0:
                    total_dot = dmg_per_tick * (spell.dot_ticks // spell.dot_interval)
                    instant = int(total_dot * 0.1 * idp_pts)
                    if instant > 0:
                        self._damage_mob(self.target, instant, is_shadow=True,
                                         spell_family=FAMILY_DEVOURING_PLAGUE)
                self._apply_shadow_weaving(self.target)

        elif family == FAMILY_PSYCHIC_SCREAM:
            # AoE Fear: fears up to 5 mobs within 8 yards for 8 seconds (16 ticks)
            px, py = self.player.x, self.player.y
            feared_count = 0
            for mob in self.mobs:
                if feared_count >= 5:
                    break
                if not mob.alive or not mob.in_combat:
                    continue
                dx = mob.x - px
                dy = mob.y - py
                dist = math.sqrt(dx * dx + dy * dy)
                if dist <= spell.spell_range:
                    mob.feared = True
                    mob.fear_remaining = 16  # 8 seconds = 16 ticks
                    # Flee direction: away from player (normalized)
                    if dist > 0.1:
                        mob.fear_dx = dx / dist
                        mob.fear_dy = dy / dist
                    else:
                        angle = self.rng.random() * math.pi * 2
                        mob.fear_dx = math.cos(angle)
                        mob.fear_dy = math.sin(angle)
                    feared_count += 1

        elif family == FAMILY_SHADOW_PROTECTION:
            # Shadow resistance buff
            val = spell_buff_value(spell_id)
            self.player.shadow_prot_remaining = spell.buff_duration
            self.player.shadow_prot_value = val

        elif family == FAMILY_DIVINE_SPIRIT:
            # Spirit buff (recalculates stats for mana regen)
            spi_bonus = spell_buff_value(spell_id)
            self.player.divine_spirit_remaining = spell.buff_duration
            self.player.divine_spirit_bonus = spi_bonus
            self.recalculate_stats()

        elif family == FAMILY_FEAR_WARD:
            # Fear immunity buff (absorbs one fear effect)
            self.player.fear_ward_remaining = spell.buff_duration

        elif family == FAMILY_HOLY_NOVA:
            # PBAoE: damage all mobs within 10 yards + heal self
            px, py = self.player.x, self.player.y
            min_dmg, max_dmg = spell_direct_value(spell_id, sp)
            hit_any = False
            for mob in self.mobs:
                if not mob.alive:
                    continue
                dx = mob.x - px
                dy = mob.y - py
                dist = math.sqrt(dx * dx + dy * dy)
                if dist <= spell.spell_range:
                    outcome = self._resolve_offensive_spell(mob.level)
                    if outcome == SPELL_MISS:
                        continue
                    dmg = self.rng.randint(min_dmg, max(min_dmg, max_dmg))
                    if outcome == SPELL_CRIT:
                        dmg = int(dmg * 1.5)
                    self._damage_mob(mob, dmg)
                    hit_any = True
            # Self-heal component (Holy Nova Heal has separate base values)
            # Heal base values per rank (from DBC, separate healing spell IDs)
            _nova_heal_base = {
                15237: (29, 5), 15430: (54, 7), 15431: (86, 11),
                27799: (124, 15), 27800: (163, 19), 27801: (220, 25),
            }
            hb, hd = _nova_heal_base.get(spell_id, (29, 5))
            heal = self.rng.randint(hb + 1, hb + hd) + int(sp * SP_COEFF_HOLY_NOVA_HEAL)
            if self.rng.random() * 100 < self.player.total_spell_crit:
                heal = int(heal * 1.5)
                self.player.spell_crits += 1
            self.player.hp = min(self.player.max_hp, self.player.hp + heal)

        elif family == FAMILY_VAMPIRIC_TOUCH:
            # Shadow DoT (slot 4) — talent-granted, mana return on ticks
            if self.target and self.target.alive:
                outcome = self._resolve_offensive_spell(self.target.level)
                if outcome == SPELL_MISS:
                    return
                dmg_per_tick = spell_dot_per_tick(spell_id, sp)
                # Apply Shadow damage modifiers to per-tick
                mod = self._shadow_damage_mod(self.target)
                dmg_per_tick = int(dmg_per_tick * mod)
                self.target.dot4_remaining = spell.dot_ticks
                self.target.dot4_timer = spell.dot_interval
                self.target.dot4_damage_per_tick = dmg_per_tick
                # Misery: VT applies spell hit debuff
                self._apply_misery(self.target)
                self._apply_shadow_weaving(self.target)

        elif family == FAMILY_DISPERSION:
            # Defensive CD: -90% damage taken, +6% mana/s for 6s
            self.player.dispersion_remaining = spell.buff_duration  # 12 ticks = 6s
            self.player.is_casting = False  # can't cast during Dispersion
            self.player.gcd_remaining = 0

        elif family == FAMILY_DISPEL_MAGIC:
            # Remove debuffs from self (in sim: no mob debuffs currently)
            # Rank 2 can also be used offensively to dispel mob buffs
            # For now: utility spell, no direct combat effect
            pass

    def _shadow_damage_mod(self, mob: Mob) -> float:
        """Calculate multiplicative Shadow damage modifier from talents.

        Combines: Darkness, Shadowform, Shadow Weaving (target debuff),
        Improved SW:Pain (for SW:P only — handled separately).
        """
        mod = 1.0
        # Darkness: +2/4/6/8/10% Shadow spell damage
        dark_pts = self._get_talent_points("darkness")
        if dark_pts > 0:
            mod *= 1.0 + 0.02 * dark_pts
        # Shadowform: +15% Shadow damage
        if self.player.shadowform_active:
            mod *= 1.15
        # Shadow Weaving: target takes +2% Shadow per stack (max 5 = +10%)
        if mob.shadow_weaving_stacks > 0:
            mod *= 1.0 + 0.02 * mob.shadow_weaving_stacks
        return mod

    def _apply_shadow_weaving(self, mob: Mob):
        """Apply or refresh Shadow Weaving debuff stack on a mob."""
        sw_pts = self._get_talent_points("shadow_weaving")
        if sw_pts <= 0:
            return
        # 33/66/100% chance to apply per Shadow spell
        chance = sw_pts / 3.0
        if self.rng.random() < chance:
            mob.shadow_weaving_stacks = min(5, mob.shadow_weaving_stacks + 1)
            mob.shadow_weaving_timer = 30  # 15s = 30 ticks

    def _apply_misery(self, mob: Mob):
        """Apply Misery debuff (spell hit increase) from SW:P, MF, VT."""
        mis_pts = self._get_talent_points("misery")
        if mis_pts > 0:
            mob.misery_stacks = 1  # binary debuff, value is mis_pts % hit

    def _vampiric_embrace_heal(self, shadow_damage: int):
        """Heal player from VE: 15% of single-target Shadow spell damage.

        Improved VE: +33/67% → 20/25%.
        """
        ve_pts = self._get_talent_points("vampiric_embrace")
        if ve_pts < 1:
            return
        pct = 0.15
        ive_pts = self._get_talent_points("improved_vampiric_embrace")
        if ive_pts == 1:
            pct = 0.20
        elif ive_pts >= 2:
            pct = 0.25
        heal = int(shadow_damage * pct)
        if heal > 0:
            self.player.hp = min(self.player.max_hp, self.player.hp + heal)

    def _damage_mob(self, mob: Mob, damage: int, is_shadow: bool = False,
                    spell_family: int = 0):
        """Apply damage to a mob, handle death.

        Args:
            is_shadow: If True, apply Shadow damage modifiers + VE healing.
            spell_family: Spell family for specific talent interactions.
        """
        # Apply Shadow talent multipliers
        if is_shadow:
            mod = self._shadow_damage_mod(mob)
            # Improved SW:Pain: +3/6% SW:Pain damage
            if spell_family == FAMILY_SW_PAIN:
                iswp_pts = self._get_talent_points("improved_sw_pain")
                if iswp_pts > 0:
                    mod *= 1.0 + 0.03 * iswp_pts
            # Twisted Faith: +2/4/6/8/10% MB/MF damage if target has SW:Pain
            if spell_family in (FAMILY_MIND_BLAST, FAMILY_MIND_FLAY):
                tf_pts = self._get_talent_points("twisted_faith")
                if tf_pts > 0 and mob.dot_remaining > 0:
                    mod *= 1.0 + 0.02 * tf_pts
            damage = int(damage * mod)
            # Shadow Weaving application
            self._apply_shadow_weaving(mob)
            # Vampiric Embrace healing
            self._vampiric_embrace_heal(damage)

        old_hp = mob.hp
        mob.hp = max(0, mob.hp - damage)
        self.damage_dealt += old_hp - mob.hp
        if not mob.in_combat:
            mob.in_combat = True
            mob.target_player = True
        if mob.hp <= 0:
            mob.alive = False
            mob.in_combat = False
            mob.target_player = False
            mob.respawn_timer = self.RESPAWN_TICKS
            self.kills += 1
            # Spirit Tap proc on kill
            st_pts = self._get_talent_points("spirit_tap")
            if st_pts > 0:
                self.player.spirit_tap_remaining = 30  # 15s = 30 ticks
                self.recalculate_stats()
            # XP reward — AzerothCore formula based on level difference
            xp = base_xp_gain(self.player.level, mob.level)
            self.player.xp_gained += xp
            self.player.xp += xp
            # Level-up check
            self._check_level_up()
            # Quest kill objective tracking
            self.on_mob_killed(mob)
            # Check if player leaves combat
            self._check_combat_end()

    def _check_combat_end(self):
        """Check if all mobs targeting player are dead."""
        for mob in self.mobs:
            if mob.alive and mob.target_player:
                return
        self.player.in_combat = False

    def _check_level_up(self):
        """Check if accumulated XP is enough for a level-up. May level multiple times."""
        p = self.player
        while p.level < MAX_LEVEL and p.xp >= XP_TABLE[p.level + 1]:
            p.level += 1
            p.leveled_up = True
            p.levels_gained += 1
            self._apply_level_stats()

    def _get_talent_points(self, talent_name: str) -> int:
        """Return current points in a talent (0 if not yet trained)."""
        return self.player.talent_points.get(talent_name, 0)

    def _assign_talent_point(self, level: int):
        """Assign talent point for the given level using the predefined build."""
        talent = get_talent_for_level(level)
        if talent is None:
            return
        p = self.player
        current = p.talent_points.get(talent, 0)
        max_pts = TALENT_DEFS[talent]["max"]
        if current < max_pts:
            p.talent_points[talent] = current + 1
        # Auto-activate Shadowform when talented
        if talent == "shadowform" and not p.shadowform_active:
            p.shadowform_active = True

    def _apply_level_stats(self):
        """Update player stats after level-up. Heals to full."""
        p = self.player
        # Assign talent point for this level
        self._assign_talent_point(p.level)
        self.recalculate_stats()
        # Full heal on level-up (matches WoW behaviour)
        p.hp = p.max_hp
        p.mana = p.max_mana

    # ─── Tick Processing ──────────────────────────────────────────

    def tick(self) -> None:
        """Advance simulation by one tick (0.5 seconds)."""
        self.tick_count += 1
        p = self.player
        self._update_chunks()
        self._update_exploration()
        self._update_quest_exploration()

        # --- Mind Flay channel ticks ---
        if p.channel_remaining > 0:
            p.channel_remaining -= 1
            p.channel_tick_timer -= 1
            if p.channel_tick_timer <= 0 and p.channel_remaining > 0:
                # Channel damage tick
                mf_spell = SPELLS.get(p.channel_spell_id)
                if mf_spell:
                    # Find channel target
                    target_mob = None
                    for mob in self.mobs:
                        if mob.uid == p.channel_target_uid and mob.alive:
                            target_mob = mob
                            break
                    if target_mob:
                        tick_dmg = spell_dot_per_tick(p.channel_spell_id,
                                                     p.total_spell_power)
                        # Mind Melt: +3/6% crit bonus (applied per tick)
                        mm_pts = self._get_talent_points("mind_melt")
                        extra_crit = mm_pts * 3.0 if mm_pts > 0 else 0.0
                        is_crit = self.rng.random() * 100 < (p.total_spell_crit + extra_crit)
                        if is_crit:
                            sp_pts = self._get_talent_points("shadow_power")
                            crit_mult = 1.5 + 0.1 * sp_pts
                            tick_dmg = int(tick_dmg * crit_mult)
                            p.spell_crits += 1
                        self._damage_mob(target_mob, tick_dmg, is_shadow=True,
                                         spell_family=FAMILY_MIND_FLAY)
                        # Pain and Suffering: MF has 33/66/100% chance to refresh SWP
                        ps_pts = self._get_talent_points("pain_and_suffering")
                        if ps_pts > 0 and target_mob.dot_remaining > 0:
                            if self.rng.random() < ps_pts / 3.0:
                                target_mob.dot_remaining = 36  # refresh to 18s
                        # Misery from Mind Flay
                        self._apply_misery(target_mob)
                    else:
                        # Target dead/gone — cancel channel
                        p.channel_remaining = 0
                p.channel_tick_timer = 2  # next tick in 1s
            if p.channel_remaining <= 0:
                # Channel finished
                p.channel_spell_id = 0
                p.channel_tick_timer = 0
                p.channel_target_uid = 0

        # --- Cast completion ---
        if p.is_casting:
            p.cast_remaining -= 1
            if p.cast_remaining <= 0:
                p.is_casting = False
                # Mind Flay ends via channel system, not _apply_spell
                if p.cast_spell_id and SPELLS.get(p.cast_spell_id, None) \
                        and SPELLS[p.cast_spell_id].spell_family != FAMILY_MIND_FLAY:
                    self._apply_spell(p.cast_spell_id)
                p.cast_spell_id = 0

        # --- GCD ---
        if p.gcd_remaining > 0:
            p.gcd_remaining -= 1

        # --- Mob AI (inlined distance to avoid method-call overhead) ---
        px, py = p.x, p.y
        _sqrt = math.sqrt
        for mob in self.mobs:
            if not mob.alive:
                if mob.respawn_timer > 0:
                    mob.respawn_timer -= 1
                    if mob.respawn_timer <= 0:
                        self._respawn_mob(mob)
                continue

            # Inline distance (saves 2 method calls per mob)
            _dx = mob.x - px
            _dy = mob.y - py
            dist = _sqrt(_dx * _dx + _dy * _dy)

            if not mob.in_combat:
                # Aggro check
                if dist <= mob.template.detect_range:
                    mob.in_combat = True
                    mob.target_player = True
                    p.in_combat = True
                    if p.is_eating:
                        p.is_eating = False
                else:
                    # Far non-combat mob: only process DoTs then skip
                    if mob.dot_remaining > 0:
                        mob.dot_remaining -= 1
                        mob.dot_timer -= 1
                        if mob.dot_timer <= 0:
                            self._damage_mob(mob, mob.dot_damage_per_tick,
                                             is_shadow=True, spell_family=FAMILY_SW_PAIN)
                            mob.dot_timer = 6
                    if mob.dot2_remaining > 0:
                        mob.dot2_remaining -= 1
                        mob.dot2_timer -= 1
                        if mob.dot2_timer <= 0:
                            self._damage_mob(mob, mob.dot2_damage_per_tick)
                            mob.dot2_timer = 6
                    if mob.dot3_remaining > 0:
                        mob.dot3_remaining -= 1
                        mob.dot3_timer -= 1
                        if mob.dot3_timer <= 0:
                            tick_dmg = mob.dot3_damage_per_tick
                            self._damage_mob(mob, tick_dmg, is_shadow=True,
                                             spell_family=FAMILY_DEVOURING_PLAGUE)
                            if mob.dot3_heals_caster:
                                p.hp = min(p.max_hp, p.hp + tick_dmg)
                            mob.dot3_timer = 6
                    if mob.dot4_remaining > 0:
                        mob.dot4_remaining -= 1
                        mob.dot4_timer -= 1
                        if mob.dot4_timer <= 0:
                            tick_dmg = mob.dot4_damage_per_tick
                            self._damage_mob(mob, tick_dmg, is_shadow=True,
                                             spell_family=FAMILY_VAMPIRIC_TOUCH)
                            mana_return = int(p.max_mana * 0.02)
                            p.mana = min(p.max_mana, p.mana + mana_return)
                            mob.dot4_timer = 6
                    continue

            if mob.target_player:
                # Leash check (inlined)
                sdx = mob.x - mob.spawn_x
                sdy = mob.y - mob.spawn_y
                if _sqrt(sdx * sdx + sdy * sdy) > self.MOB_LEASH_RANGE:
                    self._evade_mob(mob)
                    continue

                # Fear: mob flees instead of chasing/attacking
                if mob.feared:
                    mob.fear_remaining -= 1
                    if mob.fear_remaining <= 0:
                        mob.feared = False
                    else:
                        # Move in flee direction
                        move = self.MOB_SPEED
                        new_mx = mob.x + mob.fear_dx * move
                        new_my = mob.y + mob.fear_dy * move
                        if self.terrain:
                            new_mz = self.terrain.get_height(new_mx, new_my)
                            if self.terrain.check_walkable(mob.x, mob.y, mob.z, new_mx, new_my, new_mz):
                                mob.x = new_mx
                                mob.y = new_my
                                mob.z = new_mz
                        else:
                            mob.x = new_mx
                            mob.y = new_my
                    # Skip chase and melee while feared — fall through to DoTs
                else:
                    # Chase player (not feared)
                    if dist > 2.0:
                        move = min(self.MOB_SPEED, dist - 1.5)
                        new_mx = mob.x + (-_dx / dist) * move
                        new_my = mob.y + (-_dy / dist) * move
                        if self.terrain:
                            new_mz = self.terrain.get_height(new_mx, new_my)
                            if self.terrain.check_walkable(mob.x, mob.y, mob.z, new_mx, new_my, new_mz):
                                mob.x = new_mx
                                mob.y = new_my
                                mob.z = new_mz
                        else:
                            mob.x = new_mx
                            mob.y = new_my
                        # Recompute distance after move
                        _dx = mob.x - px
                        _dy = mob.y - py
                        dist = _sqrt(_dx * _dx + _dy * _dy)

                    # Melee attack (WotLK single-roll attack table)
                    if dist <= 5.0:
                        mob.attack_timer -= 1
                        if mob.attack_timer <= 0:
                            dmg = self.rng.randint(mob.template.min_damage, mob.template.max_damage)
                            roll = self.rng.random() * 100.0
                            outcome = resolve_mob_melee_attack(
                                attacker_level=mob.level,
                                defender_level=p.level,
                                defender_dodge=p.total_dodge,
                                defender_parry=p.total_parry,
                                defender_block=p.total_block,
                                defender_defense_bonus=p.total_defense,
                                defender_resilience_pct=p.total_resilience,
                                roll=roll,
                            )
                            if outcome == MELEE_MISS:
                                p.mob_misses += 1
                            elif outcome == MELEE_DODGE:
                                p.dodges += 1
                            elif outcome == MELEE_PARRY:
                                p.parries += 1
                            elif outcome == MELEE_BLOCK:
                                p.blocks += 1
                                # Block reduces damage by block_value, not a full avoid
                                dmg = max(0, dmg - p.total_block_value)
                                self._damage_player(dmg, mob)
                            elif outcome == MELEE_CRIT:
                                p.mob_crits += 1
                                dmg = int(dmg * 2.0)  # mob crit = 200% damage
                                self._damage_player(dmg, mob)
                            elif outcome == MELEE_CRUSHING:
                                p.mob_crushings += 1
                                dmg = int(dmg * 1.5)  # crushing = 150% damage
                                self._damage_player(dmg, mob)
                            else:  # MELEE_NORMAL
                                self._damage_player(dmg, mob)
                            mob.attack_timer = mob.template.attack_speed
                            p.combat_timer = 0

            # DoT processing (slot 1: SW:Pain — Shadow, VE heals)
            if mob.dot_remaining > 0:
                mob.dot_remaining -= 1
                mob.dot_timer -= 1
                if mob.dot_timer <= 0:
                    tick_dmg = mob.dot_damage_per_tick
                    self._damage_mob(mob, tick_dmg, is_shadow=True,
                                     spell_family=FAMILY_SW_PAIN)
                    mob.dot_timer = 6
            # DoT processing (slot 2: Holy Fire)
            if mob.dot2_remaining > 0:
                mob.dot2_remaining -= 1
                mob.dot2_timer -= 1
                if mob.dot2_timer <= 0:
                    self._damage_mob(mob, mob.dot2_damage_per_tick)
                    mob.dot2_timer = 6
            # DoT processing (slot 3: Devouring Plague — Shadow, heals caster)
            if mob.dot3_remaining > 0:
                mob.dot3_remaining -= 1
                mob.dot3_timer -= 1
                if mob.dot3_timer <= 0:
                    tick_dmg = mob.dot3_damage_per_tick
                    self._damage_mob(mob, tick_dmg, is_shadow=True,
                                     spell_family=FAMILY_DEVOURING_PLAGUE)
                    if mob.dot3_heals_caster:
                        p.hp = min(p.max_hp, p.hp + tick_dmg)
                    mob.dot3_timer = 6
            # DoT processing (slot 4: Vampiric Touch — Shadow, mana return)
            if mob.dot4_remaining > 0:
                mob.dot4_remaining -= 1
                mob.dot4_timer -= 1
                if mob.dot4_timer <= 0:
                    tick_dmg = mob.dot4_damage_per_tick
                    self._damage_mob(mob, tick_dmg, is_shadow=True,
                                     spell_family=FAMILY_VAMPIRIC_TOUCH)
                    # VT returns 2% of max mana per tick to caster
                    mana_return = int(p.max_mana * 0.02)
                    p.mana = min(p.max_mana, p.mana + mana_return)
                    mob.dot4_timer = 6
            # Shadow Weaving timer decay
            if mob.shadow_weaving_timer > 0:
                mob.shadow_weaving_timer -= 1
                if mob.shadow_weaving_timer <= 0:
                    mob.shadow_weaving_stacks = 0

        # --- Shield decay ---
        if p.shield_remaining > 0:
            p.shield_remaining -= 1
            if p.shield_remaining <= 0:
                p.shield_absorb = 0
        if p.shield_cooldown > 0:
            p.shield_cooldown -= 1

        # --- Spell cooldowns ---
        for sid in list(p.spell_cooldowns):
            p.spell_cooldowns[sid] -= 1
            if p.spell_cooldowns[sid] <= 0:
                del p.spell_cooldowns[sid]

        # --- HoT (Renew) ---
        if p.hot_remaining > 0:
            p.hot_remaining -= 1
            p.hot_timer -= 1
            if p.hot_timer <= 0:
                p.hp = min(p.max_hp, p.hp + p.hot_heal_per_tick)
                p.hot_timer = 6
            if p.hot_remaining <= 0:
                p.hot_heal_per_tick = 0

        # --- Buff: Inner Fire ---
        if p.inner_fire_remaining > 0:
            p.inner_fire_remaining -= 1
            if p.inner_fire_remaining <= 0:
                p.inner_fire_armor = 0
                p.inner_fire_spellpower = 0
                self.recalculate_stats()  # armor/SP changed

        # --- Buff: PW:Fortitude ---
        if p.fortitude_remaining > 0:
            p.fortitude_remaining -= 1
            if p.fortitude_remaining <= 0:
                # Remove Stamina bonus -> recalculate HP
                p.fortitude_stamina_bonus = 0
                p.fortitude_hp_bonus = 0
                self.recalculate_stats()
                p.hp = min(p.hp, p.max_hp)

        # --- Buff: Shadow Protection ---
        if p.shadow_prot_remaining > 0:
            p.shadow_prot_remaining -= 1
            if p.shadow_prot_remaining <= 0:
                p.shadow_prot_value = 0

        # --- Buff: Divine Spirit ---
        if p.divine_spirit_remaining > 0:
            p.divine_spirit_remaining -= 1
            if p.divine_spirit_remaining <= 0:
                p.divine_spirit_bonus = 0
                self.recalculate_stats()

        # --- Buff: Fear Ward ---
        if p.fear_ward_remaining > 0:
            p.fear_ward_remaining -= 1

        # --- Spirit Tap decay ---
        if p.spirit_tap_remaining > 0:
            p.spirit_tap_remaining -= 1
            if p.spirit_tap_remaining <= 0:
                self.recalculate_stats()  # Spirit bonus removed

        # --- Dispersion ---
        if p.dispersion_remaining > 0:
            p.dispersion_remaining -= 1
            # +6% mana per second = 3% per tick
            mana_regen = int(p.max_mana * 0.03)
            p.mana = min(p.max_mana, p.mana + mana_regen)
            if p.dispersion_remaining <= 0:
                pass  # Dispersion ended

        # --- Regen ---
        if p.in_combat:
            p.combat_timer += 1
        else:
            p.combat_timer += 1

        # HP regen: only out of combat, after OOC delay
        if not p.in_combat and p.combat_timer >= self.OOC_DELAY_TICKS:
            p.ooc_regen_accumulator += self.HP_REGEN_PER_TICK
            if p.ooc_regen_accumulator >= 1.0:
                heal = int(p.ooc_regen_accumulator)
                p.hp = min(p.max_hp, p.hp + heal)
                p.ooc_regen_accumulator -= heal

        # Mana regen (WotLK 5-second rule):
        #   OOC / not casting: Spirit-based regen + MP5 from gear
        #   While casting: only MP5 from gear (flat) + Meditation talent %
        mp5_per_tick = p.gear_mp5 / 5.0 * 0.5  # MP5 -> per tick (0.5s)
        if not p.is_casting:
            spi_regen = spirit_mana_regen(
                p.level, p.total_intellect, p.total_spirit, p.class_id)
            p.mana_regen_accumulator += spi_regen + mp5_per_tick
        else:
            # Meditation: +17/33/50% of Spirit-based mana regen continues while casting
            med_pts = self._get_talent_points("meditation")
            if med_pts > 0:
                spi_regen = spirit_mana_regen(
                    p.level, p.total_intellect, p.total_spirit, p.class_id)
                med_pcts = {1: 0.17, 2: 0.33, 3: 0.50}
                p.mana_regen_accumulator += spi_regen * med_pcts.get(med_pts, 0.50)
            p.mana_regen_accumulator += mp5_per_tick
        # Fallback minimum: 2% of max_mana per tick if spirit regen is too low
        min_regen = p.max_mana * self.MANA_REGEN_PCT_PER_TICK
        if not p.is_casting and p.mana_regen_accumulator < min_regen:
            p.mana_regen_accumulator = min_regen
        if p.mana_regen_accumulator >= 1.0:
            regen = int(p.mana_regen_accumulator)
            p.mana = min(p.max_mana, p.mana + regen)
            p.mana_regen_accumulator -= regen

        # --- Eat/Drink regen (5% HP + 5% Mana per second = 2.5% per tick) ---
        if p.is_eating:
            # Auto-interrupt if in combat (shouldn't happen, but safety check)
            if p.in_combat:
                p.is_eating = False
            else:
                hp_regen = int(p.max_hp * 0.025)
                mana_regen = int(p.max_mana * 0.025)
                p.hp = min(p.max_hp, p.hp + max(1, hp_regen))
                p.mana = min(p.max_mana, p.mana + max(1, mana_regen))
                # Stop eating when both HP and Mana are full
                if p.hp >= p.max_hp and p.mana >= p.max_mana:
                    p.is_eating = False

    def _damage_player(self, damage: int, attacker: 'Mob | None' = None):
        """Apply damage to player, considering armor mitigation, talents, and shield.

        Armor mitigation uses the WotLK formula from Unit::CalcArmorReducedDamage
        (Unit.cpp:2067):
          eff_level = attacker_level + 4.5*(attacker_level-59) if >59
          dr = 0.1 * armor / (8.5 * eff_level + 40)
          mitigation = dr / (1 + dr), capped at 75%
        """
        p = self.player
        # Taking damage interrupts eating
        if p.is_eating:
            p.is_eating = False
        # Dispersion: -90% damage taken
        if p.dispersion_remaining > 0:
            damage = max(1, int(damage * 0.10))
        # Shadowform: -15% physical damage taken (mob melee is physical)
        if p.shadowform_active:
            damage = max(1, int(damage * 0.85))
        # total_armor is pre-computed in recalculate_stats (gear + agi*2 + inner fire)
        if p.total_armor > 0:
            if attacker is not None:
                mob_level = attacker.level
            elif self.target and self.target.alive:
                mob_level = self.target.level
            else:
                mob_level = 1
                for m in self.mobs:
                    if m.alive and m.target_player:
                        mob_level = m.level
                        break
            eff_level = mob_level + (4.5 * (mob_level - 59) if mob_level > 59 else 0)
            dr = 0.1 * p.total_armor / (8.5 * eff_level + 40)
            mitigation = dr / (1 + dr)
            mitigation = min(mitigation, 0.75)
            damage = max(1, int(damage * (1 - mitigation)))
        if p.shield_absorb > 0:
            absorbed = min(p.shield_absorb, damage)
            p.shield_absorb -= absorbed
            damage -= absorbed
            if p.shield_absorb <= 0:
                p.shield_remaining = 0
        p.hp = max(0, p.hp - damage)

    def _respawn_mob(self, mob: Mob):
        """Respawn a dead mob at its spawn point."""
        from sim.creature_db import CreatureDB
        level = self.rng.randint(mob.template.min_level, mob.template.max_level)
        base_hp = CreatureDB.get_base_hp(level)
        mob.hp = base_hp
        mob.max_hp = base_hp
        mob.level = level
        mob.alive = True
        mob.in_combat = False
        mob.target_player = False
        mob.looted = False
        mob.x = mob.spawn_x
        mob.y = mob.spawn_y
        mob.z = mob.spawn_z
        mob.attack_timer = 0
        mob.dot_remaining = 0
        mob.dot_timer = 0
        mob.dot_damage_per_tick = 0
        mob.dot3_remaining = 0
        mob.dot3_timer = 0
        mob.dot3_damage_per_tick = 0
        mob.dot3_heals_caster = False
        mob.dot4_remaining = 0
        mob.dot4_timer = 0
        mob.dot4_damage_per_tick = 0
        mob.shadow_weaving_stacks = 0
        mob.shadow_weaving_timer = 0
        mob.misery_stacks = 0
        mob.feared = False
        mob.fear_remaining = 0
        mob.respawn_timer = 0

    def _evade_mob(self, mob: Mob):
        """Mob evades and returns to spawn."""
        mob.in_combat = False
        mob.target_player = False
        mob.hp = mob.max_hp
        mob.x = mob.spawn_x
        mob.y = mob.spawn_y
        mob.z = mob.spawn_z
        mob.attack_timer = 0
        mob.dot_remaining = 0
        mob.dot3_remaining = 0
        mob.dot3_heals_caster = False
        mob.dot4_remaining = 0
        mob.shadow_weaving_stacks = 0
        mob.shadow_weaving_timer = 0
        mob.misery_stacks = 0
        mob.feared = False
        mob.fear_remaining = 0
        self._check_combat_end()

    # ─── State Query ─────────────────────────────────────────────

    def get_nearby_mobs(self, scan_range: Optional[float] = None) -> list[dict]:
        """Get list of nearby mobs and vendors (within scan range)."""
        r = scan_range or self.SCAN_RANGE
        r_sq = r * r  # squared comparison avoids sqrt for far mobs
        px, py = self.player.x, self.player.y
        _sqrt = math.sqrt
        result = []
        for mob in self.mobs:
            dx = mob.x - px
            dy = mob.y - py
            dsq = dx * dx + dy * dy
            if dsq <= r_sq:
                d = _sqrt(dsq)
                result.append({
                    "uid": mob.uid,
                    "name": mob.template.name,
                    "level": mob.level,
                    "hp": mob.hp,
                    "max_hp": mob.max_hp,
                    "alive": mob.alive,
                    "x": mob.x,
                    "y": mob.y,
                    "z": mob.z,
                    "dist": d,
                    "target_player": mob.target_player,
                    "looted": mob.looted,
                    "attackable": 1 if mob.alive else 0,
                    "vendor": 0,
                })
        # Include quest NPCs
        for qnpc in self.quest_npcs:
            dx = qnpc.x - px
            dy = qnpc.y - py
            dsq = dx * dx + dy * dy
            if dsq <= r_sq:
                d = _sqrt(dsq)
                result.append({
                    "uid": qnpc.uid,
                    "name": qnpc.name,
                    "level": 10,
                    "hp": 100,
                    "max_hp": 100,
                    "alive": True,
                    "x": qnpc.x,
                    "y": qnpc.y,
                    "z": qnpc.z,
                    "dist": d,
                    "target_player": False,
                    "looted": False,
                    "attackable": 0,
                    "vendor": 0,
                    "questgiver": 1,
                    "entry": qnpc.entry,
                })
        # Include vendor NPCs
        for v in self.vendors:
            dx = v.x - px
            dy = v.y - py
            dsq = dx * dx + dy * dy
            if dsq <= r_sq:
                d = _sqrt(dsq)
                result.append({
                    "uid": v.uid,
                    "name": v.name,
                    "level": v.level,
                    "hp": 100,
                    "max_hp": 100,
                    "alive": True,
                    "x": v.x,
                    "y": v.y,
                    "z": v.z,
                    "dist": d,
                    "target_player": False,
                    "looted": False,
                    "attackable": 0,
                    "vendor": 1,
                })
        return result

    def get_target_info(self) -> dict:
        """Get info about current target."""
        if self.target is None:
            return {"status": "none", "hp": 0, "x": 0, "y": 0, "z": 82.0, "level": 0}
        status = "alive" if self.target.alive else "dead"
        return {
            "status": status,
            "hp": self.target.hp,
            "max_hp": self.target.max_hp,
            "x": self.target.x,
            "y": self.target.y,
            "z": self.target.z,
            "level": self.target.level,
            "has_sw_pain": self.target.dot_remaining > 0,
            "has_holy_fire": self.target.dot2_remaining > 0,
            "has_devouring_plague": self.target.dot3_remaining > 0,
            "has_vampiric_touch": self.target.dot4_remaining > 0,
        }

    def get_state_dict(self) -> dict:
        """Get full state dict matching WoWEnv format."""
        p = self.player
        t_info = self.get_target_info()
        nearby = self.get_nearby_mobs()
        return {
            "name": "SimBot",
            "hp": p.hp,
            "max_hp": p.max_hp,
            "power": p.mana,
            "max_power": p.max_mana,
            "level": p.level,
            "x": p.x,
            "y": p.y,
            "z": p.z,
            "o": p.orientation,
            "combat": "true" if p.in_combat else "false",
            "casting": "true" if p.is_casting else "false",
            "free_slots": p.free_slots,
            "equipped_upgrade": "true" if p.equipped_upgrade > 0 else "false",
            "target_status": t_info["status"],
            "target_hp": t_info["hp"],
            "target_level": t_info.get("level", 0),
            "xp_gained": p.xp_gained,
            "loot_copper": p.loot_copper,
            "loot_score": p.loot_score,
            "leveled_up": "true" if p.leveled_up else "false",
            "tx": t_info["x"],
            "ty": t_info["y"],
            "tz": t_info["z"],
            "has_shield": "true" if p.shield_remaining > 0 else "false",
            "has_renew": "true" if p.hot_remaining > 0 else "false",
            "has_inner_fire": "true" if p.inner_fire_remaining > 0 else "false",
            "has_fortitude": "true" if p.fortitude_remaining > 0 else "false",
            "mind_blast_ready": "true" if p.spell_cooldowns.get(FAMILY_MIND_BLAST, 0) <= 0 else "false",
            "holy_fire_ready": "true" if p.spell_cooldowns.get(FAMILY_HOLY_FIRE, 0) <= 0 else "false",
            "target_has_sw_pain": "true" if t_info.get("has_sw_pain") else "false",
            "target_has_holy_fire": "true" if t_info.get("has_holy_fire") else "false",
            "target_has_devouring_plague": "true" if (self.target and self.target.alive and self.target.dot3_remaining > 0) else "false",
            "target_has_vampiric_touch": "true" if t_info.get("has_vampiric_touch") else "false",
            "shadowform_active": "true" if p.shadowform_active else "false",
            "dispersion_active": "true" if p.dispersion_remaining > 0 else "false",
            "dispersion_ready": "true" if (self._get_talent_points("dispersion") > 0 and p.spell_cooldowns.get(FAMILY_DISPERSION, 0) <= 0) else "false",
            "is_channeling": "true" if p.channel_remaining > 0 else "false",
            "mind_flay_available": "true" if self._get_talent_points("mind_flay") > 0 else "false",
            "vampiric_touch_available": "true" if self._get_talent_points("vampiric_touch") > 0 else "false",
            "has_shadow_protection": "true" if p.shadow_prot_remaining > 0 else "false",
            "has_divine_spirit": "true" if p.divine_spirit_remaining > 0 else "false",
            "has_fear_ward": "true" if p.fear_ward_remaining > 0 else "false",
            "psychic_scream_ready": "true" if p.spell_cooldowns.get(FAMILY_PSYCHIC_SCREAM, 0) <= 0 else "false",
            "is_eating": "true" if p.is_eating else "false",
            "nearby_mobs": [
                {
                    "guid": str(m["uid"]),
                    "name": m["name"],
                    "level": m["level"],
                    "attackable": m["attackable"],
                    "vendor": m.get("vendor", 0),
                    "questgiver": m.get("questgiver", 0),
                    "target": "1" if m["target_player"] else "0",
                    "hp": m["hp"],
                    "x": m["x"],
                    "y": m["y"],
                }
                for m in nearby
            ],
            # Primary stats (sim-only, total = base + gear)
            "total_strength": p.total_strength,
            "total_agility": p.total_agility,
            "total_stamina": p.total_stamina,
            "total_intellect": p.total_intellect,
            "total_spirit": p.total_spirit,
            # Offensive stats
            "attack_power": p.total_attack_power,
            "ranged_ap": p.total_ranged_ap,
            "spell_power": p.total_spell_power,
            "melee_crit": p.total_melee_crit,
            "ranged_crit": p.total_ranged_crit,
            "spell_crit": p.total_spell_crit,
            "melee_haste": p.total_melee_haste,
            "ranged_haste": p.total_ranged_haste,
            "spell_haste": p.total_spell_haste,
            "hit_melee": p.total_hit_melee,
            "hit_ranged": p.total_hit_ranged,
            "hit_spell": p.total_hit_spell,
            "expertise": p.total_expertise,
            "armor_pen": p.total_armor_pen,
            # Defensive stats
            "total_armor": p.total_armor,
            "dodge": p.total_dodge,
            "parry": p.total_parry,
            "block": p.total_block,
            "block_value": p.total_block_value,
            "defense": p.total_defense,
            "resilience": p.total_resilience,
            # Gear totals (for backwards compat)
            "gear_armor": p.gear_armor,
            "gear_stamina": p.gear_stamina,
            "gear_intellect": p.gear_intellect,
            "gear_spirit": p.gear_spirit,
            # Equipment (sim-only): slot -> item summary
            "equipment": {
                EQUIPMENT_SLOT_NAMES.get(slot, str(slot)): {
                    "entry": item.entry,
                    "name": item.name,
                    "score": item.score,
                }
                for slot, item in p.equipment.items()
            },
            "equipment_slots_used": len(p.equipment),
            # Bag system (sim-only)
            "bags": {
                str(slot): {
                    "entry": bag.entry,
                    "name": bag.name,
                    "slots": bag.container_slots,
                }
                for slot, bag in p.bags.items()
            },
            "bag_slots_used": len(p.bags),
            "total_bag_slots": p.total_bag_slots,
            # Quest state (sim-only, not present in live TCP stream)
            "quest_active": len(self.active_quests) > 0,
            "quest_progress": self._get_quest_progress_ratio(),
            "quests_completed_total": self.quests_completed,
        }

    def _get_quest_progress_ratio(self) -> float:
        """Get aggregate quest completion progress as 0-1 ratio."""
        if not self.active_quests or not self.quest_db:
            return 0.0
        total_needed = 0
        total_done = 0
        for qid, prog in self.active_quests.items():
            qt = self.quest_db.templates[qid]
            for i, obj in enumerate(qt.objectives):
                total_needed += obj.count
                total_done += min(prog.counts[i], obj.count)
        return total_done / max(1, total_needed)

    def get_best_quest_npc(self) -> tuple:
        """Find the most relevant quest NPC for the bot to interact with.

        Returns (npc, npc_type) where npc_type is:
          'turn_in' — NPC can accept a completed quest
          'accept'  — NPC can give a new quest
          None      — no relevant quest NPC
        """
        if not self.quest_db:
            return None, None

        px, py = self.player.x, self.player.y
        best_npc = None
        best_type = None
        best_dist = float('inf')

        for npc in self.quest_npcs:
            dx = npc.x - px
            dy = npc.y - py
            dist = math.sqrt(dx * dx + dy * dy)

            # Turn-in NPCs have highest priority
            completable = self.quest_db.get_completable_quests(
                npc.entry, self.active_quests)
            if completable and dist < best_dist:
                best_npc = npc
                best_type = 'turn_in'
                best_dist = dist
                continue

            # Accept NPCs have second priority
            available = self.quest_db.get_available_quests(
                npc.entry, self.player.level, self.completed_quests,
                self.active_quests)
            if available and (best_type != 'turn_in') and dist < best_dist:
                best_npc = npc
                best_type = 'accept'
                best_dist = dist

        return best_npc, best_type

    def consume_events(self) -> dict:
        """Consume and reset accumulated event values (like real server)."""
        p = self.player
        events = {
            "xp_gained": p.xp_gained,
            "loot_copper": p.loot_copper,
            "loot_score": p.loot_score,
            "equipped_upgrade": p.equipped_upgrade,
            "leveled_up": p.leveled_up,
            "levels_gained": p.levels_gained,
            "loot_items": list(p.loot_items),
            "loot_failed": list(p.loot_failed),
            "sell_copper": p.sell_copper,
            "items_sold": p.items_sold,
            "new_areas": self._new_areas,
            "new_zones": self._new_zones,
            "new_maps": self._new_maps,
            "quest_xp": p.quest_xp_gained,
            "quest_copper": p.quest_copper_gained,
            "quests_completed": p.quests_completed_tick,
            # Combat event counters
            "dodges": p.dodges,
            "parries": p.parries,
            "blocks": p.blocks,
            "mob_misses": p.mob_misses,
            "mob_crits": p.mob_crits,
            "mob_crushings": p.mob_crushings,
            "spell_misses": p.spell_misses,
            "spell_crits": p.spell_crits,
        }
        p.xp_gained = 0
        p.loot_copper = 0
        p.loot_score = 0
        p.equipped_upgrade = 0.0
        p.leveled_up = False
        p.levels_gained = 0
        p.loot_items.clear()
        p.loot_failed.clear()
        p.sell_copper = 0
        p.items_sold = 0
        p.quest_xp_gained = 0
        p.quest_copper_gained = 0
        p.quests_completed_tick = 0
        # Reset combat event counters
        p.dodges = 0
        p.parries = 0
        p.blocks = 0
        p.mob_misses = 0
        p.mob_crits = 0
        p.mob_crushings = 0
        p.spell_misses = 0
        p.spell_crits = 0
        self._new_areas = 0
        self._new_zones = 0
        self._new_maps = 0
        return events
