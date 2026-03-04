"""
WoW Simulation Data Models.

Data classes for the combat simulation: items, spells, mobs, vendors,
quest NPCs, player state, and mob instances. Also includes hardcoded
game data (spell definitions, mob templates, spawn positions, vendor data).
"""

from dataclasses import dataclass, field

from sim.constants import CLASS_PRIEST, DEFAULT_BACKPACK_SLOTS


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
    mana_cost: int
    min_damage: int = 0
    max_damage: int = 0
    min_heal: int = 0
    max_heal: int = 0
    spell_range: float = 0.0
    is_dot: bool = False
    dot_damage: int = 0
    dot_ticks: int = 0     # total duration in ticks
    dot_interval: int = 6  # ticks between dot ticks (3s = 6 ticks)
    is_shield: bool = False
    shield_absorb: int = 0
    shield_duration: int = 0  # ticks
    gcd_ticks: int = 3    # 1.5s = 3 ticks
    cooldown_ticks: int = 0   # spell-specific cooldown (ticks), 0 = none
    is_hot: bool = False
    hot_heal: int = 0      # total HoT healing
    hot_ticks: int = 0     # total duration in ticks
    hot_interval: int = 6  # ticks between hot ticks (3s = 6 ticks)
    is_buff: bool = False
    buff_duration: int = 0  # ticks


SPELLS = {
    585: SpellDef(
        id=585, name="Smite",
        cast_ticks=3, mana_cost=6,
        min_damage=13, max_damage=17,
        spell_range=30.0,
    ),
    2050: SpellDef(
        id=2050, name="Lesser Heal",
        cast_ticks=3, mana_cost=11,
        min_heal=46, max_heal=56,
        spell_range=0.0,  # self-cast
    ),
    589: SpellDef(
        id=589, name="Shadow Word: Pain",
        cast_ticks=0, mana_cost=25,  # instant
        spell_range=30.0,
        is_dot=True,
        dot_damage=30,     # total 30 over 18s = ~5 per tick
        dot_ticks=36,      # 18s = 36 ticks
        dot_interval=6,    # tick every 3s
    ),
    17: SpellDef(
        id=17, name="Power Word: Shield",
        cast_ticks=0, mana_cost=25,  # instant
        is_shield=True,
        shield_absorb=44,
        shield_duration=60,  # 30s = 60 ticks
    ),
    8092: SpellDef(
        id=8092, name="Mind Blast",
        cast_ticks=3, mana_cost=50,  # 1.5s cast
        min_damage=39, max_damage=43,
        spell_range=30.0,
        cooldown_ticks=16,  # 8s = 16 ticks
    ),
    139: SpellDef(
        id=139, name="Renew",
        cast_ticks=0, mana_cost=30,  # instant
        is_hot=True,
        hot_heal=45,       # total heal over 15s
        hot_ticks=30,      # 15s = 30 ticks
        hot_interval=6,    # tick every 3s = 5 HoT ticks
    ),
    14914: SpellDef(
        id=14914, name="Holy Fire",
        cast_ticks=4, mana_cost=40,  # 2.0s cast
        min_damage=15, max_damage=20,
        spell_range=30.0,
        is_dot=True,
        dot_damage=12,     # total 12 over 6s
        dot_ticks=12,      # 6s = 12 ticks
        dot_interval=6,    # tick every 3s = 2 DoT ticks
        cooldown_ticks=20,  # 10s = 20 ticks
    ),
    588: SpellDef(
        id=588, name="Inner Fire",
        cast_ticks=0, mana_cost=20,  # instant
        is_buff=True,
        buff_duration=400,  # 200s ≈ 3.3 min = 400 ticks
    ),
    1243: SpellDef(
        id=1243, name="Power Word: Fortitude",
        cast_ticks=0, mana_cost=30,  # instant
        is_buff=True,
        buff_duration=1200,  # 600s = 10 min = 1200 ticks
    ),
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
    # Buff: PW:Fortitude
    fortitude_remaining: int = 0   # ticks
    fortitude_hp_bonus: int = 0
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
    # Quest tracking (consume-on-read)
    quest_xp_gained: int = 0            # XP from quest turn-ins this tick
    quest_copper_gained: int = 0        # copper from quest turn-ins this tick
    quests_completed_tick: int = 0      # quests completed this tick (consume-on-read)

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
    looted: bool = False
    spawn_x: float = 0.0
    spawn_y: float = 0.0
    spawn_z: float = 82.0


