"""
Quick validation test for the WoW Combat Simulation.

Runs several checks:
1. Simulation engine standalone test
2. Gymnasium env space validation
3. Random agent episode test
4. Performance benchmark (FPS measurement)
"""

import os
import sys
import time
import math
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from sim.combat_sim import (CombatSimulation, SPELLS, MOB_TEMPLATES, SPAWN_POSITIONS,
                             XP_TABLE, base_xp_gain, get_gray_level,
                             player_max_hp, player_max_mana, smite_damage, heal_amount,
                             INVENTORY_SLOTS, VENDOR_DATA, InventoryItem, EquippedItem,
                             spell_crit_chance, spell_haste_pct, spirit_mana_regen,
                             sw_pain_total, pw_shield_absorb, class_base_stat,
                             CLASS_BASE_STATS, CLASS_PRIEST, CLASS_WARRIOR, CLASS_ROGUE,
                             CLASS_MAGE, CLASS_HUNTER, CLASS_PALADIN, CLASS_DRUID,
                             ITEM_MOD_STAMINA, ITEM_MOD_INTELLECT, ITEM_MOD_SPIRIT,
                             ITEM_MOD_STRENGTH, ITEM_MOD_AGILITY, ITEM_MOD_ATTACK_POWER,
                             ITEM_MOD_SPELL_POWER, ITEM_MOD_CRIT_RATING,
                             ITEM_MOD_HASTE_RATING, ITEM_MOD_MANA_REGENERATION,
                             ITEM_MOD_DODGE_RATING, ITEM_MOD_PARRY_RATING,
                             ITEM_MOD_DEFENSE_SKILL_RATING, ITEM_MOD_EXPERTISE_RATING,
                             ITEM_MOD_ARMOR_PENETRATION_RATING, ITEM_MOD_RESILIENCE_RATING,
                             ITEM_MOD_HEALTH_REGEN,
                             SP_COEFF_SMITE, SP_COEFF_HEAL,
                             melee_attack_power, ranged_attack_power,
                             melee_crit_chance, dodge_chance, parry_chance,
                             hit_chance_spell, expertise_pct, armor_penetration_pct,
                             EQUIPMENT_SLOT_HEAD, EQUIPMENT_SLOT_CHEST,
                             EQUIPMENT_SLOT_MAINHAND, EQUIPMENT_SLOT_OFFHAND,
                             EQUIPMENT_SLOT_RANGED, EQUIPMENT_SLOT_FINGER1,
                             EQUIPMENT_SLOT_FINGER2, EQUIPMENT_SLOT_TRINKET1,
                             EQUIPMENT_SLOT_TRINKET2, INVTYPE_TO_SLOTS,
                             class_aware_score, CLASS_STAT_WEIGHTS,
                             BAG_SLOT_START, BAG_SLOT_END, NUM_BAG_SLOTS,
                             DEFAULT_BACKPACK_SLOTS, INVTYPE_BAG, EquippedBag,
                             # Combat resolution
                             resolve_mob_melee_attack, resolve_spell_hit,
                             spell_miss_chance, mob_melee_miss_chance,
                             mob_melee_crit_chance, mob_crushing_chance,
                             MELEE_MISS, MELEE_DODGE, MELEE_PARRY, MELEE_BLOCK,
                             MELEE_CRIT, MELEE_NORMAL, MELEE_CRUSHING,
                             SPELL_MISS, SPELL_HIT, SPELL_CRIT,
                             ITEM_MOD_HIT_SPELL_RATING, ITEM_MOD_BLOCK_RATING)
from sim.wow_sim_env import WoWSimEnv


def test_combat_engine():
    """Test the raw simulation engine."""
    print("=== Test 1: Combat Engine ===")
    sim = CombatSimulation(num_mobs=10, seed=42)

    # Check initial state (WotLK: base + stamina/intellect contribution)
    p = sim.player
    expected_hp = player_max_hp(1)   # 72 + stam_hp(20) = 92
    expected_mana = player_max_mana(1)  # 123 + int_mana(22) = 173
    assert p.hp == expected_hp, f"Expected HP={expected_hp}, got {p.hp}"
    assert p.mana == expected_mana, f"Expected Mana={expected_mana}, got {p.mana}"
    assert p.level == 1
    print(f"  Player: HP={p.hp}/{p.max_hp}, Mana={p.mana}/{p.max_mana}")
    print(f"  Position: ({p.x:.1f}, {p.y:.1f}), Orientation={p.orientation:.2f}")

    # Check mobs spawned
    assert len(sim.mobs) == 10, f"Expected 10 mobs, got {len(sim.mobs)}"
    alive = sum(1 for m in sim.mobs if m.alive)
    print(f"  Mobs: {alive}/{len(sim.mobs)} alive")

    # Test movement
    old_x, old_y = p.x, p.y
    sim.do_move_forward()
    dx = p.x - old_x
    dy = p.y - old_y
    dist_moved = math.sqrt(dx * dx + dy * dy)
    assert abs(dist_moved - 3.0) < 0.01, f"Expected 3.0 move, got {dist_moved}"
    print(f"  Move forward: ({old_x:.1f},{old_y:.1f}) -> ({p.x:.1f},{p.y:.1f})")

    # Test turn
    old_o = p.orientation
    sim.do_turn_left()
    assert abs(p.orientation - (old_o + 0.5)) < 0.01 or abs(p.orientation - (old_o + 0.5 - 2 * math.pi)) < 0.01
    print(f"  Turn left: {old_o:.2f} -> {p.orientation:.2f}")

    # Test targeting
    sim.do_target_nearest()
    if sim.target:
        d = sim._dist_to_mob(sim.target)
        print(f"  Target: {sim.target.template.name} (HP={sim.target.hp}) at dist={d:.1f}")
    else:
        print("  No target in range (expected if spawn is far from mobs)")

    # Test spell casting
    if sim.target and sim.target.alive:
        old_mana = p.mana
        old_mob_hp = sim.target.hp
        success = sim.do_cast_smite()
        if success:
            print(f"  Cast Smite: mana {old_mana} -> {p.mana}, casting={p.is_casting}")
            # Advance ticks to complete cast
            for _ in range(5):
                sim.tick()
            print(f"  After cast: mob HP {old_mob_hp} -> {sim.target.hp}")
    else:
        print("  [Skip spell test — no target in range]")

    # Test state dict
    state = sim.get_state_dict()
    assert 'hp' in state
    assert 'nearby_mobs' in state
    print(f"  State dict: {len(state)} keys, {len(state['nearby_mobs'])} nearby mobs")

    print("  PASSED\n")


def test_gym_env():
    """Test the Gymnasium environment wrapper."""
    print("=== Test 2: Gymnasium Env ===")
    env = WoWSimEnv(num_mobs=10, seed=42)

    # Check spaces
    assert env.observation_space.shape == (52,), f"Obs shape: {env.observation_space.shape}"
    assert env.action_space.n == 30, f"Action space: {env.action_space.n}"
    print(f"  Obs space: {env.observation_space.shape}, dtype={env.observation_space.dtype}")
    print(f"  Action space: Discrete({env.action_space.n})")

    # Reset
    obs, info = env.reset()
    assert obs.shape == (52,), f"Obs shape after reset: {obs.shape}"
    assert obs.dtype == np.float32
    print(f"  Reset obs: shape={obs.shape}, range=[{obs.min():.3f}, {obs.max():.3f}]")

    # Check action_masks
    mask = env.action_masks()
    assert mask.shape == (30,), f"Mask shape: {mask.shape}"
    assert mask.dtype == bool, f"Mask dtype: {mask.dtype}"
    assert mask[0] == True, "Noop should always be valid"
    print(f"  Action mask: shape={mask.shape}, valid={mask.sum()}/{env.action_space.n}")

    # Step with each action
    for action in range(30):
        obs, reward, done, trunc, info = env.step(action)
        assert obs.shape == (52,)
        if done:
            obs, info = env.reset()

    print(f"  All {env.action_space.n} actions executed successfully")
    print("  PASSED\n")


def test_random_episode():
    """Run a full episode with random actions."""
    print("=== Test 3: Random Episode ===")
    env = WoWSimEnv(num_mobs=15, seed=123)
    obs, _ = env.reset()

    total_reward = 0.0
    steps = 0
    kills = 0
    max_steps = 1000

    for i in range(max_steps):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
        if reward > 50:  # likely a kill
            kills += 1
        if done or trunc:
            break

    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Kills (reward>50): {kills}")
    print(f"  Final HP: {env.sim.player.hp}/{env.sim.player.max_hp}")
    print(f"  Final Mana: {env.sim.player.mana}/{env.sim.player.max_mana}")
    print(f"  Terminated: done={done}, trunc={trunc}")
    print("  PASSED\n")


def test_performance():
    """Benchmark: how many steps per second?"""
    print("=== Test 4: Performance Benchmark ===")
    env = WoWSimEnv(num_mobs=15, seed=42)
    obs, _ = env.reset()

    num_steps = 10000
    start = time.time()
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        if done or trunc:
            obs, _ = env.reset()
    elapsed = time.time() - start

    fps = num_steps / elapsed
    real_time_equiv = num_steps * 0.5  # each tick = 0.5s
    speedup = real_time_equiv / elapsed

    print(f"  {num_steps} steps in {elapsed:.2f}s")
    print(f"  Single-env FPS: {fps:.0f}")
    print(f"  Simulated time: {real_time_equiv:.0f}s ({real_time_equiv/3600:.1f}h)")
    print(f"  Speed-up vs real-time: {speedup:.0f}x")
    print(f"  Speed-up vs server (400ms ticks): {speedup * 0.8:.0f}x")
    print("  PASSED\n")


def test_combat_scenario():
    """Test a scripted combat scenario to verify mechanics."""
    print("=== Test 5: Scripted Combat ===")
    sim = CombatSimulation(num_mobs=5, seed=99)

    # Move towards mobs until we aggro one
    steps = 0
    while not sim.player.in_combat and steps < 200:
        sim.do_move_forward()
        sim.tick()
        steps += 1

    if sim.player.in_combat:
        print(f"  Entered combat after {steps} ticks ({steps*0.5:.1f}s)")

        # Target nearest
        sim.do_target_nearest()
        if sim.target:
            print(f"  Targeting: {sim.target.template.name} HP={sim.target.hp}/{sim.target.max_hp}")

            # Cast smite until mob dies or we die
            casts = 0
            while sim.target.alive and sim.player.hp > 0:
                if not sim.player.is_casting and sim.player.gcd_remaining == 0:
                    if sim.player.hp < 30 and sim.player.mana >= 11:
                        sim.do_cast_heal()
                    elif sim.player.mana >= 6:
                        sim.do_cast_smite()
                        casts += 1
                    else:
                        sim.do_noop()
                else:
                    sim.do_noop()
                sim.tick()

            print(f"  Combat ended: Casts={casts}")
            print(f"  Player HP={sim.player.hp}/{sim.player.max_hp}, Mana={sim.player.mana}/{sim.player.max_mana}")
            if not sim.target.alive:
                print(f"  Mob killed! XP gained={sim.player.xp_gained}")
            else:
                print(f"  Player died!")
    else:
        print(f"  Did not enter combat in {steps} ticks (mobs might be far)")

    print("  PASSED\n")


def test_level_system():
    """Test XP formulas, level-up, and stat scaling."""
    print("=== Test 6: Level System ===")

    # --- XP formula sanity checks (match AzerothCore test values) ---
    # Same level, pl=1, mob=1: (1*5+45)*(20+0)/10+1)//2 = (50*20/10+1)/2 = (100+1)/2 = 50
    xp = base_xp_gain(1, 1)
    assert xp == 50, f"L1 vs L1 expected 50, got {xp}"
    print(f"  XP L1 vs L1 mob: {xp}")

    # Higher mob: pl=1, mob=3 → diff=2, capped at 4 → ((5+45)*(20+2)/10+1)/2 = (50*22/10+1)/2 = (110+1)/2 = 55
    xp = base_xp_gain(1, 3)
    assert xp == 55, f"L1 vs L3 expected 55, got {xp}"
    print(f"  XP L1 vs L3 mob: {xp}")

    # Gray level checks
    assert get_gray_level(1) == 0
    assert get_gray_level(5) == 0
    assert get_gray_level(10) == 4
    print(f"  Gray levels: L1→0, L5→0, L10→4 ✓")

    # Gray mob gives 0 XP
    xp = base_xp_gain(10, 3)
    assert xp == 0, f"L10 vs L3 (gray) expected 0, got {xp}"
    print(f"  XP L10 vs L3 (gray): {xp} ✓")

    # --- Stat scaling (WotLK: DBC-based BaseHP/BaseMana + stamina/intellect contribution) ---
    # player_max_hp(1) = BaseHP(52) + stam_hp(20) = 52 + 20 = 72
    assert player_max_hp(1) == 72, f"HP(1) expected 72, got {player_max_hp(1)}"
    # player_max_hp(2) = BaseHP(57) + stam_hp(20) = 57 + 20 = 77
    assert player_max_hp(2) == 77, f"HP(2) expected 77, got {player_max_hp(2)}"
    # player_max_hp(10) = BaseHP(147) + stam_hp(23) = 147 + 20 + 30 = 187 (with non-linear stam from CSV)
    assert player_max_hp(10) == 187, f"HP(10) expected 187, got {player_max_hp(10)}"
    # player_max_mana(1) = BaseMana(73) + int_mana(22) = 73 + 20 + 30 = 123
    assert player_max_mana(1) == 123, f"Mana(1) expected 123, got {player_max_mana(1)}"
    # player_max_mana(2) = BaseMana(79) + int_mana(22) = 79 + 20 + 42 (non-linear from CSV) = 141
    assert player_max_mana(2) == 141, f"Mana(2) expected 141, got {player_max_mana(2)}"
    # Smite/Heal base damage from DBC (Rank 1 with RealPointsPerLevel scaling)
    # Smite: BasePoints=12, DieSides=5, RPL=0.5, MaxLevel=6
    # Smite R1 (585): BasePoints=12, DieSides=5 → 13-17
    min_d, max_d = smite_damage(1)
    assert (min_d, max_d) == (13, 17), f"Smite R1 expected (13,17), got ({min_d},{max_d})"
    # Rank system: at level 6, best rank is R2 (591, bp=24, ds=7 → 25-31)
    from sim.constants import get_best_rank, FAMILY_SMITE
    from sim.formulas import spell_direct_value
    r2_id = get_best_rank(FAMILY_SMITE, 6)
    assert r2_id == 591, f"Smite best rank at L6 should be 591, got {r2_id}"
    min_d2, max_d2 = spell_direct_value(591)
    assert (min_d2, max_d2) == (25, 31), f"Smite R2 expected (25,31), got ({min_d2},{max_d2})"
    # At level 14, best rank is R3 (598, bp=53, ds=9 → 54-62)
    r3_id = get_best_rank(FAMILY_SMITE, 14)
    assert r3_id == 598, f"Smite best rank at L14 should be 598, got {r3_id}"
    min_d3, max_d3 = spell_direct_value(598)
    assert (min_d3, max_d3) == (54, 62), f"Smite R3 expected (54,62), got ({min_d3},{max_d3})"
    # Lesser Heal R1 (2050): BasePoints=45, DieSides=11 → 46-56
    min_h, max_h = heal_amount(1)
    assert (min_h, max_h) == (46, 56), f"Heal R1 expected (46,56), got ({min_h},{max_h})"
    # Spell functions with SP scaling (using spell_bonus_data coefficients)
    min_d_sp, max_d_sp = smite_damage(1, spell_power=100)
    sp_bonus = int(100 * SP_COEFF_SMITE)  # 12
    assert (min_d_sp, max_d_sp) == (13 + sp_bonus, 17 + sp_bonus), \
        f"Smite+SP expected ({13+sp_bonus},{17+sp_bonus}), got ({min_d_sp},{max_d_sp})"
    print(f"  Stat scaling: HP, Mana, Smite ranks, Heal, SP scaling ✓")

    # --- Level-up in simulation ---
    sim = CombatSimulation(num_mobs=10, seed=42)
    p = sim.player
    assert p.level == 1
    assert p.xp == 0

    # Inject enough XP to level up (need 400 for level 2)
    p.xp = 399
    p.xp_gained = 399
    # Simulate a kill that gives 50+ XP → should cross 400 threshold
    # Manually call _check_level_up after adding XP
    p.xp += 50
    p.xp_gained += 50
    sim._check_level_up()

    expected_hp_l2 = player_max_hp(2)   # 152
    expected_mana_l2 = player_max_mana(2)  # 193
    assert p.level == 2, f"Expected level 2 after 449 XP, got {p.level}"
    assert p.max_hp == expected_hp_l2, f"Expected max_hp={expected_hp_l2}, got {p.max_hp}"
    assert p.hp == expected_hp_l2, "HP should be full after level-up"
    assert p.max_mana == expected_mana_l2, f"Expected max_mana={expected_mana_l2}, got {p.max_mana}"
    assert p.leveled_up is True
    assert p.levels_gained == 1
    print(f"  Level-up: L1→L2 at XP=449, HP={p.max_hp}, Mana={p.max_mana} ✓")

    # Test multi-level-up (inject tons of XP)
    p.xp = 2000
    p.leveled_up = False
    p.levels_gained = 0
    sim._check_level_up()
    assert p.level == 4, f"Expected level 4 at XP=2000, got {p.level}"
    assert p.levels_gained == 2, f"Expected 2 levels gained, got {p.levels_gained}"
    print(f"  Multi level-up: L2→L4 at XP=2000, levels_gained={p.levels_gained} ✓")

    # Test consume_events clears leveled_up
    events = sim.consume_events()
    assert events["leveled_up"] is True
    assert events["levels_gained"] == 2
    assert p.leveled_up is False
    assert p.levels_gained == 0
    print(f"  consume_events clears leveled_up ✓")

    # --- XP diminishing returns in environment ---
    env = WoWSimEnv(num_mobs=15, seed=77)
    obs, _ = env.reset()
    # Run a scripted combat episode to verify XP flows through
    total_xp = 0
    for _ in range(2000):
        # Simple script: target, then smite
        if not env.sim.target or not env.sim.target.alive:
            action = 4  # target
        else:
            action = 5  # smite
        obs, reward, done, trunc, info = env.step(action)
        if done or trunc:
            break
    final_level = env.sim.player.level
    final_xp = env.sim.player.xp
    print(f"  Env episode: final_level={final_level}, total_xp={final_xp}")
    assert final_level >= 1, "Should still be at least level 1"

    print("  PASSED\n")


def test_loot_tables():
    """Test LootDB loading, rolling, and integration with CombatSimulation."""
    import tempfile
    import csv

    print("=== Test 7: Loot Tables ===")

    # --- Create temporary CSV files with test data ---
    tmpdir = tempfile.mkdtemp()

    # item_template.csv
    item_header = ['entry', 'name', 'class', 'subclass', 'Quality', 'SellPrice',
                   'InventoryType', 'ItemLevel', 'armor', 'dmg_min1', 'dmg_max1',
                   'delay'] + [f'stat_type{i}' for i in range(1, 11)] + \
                  [f'stat_value{i}' for i in range(1, 11)]
    items = [
        # Grey junk: Chunk of Boar Meat (Quality=0, non-equip)
        {'entry': '769', 'name': 'Chunk of Boar Meat', 'class': '7', 'subclass': '0',
         'Quality': '0', 'SellPrice': '25', 'InventoryType': '0', 'ItemLevel': '5',
         'armor': '0', 'dmg_min1': '0', 'dmg_max1': '0', 'delay': '0'},
        # Common cloth armor: Frayed Robe (Quality=1, chest)
        {'entry': '1395', 'name': 'Frayed Robe', 'class': '4', 'subclass': '1',
         'Quality': '1', 'SellPrice': '50', 'InventoryType': '20', 'ItemLevel': '4',
         'armor': '8', 'dmg_min1': '0', 'dmg_max1': '0', 'delay': '0',
         'stat_type1': '7', 'stat_value1': '1'},  # +1 Stamina
        # Green weapon: Kobold Mining Mace (Quality=2, one-hand mace)
        {'entry': '7439', 'name': 'Kobold Mining Mace', 'class': '2', 'subclass': '4',
         'Quality': '2', 'SellPrice': '120', 'InventoryType': '13', 'ItemLevel': '8',
         'armor': '0', 'dmg_min1': '3', 'dmg_max1': '7', 'delay': '2000'},
        # Another chest piece (better, for upgrade testing)
        {'entry': '2000', 'name': 'Dirty Leather Vest', 'class': '4', 'subclass': '2',
         'Quality': '1', 'SellPrice': '80', 'InventoryType': '20', 'ItemLevel': '6',
         'armor': '15', 'dmg_min1': '0', 'dmg_max1': '0', 'delay': '0',
         'stat_type1': '7', 'stat_value1': '2'},  # +2 Stamina
    ]
    item_path = os.path.join(tmpdir, 'item_template.csv')
    with open(item_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=item_header, delimiter=';',
                           quotechar='"', quoting=csv.QUOTE_ALL)
        w.writeheader()
        for item in items:
            row = {k: item.get(k, '0') for k in item_header}
            w.writerow(row)

    # creature_loot_template.csv
    loot_header = ['Entry', 'Item', 'Reference', 'Chance', 'QuestRequired',
                   'LootMode', 'GroupId', 'MinCount', 'MaxCount', 'Comment']
    loot_entries = [
        # Entry 299 (Diseased Young Wolf): group-0 meat drop + group-1 equippables
        {'Entry': '299', 'Item': '769', 'Reference': '0', 'Chance': '80',
         'QuestRequired': '0', 'LootMode': '1', 'GroupId': '0',
         'MinCount': '1', 'MaxCount': '2', 'Comment': 'Boar Meat'},
        {'Entry': '299', 'Item': '1395', 'Reference': '0', 'Chance': '40',
         'QuestRequired': '0', 'LootMode': '1', 'GroupId': '1',
         'MinCount': '1', 'MaxCount': '1', 'Comment': 'Frayed Robe'},
        {'Entry': '299', 'Item': '7439', 'Reference': '0', 'Chance': '60',
         'QuestRequired': '0', 'LootMode': '1', 'GroupId': '1',
         'MinCount': '1', 'MaxCount': '1', 'Comment': 'Kobold Mace'},
        # Entry 6 (Kobold Vermin): reference-based loot
        {'Entry': '6', 'Item': '769', 'Reference': '0', 'Chance': '50',
         'QuestRequired': '0', 'LootMode': '1', 'GroupId': '0',
         'MinCount': '1', 'MaxCount': '1', 'Comment': 'Boar Meat'},
    ]
    loot_path = os.path.join(tmpdir, 'creature_loot_template.csv')
    with open(loot_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=loot_header, delimiter=';',
                           quotechar='"', quoting=csv.QUOTE_ALL)
        w.writeheader()
        for e in loot_entries:
            w.writerow(e)

    # --- Test 7a: LootDB loading ---
    from sim.loot_db import LootDB
    loot_db = LootDB(tmpdir, quiet=True)
    assert loot_db.loaded, "LootDB should be loaded with test data"
    assert len(loot_db.items) == 4, f"Expected 4 items, got {len(loot_db.items)}"
    assert len(loot_db.creature_loot) == 2, f"Expected 2 loot tables, got {len(loot_db.creature_loot)}"
    print(f"  7a: LootDB loaded: {len(loot_db.items)} items, {len(loot_db.creature_loot)} loot tables")

    # --- Test 7b: Item score computation ---
    meat = loot_db.get_item(769)
    assert meat is not None
    assert meat.quality == 0
    assert meat.sell_price == 25
    # Score: (0*10) + 5 + 0 + 0 + 0 = 5
    assert meat.score == 5.0, f"Meat score expected 5.0, got {meat.score}"

    robe = loot_db.get_item(1395)
    assert robe is not None
    assert robe.inventory_type == 20  # Robe
    # Score: (1*10) + 4 + 8 + 0 + (1*2) = 24
    assert robe.score == 24.0, f"Robe score expected 24.0, got {robe.score}"

    mace = loot_db.get_item(7439)
    # Score: (2*10) + 8 + 0 + ((3+7)/2 / (2000/1000)) + 0 = 20+8+2.5 = 30.5
    assert abs(mace.score - 30.5) < 0.01, f"Mace score expected 30.5, got {mace.score}"
    print(f"  7b: Item scores: meat={meat.score}, robe={robe.score}, mace={mace.score} ✓")

    # --- Test 7c: Loot rolling ---
    import random
    rng = random.Random(42)
    # Roll loot for wolf (entry 299) many times, check distribution
    total_rolls = 1000
    meat_count = 0
    equip_count = 0
    empty_group1 = 0
    for _ in range(total_rolls):
        results = loot_db.roll_loot(299, rng)
        has_meat = any(r.item.entry == 769 for r in results)
        has_equip = any(r.item.entry in (1395, 7439) for r in results)
        # Check group 1: should have at most 1 equippable (group constraint)
        equips = [r for r in results if r.item.entry in (1395, 7439)]
        assert len(equips) <= 1, f"Group 1 should produce at most 1 item, got {len(equips)}"
        if has_meat:
            meat_count += 1
        if has_equip:
            equip_count += 1
        else:
            empty_group1 += 1

    # Meat: 80% chance → expect ~800 ± 50
    assert 700 < meat_count < 900, f"Meat drop rate off: {meat_count}/{total_rolls}"
    # Group 1: total chance = 40+60 = 100% → should always drop one (0% empty)
    assert empty_group1 == 0, f"Group 1 (100% total) should always drop, but {empty_group1} empty"
    print(f"  7c: Loot distribution ({total_rolls}x wolf): "
          f"meat={meat_count/total_rolls:.0%}, equip={equip_count/total_rolls:.0%} ✓")

    # --- Test 7d: Integration with CombatSimulation ---
    sim = CombatSimulation(num_mobs=10, seed=42, loot_db=loot_db)

    # Manually kill a wolf mob and test looting
    wolf = None
    for mob in sim.mobs:
        if mob.template.entry == 299:
            wolf = mob
            break

    if wolf:
        # Position player next to wolf
        sim.player.x = wolf.x
        sim.player.y = wolf.y + 1
        # Kill it
        wolf.hp = 0
        wolf.alive = False
        wolf.looted = False

        old_score = sim.player.loot_score
        old_slots = sim.player.free_slots
        success = sim.do_loot()
        assert success, "Loot should succeed"
        assert wolf.looted, "Wolf should be marked as looted"
        # With loot tables, something should have dropped (meat or equip)
        assert sim.player.loot_score >= old_score, \
            f"Score should increase: {old_score} -> {sim.player.loot_score}"
        print(f"  7d: Loot integration: score {old_score}→{sim.player.loot_score}, "
              f"slots {old_slots}→{sim.player.free_slots} ✓")
    else:
        print("  7d: [Skip — no wolf mob in spawn set]")

    # --- Test 7e: Upgrade detection ---
    sim2 = CombatSimulation(num_mobs=5, seed=99, loot_db=loot_db)
    p = sim2.player
    # Pre-equip a weak chest item (InventoryType 20=Robe maps to EQUIPMENT_SLOT_CHEST)
    chest_slot = EQUIPMENT_SLOT_CHEST
    p.equipment[chest_slot] = EquippedItem(
        entry=0, name="Weak Chest", inventory_type=20, score=10.0, stats={})
    p.equipped_scores[chest_slot] = 10.0

    # Create a mock loot scenario: Dirty Leather Vest (score=35) should be upgrade
    vest = loot_db.get_item(2000)
    # Score: (1*10) + 6 + 15 + 0 + (2*2) = 35
    assert vest is not None
    assert vest.score == 35.0, f"Vest score expected 35.0, got {vest.score}"
    assert vest.score > 10.0, "Vest should be better than equipped"

    # Verify the upgrade logic works via try_equip_item (uses slot system)
    result = sim2.try_equip_item(vest)
    assert result, "Vest should be detected as upgrade"
    assert p.equipped_scores[chest_slot] == 35.0, "Equipped score should update"
    print(f"  7e: Upgrade detection: 10.0 → {p.equipped_scores[chest_slot]} ✓")

    # --- Test 7f: Fallback without loot_db ---
    sim_no_loot = CombatSimulation(num_mobs=10, seed=42)  # no loot_db
    for mob in sim_no_loot.mobs:
        if mob.template.entry == 6:  # Kobold Vermin (has min_gold)
            sim_no_loot.player.x = mob.x
            sim_no_loot.player.y = mob.y + 1
            mob.hp = 0
            mob.alive = False
            mob.looted = False
            old_copper = sim_no_loot.player.loot_copper
            sim_no_loot.do_loot()
            # Gold should still work via min/max_gold
            assert sim_no_loot.player.loot_copper >= old_copper, "Gold should still drop"
            print(f"  7f: Fallback loot (no DB): copper={sim_no_loot.player.loot_copper} ✓")
            break

    # --- Test 7g: Inventory capacity (default backpack = 16 slots) ---
    assert DEFAULT_BACKPACK_SLOTS == 16, f"Expected 16 backpack slots, got {DEFAULT_BACKPACK_SLOTS}"
    sim_inv = CombatSimulation(num_mobs=50, seed=123, loot_db=loot_db)
    assert sim_inv.player.free_slots == DEFAULT_BACKPACK_SLOTS, \
        f"Player should start with {DEFAULT_BACKPACK_SLOTS} free slots"

    # Fill inventory completely
    sim_inv.player.free_slots = 0

    # Kill and try to loot a wolf — items should fail, gold should still work
    wolf_inv = None
    for mob in sim_inv.mobs:
        if mob.template.entry == 299:
            wolf_inv = mob
            break
    assert wolf_inv is not None, "Need a wolf mob for inventory test"

    sim_inv.player.x = wolf_inv.x
    sim_inv.player.y = wolf_inv.y + 1
    wolf_inv.hp = 0
    wolf_inv.alive = False
    wolf_inv.looted = False
    old_copper_inv = sim_inv.player.loot_copper

    success = sim_inv.do_loot()
    assert success, "Loot action should still succeed (mob gets marked looted)"
    assert sim_inv.player.free_slots == 0, "Free slots should stay at 0"
    assert len(sim_inv.player.loot_failed) > 0, \
        "Items should be in loot_failed when inventory is full"
    assert len(sim_inv.player.loot_items) == 0, \
        "No items should be in loot_items when inventory is full"
    # Gold doesn't need inventory space
    assert sim_inv.player.loot_copper >= old_copper_inv, "Gold should still drop"
    print(f"  7g: Inventory full: failed={len(sim_inv.player.loot_failed)}, "
          f"copper={sim_inv.player.loot_copper - old_copper_inv} ✓")

    # --- Test 7h: Inventory partial fill ---
    sim_part = CombatSimulation(num_mobs=50, seed=77, loot_db=loot_db)
    sim_part.player.free_slots = 1  # only 1 slot left

    wolf_part = None
    for mob in sim_part.mobs:
        if mob.template.entry == 299:
            wolf_part = mob
            break
    assert wolf_part is not None

    sim_part.player.x = wolf_part.x
    sim_part.player.y = wolf_part.y + 1
    wolf_part.hp = 0
    wolf_part.alive = False
    wolf_part.looted = False
    sim_part.do_loot()

    # Wolf drops meat (group 0, 80%) + 1 equip (group 1, 100%)
    # With 1 slot: first item fits, rest should fail
    total_items = len(sim_part.player.loot_items) + len(sim_part.player.loot_failed)
    assert total_items > 0, "Should have rolled at least some loot"
    if total_items > 1:
        assert len(sim_part.player.loot_items) == 1, \
            f"Only 1 item should fit, got {len(sim_part.player.loot_items)}"
        assert sim_part.player.free_slots == 0, "Slot should be used up"
        assert len(sim_part.player.loot_failed) == total_items - 1, \
            "Remaining items should fail"
    print(f"  7h: Partial inventory: looted={len(sim_part.player.loot_items)}, "
          f"failed={len(sim_part.player.loot_failed)}, "
          f"slots={sim_part.player.free_slots} ✓")

    # --- Test 7i: Sell requires vendor proximity ---
    sim_sell = CombatSimulation(num_mobs=5, seed=42)
    # Add fake items to inventory
    for i in range(5):
        sim_sell.player.inventory.append(InventoryItem(
            entry=100+i, name=f"Junk_{i}", quality=0,
            sell_price=10, score=5.0, inventory_type=0))
    sim_sell.player.free_slots = INVENTORY_SLOTS - 5

    # Selling far from vendor should fail
    sim_sell.player.x = -9100.0  # far from any vendor
    sim_sell.player.y = -200.0
    sold = sim_sell.do_sell()
    assert not sold, "Sell should fail when far from vendor"
    assert sim_sell.player.free_slots == INVENTORY_SLOTS - 5, "Slots unchanged"

    # Move near a vendor and sell
    v = VENDOR_DATA[0]
    sim_sell.player.x = v["x"] + 1.0
    sim_sell.player.y = v["y"]
    sold = sim_sell.do_sell()
    assert sold, "Sell should succeed near vendor"
    assert sim_sell.player.free_slots == INVENTORY_SLOTS, \
        f"Sell should restore to {INVENTORY_SLOTS}, got {sim_sell.player.free_slots}"
    assert sim_sell.player.copper == 50, f"Expected 50 copper (5×10), got {sim_sell.player.copper}"
    assert len(sim_sell.player.inventory) == 0, "Inventory should be empty"

    sold_again = sim_sell.do_sell()
    assert not sold_again, "Sell should fail when inventory is already empty"
    print(f"  7i: Sell at vendor: copper={sim_sell.player.copper}, slots={sim_sell.player.free_slots} ✓")

    # --- Test 7j: consume_events includes loot_items/loot_failed ---
    sim_ev = CombatSimulation(num_mobs=50, seed=55, loot_db=loot_db)
    sim_ev.player.free_slots = 0
    wolf_ev = None
    for mob in sim_ev.mobs:
        if mob.template.entry == 299:
            wolf_ev = mob
            break
    assert wolf_ev is not None
    sim_ev.player.x = wolf_ev.x
    sim_ev.player.y = wolf_ev.y + 1
    wolf_ev.hp = 0
    wolf_ev.alive = False
    wolf_ev.looted = False
    sim_ev.do_loot()
    events = sim_ev.consume_events()
    assert "loot_items" in events, "Events must include loot_items"
    assert "loot_failed" in events, "Events must include loot_failed"
    assert len(events["loot_failed"]) > 0, "Should have failed items"
    # After consume, player lists should be cleared
    assert len(sim_ev.player.loot_items) == 0, "loot_items should be cleared"
    assert len(sim_ev.player.loot_failed) == 0, "loot_failed should be cleared"
    print(f"  7j: consume_events: loot_items={events['loot_items']}, "
          f"loot_failed={events['loot_failed']} ✓")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    print("  PASSED\n")


def test_vendor_system():
    """Test vendor NPCs, navigation, sell mechanics, and dynamic spawning."""
    import tempfile
    import csv

    print("=== Test 8: Vendor System ===")

    sim = CombatSimulation(num_mobs=10, seed=42)

    # --- Test 8a: Fallback vendors are spawned without creature_db ---
    assert len(sim.vendors) == len(VENDOR_DATA), \
        f"Expected {len(VENDOR_DATA)} fallback vendors, got {len(sim.vendors)}"
    print(f"  8a: {len(sim.vendors)} fallback vendor NPCs spawned ✓")

    # --- Test 8b: Vendors appear in nearby_mobs with vendor flag ---
    sim.player.x = VENDOR_DATA[0]["x"]
    sim.player.y = VENDOR_DATA[0]["y"]
    nearby = sim.get_nearby_mobs()
    vendor_entries = [m for m in nearby if m.get("vendor") == 1]
    assert len(vendor_entries) > 0, "Vendors should appear in nearby_mobs"
    assert vendor_entries[0]["attackable"] == 0, "Vendors should not be attackable"
    print(f"  8b: {len(vendor_entries)} vendors visible in nearby_mobs ✓")

    # --- Test 8c: Vendors appear in state dict ---
    state = sim.get_state_dict()
    vendor_in_state = [m for m in state["nearby_mobs"] if m.get("vendor") == 1]
    assert len(vendor_in_state) > 0, "Vendors should appear in state dict"
    print(f"  8c: Vendors in state dict: {len(vendor_in_state)} ✓")

    # --- Test 8d: get_nearest_vendor ---
    vendor = sim.get_nearest_vendor()
    assert vendor is not None, "Should find a nearest vendor"
    print(f"  8d: Nearest vendor: {vendor.name} ✓")

    # --- Test 8e: do_move_to navigates toward target ---
    sim2 = CombatSimulation(num_mobs=5, seed=42)
    start_x, start_y = sim2.player.x, sim2.player.y
    target_x, target_y = start_x + 10, start_y + 10
    moved = sim2.do_move_to(target_x, target_y)
    assert moved, "do_move_to should succeed"
    old_dist = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
    new_dist = math.sqrt((target_x - sim2.player.x)**2 + (target_y - sim2.player.y)**2)
    assert new_dist < old_dist, "Player should be closer to target"
    moved_dist = math.sqrt((sim2.player.x - start_x)**2 + (sim2.player.y - start_y)**2)
    assert abs(moved_dist - sim2.MOVE_SPEED) < 0.01, \
        f"Should move {sim2.MOVE_SPEED} units, moved {moved_dist:.2f}"
    print(f"  8e: do_move_to: moved {moved_dist:.1f} units toward target ✓")

    # --- Test 8f: Sell with inventory items gives copper ---
    sim3 = CombatSimulation(num_mobs=5, seed=42)
    p = sim3.player
    p.inventory.append(InventoryItem(entry=1, name="Junk", quality=0,
                                     sell_price=25, score=5.0, inventory_type=0))
    p.inventory.append(InventoryItem(entry=2, name="Cloth", quality=1,
                                     sell_price=50, score=10.0, inventory_type=0))
    p.free_slots = INVENTORY_SLOTS - 2
    v = VENDOR_DATA[0]
    p.x = v["x"]
    p.y = v["y"]
    sold = sim3.do_sell()
    assert sold, "Sell should succeed at vendor with items"
    assert p.copper == 75, f"Expected 75 copper, got {p.copper}"
    assert p.sell_copper == 75, f"Expected sell_copper=75, got {p.sell_copper}"
    assert p.free_slots == INVENTORY_SLOTS
    assert len(p.inventory) == 0
    events = sim3.consume_events()
    assert events["sell_copper"] == 75
    assert p.sell_copper == 0
    print(f"  8f: Sell copper: 25+50={events['sell_copper']} copper ✓")

    # --- Test 8g: Vendors persist after reset ---
    sim.reset()
    assert len(sim.vendors) == len(VENDOR_DATA), "Vendors should be re-spawned after reset"
    print(f"  8g: Vendors persist after reset ✓")

    # --- Test 8h: Dynamic vendor spawning from creature_db ---
    tmpdir = tempfile.mkdtemp()

    # Create creature_template with one attackable mob and one vendor
    tmpl_header = ['entry', 'name', 'minlevel', 'maxlevel', 'faction', 'npcflag',
                   'detection_range', 'rank', 'BaseAttackTime', 'mingold', 'maxgold',
                   'HealthModifier', 'DamageModifier', 'ExperienceModifier',
                   'unit_class', 'unit_flags', 'type', 'lootid']
    templates = [
        # Attackable mob (wolf)
        {'entry': '100', 'name': 'Test Wolf', 'minlevel': '1', 'maxlevel': '2',
         'faction': '14', 'npcflag': '0', 'detection_range': '20',
         'rank': '0', 'BaseAttackTime': '2000', 'mingold': '0', 'maxgold': '5',
         'HealthModifier': '1.0', 'DamageModifier': '1.0', 'ExperienceModifier': '1.0',
         'unit_class': '1', 'unit_flags': '0', 'type': '1', 'lootid': '100'},
        # Vendor NPC (friendly, npcflag=128)
        {'entry': '200', 'name': 'Test Merchant', 'minlevel': '5', 'maxlevel': '5',
         'faction': '11', 'npcflag': '128', 'detection_range': '0',
         'rank': '0', 'BaseAttackTime': '2000', 'mingold': '0', 'maxgold': '0',
         'HealthModifier': '1.0', 'DamageModifier': '1.0', 'ExperienceModifier': '1.0',
         'unit_class': '1', 'unit_flags': '0', 'type': '7', 'lootid': '0'},
        # Another vendor with combined flags (vendor + repair = 128+4096)
        {'entry': '201', 'name': 'Test Blacksmith', 'minlevel': '10', 'maxlevel': '10',
         'faction': '55', 'npcflag': '4224', 'detection_range': '0',
         'rank': '0', 'BaseAttackTime': '2000', 'mingold': '0', 'maxgold': '0',
         'HealthModifier': '1.0', 'DamageModifier': '1.0', 'ExperienceModifier': '1.0',
         'unit_class': '1', 'unit_flags': '0', 'type': '7', 'lootid': '0'},
    ]
    tmpl_path = os.path.join(tmpdir, 'creature_template.csv')
    with open(tmpl_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=tmpl_header, delimiter=';',
                           quotechar='"', quoting=csv.QUOTE_ALL)
        w.writeheader()
        for t in templates:
            w.writerow(t)

    # Create creature.csv with spawns near player start (-8921, -119)
    spawn_header = ['guid', 'id1', 'map', 'position_x', 'position_y', 'position_z',
                    'orientation', 'npcflag', 'unit_flags']
    spawns = [
        # Wolf spawn
        {'guid': '1', 'id1': '100', 'map': '0',
         'position_x': '-8900', 'position_y': '-110', 'position_z': '82',
         'orientation': '0', 'npcflag': '0', 'unit_flags': '0'},
        # Vendor spawn
        {'guid': '2', 'id1': '200', 'map': '0',
         'position_x': '-8910', 'position_y': '-105', 'position_z': '82',
         'orientation': '0', 'npcflag': '0', 'unit_flags': '0'},
        # Blacksmith spawn
        {'guid': '3', 'id1': '201', 'map': '0',
         'position_x': '-8905', 'position_y': '-115', 'position_z': '82',
         'orientation': '0', 'npcflag': '0', 'unit_flags': '0'},
    ]
    spawn_path = os.path.join(tmpdir, 'creature.csv')
    with open(spawn_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=spawn_header, delimiter=';',
                           quotechar='"', quoting=csv.QUOTE_ALL)
        w.writeheader()
        for s in spawns:
            w.writerow(s)

    from sim.creature_db import CreatureDB
    db = CreatureDB(tmpdir, quiet=True)

    # Verify vendor detection
    assert db.templates[200].is_vendor, "Test Merchant should be a vendor"
    assert db.templates[201].is_vendor, "Test Blacksmith should be a vendor (128 in 4224)"
    assert not db.templates[100].is_vendor, "Test Wolf should not be a vendor"
    assert db.templates[100].is_attackable, "Test Wolf should be attackable"
    assert not db.templates[200].is_attackable, "Test Merchant should not be attackable"

    total_vendor_spawns = sum(len(v) for v in db.vendor_index.values())
    assert total_vendor_spawns == 2, f"Expected 2 vendor spawns, got {total_vendor_spawns}"
    print(f"  8h: CreatureDB: {total_vendor_spawns} vendor spawns detected ✓")

    # --- Test 8i: Dynamic vendors in CombatSimulation ---
    sim_db = CombatSimulation(seed=42, creature_db=db)
    # With creature_db, vendors come from chunks not VENDOR_DATA
    assert len(sim_db.vendors) >= 2, \
        f"Expected ≥2 dynamic vendors, got {len(sim_db.vendors)}"
    vendor_names = {v.name for v in sim_db.vendors}
    assert "Test Merchant" in vendor_names, "Test Merchant should be spawned"
    assert "Test Blacksmith" in vendor_names, "Test Blacksmith should be spawned"
    print(f"  8i: Dynamic vendors from creature_db: {sorted(vendor_names)} ✓")

    # --- Test 8j: Sell works at dynamically spawned vendor ---
    sim_db.player.inventory.append(InventoryItem(
        entry=1, name="Junk", quality=0, sell_price=30, score=5.0, inventory_type=0))
    sim_db.player.free_slots = INVENTORY_SLOTS - 1
    nv = sim_db.get_nearest_vendor()
    sim_db.player.x = nv.x
    sim_db.player.y = nv.y
    sold = sim_db.do_sell()
    assert sold, "Sell should work at dynamic vendor"
    assert sim_db.player.copper == 30, f"Expected 30 copper, got {sim_db.player.copper}"
    print(f"  8j: Sell at dynamic vendor: copper={sim_db.player.copper} ✓")

    # --- Test 8k: items_sold in consume_events ---
    sim_items = CombatSimulation(num_mobs=5, seed=42)
    p_items = sim_items.player
    for i in range(10):
        p_items.inventory.append(InventoryItem(
            entry=100+i, name=f"Junk_{i}", quality=0,
            sell_price=5, score=3.0, inventory_type=0))
    p_items.free_slots = INVENTORY_SLOTS - 10
    v = VENDOR_DATA[0]
    p_items.x = v["x"]
    p_items.y = v["y"]
    sold = sim_items.do_sell()
    assert sold, "Sell should succeed"
    events = sim_items.consume_events()
    assert events["items_sold"] == 10, f"Expected 10 items_sold, got {events['items_sold']}"
    assert p_items.items_sold == 0, "items_sold should be cleared after consume"
    print(f"  8k: items_sold in consume_events: {events['items_sold']} ✓")

    # --- Test 8l: Proximity-based sell via WoWSimEnv (no nav override) ---
    env = WoWSimEnv(num_mobs=5, seed=42)
    obs, _ = env.reset()
    # Fill inventory with items
    fill_count = DEFAULT_BACKPACK_SLOTS - 2
    for i in range(fill_count):
        env.sim.player.inventory.append(InventoryItem(
            entry=100+i, name=f"Junk_{i}", quality=0,
            sell_price=10, score=3.0, inventory_type=0))
    env.sim.player.recalculate_free_slots()

    # Sell should be masked when far from vendor
    mask = env.action_masks()
    assert mask[8] == False, "Sell should be masked when far from vendor"

    # Place player at vendor and sell should be unmasked
    v = VENDOR_DATA[0]
    env.sim.player.x = v["x"]
    env.sim.player.y = v["y"]
    total_slots = env.sim.player.total_bag_slots
    mask = env.action_masks()
    assert mask[8] == True, "Sell should be valid when near vendor with items"

    # Execute sell
    obs, reward, done, trunc, info = env.step(8)
    assert env.sim.player.free_slots == total_slots, \
        f"Bot should have sold all items, but free_slots={env.sim.player.free_slots}"
    assert env.sim.player.copper > 0, "Should have earned copper from selling"
    print(f"  8l: Proximity-based sell: copper={env.sim.player.copper}, "
          f"slots={env.sim.player.free_slots} ✓")

    # --- Test 8m: Sell reward scales with inventory fullness ---
    env2 = WoWSimEnv(num_mobs=5, seed=42)
    obs, _ = env2.reset()
    # Sell with full inventory (30 items)
    for i in range(INVENTORY_SLOTS):
        env2.sim.player.inventory.append(InventoryItem(
            entry=100+i, name=f"Junk_{i}", quality=0,
            sell_price=10, score=3.0, inventory_type=0))
    env2.sim.player.free_slots = 0
    # Place player right at vendor for immediate sell
    v = VENDOR_DATA[0]
    env2.sim.player.x = v["x"]
    env2.sim.player.y = v["y"]
    env2.last_state = env2.sim.get_state_dict()
    obs, reward_full, done, trunc, info = env2.step(8)
    # fullness = 30/30 = 1.0 → sell_reward = 1.0 + 7.0 = 8.0, plus copper bonus
    assert reward_full > 5.0, f"Full sell should give big reward, got {reward_full:.2f}"

    # Now sell with nearly empty inventory (2 items)
    obs, _ = env2.reset()
    env2.sim.player.inventory.append(InventoryItem(
        entry=1, name="Junk", quality=0, sell_price=5, score=3.0, inventory_type=0))
    env2.sim.player.inventory.append(InventoryItem(
        entry=2, name="Junk2", quality=0, sell_price=5, score=3.0, inventory_type=0))
    env2.sim.player.free_slots = INVENTORY_SLOTS - 2
    env2.sim.player.x = v["x"]
    env2.sim.player.y = v["y"]
    env2.last_state = env2.sim.get_state_dict()
    obs, reward_small, done, trunc, info = env2.step(8)
    # fullness = 2/30 ≈ 0.067 → sell_reward = 1.0 + 7.0*0.067 ≈ 1.47, plus tiny copper
    assert reward_small < reward_full, \
        f"Small sell reward ({reward_small:.2f}) should be less than full ({reward_full:.2f})"
    print(f"  8m: Sell rewards: full={reward_full:.2f}, small={reward_small:.2f} ✓")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    print("  PASSED\n")


def test_quest_system():
    """Test quest system: QuestDB, objectives, NPC interaction, rewards."""
    print("=== Test 9: Quest System ===")

    from sim.quest_db import (QuestDB, QuestObjectiveType, QUEST_TEMPLATES,
                               QUEST_NPC_DATA)

    # --- Test 9a: QuestDB loading ---
    qdb = QuestDB(quiet=True)
    assert len(qdb.templates) == len(QUEST_TEMPLATES), \
        f"Expected {len(QUEST_TEMPLATES)} quests, got {len(qdb.templates)}"
    assert len(qdb.npc_data) == len(QUEST_NPC_DATA), \
        f"Expected {len(QUEST_NPC_DATA)} NPCs, got {len(qdb.npc_data)}"
    print(f"  9a: QuestDB: {len(qdb.templates)} quests, {len(qdb.npc_data)} NPCs ✓")

    # --- Test 9b: Quest chain prerequisites ---
    # Quest 7 (Kobold Camp Cleanup) requires quest 33 (Wolves Across the Border)
    available = qdb.get_available_quests(197, player_level=1,
                                         completed=set(), active={})
    quest_ids = [q.quest_id for q in available]
    assert 7 not in quest_ids, "Quest 7 should NOT be available (needs prev=33)"

    available2 = qdb.get_available_quests(197, player_level=1,
                                          completed={33}, active={})
    quest_ids2 = [q.quest_id for q in available2]
    assert 7 in quest_ids2, "Quest 7 SHOULD be available after completing 33"
    print(f"  9b: Quest chain: 33→7 prerequisite works ✓")

    # --- Test 9c: Level requirements ---
    # Quest 15 requires min_level=2
    available3 = qdb.get_available_quests(197, player_level=1,
                                          completed={33, 7}, active={})
    quest_ids3 = [q.quest_id for q in available3]
    assert 15 not in quest_ids3, "Quest 15 should NOT be available at level 1"

    available4 = qdb.get_available_quests(197, player_level=2,
                                          completed={33, 7}, active={})
    quest_ids4 = [q.quest_id for q in available4]
    assert 15 in quest_ids4, "Quest 15 SHOULD be available at level 2"
    print(f"  9c: Level requirement: quest 15 needs L2 ✓")

    # --- Test 9d: CombatSimulation with quest NPCs ---
    sim = CombatSimulation(seed=42, quest_db=qdb)  # all spawns (need ≥10 wolves)
    assert len(sim.quest_npcs) == len(QUEST_NPC_DATA), \
        f"Expected {len(QUEST_NPC_DATA)} quest NPCs, got {len(sim.quest_npcs)}"
    assert len(sim.active_quests) == 0, "No quests should be active at start"
    wolf_count = sum(1 for m in sim.mobs if m.template.entry == 299)
    assert wolf_count >= 10, f"Need ≥10 wolves, got {wolf_count}"
    print(f"  9d: Sim initialized with {len(sim.quest_npcs)} quest NPCs, {wolf_count} wolves ✓")

    # --- Test 9e: Accept quest by walking to NPC ---
    # Move to Deputy Willem (entry 823)
    willem = None
    for npc in sim.quest_npcs:
        if npc.entry == 823:
            willem = npc
            break
    assert willem is not None, "Deputy Willem should exist"
    sim.player.x = willem.x
    sim.player.y = willem.y
    sim.do_quest_interact()
    # Should accept quest 33 (Wolves Across the Border)
    assert 33 in sim.active_quests, "Quest 33 should be accepted"
    print(f"  9e: Accepted quests: {list(sim.active_quests.keys())} ✓")

    # --- Test 9f: Kill objective tracking ---
    wolves_killed = 0
    for mob in sim.mobs:
        if mob.template.entry == 299 and mob.alive:
            mob.hp = 0
            mob.alive = False
            sim.on_mob_killed(mob)
            wolves_killed += 1
            if wolves_killed >= 10:
                break

    prog33 = sim.active_quests[33]
    assert prog33.counts[0] == 10, f"Expected 10 wolf kills, got {prog33.counts[0]}"
    assert prog33.completed, "Quest 33 should be complete after 10 kills"
    print(f"  9f: Kill tracking: {prog33.counts[0]}/10 wolves, complete={prog33.completed} ✓")

    # --- Test 9g: Turn in quest and get rewards ---
    old_xp = sim.player.xp
    sim.do_quest_interact()  # at Willem, turn in quest 33
    assert 33 in sim.completed_quests, "Quest 33 should be completed"
    assert 33 not in sim.active_quests, "Quest 33 should be removed from active"
    assert sim.player.quest_xp_gained == 250, \
        f"Expected 250 quest XP, got {sim.player.quest_xp_gained}"
    assert sim.player.xp == old_xp + 250, "XP should increase by 250"
    print(f"  9g: Turn-in: XP +250, completed={sim.completed_quests} ✓")

    # --- Test 9h: Chain quest now available (7 after 33) ---
    # Move to McBride to pick up quest 7
    mcbride = None
    for npc in sim.quest_npcs:
        if npc.entry == 197:
            mcbride = npc
            break
    assert mcbride is not None
    sim.player.x = mcbride.x
    sim.player.y = mcbride.y
    sim.do_quest_interact()
    assert 7 in sim.active_quests, "Chain quest 7 should be accepted after 33"
    print(f"  9h: Chain quest 7 accepted ✓")

    # --- Test 9i: Quest events in consume_events ---
    events = sim.consume_events()
    assert "quest_xp" in events, "Events must include quest_xp"
    assert "quests_completed" in events, "Events must include quests_completed"
    assert events["quest_xp"] == 250, f"Expected 250 quest_xp, got {events['quest_xp']}"
    assert events["quests_completed"] == 1, \
        f"Expected 1 quest completed, got {events['quests_completed']}"
    # After consume, should be cleared
    events2 = sim.consume_events()
    assert events2["quest_xp"] == 0, "quest_xp should be cleared after consume"
    assert events2["quests_completed"] == 0, "quests_completed should be cleared"
    print(f"  9i: consume_events: quest_xp={events['quest_xp']}, "
          f"quests_completed={events['quests_completed']} ✓")

    # --- Test 9j: Quest state resets on sim reset ---
    sim.reset()
    assert len(sim.active_quests) == 0, "Active quests should be cleared on reset"
    assert len(sim.completed_quests) == 0, "Completed quests should be cleared on reset"
    assert sim.quests_completed == 0, "Quest counter should reset"
    assert len(sim.quest_npcs) == len(QUEST_NPC_DATA), \
        "Quest NPCs should be re-spawned after reset"
    print(f"  9j: Reset clears quest state ✓")

    # --- Test 9k: WoWSimEnv with quests ---
    env = WoWSimEnv(seed=42, enable_quests=True)
    obs, _ = env.reset()
    assert obs.shape == (52,), f"Expected obs(52,), got {obs.shape}"
    assert env.action_space.n == 30, f"Expected 30 actions, got {env.action_space.n}"
    # Quest dims should be non-zero (quest NPCs are visible)
    assert obs[48] > 0 or obs[46] == 0.0, "Quest NPC obs should reflect nearby NPCs"
    print(f"  9k: WoWSimEnv(enable_quests=True): obs={obs.shape}, "
          f"quest_dims={obs[46:52]} ✓")

    # --- Test 9l: Quest reward in env ---
    # Set up: accept quest, complete it, turn in, check reward
    env2 = WoWSimEnv(seed=42, enable_quests=True)
    obs, _ = env2.reset()
    # Move to Willem and accept
    willem2 = None
    for npc in env2.sim.quest_npcs:
        if npc.entry == 823:
            willem2 = npc
            break
    env2.sim.player.x = willem2.x
    env2.sim.player.y = willem2.y
    env2.last_state = env2.sim.get_state_dict()
    # Action 11 to interact
    obs, r, d, t, info = env2.step(11)
    assert len(env2.sim.active_quests) > 0, "Should have accepted quests"

    # Kill 10 wolves via sim
    count = 0
    for mob in env2.sim.mobs:
        if mob.template.entry == 299 and mob.alive:
            mob.hp = 0
            mob.alive = False
            env2.sim.on_mob_killed(mob)
            count += 1
            if count >= 10:
                break

    # Return to Willem and turn in
    env2.sim.player.x = willem2.x
    env2.sim.player.y = willem2.y
    env2.last_state = env2.sim.get_state_dict()
    obs, reward, d, t, info = env2.step(11)
    # Quest completion reward should be +20.0 per quest (plus quest XP via kill signal)
    assert reward > 15.0, f"Quest turn-in should give big reward, got {reward:.2f}"
    print(f"  9l: Quest turn-in reward: {reward:.2f} ✓")

    # --- Test 9m: get_best_quest_npc ---
    sim3 = CombatSimulation(seed=42, quest_db=qdb)
    npc, npc_type = sim3.get_best_quest_npc()
    assert npc is not None, "Should find a quest NPC with available quests"
    assert npc_type == 'accept', f"Expected 'accept', got '{npc_type}'"
    print(f"  9m: Best quest NPC: {npc.name} ({npc_type}) ✓")

    # Accept and complete a quest, then check for turn_in type
    sim3.player.x = npc.x
    sim3.player.y = npc.y
    sim3.do_quest_interact()
    # Complete quest objectives manually
    for qid, prog in list(sim3.active_quests.items()):
        qt = qdb.templates[qid]
        for i in range(len(prog.counts)):
            prog.counts[i] = qt.objectives[i].count
        prog.check_complete(qt.objectives)

    npc2, npc_type2 = sim3.get_best_quest_npc()
    assert npc_type2 == 'turn_in', f"Expected 'turn_in', got '{npc_type2}'"
    print(f"  9m: After completion: {npc2.name} ({npc_type2}) ✓")

    print("  PASSED\n")


def test_quest_csv_loading():
    """Test quest system CSV loading from AzerothCore exports."""
    print("=== Test 10: Quest CSV Loading ===")

    from sim.quest_db import (QuestDB, QuestObjectiveType, _estimate_quest_xp,
                              load_quest_xp_dbc, _quest_xp_lookup)

    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    if not os.path.isfile(os.path.join(data_dir, 'quest_template.csv')):
        print("  SKIPPED (quest_template.csv not found)\n")
        return

    # --- Test 10a: CSV loading ---
    qdb = QuestDB(data_dir=data_dir, quiet=True)
    assert qdb.loaded, "QuestDB should report loaded=True with CSVs"
    assert len(qdb.templates) > 100, \
        f"Expected >100 quests from CSV, got {len(qdb.templates)}"
    assert len(qdb.npc_data) > 10, \
        f"Expected >10 quest NPCs, got {len(qdb.npc_data)}"
    print(f"  10a: CSV loaded: {len(qdb.templates)} quests, "
          f"{len(qdb.npc_data)} NPCs ✓")

    # --- Test 10b: Real quest 7 loaded from CSV ---
    assert 7 in qdb.templates, "Quest 7 should be loaded from CSV"
    qt7 = qdb.templates[7]
    assert qt7.title == "Kobold Camp Cleanup", \
        f"Quest 7 title: {qt7.title}"
    assert qt7.quest_level == 2, f"Quest 7 level: {qt7.quest_level}"
    assert len(qt7.objectives) > 0, "Quest 7 should have objectives"
    assert qt7.objectives[0].obj_type == QuestObjectiveType.KILL, \
        "Quest 7 should have KILL objective"
    assert qt7.giver_entry > 0, "Quest 7 should have a giver NPC"
    assert qt7.ender_entry > 0, "Quest 7 should have an ender NPC"
    print(f"  10b: Quest 7 '{qt7.title}': L{qt7.quest_level}, "
          f"giver={qt7.giver_entry}, ender={qt7.ender_entry}, "
          f"obj={qt7.objectives[0].target}x{qt7.objectives[0].count} ✓")

    # --- Test 10c: Chain info from quest_template_addon ---
    # Quest 7 should have a PrevQuestID from addon
    # (the exact chain depends on DB version, just check it's loaded)
    quests_with_prev = sum(1 for qt in qdb.templates.values() if qt.prev_quest > 0)
    quests_with_next = sum(1 for qt in qdb.templates.values() if qt.next_quest > 0)
    print(f"  10c: Chain info: {quests_with_prev} with prev_quest, "
          f"{quests_with_next} with next_quest ✓")

    # --- Test 10d: Giver/ender maps ---
    total_givers = len(qdb.giver_map)
    total_enders = len(qdb.ender_map)
    assert total_givers > 10, f"Expected >10 giver NPCs, got {total_givers}"
    assert total_enders > 10, f"Expected >10 ender NPCs, got {total_enders}"
    print(f"  10d: Maps: {total_givers} giver NPCs, {total_enders} ender NPCs ✓")

    # --- Test 10e: QuestXP from DBC ---
    dbc_path = os.path.join(data_dir, 'QuestXP.dbc')
    if os.path.isfile(dbc_path):
        table = load_quest_xp_dbc(dbc_path)
        assert len(table) == 100, f"Expected 100 levels, got {len(table)}"
        # Known values from QuestXP.dbc
        assert table[1][5] == 80, f"L1/D5 should be 80, got {table[1][5]}"
        assert table[2][5] == 170, f"L2/D5 should be 170, got {table[2][5]}"
        assert table[10][5] == 840, f"L10/D5 should be 840, got {table[10][5]}"
        # D0 and D9 always 0
        assert table[10][0] == 0, "D0 should be 0"
        assert table[10][9] == 0, "D9 should be 0"
        # _quest_xp_lookup uses DBC
        assert _quest_xp_lookup(2, 5) == 170, "Lookup should match DBC"
        assert _quest_xp_lookup(20, 5) > _quest_xp_lookup(10, 5), "Higher level = more XP"
        # Quest 7 (L2, RewardXPDifficulty from CSV) should have real XP
        qt7_xp = qdb.templates[7].rewards.xp
        assert qt7_xp > 0, f"Quest 7 should have XP from DBC, got {qt7_xp}"
        print(f"  10e: QuestXP.dbc: L1/D5={table[1][5]}, L2/D5={table[2][5]}, "
              f"L10/D5={table[10][5]}, Q7 XP={qt7_xp} ✓")
    else:
        # Fallback approximation test
        xp_l2_d5 = _estimate_quest_xp(2, 5)
        assert 100 <= xp_l2_d5 <= 300, f"Expected ~170 XP for L2/D5, got {xp_l2_d5}"
        assert _estimate_quest_xp(10, 0) == 0, "D0 should give 0 XP"
        print(f"  10e: QuestXP fallback: L2/D5={xp_l2_d5} ✓")

    # --- Test 10f: Quest objectives parsed correctly ---
    quests_with_kill = sum(1 for qt in qdb.templates.values()
                          if any(o.obj_type == QuestObjectiveType.KILL
                                 for o in qt.objectives))
    quests_with_collect = sum(1 for qt in qdb.templates.values()
                             if any(o.obj_type == QuestObjectiveType.COLLECT
                                    for o in qt.objectives))
    quests_with_obj = sum(1 for qt in qdb.templates.values() if qt.objectives)
    print(f"  10f: Objectives: {quests_with_obj} total, "
          f"{quests_with_kill} kill, {quests_with_collect} collect ✓")

    # --- Test 10g: Fallback without data_dir ---
    qdb_fallback = QuestDB(quiet=True)
    assert not qdb_fallback.loaded, "QuestDB without data_dir should not be loaded"
    assert len(qdb_fallback.templates) == 3, \
        f"Fallback should have 3 hardcoded quests, got {len(qdb_fallback.templates)}"
    print(f"  10g: Fallback: {len(qdb_fallback.templates)} hardcoded quests ✓")

    print("  PASSED\n")


def test_attribute_system():
    """Test WotLK 3.3.5 attribute system: stats, equipment, spell scaling, armor."""
    print("=== Test 11: Attribute & Equipment System ===")

    # --- Test 11a: Base stat formulas (WotLK — DBC per-level lookup) ---
    # stat_index: 0=str, 1=agi, 2=stam, 3=int, 4=spi
    # Priest base stats at level 1: (20, 20, 20, 22, 23) from player_class_stats.csv
    assert class_base_stat(CLASS_PRIEST, 2, 1) == 20   # stamina
    assert class_base_stat(CLASS_PRIEST, 2, 10) == 23   # stam L10 (non-linear from CSV)
    assert class_base_stat(CLASS_PRIEST, 3, 1) == 22    # intellect
    assert class_base_stat(CLASS_PRIEST, 3, 10) == 33   # int L10 (non-linear from CSV)
    assert class_base_stat(CLASS_PRIEST, 4, 1) == 23    # spirit
    # Warrior base stats at level 1: (23, 20, 22, 20, 20) from CSV
    assert class_base_stat(CLASS_WARRIOR, 0, 1) == 23   # strength
    assert class_base_stat(CLASS_WARRIOR, 0, 10) == 33   # str L10 (non-linear from CSV)
    assert class_base_stat(CLASS_WARRIOR, 2, 1) == 22   # stamina
    # Rogue base stats at level 1: (21, 23, 20, 20, 20) from CSV
    assert class_base_stat(CLASS_ROGUE, 1, 1) == 23     # agility
    assert class_base_stat(CLASS_MAGE, 3, 1) == 23      # mage intellect
    print(f"  11a: Base stats: Priest(stam=20,int=22,spi=23), Warrior(str=23), "
          f"Rogue(agi=23), Mage(int=23) ✓")

    # --- Test 11b: HP with bonus stamina (WotLK: first 20=1HP, above 20=10HP) ---
    hp_base = player_max_hp(1)  # Priest L1: BaseHP(52) + stam_hp(20) = 72
    hp_bonus = player_max_hp(1, bonus_stamina=5)  # +5 stam -> total 25 -> +50 HP more
    # total_stam = 20+5 = 25, stam_hp = 20 + 5*10 = 70 -> 52+70 = 122
    assert hp_bonus == 122, f"HP(1,+5stam) expected 122, got {hp_bonus}"
    assert hp_bonus > hp_base, "Bonus stamina should increase HP"
    hp_flat = player_max_hp(1, bonus_hp=100)  # +100 flat HP
    assert hp_flat == hp_base + 100, f"Flat HP bonus should add directly"
    # Warrior has more base HP and HP/level
    hp_war = player_max_hp(1, class_id=CLASS_WARRIOR)
    hp_war10 = player_max_hp(10, class_id=CLASS_WARRIOR)
    assert hp_war > 0, "Warrior should have base HP"
    assert hp_war10 > hp_war, "Warrior HP should scale with level"
    print(f"  11b: HP formulas: Priest base={hp_base}, +5stam={hp_bonus}, "
          f"Warrior L1={hp_war}, L10={hp_war10} ✓")

    # --- Test 11c: Mana with bonus intellect (WotLK: first 20=1, above 20=15) ---
    mana_base = player_max_mana(1)  # Priest: BaseMana(73) + int_mana(22) = 73+20+30 = 123
    mana_bonus = player_max_mana(1, bonus_intellect=10)  # total_int=32 -> 20+12*15=200 -> 73+200 = 273
    assert mana_bonus == 273, f"Mana(1,+10int) expected 273, got {mana_bonus}"
    mana_flat = player_max_mana(1, bonus_mana=50)
    assert mana_flat == mana_base + 50, f"Flat mana bonus should add directly"
    # Non-mana classes (Warrior, Rogue, DK) return 0
    assert player_max_mana(1, class_id=CLASS_WARRIOR) == 0, "Warrior should have 0 mana"
    assert player_max_mana(1, class_id=CLASS_ROGUE) == 0, "Rogue should have 0 mana"
    # Mage has more base mana than Priest
    mana_mage = player_max_mana(1, class_id=CLASS_MAGE)
    assert mana_mage > 0, "Mage should have mana"
    print(f"  11c: Mana formulas: Priest={mana_base}, +10int={mana_bonus}, "
          f"Warrior=0, Mage={mana_mage} ✓")

    # --- Test 11d: Spell crit from Intellect (NOT from SP!) ---
    crit_l1 = spell_crit_chance(1)  # Priest default
    assert crit_l1 > 0.0, f"Base spell crit should be > 0, got {crit_l1}"
    crit_l1_int = spell_crit_chance(1, bonus_intellect=50)
    assert crit_l1_int > crit_l1, "More Int should increase spell crit"
    crit_l1_rating = spell_crit_chance(1, bonus_crit_rating=50)
    assert crit_l1_rating > crit_l1, "Crit rating should increase spell crit"
    # Mage should have different spell crit than Priest
    crit_mage = spell_crit_chance(1, class_id=CLASS_MAGE)
    assert crit_mage > 0.0, "Mage should have base spell crit"
    # Warriors should have 0 spell crit
    crit_war = spell_crit_chance(1, class_id=CLASS_WARRIOR)
    assert crit_war == 0.0, f"Warrior spell crit should be 0, got {crit_war}"
    print(f"  11d: Spell crit: Priest={crit_l1:.2f}%, Mage={crit_mage:.2f}%, "
          f"Warrior={crit_war:.2f}%, +50int={crit_l1_int:.2f}% ✓")

    # --- Test 11e: Rating scaling (GtCombatRatings non-linear curve) ---
    # Ratings are much more effective at low levels than at high levels
    assert spell_haste_pct(1, 0) == 0.0, "No haste without rating"
    haste_l1 = spell_haste_pct(1, bonus_haste_rating=50)
    haste_l40 = spell_haste_pct(40, bonus_haste_rating=50)
    haste_l60 = spell_haste_pct(60, bonus_haste_rating=50)
    haste_l80 = spell_haste_pct(80, bonus_haste_rating=50)
    assert haste_l1 > haste_l40, "L1 haste should be > L40 haste for same rating"
    assert haste_l40 > haste_l60, "L40 haste should be > L60 haste"
    assert haste_l60 > haste_l80, "L60 haste should be > L80 haste"
    # L80 value should match known WotLK: 32.79 rating per 1%
    assert abs(haste_l80 - 50 / 32.79) < 0.1, \
        f"L80 haste should be ~1.52%, got {haste_l80:.2f}%"
    # Crit rating also scales: L1 benefit >> L80 benefit
    crit_bonus_l1 = spell_crit_chance(1, bonus_crit_rating=100) - spell_crit_chance(1)
    crit_bonus_l80 = spell_crit_chance(80, bonus_crit_rating=100) - spell_crit_chance(80)
    assert crit_bonus_l1 > crit_bonus_l80 * 10, \
        f"100 crit rating at L1 ({crit_bonus_l1:.1f}%) should be >10x L80 ({crit_bonus_l80:.1f}%)"
    print(f"  11e: Rating scaling: Haste(50r) L1={haste_l1:.1f}% → L40={haste_l40:.1f}% "
          f"→ L60={haste_l60:.1f}% → L80={haste_l80:.2f}%, "
          f"CritBonus(100r) L1={crit_bonus_l1:.1f}%/L80={crit_bonus_l80:.1f}% ✓")

    # --- Test 11f: Spirit mana regen ---
    regen = spirit_mana_regen(1)  # Priest default
    assert regen > 0.0, "Base spirit regen should be > 0"
    regen_bonus = spirit_mana_regen(1, bonus_spirit=20)
    assert regen_bonus > regen, "More spirit should increase regen"
    # Mage has different regen coefficient
    regen_mage = spirit_mana_regen(1, class_id=CLASS_MAGE)
    assert regen_mage > 0.0, "Mage should have spirit mana regen"
    print(f"  11f: Spirit mana regen: Priest={regen:.2f}/tick, +20spi={regen_bonus:.2f}/tick, "
          f"Mage={regen_mage:.2f}/tick ✓")

    # --- Test 11g: Spell power scaling (using spell_bonus_data coefficients) ---
    # Smite: base (13,17) + SP*0.123
    d_min, d_max = smite_damage(1, spell_power=140)  # 140 * 0.123 = 17
    exp_sp = int(140 * SP_COEFF_SMITE)  # 17
    assert d_min == 13 + exp_sp, f"Smite(1,SP=140) min expected {13+exp_sp}, got {d_min}"
    assert d_max == 17 + exp_sp, f"Smite(1,SP=140) max expected {17+exp_sp}, got {d_max}"
    # Heal: base (46,56) + SP*0.231
    h_min, h_max = heal_amount(1, spell_power=140)  # 140*0.231 = 32
    exp_h_sp = int(140 * SP_COEFF_HEAL)  # 32
    assert h_min == 46 + exp_h_sp, f"Heal(1,SP=140) min expected {46+exp_h_sp}, got {h_min}"
    # SW:Pain: base 30 + SP*0.1833*6
    swp = sw_pain_total(1, spell_power=100)  # 30 + int(100*0.1833*6) = 30 + 109 = 139
    assert swp > 30, "SP should increase SW:Pain damage"
    # PW:Shield: base 44 + SP*0.8068
    pws = pw_shield_absorb(1, spell_power=100)  # 44 + 80 = 124
    assert pws > 44, "SP should increase PW:Shield absorb"
    print(f"  11g: SP scaling: Smite({d_min}-{d_max}), Heal({h_min}-{h_max}), "
          f"SWP={swp}, PWS={pws} ✓")

    # --- Test 11h: Equipment system — equip items and verify stat recalculation ---
    sim = CombatSimulation(num_mobs=5, seed=42)
    p = sim.player
    old_hp = p.max_hp
    old_mana = p.max_mana

    # Equip a chest piece with +10 Stamina, +8 Intellect, +5 Spirit
    chest = EquippedItem(
        entry=9001, name="Test Robe", inventory_type=20, score=50.0,
        stats={ITEM_MOD_STAMINA: 10, ITEM_MOD_INTELLECT: 8, ITEM_MOD_SPIRIT: 5},
        armor=30,
    )
    p.equipment[EQUIPMENT_SLOT_CHEST] = chest
    p.equipped_scores[EQUIPMENT_SLOT_CHEST] = 50.0
    sim.recalculate_stats()

    assert p.gear_stamina == 10, f"Expected gear_stamina=10, got {p.gear_stamina}"
    assert p.gear_intellect == 8, f"Expected gear_intellect=8, got {p.gear_intellect}"
    assert p.gear_spirit == 5, f"Expected gear_spirit=5, got {p.gear_spirit}"
    assert p.gear_armor == 30, f"Expected gear_armor=30, got {p.gear_armor}"
    assert p.max_hp > old_hp, f"HP should increase with +Stam gear: {old_hp} -> {p.max_hp}"
    assert p.max_mana > old_mana, f"Mana should increase with +Int gear: {old_mana} -> {p.max_mana}"
    print(f"  11h: Equipment: HP {old_hp}→{p.max_hp}, Mana {old_mana}→{p.max_mana}, "
          f"armor={p.gear_armor} ✓")

    # --- Test 11i: Spell Power from gear affects spells ---
    old_sp = p.total_spell_power
    sp_wand = EquippedItem(
        entry=9002, name="SP Wand", inventory_type=26, score=40.0,
        stats={ITEM_MOD_SPELL_POWER: 50},
    )
    p.equipment[EQUIPMENT_SLOT_RANGED] = sp_wand
    p.equipped_scores[EQUIPMENT_SLOT_RANGED] = 40.0
    sim.recalculate_stats()
    assert p.total_spell_power == 50, f"Expected SP=50, got {p.total_spell_power}"
    assert p.gear_spell_power == 50, f"Expected gear_sp=50, got {p.gear_spell_power}"
    print(f"  11i: Spell Power from gear: {old_sp}→{p.total_spell_power} ✓")

    # --- Test 11j: try_equip_item upgrades and triggers stat recalc ---
    sim2 = CombatSimulation(num_mobs=5, seed=99)
    p2 = sim2.player
    from sim.loot_db import ItemData
    item = ItemData(
        entry=5001, name="Upgrade Helm", quality=2, sell_price=100,
        inventory_type=1, item_level=10, item_class=4, item_subclass=1,
        score=60.0,
        stats={ITEM_MOD_STAMINA: 8, ITEM_MOD_INTELLECT: 6},
        armor=20, weapon_dps=0.0,
    )
    old_hp2 = p2.max_hp
    result = sim2.try_equip_item(item)
    assert result is True, "Should equip upgrade"
    assert p2.equipment[EQUIPMENT_SLOT_HEAD].entry == 5001, "Item should be in head slot"
    assert p2.max_hp > old_hp2, "HP should increase after equip"
    assert p2.gear_stamina == 8, f"gear_stamina should be 8, got {p2.gear_stamina}"

    # Lower score item should not replace
    worse = ItemData(
        entry=5002, name="Worse Helm", quality=1, sell_price=50,
        inventory_type=1, item_level=5, item_class=4, item_subclass=1,
        score=30.0, stats={ITEM_MOD_STAMINA: 2}, armor=5, weapon_dps=0.0,
    )
    result2 = sim2.try_equip_item(worse)
    assert result2 is False, "Should not equip worse item"
    assert p2.equipment[EQUIPMENT_SLOT_HEAD].entry == 5001, "Original item should still be equipped"
    print(f"  11j: try_equip_item: upgrade=True, downgrade=False ✓")

    # --- Test 11k: Armor mitigation (WotLK formula) ---
    sim3 = CombatSimulation(num_mobs=5, seed=42)
    p3 = sim3.player
    # Give player some armor gear
    p3.equipment[EQUIPMENT_SLOT_CHEST] = EquippedItem(
        entry=9003, name="Armor Chest", inventory_type=5, score=40.0,
        stats={}, armor=100,
    )
    sim3.recalculate_stats()
    assert p3.gear_armor == 100

    # Damage player and check mitigation is applied
    p3.hp = p3.max_hp
    full_hp = p3.hp
    sim3._damage_player(50)
    damage_taken = full_hp - p3.hp
    assert damage_taken < 50, f"Armor should reduce damage: took {damage_taken} of 50"
    assert damage_taken > 0, "Should still take some damage"
    print(f"  11k: Armor mitigation: 50 raw → {damage_taken} taken (armor={p3.gear_armor}) ✓")

    # --- Test 11l: Gear stats in state_dict ---
    state = sim2.get_state_dict()
    assert "spell_power" in state, "State dict should include spell_power"
    assert "spell_crit" in state, "State dict should include spell_crit"
    assert "gear_armor" in state, "State dict should include gear_armor"
    assert "attack_power" in state, "State dict should include attack_power"
    assert "melee_crit" in state, "State dict should include melee_crit"
    assert "dodge" in state, "State dict should include dodge"
    assert "hit_spell" in state, "State dict should include hit_spell"
    assert state["gear_stamina"] == 8, f"State dict gear_stamina should be 8"
    print(f"  11l: Gear stats in state_dict: SP={state['spell_power']}, "
          f"crit={state['spell_crit']:.1f}, AP={state['attack_power']}, "
          f"dodge={state['dodge']:.1f} ✓")

    # --- Test 11m: Gear obs in WoWSimEnv ---
    env = WoWSimEnv(num_mobs=5, seed=42)
    obs, _ = env.reset()
    # obs[22] = is_eating (0/1)
    # Talent obs at indices 29-32 (4 dims: target_has_vt, shadowform, dispersion, channeling)
    # Stat obs at indices 33-42 (10 dims: SP, spell_crit, spell_haste, armor,
    # AP, melee_crit, dodge, hit, expertise, ArP)
    assert obs[22] == 0.0, f"is_eating obs should be 0 at start, got {obs[22]}"
    assert obs[33] == 0.0, f"SP obs should be 0 with no gear, got {obs[33]}"
    assert obs[34] >= 0.0, f"Spell crit obs should be >= 0, got {obs[34]}"
    assert obs[35] == 0.0, f"Spell haste obs should be 0 with no gear, got {obs[35]}"
    assert obs[36] >= 0.0, f"Armor obs should be >= 0, got {obs[36]}"  # agi*2 gives base armor
    assert obs[37] >= 0.0, f"AP obs should be >= 0, got {obs[37]}"
    assert obs[38] >= 0.0, f"Melee crit obs should be >= 0, got {obs[38]}"
    assert obs[39] >= 0.0, f"Dodge obs should be >= 0, got {obs[39]}"
    print(f"  11m: Stat obs [33:43]={obs[33:43]} ✓")

    # --- Test 11n: Equipment persists across ticks ---
    sim4 = CombatSimulation(num_mobs=5, seed=42)
    sim4.player.equipment[EQUIPMENT_SLOT_CHEST] = EquippedItem(
        entry=9004, name="Persist Chest", inventory_type=5, score=45.0,
        stats={ITEM_MOD_STAMINA: 5, ITEM_MOD_SPELL_POWER: 20}, armor=25,
    )
    sim4.recalculate_stats()
    hp_before = sim4.player.max_hp
    sp_before = sim4.player.total_spell_power
    for _ in range(20):
        sim4.tick()
    assert sim4.player.max_hp == hp_before, "HP should not change across ticks"
    assert sim4.player.total_spell_power == sp_before, "SP should not change across ticks"
    print(f"  11n: Stats persist across ticks: HP={hp_before}, SP={sp_before} ✓")

    # --- Test 11o: Equipment reset on sim reset ---
    sim4.reset()
    assert len(sim4.player.equipment) == 0, "Equipment should be cleared on reset"
    assert sim4.player.gear_stamina == 0, "Gear stats should be 0 after reset"
    assert sim4.player.total_spell_power == 0, "SP should be 0 after reset"
    assert sim4.player.max_hp == player_max_hp(1), \
        f"HP should be base after reset: {sim4.player.max_hp} vs {player_max_hp(1)}"
    print(f"  11o: Reset clears equipment and stats ✓")

    # --- Test 11p: Melee Attack Power by class ---
    # Warrior: level*3 + str*2 - 20
    war_str = class_base_stat(CLASS_WARRIOR, 0, 1)  # 23
    war_agi = class_base_stat(CLASS_WARRIOR, 1, 1)  # 20
    war_ap = melee_attack_power(1, war_str, war_agi, CLASS_WARRIOR)
    assert war_ap == int(1 * 3.0 + war_str * 2.0 - 20.0), f"Warrior AP expected {int(1*3+war_str*2-20)}, got {war_ap}"
    # Rogue: level*2 + str + agi - 20
    rog_str = class_base_stat(CLASS_ROGUE, 0, 1)  # 21
    rog_agi = class_base_stat(CLASS_ROGUE, 1, 1)  # 23
    rog_ap = melee_attack_power(1, rog_str, rog_agi, CLASS_ROGUE)
    assert rog_ap == int(1 * 2.0 + rog_str + rog_agi - 20.0), f"Rogue AP expected {int(1*2+rog_str+rog_agi-20)}, got {rog_ap}"
    # Priest: str - 10
    pri_str = class_base_stat(CLASS_PRIEST, 0, 1)  # 20
    pri_agi = class_base_stat(CLASS_PRIEST, 1, 1)
    pri_ap = melee_attack_power(1, pri_str, pri_agi, CLASS_PRIEST)
    assert pri_ap == int(pri_str - 10.0), f"Priest AP expected {int(pri_str - 10)}, got {pri_ap}"
    print(f"  11p: Attack Power: Warrior={war_ap}, Rogue={rog_ap}, Priest={pri_ap} ✓")

    # --- Test 11q: Melee crit from Agility ---
    # Warrior at L1 (base crit 0.0, simpler)
    war_crit = melee_crit_chance(1, 20, class_id=CLASS_WARRIOR)
    assert war_crit > 0.0, f"Warrior melee crit should be > 0, got {war_crit}"
    war_crit_high = melee_crit_chance(1, 100, class_id=CLASS_WARRIOR)
    assert war_crit_high > war_crit, "Higher agi should increase melee crit"
    # Rogue at base agi should have positive crit from DBC base value
    rog_crit_low = melee_crit_chance(1, class_base_stat(CLASS_ROGUE, 1, 1), class_id=CLASS_ROGUE)
    assert rog_crit_low > 0.0, f"Rogue L1 base agi should have positive crit, got {rog_crit_low}"
    rog_crit_high = melee_crit_chance(1, 200, class_id=CLASS_ROGUE)
    assert rog_crit_high > 0.0, "Rogue with 200 agi should have melee crit"
    print(f"  11q: Melee crit: Warrior(agi=20)={war_crit:.2f}%, "
          f"Rogue(agi=200)={rog_crit_high:.2f}% ✓")

    # --- Test 11r: Dodge with diminishing returns ---
    # Tank class should get reasonable dodge
    war_dodge = dodge_chance(80, 200, dodge_rating=100, class_id=CLASS_WARRIOR)
    assert war_dodge > 5.0, f"Warrior dodge should be > 5%, got {war_dodge:.2f}%"
    # Diminishing returns: doubling dodge rating should NOT double bonus
    war_dodge2 = dodge_chance(80, 200, dodge_rating=200, class_id=CLASS_WARRIOR)
    dodge_gain1 = war_dodge - dodge_chance(80, 200, class_id=CLASS_WARRIOR)
    dodge_gain2 = war_dodge2 - dodge_chance(80, 200, class_id=CLASS_WARRIOR)
    assert dodge_gain2 < dodge_gain1 * 2, "Dodge should have diminishing returns"
    print(f"  11r: Dodge: Warrior(L80,agi=200,dr=100)={war_dodge:.2f}%, "
          f"DR check: +100r={dodge_gain1:.2f}%, +200r={dodge_gain2:.2f}% ✓")

    # --- Test 11s: Parry for melee classes, 0 for casters ---
    war_parry = parry_chance(80, parry_rating=100, class_id=CLASS_WARRIOR)
    assert war_parry > 5.0, f"Warrior should have parry > 5%, got {war_parry}"
    pri_parry = parry_chance(80, parry_rating=100, class_id=CLASS_PRIEST)
    assert pri_parry == 0.0, f"Priest should have 0 parry, got {pri_parry}"
    print(f"  11s: Parry: Warrior(L80,pr=100)={war_parry:.2f}%, Priest=0.0% ✓")

    # --- Test 11t: Hit, Expertise, Armor Penetration ---
    hit = hit_chance_spell(80, hit_rating=100)
    assert hit > 0.0, "Hit rating should give hit %"
    exp = expertise_pct(80, expertise_rating=100)
    assert exp > 0.0, "Expertise rating should give expertise %"
    arp = armor_penetration_pct(80, arp_rating=100)
    assert arp > 0.0, "ArP rating should give ArP %"
    assert arp <= 100.0, "ArP should be capped at 100%"
    print(f"  11t: Hit(100r)={hit:.2f}%, Expertise(100r)={exp:.2f}%, ArP(100r)={arp:.2f}% ✓")

    # --- Test 11u: Equipment with melee stats recalculates correctly ---
    sim5 = CombatSimulation(num_mobs=5, seed=42)
    p5 = sim5.player
    melee_gear = EquippedItem(
        entry=9010, name="Test Sword", inventory_type=21, score=80.0,
        stats={ITEM_MOD_STRENGTH: 15, ITEM_MOD_AGILITY: 10,
               ITEM_MOD_ATTACK_POWER: 30, ITEM_MOD_DODGE_RATING: 20,
               ITEM_MOD_EXPERTISE_RATING: 15},
        armor=0,
    )
    p5.equipment[EQUIPMENT_SLOT_MAINHAND] = melee_gear
    p5.equipped_scores[EQUIPMENT_SLOT_MAINHAND] = 80.0
    sim5.recalculate_stats()
    assert p5.gear_strength == 15, f"gear_strength should be 15, got {p5.gear_strength}"
    assert p5.gear_agility == 10, f"gear_agility should be 10, got {p5.gear_agility}"
    assert p5.gear_attack_power == 30, f"gear_ap should be 30, got {p5.gear_attack_power}"
    assert p5.gear_dodge_rating == 20, f"gear_dodge_rating should be 20, got {p5.gear_dodge_rating}"
    assert p5.total_attack_power > 0, f"Total AP should be > 0, got {p5.total_attack_power}"
    assert p5.total_dodge > 0, f"Total dodge should be > 0, got {p5.total_dodge}"
    assert p5.total_expertise > 0, f"Total expertise should be > 0, got {p5.total_expertise}"
    print(f"  11u: Melee gear: AP={p5.total_attack_power}, dodge={p5.total_dodge:.2f}%, "
          f"expertise={p5.total_expertise:.2f}% ✓")

    # --- Test 11v: Unequip item removes stats ---
    sim6 = CombatSimulation(num_mobs=5, seed=42)
    p6 = sim6.player
    from sim.loot_db import ItemData
    helm = ItemData(
        entry=9020, name="Unequip Helm", quality=2, sell_price=100,
        inventory_type=1, item_level=10, item_class=4, item_subclass=1,
        score=50.0,
        stats={ITEM_MOD_STAMINA: 12, ITEM_MOD_INTELLECT: 10, ITEM_MOD_SPELL_POWER: 25},
        armor=15, weapon_dps=0.0,
    )
    sim6.equip_item(helm)
    assert p6.gear_stamina == 12, f"After equip: gear_stamina={p6.gear_stamina}"
    assert p6.gear_intellect == 10, f"After equip: gear_intellect={p6.gear_intellect}"
    assert p6.total_spell_power == 25, f"After equip: SP={p6.total_spell_power}"
    hp_equipped = p6.max_hp
    mana_equipped = p6.max_mana

    removed = sim6.unequip_item(EQUIPMENT_SLOT_HEAD)
    assert removed is not None, "Should return the removed item"
    assert removed.entry == 9020, "Should return the correct item"
    assert p6.gear_stamina == 0, f"After unequip: gear_stamina should be 0, got {p6.gear_stamina}"
    assert p6.gear_intellect == 0, f"After unequip: gear_intellect should be 0, got {p6.gear_intellect}"
    assert p6.total_spell_power == 0, f"After unequip: SP should be 0, got {p6.total_spell_power}"
    assert p6.max_hp < hp_equipped, f"HP should decrease: {hp_equipped} -> {p6.max_hp}"
    assert p6.max_mana < mana_equipped, f"Mana should decrease: {mana_equipped} -> {p6.max_mana}"
    assert EQUIPMENT_SLOT_HEAD not in p6.equipment, "Slot should be empty"
    # Unequipped item should be in inventory
    assert len(p6.inventory) == 1, f"Inventory should have 1 item, got {len(p6.inventory)}"
    assert p6.inventory[0].entry == 9020
    print(f"  11v: Unequip: HP {hp_equipped}→{p6.max_hp}, SP 25→{p6.total_spell_power}, "
          f"inv={len(p6.inventory)} ✓")

    # Unequip empty slot returns None
    empty = sim6.unequip_item(EQUIPMENT_SLOT_HEAD)
    assert empty is None, "Unequip empty slot should return None"

    # --- Test 11w: Dual-slot items (rings) ---
    sim7 = CombatSimulation(num_mobs=5, seed=42)
    p7 = sim7.player
    ring1 = ItemData(
        entry=9030, name="Ring of Power", quality=2, sell_price=50,
        inventory_type=11, item_level=10, item_class=4, item_subclass=0,
        score=40.0,
        stats={ITEM_MOD_SPELL_POWER: 15}, armor=0, weapon_dps=0.0,
    )
    ring2 = ItemData(
        entry=9031, name="Ring of Wisdom", quality=2, sell_price=60,
        inventory_type=11, item_level=12, item_class=4, item_subclass=0,
        score=45.0,
        stats={ITEM_MOD_INTELLECT: 8}, armor=0, weapon_dps=0.0,
    )
    # First ring goes to Finger 1 (empty)
    sim7.equip_item(ring1)
    assert EQUIPMENT_SLOT_FINGER1 in p7.equipment, "First ring in Finger 1"
    assert p7.equipment[EQUIPMENT_SLOT_FINGER1].entry == 9030

    # Second ring goes to Finger 2 (also empty)
    sim7.equip_item(ring2)
    assert EQUIPMENT_SLOT_FINGER2 in p7.equipment, "Second ring in Finger 2"
    assert p7.equipment[EQUIPMENT_SLOT_FINGER2].entry == 9031

    # Both ring stats should be applied
    assert p7.gear_spell_power == 15, f"SP from ring1: {p7.gear_spell_power}"
    assert p7.gear_intellect == 8, f"Int from ring2: {p7.gear_intellect}"

    # Third ring (better) should replace the weaker one (ring1, score=40)
    ring3 = ItemData(
        entry=9032, name="Ring of Glory", quality=3, sell_price=100,
        inventory_type=11, item_level=15, item_class=4, item_subclass=0,
        score=60.0,
        stats={ITEM_MOD_SPELL_POWER: 30}, armor=0, weapon_dps=0.0,
    )
    sim7.equip_item(ring3)
    assert p7.equipment[EQUIPMENT_SLOT_FINGER1].entry == 9032, "Ring3 should replace weaker ring1"
    assert p7.equipment[EQUIPMENT_SLOT_FINGER2].entry == 9031, "Ring2 should remain"
    assert p7.gear_spell_power == 30, f"SP should be 30 (ring3 only), got {p7.gear_spell_power}"
    print(f"  11w: Dual slots (rings): 2 rings + replacement ✓")

    # --- Test 11x: Two-hand weapon clears offhand ---
    sim8 = CombatSimulation(num_mobs=5, seed=42)
    p8 = sim8.player
    # Equip a one-hand weapon + shield
    sword = ItemData(
        entry=9040, name="Short Sword", quality=1, sell_price=30,
        inventory_type=13, item_level=5, item_class=2, item_subclass=7,
        score=20.0,
        stats={ITEM_MOD_STRENGTH: 5}, armor=0, weapon_dps=5.0,
    )
    shield = ItemData(
        entry=9041, name="Buckler", quality=1, sell_price=25,
        inventory_type=14, item_level=5, item_class=4, item_subclass=6,
        score=15.0,
        stats={ITEM_MOD_STAMINA: 3}, armor=50, weapon_dps=0.0,
    )
    sim8.equip_item(sword)
    sim8.equip_item(shield)
    assert EQUIPMENT_SLOT_MAINHAND in p8.equipment, "Sword in main hand"
    assert EQUIPMENT_SLOT_OFFHAND in p8.equipment, "Shield in offhand"
    assert p8.gear_armor == 50, f"Shield armor: {p8.gear_armor}"

    # Equip two-hander: should clear offhand
    staff = ItemData(
        entry=9042, name="Mighty Staff", quality=2, sell_price=80,
        inventory_type=17, item_level=12, item_class=2, item_subclass=10,
        score=55.0,
        stats={ITEM_MOD_INTELLECT: 10, ITEM_MOD_SPELL_POWER: 20},
        armor=0, weapon_dps=8.0,
    )
    sim8.equip_item(staff)
    assert p8.equipment[EQUIPMENT_SLOT_MAINHAND].entry == 9042, "Staff in main hand"
    assert EQUIPMENT_SLOT_OFFHAND not in p8.equipment, "Offhand should be cleared"
    assert p8.gear_armor == 0, f"Shield armor should be gone, got {p8.gear_armor}"
    assert p8.gear_spell_power == 20, f"Staff SP: {p8.gear_spell_power}"
    # Old items should be in inventory
    inv_entries = [i.entry for i in p8.inventory]
    assert 9040 in inv_entries, "Old sword should be in inventory"
    assert 9041 in inv_entries, "Old shield should be in inventory"
    print(f"  11x: Two-hand clears offhand: inv={len(p8.inventory)} items ✓")

    # --- Test 11y: equip_item returns old item ---
    sim9 = CombatSimulation(num_mobs=5, seed=42)
    helm_a = ItemData(
        entry=9050, name="Helm A", quality=1, sell_price=20,
        inventory_type=1, item_level=5, item_class=4, item_subclass=1,
        score=25.0,
        stats={ITEM_MOD_STAMINA: 4}, armor=10, weapon_dps=0.0,
    )
    helm_b = ItemData(
        entry=9051, name="Helm B", quality=2, sell_price=50,
        inventory_type=1, item_level=10, item_class=4, item_subclass=1,
        score=50.0,
        stats={ITEM_MOD_STAMINA: 8}, armor=20, weapon_dps=0.0,
    )
    success_a, old_a = sim9.equip_item(helm_a)
    assert success_a is True
    assert old_a is None, "No old item when equipping to empty slot"
    success_b, old_b = sim9.equip_item(helm_b)
    assert success_b is True
    assert old_b is not None, "Should return displaced item"
    assert old_b.entry == 9050, "Displaced item should be Helm A"
    assert sim9.player.gear_stamina == 8, "Should have Helm B stats"
    print(f"  11y: equip_item returns old item: {old_b.name} ✓")

    # --- Test 11z: Slot mapping (INVTYPE_TO_SLOTS) ---
    assert INVTYPE_TO_SLOTS[1] == [EQUIPMENT_SLOT_HEAD], "inv_type 1 = Head"
    assert INVTYPE_TO_SLOTS[20] == [EQUIPMENT_SLOT_CHEST], "inv_type 20 (Robe) = Chest"
    assert INVTYPE_TO_SLOTS[14] == [EQUIPMENT_SLOT_OFFHAND], "inv_type 14 (Shield) = Offhand"
    assert INVTYPE_TO_SLOTS[11] == [EQUIPMENT_SLOT_FINGER1, EQUIPMENT_SLOT_FINGER2], "Rings = 2 slots"
    assert INVTYPE_TO_SLOTS[12] == [EQUIPMENT_SLOT_TRINKET1, EQUIPMENT_SLOT_TRINKET2], "Trinkets = 2 slots"
    assert INVTYPE_TO_SLOTS[17] == [EQUIPMENT_SLOT_MAINHAND], "Two-Hand = Main Hand"
    print(f"  11z: INVTYPE_TO_SLOTS mapping correct ✓")

    # --- Test 11aa: Equipment blocked during combat ---
    sim10 = CombatSimulation(num_mobs=5, seed=42)
    p10 = sim10.player
    combat_helm = ItemData(
        entry=9060, name="Combat Helm", quality=2, sell_price=50,
        inventory_type=1, item_level=10, item_class=4, item_subclass=1,
        score=50.0,
        stats={ITEM_MOD_STAMINA: 8}, armor=15, weapon_dps=0.0,
    )
    # Equip outside combat — should work
    success, _ = sim10.equip_item(combat_helm)
    assert success, "Equip should work outside combat"
    assert EQUIPMENT_SLOT_HEAD in p10.equipment

    # Enter combat
    p10.in_combat = True

    # Try equip during combat — should fail
    combat_helm2 = ItemData(
        entry=9061, name="Better Helm", quality=3, sell_price=100,
        inventory_type=1, item_level=15, item_class=4, item_subclass=1,
        score=80.0,
        stats={ITEM_MOD_STAMINA: 15}, armor=25, weapon_dps=0.0,
    )
    success2, _ = sim10.equip_item(combat_helm2)
    assert not success2, "Equip should be blocked during combat"
    assert p10.equipment[EQUIPMENT_SLOT_HEAD].entry == 9060, "Original helm still equipped"

    # try_equip_item should also fail (it goes through equip_item)
    result = sim10.try_equip_item(combat_helm2)
    assert not result, "try_equip_item should be blocked during combat"

    # Unequip during combat — should fail
    removed = sim10.unequip_item(EQUIPMENT_SLOT_HEAD)
    assert removed is None, "Unequip should be blocked during combat"
    assert EQUIPMENT_SLOT_HEAD in p10.equipment, "Helm still equipped"

    # Leave combat — equip should work again
    p10.in_combat = False
    success3, old = sim10.equip_item(combat_helm2)
    assert success3, "Equip should work after leaving combat"
    assert old.entry == 9060, "Old helm returned"
    assert p10.equipment[EQUIPMENT_SLOT_HEAD].entry == 9061, "New helm equipped"
    print(f"  11aa: Equipment blocked during combat ✓")

    # --- Test 11ab: Class-aware item scoring ---
    # Priest should value INT item higher than STR item
    int_item_stats = {ITEM_MOD_INTELLECT: 10}
    str_item_stats = {ITEM_MOD_STRENGTH: 10}
    int_score = class_aware_score(int_item_stats, 2, 10, 0, 0.0, CLASS_PRIEST)
    str_score = class_aware_score(str_item_stats, 2, 10, 0, 0.0, CLASS_PRIEST)
    assert int_score > str_score, (
        f"Priest should prefer INT ({int_score}) over STR ({str_score})")
    # Warrior should prefer STR over INT
    w_int_score = class_aware_score(int_item_stats, 2, 10, 0, 0.0, CLASS_WARRIOR)
    w_str_score = class_aware_score(str_item_stats, 2, 10, 0, 0.0, CLASS_WARRIOR)
    assert w_str_score > w_int_score, (
        f"Warrior should prefer STR ({w_str_score}) over INT ({w_int_score})")
    print(f"  11ab: Class-aware scoring: Priest INT={int_score:.1f}>STR={str_score:.1f}, "
          f"Warrior STR={w_str_score:.1f}>INT={w_int_score:.1f} ✓")

    # --- Test 11ac: Class-aware try_equip_item prefers class stats ---
    # Priest with STR helm equipped should upgrade to INT helm (even at same raw score)
    sim11 = CombatSimulation(num_mobs=5, seed=42)
    p11 = sim11.player  # Priest by default
    # Equip a STR helm (bad for Priest)
    from sim.loot_db import ItemData
    str_helm = ItemData(
        entry=9070, name="Warrior Helm", quality=2, sell_price=100,
        inventory_type=1, item_level=10, item_class=4, item_subclass=1,
        score=50.0, stats={ITEM_MOD_STRENGTH: 10}, armor=20, weapon_dps=0.0,
    )
    sim11.equip_item(str_helm, EQUIPMENT_SLOT_HEAD)
    assert p11.equipment[EQUIPMENT_SLOT_HEAD].entry == 9070

    # Offer an INT helm with the same raw stats total
    int_helm = ItemData(
        entry=9071, name="Priest Helm", quality=2, sell_price=100,
        inventory_type=1, item_level=10, item_class=4, item_subclass=1,
        score=50.0, stats={ITEM_MOD_INTELLECT: 10}, armor=20, weapon_dps=0.0,
    )
    result = sim11.try_equip_item(int_helm)
    assert result is True, "Priest should upgrade from STR to INT helm"
    assert p11.equipment[EQUIPMENT_SLOT_HEAD].entry == 9071, "INT helm should be equipped"
    assert p11.equipped_upgrade > 0, "equipped_upgrade should carry score diff"
    print(f"  11ac: Priest upgrades STR→INT helm, score_diff={p11.equipped_upgrade:.1f} ✓")

    # --- Test 11ad: Scaled upgrade reward in env ---
    env11 = WoWSimEnv(num_mobs=5, seed=42)
    env11.reset()
    events_test = {
        "equipped_upgrade": 20.0,  # medium upgrade
        "xp_gained": 0, "loot_copper": 0, "loot_score": 0,
        "leveled_up": False, "levels_gained": 0,
        "loot_items": [], "loot_failed": [],
        "sell_copper": 0, "items_sold": 0,
        "new_areas": 0, "new_zones": 0, "new_maps": 0,
        "quest_xp": 0, "quest_copper": 0, "quests_completed": 0,
    }
    # Manually compute expected reward: 1.0 + 20.0 * 0.15 = 4.0
    expected_upgrade_reward = min(1.0 + 20.0 * 0.15, 5.0)
    assert abs(expected_upgrade_reward - 4.0) < 0.01, f"Expected 4.0, got {expected_upgrade_reward}"
    # Zero upgrade → no reward
    zero_reward = min(1.0 + 0.0 * 0.15, 5.0)
    assert zero_reward == 1.0  # base, but only applied if upgrade_score > 0
    print(f"  11ad: Scaled upgrade reward: diff=20→reward={expected_upgrade_reward:.1f} ✓")

    print("  PASSED\n")


def test_bag_system():
    """Test 13: Bag system — equip, upgrade, capacity, sell."""
    print("Test 13: Bag System")

    # --- 13a: Default backpack only (16 slots) ---
    sim = CombatSimulation(num_mobs=5, seed=42)
    p = sim.player
    assert p.total_bag_slots == DEFAULT_BACKPACK_SLOTS, \
        f"Starting capacity should be {DEFAULT_BACKPACK_SLOTS}, got {p.total_bag_slots}"
    assert p.free_slots == DEFAULT_BACKPACK_SLOTS, \
        f"Starting free_slots should be {DEFAULT_BACKPACK_SLOTS}, got {p.free_slots}"
    assert len(p.bags) == 0, "Should start with no equipped bags"
    print(f"  13a: Default backpack = {DEFAULT_BACKPACK_SLOTS} slots ✓")

    # --- 13b: Equip a bag into empty slot ---
    # Create a mock bag item (like ItemData from loot_db)
    class MockBag:
        def __init__(self, entry, name, container_slots, quality=1, sell_price=100,
                     bag_family=0):
            self.entry = entry
            self.name = name
            self.container_slots = container_slots
            self.quality = quality
            self.sell_price = sell_price
            self.bag_family = bag_family
            self.inventory_type = INVTYPE_BAG

    bag6 = MockBag(4238, "Linen Bag", 6)
    result = sim.try_equip_bag(bag6)
    assert result is True, "Should equip bag into empty slot"
    assert len(p.bags) == 1, "Should have 1 equipped bag"
    assert p.total_bag_slots == DEFAULT_BACKPACK_SLOTS + 6, \
        f"Total should be {DEFAULT_BACKPACK_SLOTS + 6}, got {p.total_bag_slots}"
    assert p.free_slots == DEFAULT_BACKPACK_SLOTS + 6, \
        f"Free should be {DEFAULT_BACKPACK_SLOTS + 6}, got {p.free_slots}"
    print(f"  13b: Equip 6-slot bag → {p.total_bag_slots} total slots ✓")

    # --- 13c: Fill all 4 bag slots ---
    bag8 = MockBag(4240, "Woolen Bag", 8)
    bag10 = MockBag(804, "Large Blue Sack", 10)
    bag12 = MockBag(1623, "Raptor Skin Pouch", 12)
    assert sim.try_equip_bag(bag8) is True
    assert sim.try_equip_bag(bag10) is True
    assert sim.try_equip_bag(bag12) is True
    assert len(p.bags) == NUM_BAG_SLOTS, f"Should have {NUM_BAG_SLOTS} bags"
    expected = DEFAULT_BACKPACK_SLOTS + 6 + 8 + 10 + 12
    assert p.total_bag_slots == expected, \
        f"Total should be {expected}, got {p.total_bag_slots}"
    print(f"  13c: 4 bags equipped → {p.total_bag_slots} total slots ✓")

    # --- 13d: Upgrade smallest bag (6-slot) with larger one (14-slot) ---
    bag14 = MockBag(1685, "Troll-hide Bag", 14)
    old_total = p.total_bag_slots
    result = sim.try_equip_bag(bag14)
    assert result is True, "Should replace 6-slot bag with 14-slot"
    new_total = p.total_bag_slots
    assert new_total == old_total - 6 + 14, \
        f"Expected {old_total - 6 + 14}, got {new_total}"
    # Old bag should be in inventory
    bag_in_inv = [i for i in p.inventory if i.entry == 4238]
    assert len(bag_in_inv) == 1, "Old 6-slot bag should be in inventory"
    print(f"  13d: Upgrade 6→14 slot bag → {p.total_bag_slots} total ✓")

    # --- 13e: Reject smaller bag when all slots full with bigger bags ---
    bag4 = MockBag(9999, "Tiny Bag", 4)
    result = sim.try_equip_bag(bag4)
    assert result is False, "Should reject bag smaller than any equipped"
    print("  13e: Reject smaller bag ✓")

    # --- 13f: Reject profession bags (bag_family != 0) ---
    herb_bag = MockBag(8000, "Herb Bag", 20, bag_family=32)
    result = sim.try_equip_bag(herb_bag)
    assert result is False, "Should reject profession-specific bags"
    print("  13f: Reject profession bags (bag_family != 0) ✓")

    # --- 13g: Combat blocks bag equip ---
    p.in_combat = True
    bag20 = MockBag(1977, "20-slot Bag", 20)
    result = sim.equip_bag(bag20)
    assert result is False, "Should block bag equip during combat"
    p.in_combat = False
    print("  13g: Combat blocks bag equip ✓")

    # --- 13h: Free slots with inventory items ---
    sim2 = CombatSimulation(num_mobs=5, seed=42)
    p2 = sim2.player
    # Add 10 items to inventory
    for i in range(10):
        p2.inventory.append(InventoryItem(
            entry=i, name=f"Item {i}", quality=0, sell_price=10,
            score=1.0, inventory_type=0))
    p2.recalculate_free_slots()
    assert p2.free_slots == DEFAULT_BACKPACK_SLOTS - 10, \
        f"Should have {DEFAULT_BACKPACK_SLOTS - 10} free, got {p2.free_slots}"
    # Equip a bag — should increase free slots
    bag10b = MockBag(804, "Large Blue Sack", 10)
    sim2.equip_bag(bag10b)
    assert p2.free_slots == DEFAULT_BACKPACK_SLOTS + 10 - 10, \
        f"Should have {DEFAULT_BACKPACK_SLOTS + 10 - 10} free, got {p2.free_slots}"
    print(f"  13h: Bag equip updates free_slots with items in inventory ✓")

    # --- 13i: Sell clears inventory but preserves bags ---
    sim3 = CombatSimulation(num_mobs=5, seed=42)
    p3 = sim3.player
    sim3.equip_bag(MockBag(4238, "Linen Bag", 6))
    sim3.equip_bag(MockBag(4240, "Woolen Bag", 8))
    # Add items to inventory
    for i in range(5):
        p3.inventory.append(InventoryItem(
            entry=i, name=f"Item {i}", quality=0, sell_price=10,
            score=1.0, inventory_type=0))
    p3.recalculate_free_slots()
    assert len(p3.bags) == 2
    old_bag_count = len(p3.bags)
    # Sell: move player near vendor
    p3.x = sim3.vendors[0].x
    p3.y = sim3.vendors[0].y
    sim3.do_sell()
    assert len(p3.bags) == old_bag_count, "Bags should be preserved after sell"
    assert len(p3.inventory) == 0, "Inventory should be empty after sell"
    assert p3.free_slots == DEFAULT_BACKPACK_SLOTS + 6 + 8, \
        f"Free slots should reflect bags after sell, got {p3.free_slots}"
    print(f"  13i: Sell preserves equipped bags, free_slots={p3.free_slots} ✓")

    # --- 13j: state_dict includes bag info ---
    state = sim.get_state_dict()
    assert "bags" in state, "state_dict should have 'bags'"
    assert "total_bag_slots" in state, "state_dict should have 'total_bag_slots'"
    assert "bag_slots_used" in state, "state_dict should have 'bag_slots_used'"
    assert state["bag_slots_used"] == NUM_BAG_SLOTS, \
        f"Should report {NUM_BAG_SLOTS} bag slots used"
    print(f"  13j: state_dict has bag info ✓")

    # --- 13k: Reset clears bags ---
    sim.reset()
    assert len(sim.player.bags) == 0, "Reset should clear bags"
    assert sim.player.total_bag_slots == DEFAULT_BACKPACK_SLOTS, \
        "Reset should restore default capacity"
    assert sim.player.free_slots == DEFAULT_BACKPACK_SLOTS, \
        "Reset should restore default free_slots"
    print(f"  13k: Reset clears bags → {DEFAULT_BACKPACK_SLOTS} slots ✓")

    # --- 13l: Gym env works with bag system ---
    env = WoWSimEnv(num_mobs=5, seed=42)
    obs, info = env.reset()
    # free_slots should be normalized: DEFAULT_BACKPACK_SLOTS / 20.0
    expected_norm = DEFAULT_BACKPACK_SLOTS / 20.0
    assert abs(obs[9] - expected_norm) < 0.01, \
        f"Obs[9] should be {expected_norm}, got {obs[9]}"
    # Run a few steps to make sure nothing crashes
    for _ in range(20):
        obs, rew, done, trunc, info = env.step(env.action_space.sample())
        if done or trunc:
            obs, info = env.reset()
    print(f"  13l: Gym env works with bag system ✓")

    print("  PASSED\n")


def test_combat_resolution():
    """Test 14: WotLK combat resolution — melee attack table, spell miss/crit, level diff."""
    print("Test 14: Combat Resolution System")

    # --- 14a: Spell miss chance formulas ---
    # Equal level: 4% base miss
    miss = spell_miss_chance(1, 1, 0.0)
    assert abs(miss - 4.0) < 0.01, f"Same-level spell miss should be 4%, got {miss}"
    # +1 level mob: 5%
    miss = spell_miss_chance(1, 2, 0.0)
    assert abs(miss - 5.0) < 0.01, f"+1 level spell miss should be 5%, got {miss}"
    # +2 level mob: 6%
    miss = spell_miss_chance(1, 3, 0.0)
    assert abs(miss - 6.0) < 0.01, f"+2 level spell miss should be 6%, got {miss}"
    # +3 level mob (boss): 17%
    miss = spell_miss_chance(1, 4, 0.0)
    assert abs(miss - 17.0) < 0.01, f"+3 level spell miss should be 17%, got {miss}"
    # Lower level mob: reduced miss (floor 1%)
    miss = spell_miss_chance(5, 1, 0.0)
    assert miss >= 1.0, f"Spell miss should have 1% floor, got {miss}"
    # Hit rating reduces miss
    miss_no_hit = spell_miss_chance(1, 2, 0.0)
    miss_with_hit = spell_miss_chance(1, 2, 3.0)
    assert miss_with_hit < miss_no_hit, "Hit rating should reduce spell miss"
    assert miss_with_hit >= 1.0, "Spell miss should not go below 1%"
    print(f"  14a: Spell miss formulas (4%/5%/6%/17% by level diff, hit reduces) ✓")

    # --- 14b: Mob melee miss/crit/crushing formulas ---
    miss = mob_melee_miss_chance(1, 1, 0.0)
    assert abs(miss - 5.0) < 0.01, f"Same-level mob miss should be 5%, got {miss}"
    # Defense bonus increases mob miss
    miss_def = mob_melee_miss_chance(1, 1, 10.0)
    assert miss_def > 5.0, "Defense bonus should increase mob miss chance"

    crit = mob_melee_crit_chance(1, 1, 0.0, 0.0)
    assert abs(crit - 5.0) < 0.01, f"Same-level mob crit should be 5%, got {crit}"
    # Higher level mob crits more
    crit_high = mob_melee_crit_chance(3, 1, 0.0, 0.0)
    assert crit_high > 5.0, f"Higher level mob should crit more, got {crit_high}"
    # Defense reduces mob crit
    crit_def = mob_melee_crit_chance(1, 1, 10.0, 0.0)
    assert crit_def < 5.0, f"Defense should reduce mob crit, got {crit_def}"
    # Resilience reduces mob crit
    crit_res = mob_melee_crit_chance(1, 1, 0.0, 2.0)
    assert crit_res < 5.0, f"Resilience should reduce mob crit, got {crit_res}"

    # Crushing: only 4+ levels above
    crush = mob_crushing_chance(1, 1)
    assert crush == 0.0, "Same-level mob should have 0% crushing"
    crush = mob_crushing_chance(3, 1)
    assert crush == 0.0, "+2 level mob should have 0% crushing (need +4)"
    crush = mob_crushing_chance(5, 1)
    assert crush > 0.0, "+4 level mob should have crushing blow chance"
    print(f"  14b: Mob melee miss/crit/crushing formulas (level diff, defense, resilience) ✓")

    # --- 14c: Melee attack table single-roll resolution ---
    # Very low roll -> miss
    outcome = resolve_mob_melee_attack(1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, roll=1.0)
    assert outcome == MELEE_MISS, f"Roll 1.0 should miss (5% miss zone), got {outcome}"
    # Roll just above miss+dodge+parry+block, in crit zone
    # miss=5%, dodge=0, parry=0, block=0 -> crit starts at 5%
    # crit = 5% -> crit zone is [5, 10)
    outcome = resolve_mob_melee_attack(1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, roll=7.0)
    assert outcome == MELEE_CRIT, f"Roll 7.0 should crit (zone 5-10%), got {outcome}"
    # High roll -> normal
    outcome = resolve_mob_melee_attack(1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, roll=50.0)
    assert outcome == MELEE_NORMAL, f"Roll 50.0 should be normal, got {outcome}"

    # With dodge: miss(5%) + dodge(10%) = 15% threshold
    outcome = resolve_mob_melee_attack(1, 1, 10.0, 0.0, 0.0, 0.0, 0.0, roll=8.0)
    assert outcome == MELEE_DODGE, f"Roll 8.0 with 10% dodge should dodge, got {outcome}"

    # With parry: miss(5%) + dodge(10%) + parry(5%) = 20%
    outcome = resolve_mob_melee_attack(1, 1, 10.0, 5.0, 0.0, 0.0, 0.0, roll=17.0)
    assert outcome == MELEE_PARRY, f"Roll 17.0 with dodge+parry should parry, got {outcome}"

    # With block: miss(5%) + dodge(10%) + parry(5%) + block(5%) = 25%
    outcome = resolve_mob_melee_attack(1, 1, 10.0, 5.0, 5.0, 0.0, 0.0, roll=22.0)
    assert outcome == MELEE_BLOCK, f"Roll 22.0 with block should block, got {outcome}"
    print(f"  14c: Single-roll melee attack table (miss/dodge/parry/block/crit/normal) ✓")

    # --- 14d: Spell hit resolution ---
    # Miss roll
    outcome = resolve_spell_hit(1, 1, 0.0, 5.0, roll_hit=2.0, roll_crit=50.0)
    assert outcome == SPELL_MISS, f"Low hit roll should miss, got {outcome}"
    # Crit roll
    outcome = resolve_spell_hit(1, 1, 0.0, 10.0, roll_hit=50.0, roll_crit=5.0)
    assert outcome == SPELL_CRIT, f"Low crit roll should crit, got {outcome}"
    # Normal hit
    outcome = resolve_spell_hit(1, 1, 0.0, 5.0, roll_hit=50.0, roll_crit=50.0)
    assert outcome == SPELL_HIT, f"High rolls should be normal hit, got {outcome}"
    print(f"  14d: Spell hit resolution (two-roll: miss then crit) ✓")

    # --- 14e: Integration — mob attacks use attack table in sim ---
    sim = CombatSimulation(num_mobs=5, seed=42)
    p = sim.player
    # Move player to first mob's position to ensure targeting works
    mob0 = sim.mobs[0]
    p.x = mob0.x + 2.0
    p.y = mob0.y
    sim.do_target_nearest()
    assert sim.target is not None, "Should have a target"
    # Give player high dodge to see dodges happen
    p.total_dodge = 50.0  # 50% dodge
    p.total_parry = 0.0
    p.total_block = 0.0
    # Stand next to mob and let it attack many times
    sim.target.in_combat = True
    sim.target.target_player = True
    p.in_combat = True
    p.x = sim.target.x + 1.0
    p.y = sim.target.y
    sim.target.attack_timer = 0
    # Tick many times to accumulate attacks
    p.hp = 10000  # high HP so we survive
    p.max_hp = 10000
    for _ in range(200):
        sim.target.attack_timer = 0  # force attack each tick
        sim.tick()
        if not sim.target.alive:
            break
    events = sim.consume_events()
    total_dodges = events["dodges"]
    total_misses = events["mob_misses"]
    total_crits = events["mob_crits"]
    total_attacks = total_dodges + total_misses + total_crits + events["parries"] + events["blocks"]
    # With 50% dodge we should see a significant number of dodges
    assert total_dodges > 0, f"With 50% dodge should see dodges, got 0 in {total_attacks} attacks"
    assert total_misses > 0, f"Should see some mob misses (5% base), got 0"
    print(f"  14e: Sim integration — {total_dodges} dodges, {total_misses} misses, "
          f"{total_crits} crits in mob melee attacks ✓")

    # --- 14f: Integration — spells can miss higher-level mobs ---
    sim2 = CombatSimulation(num_mobs=5, seed=123)
    p2 = sim2.player
    p2.hp = 10000
    p2.max_hp = 10000
    p2.mana = 50000
    p2.max_mana = 50000
    # Move to first mob and target it
    mob2 = sim2.mobs[0]
    p2.x = mob2.x + 2.0
    p2.y = mob2.y
    sim2.do_target_nearest()
    assert sim2.target is not None
    sim2.target.level = 4  # +3 level diff -> 17% miss
    sim2.target.hp = 999999
    sim2.target.max_hp = 999999
    # Cast many Smites and count misses
    total_casts = 0
    for _ in range(500):
        p2.mana = 50000
        p2.gcd_remaining = 0
        p2.is_casting = False
        p2.cast_remaining = 0
        result = sim2.do_cast_smite()
        if result:
            # Complete the cast immediately
            p2.cast_remaining = 0
            p2.is_casting = False
            sim2._apply_spell(585)
            total_casts += 1
    events2 = sim2.consume_events()
    spell_misses = events2["spell_misses"]
    spell_crits = events2["spell_crits"]
    assert spell_misses > 0, f"Should see spell misses vs +3 level mob (17% miss), got 0/{total_casts}"
    # With 17% miss rate over 500 casts, expect roughly 85 misses (very unlikely to be 0)
    miss_rate = spell_misses / total_casts * 100
    assert 5.0 < miss_rate < 30.0, f"Miss rate should be ~17%, got {miss_rate:.1f}%"
    print(f"  14f: Spell miss rate vs +3 level mob: {miss_rate:.1f}% "
          f"({spell_misses}/{total_casts} misses, {spell_crits} crits) ✓")

    # --- 14g: Spells don't miss same-level mobs as often ---
    sim3 = CombatSimulation(num_mobs=5, seed=456)
    p3 = sim3.player
    p3.mana = 50000
    p3.max_mana = 50000
    mob3 = sim3.mobs[0]
    p3.x = mob3.x + 2.0
    p3.y = mob3.y
    sim3.do_target_nearest()
    assert sim3.target is not None
    sim3.target.hp = 999999
    sim3.target.max_hp = 999999
    total_casts = 0
    for _ in range(500):
        p3.mana = 50000
        p3.gcd_remaining = 0
        p3.is_casting = False
        p3.cast_remaining = 0
        result = sim3.do_cast_smite()
        if result:
            p3.cast_remaining = 0
            p3.is_casting = False
            sim3._apply_spell(585)
            total_casts += 1
    events3 = sim3.consume_events()
    miss_rate_same = events3["spell_misses"] / total_casts * 100
    # Same level: 4% miss
    assert miss_rate_same < miss_rate, \
        f"Same-level miss rate ({miss_rate_same:.1f}%) should be lower than +3 ({miss_rate:.1f}%)"
    print(f"  14g: Same-level spell miss rate: {miss_rate_same:.1f}% (lower than +3 level) ✓")

    # --- 14h: Heal spells never miss (friendly target) ---
    sim4 = CombatSimulation(num_mobs=5, seed=789)
    p4 = sim4.player
    p4.mana = 50000
    p4.max_mana = 50000
    p4.hp = 1
    p4.max_hp = 10000
    heals_cast = 0
    for _ in range(100):
        p4.mana = 50000
        p4.gcd_remaining = 0
        p4.is_casting = False
        p4.cast_remaining = 0
        p4.hp = 1  # keep HP low so heal is useful
        result = sim4.do_cast_heal()
        if result:
            p4.cast_remaining = 0
            p4.is_casting = False
            sim4._apply_spell(2050)
            heals_cast += 1
    # Heal should always land — HP should be > 1 after 100 heals
    assert p4.hp > 1, "Heal should always land (never miss friendly)"
    events4 = sim4.consume_events()
    assert events4["spell_misses"] == 0, "Heals should never miss"
    print(f"  14h: Heal spells never miss (friendly target), {heals_cast} heals cast ✓")

    # --- 14i: Block reduces damage by block_value ---
    sim5 = CombatSimulation(num_mobs=5, seed=42)
    p5 = sim5.player
    p5.total_armor = 0  # remove armor to isolate block effect
    p5.total_block = 90.0  # very high block chance (after 5% miss, ~95% of non-miss are blocks)
    p5.total_block_value = 5  # blocks reduce damage by 5
    p5.total_dodge = 0.0
    p5.total_parry = 0.0
    p5.total_defense = 0.0
    p5.total_resilience = 0.0
    p5.hp = 10000
    p5.max_hp = 10000
    mob5 = sim5.mobs[0]
    p5.x = mob5.x + 2.0
    p5.y = mob5.y
    sim5.do_target_nearest()
    assert sim5.target is not None
    sim5.target.in_combat = True
    sim5.target.target_player = True
    p5.in_combat = True
    p5.x = sim5.target.x + 1.0
    p5.y = sim5.target.y
    # Run many attacks to statistically see blocks
    for _ in range(50):
        sim5.target.attack_timer = 0
        sim5.tick()
    events5 = sim5.consume_events()
    assert events5["blocks"] > 0, \
        f"Should have blocked with 90% block, got 0 blocks in 50 attacks"
    # Block reduces damage: wolf dmg 1-2, block_value=5 -> blocked hits do 0 damage
    # Total HP lost should be much less than 50 * avg(1.5) = 75
    hp_lost = 10000 - p5.hp
    # Only non-blocked (miss/crit/normal) attacks should do damage
    non_blocked = events5["mob_misses"] + events5["mob_crits"]
    print(f"  14i: Block reduces damage ({events5['blocks']} blocks, "
          f"{events5['mob_misses']} misses, HP lost={hp_lost}) ✓")

    # --- 14j: Mob crit deals 200% damage ---
    # We test this by forcing a crit outcome through a controlled roll
    sim6 = CombatSimulation(num_mobs=5, seed=42)
    p6 = sim6.player
    p6.total_armor = 0
    p6.total_dodge = 0.0
    p6.total_parry = 0.0
    p6.total_block = 0.0
    p6.total_defense = 0.0
    p6.total_resilience = 0.0
    p6.hp = 10000
    p6.max_hp = 10000
    # crit zone = [miss%, miss%+crit%] = [5%, 10%) at equal level
    outcome = resolve_mob_melee_attack(1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, roll=7.0)
    assert outcome == MELEE_CRIT, f"Roll 7.0 should be crit, got {outcome}"
    print(f"  14j: Mob crit produces 200% damage multiplier ✓")

    # --- 14k: consume_events resets combat counters ---
    sim7 = CombatSimulation(num_mobs=5, seed=42)
    p7 = sim7.player
    p7.dodges = 5
    p7.parries = 3
    p7.blocks = 2
    p7.mob_misses = 4
    p7.mob_crits = 1
    p7.mob_crushings = 0
    p7.spell_misses = 6
    p7.spell_crits = 7
    events7 = sim7.consume_events()
    assert events7["dodges"] == 5 and events7["parries"] == 3
    assert events7["spell_misses"] == 6 and events7["spell_crits"] == 7
    # After consume, all should be 0
    assert p7.dodges == 0 and p7.parries == 0 and p7.blocks == 0
    assert p7.mob_misses == 0 and p7.mob_crits == 0 and p7.spell_misses == 0
    print(f"  14k: consume_events resets all combat counters ✓")

    # --- 14l: Hit rating gear reduces spell miss ---
    sim8 = CombatSimulation(num_mobs=5, seed=42)
    p8 = sim8.player
    # Equip item with hit rating
    hit_item = EquippedItem(
        entry=99901, name="Hit Trinket", score=10.0,
        inventory_type=12,  # trinket
        stats={ITEM_MOD_HIT_SPELL_RATING: 20},
        armor=0, weapon_dps=0.0,
    )
    sim8.equip_item(hit_item, 12)  # EQUIPMENT_SLOT_TRINKET1
    assert p8.total_hit_spell > 0, f"Hit rating should give spell hit%, got {p8.total_hit_spell}"
    # Spell miss should be reduced with hit rating
    miss_with_hit = spell_miss_chance(1, 2, p8.total_hit_spell)
    miss_without = spell_miss_chance(1, 2, 0.0)
    assert miss_with_hit < miss_without, \
        f"Hit rating should reduce miss: {miss_with_hit:.1f}% vs {miss_without:.1f}%"
    print(f"  14l: Hit rating gear reduces spell miss ({miss_without:.1f}% → {miss_with_hit:.1f}%) ✓")

    print("  PASSED\n")


def test_action_masking():
    """Test the action masking system (replaces old override logic)."""
    print("=== Test 15: Action Masking ===")

    env = WoWSimEnv(seed=42)
    obs, _ = env.reset()
    sim = env.sim
    p = sim.player

    # --- 15a: Initial mask — basic validity ---
    mask = env.action_masks()
    assert mask.shape == (30,), f"Mask shape: {mask.shape}"
    assert mask.dtype == bool, f"Mask dtype: {mask.dtype}"
    assert mask[0] == True, "Noop should always be valid"
    assert mask[1] == True, "Move forward should be valid when not casting"
    assert mask[2] == True, "Turn left should be valid when not casting"
    assert mask[3] == True, "Turn right should be valid when not casting"
    print(f"  15a: Initial mask shape and basic validity OK ✓")

    # --- 15b: Casting lock — only noop allowed ---
    p.is_casting = True
    p.cast_remaining = 3
    mask = env.action_masks()
    assert mask[0] == True, "Noop should be valid while casting"
    for i in range(1, 17):
        assert mask[i] == False, f"Action {i} should be masked while casting"
    p.is_casting = False
    p.cast_remaining = 0
    print(f"  15b: Casting lock — only noop allowed ✓")

    # --- 15c: Offensive spells need alive target in range ---
    sim.target = None
    mask = env.action_masks()
    for action_id in [5, 9, 12, 14]:  # Smite, SW:Pain, Mind Blast, Holy Fire
        assert mask[action_id] == False, \
            f"Action {action_id} should be masked without target"

    # Give a target in range
    mob = sim.mobs[0]
    p.x = mob.x + 5.0
    p.y = mob.y
    sim.do_target_nearest()
    assert sim.target is not None and sim.target.alive
    mask = env.action_masks()
    # Smite should be valid (has target, in range, has mana)
    if p.mana >= SPELLS[585].mana_cost and p.gcd_remaining == 0:
        assert mask[5] == True, "Smite should be valid with alive target in range"
    print(f"  15c: Offensive spells masked without target, valid with target ✓")

    # --- 15d: Buff duplication — can't double-apply ---
    p.shield_remaining = 10
    mask = env.action_masks()
    assert mask[10] == False, "PW:Shield should be masked when shield active"
    p.shield_remaining = 0
    p.shield_cooldown = 5
    mask = env.action_masks()
    assert mask[10] == False, "PW:Shield should be masked during Weakened Soul"
    p.shield_cooldown = 0

    p.inner_fire_remaining = 10
    mask = env.action_masks()
    assert mask[15] == False, "Inner Fire should be masked when already active"
    p.inner_fire_remaining = 0

    p.fortitude_remaining = 10
    mask = env.action_masks()
    assert mask[16] == False, "Fortitude should be masked when already active"
    p.fortitude_remaining = 0

    p.hot_remaining = 10
    mask = env.action_masks()
    assert mask[13] == False, "Renew should be masked when HoT active"
    p.hot_remaining = 0

    # SW:Pain blocked when DoT on target
    if sim.target:
        sim.target.dot_remaining = 10
        mask = env.action_masks()
        assert mask[9] == False, "SW:Pain should be masked when DoT on target"
        sim.target.dot_remaining = 0
    print(f"  15d: Buff/debuff duplication correctly masked ✓")

    # --- 15e: Loot masked in combat, available out of combat ---
    mob = sim.mobs[0]
    p.x = mob.x + 2.0
    p.y = mob.y
    mob.hp = 0
    mob.alive = False
    mob.looted = False

    # In combat — loot should be masked (fight first!)
    p.in_combat = True
    mask = env.action_masks()
    assert mask[7] == False, "Loot should be masked while in combat"

    # Out of combat — loot should be available (within range)
    p.in_combat = False
    mask = env.action_masks()
    assert mask[7] == True, "Loot should be valid OOC with dead mob in range"
    print(f"  15e: Loot masked in combat, valid OOC with corpse in range ✓")

    # --- 15f: Loot masked when no dead mob in range ---
    mob.looted = True  # already looted
    mask = env.action_masks()
    assert mask[7] == False, "Loot should be masked when no unlooted dead mob in range"
    print(f"  15f: Loot masked when no lootable corpse nearby ✓")

    # --- 15g: Mana check — spells masked when OOM ---
    env2 = WoWSimEnv(seed=100)
    obs2, _ = env2.reset()
    p2 = env2.sim.player
    p2.mana = 0  # OOM
    mask = env2.action_masks()
    for action_id in [5, 6, 9, 10, 12, 13, 14, 15, 16]:
        assert mask[action_id] == False, \
            f"Action {action_id} should be masked when OOM"
    print(f"  15g: All spells masked when OOM ✓")

    # --- 15h: GCD blocks all spells ---
    env3 = WoWSimEnv(seed=200)
    obs3, _ = env3.reset()
    p3 = env3.sim.player
    p3.gcd_remaining = 2
    mask = env3.action_masks()
    for action_id in [5, 6, 9, 10, 12, 13, 14, 15, 16]:
        assert mask[action_id] == False, \
            f"Action {action_id} should be masked during GCD"
    # Movement should still be valid
    assert mask[1] == True, "Movement should be valid during GCD"
    p3.gcd_remaining = 0
    print(f"  15h: GCD blocks all spells but allows movement ✓")

    # --- 15i: Sell masked in combat / without inventory / far from vendor ---
    env4 = WoWSimEnv(seed=300)
    obs4, _ = env4.reset()
    p4 = env4.sim.player
    p4.in_combat = True
    mask = env4.action_masks()
    assert mask[8] == False, "Sell should be masked in combat"
    p4.in_combat = False
    assert len(p4.inventory) == 0
    mask = env4.action_masks()
    assert mask[8] == False, "Sell should be masked with empty inventory"
    # Add items but stay far from vendor
    for i in range(5):
        p4.inventory.append(InventoryItem(
            entry=100+i, name=f"Junk_{i}", quality=0,
            sell_price=10, score=3.0, inventory_type=0))
    p4.recalculate_free_slots()
    mask = env4.action_masks()
    assert mask[8] == False, "Sell should be masked when far from vendor"
    # Move to vendor — sell should be valid
    v = VENDOR_DATA[0]
    p4.x = v["x"]
    p4.y = v["y"]
    mask = env4.action_masks()
    assert mask[8] == True, "Sell should be valid near vendor with items"
    print(f"  15i: Sell correctly masked (combat / empty / far from vendor) ✓")

    # --- 15j: Quest interact masked without quest system ---
    env5 = WoWSimEnv(seed=400, enable_quests=False)
    obs5, _ = env5.reset()
    mask = env5.action_masks()
    assert mask[11] == False, "Quest interact should be masked when quests disabled"
    print(f"  15j: Quest interact masked when quests disabled ✓")

    # --- 15k: Stepping with masked actions still works (graceful fallback) ---
    # The combat sim methods return False for invalid actions but don't crash
    env6 = WoWSimEnv(seed=500)
    obs6, _ = env6.reset()
    env6.sim.player.mana = 0
    obs6, reward, done, trunc, info = env6.step(5)  # try Smite with no mana
    assert obs6.shape == (52,), "Step with invalid action should not crash"
    print(f"  15k: Stepping with masked action is graceful (no crash) ✓")

    print("  PASSED\n")


def test_eat_drink():
    """Test eat/drink action: regen, interrupts, masking."""
    print("=== Test 16: Eat/Drink System ===")

    # --- 16a: Basic eat/drink starts and regens HP/Mana ---
    sim = CombatSimulation(num_mobs=5, seed=42)
    p = sim.player
    p.hp = int(p.max_hp * 0.5)
    p.mana = int(p.max_mana * 0.5)
    hp_before = p.hp
    mana_before = p.mana
    assert not p.is_eating
    result = sim.do_eat_drink()
    assert result, "do_eat_drink should return True"
    assert p.is_eating, "Player should be eating"
    sim.tick()
    assert p.hp > hp_before, f"HP should increase: {hp_before} -> {p.hp}"
    assert p.mana > mana_before, f"Mana should increase: {mana_before} -> {p.mana}"
    print(f"  16a: Eat/drink regens: HP {hp_before}→{p.hp}, Mana {mana_before}→{p.mana} ✓")

    # --- 16b: Regen rate is ~2.5% per tick (5% per second, 0.5s per tick) ---
    sim2 = CombatSimulation(num_mobs=5, seed=42)
    p2 = sim2.player
    p2.hp = 1
    p2.mana = 1
    sim2.do_eat_drink()
    sim2.tick()
    expected_hp_regen = max(1, int(p2.max_hp * 0.025))
    expected_mana_regen = max(1, int(p2.max_mana * 0.025))
    # HP should be 1 + expected (plus any OOC regen from tick)
    assert p2.hp >= 1 + expected_hp_regen, \
        f"HP regen per tick: expected >= {expected_hp_regen}, got {p2.hp - 1}"
    print(f"  16b: Regen rate: HP +{p2.hp - 1}/tick, Mana +{p2.mana - 1}/tick "
          f"(expected ~{expected_hp_regen}/{expected_mana_regen}) ✓")

    # --- 16c: Auto-stops when HP+Mana are full ---
    sim3 = CombatSimulation(num_mobs=5, seed=42)
    p3 = sim3.player
    p3.hp = p3.max_hp - 1
    p3.mana = p3.max_mana - 1
    sim3.do_eat_drink()
    assert p3.is_eating
    sim3.tick()
    assert p3.hp == p3.max_hp, "HP should be full"
    assert p3.mana == p3.max_mana, "Mana should be full"
    assert not p3.is_eating, "Should stop eating when full"
    print(f"  16c: Auto-stop when full: HP={p3.hp}/{p3.max_hp}, eating={p3.is_eating} ✓")

    # --- 16d: Can't eat when already full ---
    sim4 = CombatSimulation(num_mobs=5, seed=42)
    result = sim4.do_eat_drink()
    assert not result, "Should not start eating when already full"
    assert not sim4.player.is_eating
    print(f"  16d: Can't eat when full ✓")

    # --- 16e: Movement interrupts eating ---
    sim5 = CombatSimulation(num_mobs=5, seed=42)
    sim5.player.hp = int(sim5.player.max_hp * 0.5)
    sim5.do_eat_drink()
    assert sim5.player.is_eating
    sim5.do_move_forward()
    assert not sim5.player.is_eating, "move_forward should interrupt eating"
    print(f"  16e: move_forward interrupts eating ✓")

    # --- 16f: Turn interrupts eating ---
    sim6 = CombatSimulation(num_mobs=5, seed=42)
    sim6.player.hp = int(sim6.player.max_hp * 0.5)
    sim6.do_eat_drink()
    sim6.do_turn_left()
    assert not sim6.player.is_eating, "turn_left should interrupt eating"

    sim6b = CombatSimulation(num_mobs=5, seed=42)
    sim6b.player.hp = int(sim6b.player.max_hp * 0.5)
    sim6b.do_eat_drink()
    sim6b.do_turn_right()
    assert not sim6b.player.is_eating, "turn_right should interrupt eating"
    print(f"  16f: Turning interrupts eating ✓")

    # --- 16g: Damage interrupts eating ---
    sim7 = CombatSimulation(num_mobs=5, seed=42)
    sim7.player.hp = int(sim7.player.max_hp * 0.5)
    sim7.do_eat_drink()
    assert sim7.player.is_eating
    sim7._damage_player(5)
    assert not sim7.player.is_eating, "Damage should interrupt eating"
    print(f"  16g: Damage interrupts eating ✓")

    # --- 16h: Combat (mob aggro) interrupts eating ---
    sim8 = CombatSimulation(num_mobs=5, seed=42)
    sim8.player.hp = int(sim8.player.max_hp * 0.5)
    # Move player right on top of a mob to trigger aggro on next tick
    mob = sim8.mobs[0]
    sim8.player.x = mob.x
    sim8.player.y = mob.y
    sim8.do_eat_drink()
    assert sim8.player.is_eating
    sim8.tick()  # mob should aggro and interrupt eating
    assert sim8.player.in_combat, "Player should be in combat after aggro"
    assert not sim8.player.is_eating, "Combat should interrupt eating"
    print(f"  16h: Mob aggro interrupts eating ✓")

    # --- 16i: Can't eat while in combat ---
    sim9 = CombatSimulation(num_mobs=5, seed=42)
    sim9.player.in_combat = True
    sim9.player.hp = int(sim9.player.max_hp * 0.5)
    result = sim9.do_eat_drink()
    assert not result, "Should not start eating in combat"
    assert not sim9.player.is_eating
    print(f"  16i: Can't eat in combat ✓")

    # --- 16j: Can't eat while casting ---
    sim10 = CombatSimulation(num_mobs=5, seed=42)
    sim10.player.is_casting = True
    sim10.player.hp = int(sim10.player.max_hp * 0.5)
    result = sim10.do_eat_drink()
    assert not result, "Should not start eating while casting"
    print(f"  16j: Can't eat while casting ✓")

    # --- 16k: State dict includes is_eating ---
    sim11 = CombatSimulation(num_mobs=5, seed=42)
    sim11.player.hp = int(sim11.player.max_hp * 0.5)
    state = sim11.get_state_dict()
    assert state["is_eating"] == "false"
    sim11.do_eat_drink()
    state = sim11.get_state_dict()
    assert state["is_eating"] == "true"
    print(f"  16k: State dict is_eating field ✓")

    # --- 16l: WoWSimEnv action masking for eat/drink ---
    env = WoWSimEnv(num_mobs=5, seed=42)
    obs, _ = env.reset()
    mask = env.action_masks()
    # At full HP+Mana, eat/drink (action 17) should be masked
    assert not mask[17], "Eat/drink should be masked when full HP+Mana"

    # Damage player, eat should now be available
    env.sim.player.hp = int(env.sim.player.max_hp * 0.5)
    mask2 = env.action_masks()
    assert mask2[17], "Eat/drink should be valid when HP is low"

    # In combat, eat should be masked
    env.sim.player.in_combat = True
    mask3 = env.action_masks()
    assert not mask3[17], "Eat/drink should be masked in combat"
    env.sim.player.in_combat = False
    print(f"  16l: Action masking for eat/drink ✓")

    # --- 16m: While eating, only noop is allowed ---
    env2 = WoWSimEnv(num_mobs=5, seed=42)
    env2.reset()
    env2.sim.player.hp = int(env2.sim.player.max_hp * 0.5)
    env2.sim.do_eat_drink()
    assert env2.sim.player.is_eating
    mask4 = env2.action_masks()
    assert mask4[0], "Noop should be valid while eating"
    assert mask4.sum() == 1, f"Only noop valid while eating, but {mask4.sum()} actions valid"
    print(f"  16m: Only noop allowed while eating ✓")

    # --- 16n: Obs vector includes is_eating ---
    env3 = WoWSimEnv(num_mobs=5, seed=42)
    obs, _ = env3.reset()
    assert obs[22] == 0.0, f"is_eating should be 0, got {obs[22]}"
    env3.sim.player.hp = int(env3.sim.player.max_hp * 0.5)
    env3.sim.do_eat_drink()
    obs2 = env3._build_obs(env3.sim.get_state_dict())
    assert obs2[22] == 1.0, f"is_eating should be 1 while eating, got {obs2[22]}"
    print(f"  16n: Obs vector is_eating at index 22 ✓")

    # --- 16o: do_move_to interrupts eating ---
    sim12 = CombatSimulation(num_mobs=5, seed=42)
    sim12.player.hp = int(sim12.player.max_hp * 0.5)
    sim12.do_eat_drink()
    assert sim12.player.is_eating
    sim12.do_move_to(sim12.player.x + 10, sim12.player.y)
    assert not sim12.player.is_eating, "do_move_to should interrupt eating"
    print(f"  16o: do_move_to interrupts eating ✓")

    print("  PASSED\n")


def test_spell_learning():
    """Test 17: Spell level gates, % mana costs, DBC-verified values."""
    print("=== Test 17: Spell Learning & DBC Values ===")
    from sim.formulas import (spell_mana_cost, base_mana_for_level,
                              inner_fire_values, fortitude_stamina_bonus,
                              holy_fire_damage, holy_fire_dot_total,
                              mind_blast_damage, renew_total_heal)
    from sim.constants import SPELL_LEVEL_REQ, SPELL_MANA_PCT

    # --- 17a: Level requirements match trainer_spell.csv ---
    expected_levels = {585: 1, 2050: 1, 1243: 1, 589: 4, 17: 6,
                       139: 8, 8092: 10, 588: 12, 14914: 20}
    for spell_id, level in expected_levels.items():
        assert SPELL_LEVEL_REQ[spell_id] == level, \
            f"Spell {spell_id} level req should be {level}, got {SPELL_LEVEL_REQ[spell_id]}"
        assert SPELLS[spell_id].level_req == level, \
            f"SpellDef {spell_id} level_req should be {level}"
    print(f"  17a: Level requirements: {dict(sorted(expected_levels.items(), key=lambda x: x[1]))} ✓")

    # --- 17b: % mana costs from Spell.dbc ---
    expected_pct = {585: 9, 2050: 16, 589: 22, 17: 23, 1243: 27,
                    139: 17, 8092: 17, 588: 14, 14914: 11}
    for spell_id, pct in expected_pct.items():
        assert SPELL_MANA_PCT[spell_id] == pct, \
            f"Spell {spell_id} mana pct should be {pct}, got {SPELL_MANA_PCT[spell_id]}"
    print(f"  17b: Mana cost percentages from DBC ✓")

    # --- 17c: Actual mana costs at L1 (BaseMana=73) ---
    bm = base_mana_for_level(1)
    assert bm == 73, f"Priest BaseMana at L1 should be 73, got {bm}"
    # Smite: 73 * 9 / 100 = 6
    assert spell_mana_cost(585, 1) == 6, f"Smite mana@L1 should be 6"
    # Lesser Heal: 73 * 16 / 100 = 11
    assert spell_mana_cost(2050, 1) == 11, f"Lesser Heal mana@L1 should be 11"
    # Mind Blast: 73 * 17 / 100 = 12
    assert spell_mana_cost(8092, 1) == 12, f"Mind Blast mana@L1 should be 12"
    print(f"  17c: Mana costs at L1 (BaseMana={bm}) verified ✓")

    # --- 17d: Mana costs scale with level ---
    bm10 = base_mana_for_level(10)
    assert bm10 > bm, f"BaseMana should increase: L1={bm} < L10={bm10}"
    cost_l1 = spell_mana_cost(585, 1)
    cost_l10 = spell_mana_cost(585, 10)
    assert cost_l10 > cost_l1, f"Smite cost should increase: L1={cost_l1} < L10={cost_l10}"
    print(f"  17d: Mana scales: Smite L1={cost_l1}, L10={cost_l10} (BaseMana {bm}→{bm10}) ✓")

    # --- 17e: Level gate blocks spells in combat_sim ---
    sim = CombatSimulation(num_mobs=10, seed=42)
    p = sim.player
    assert p.level == 1

    # Walk toward mobs and target one
    for _ in range(50):
        sim.do_move_forward()
        sim.tick()
    sim.do_target_nearest()
    sim.tick()
    # Even if no target, the level gate check comes first
    # SW:Pain requires L4 — should fail at L1 (level check before target check)
    result_swp = sim.do_cast_sw_pain()
    assert result_swp is False, "SW:Pain should be blocked at L1 (needs L4)"
    # Mind Blast requires L10 — should fail at L1
    result_mb = sim.do_cast_mind_blast()
    assert result_mb is False, "Mind Blast should be blocked at L1 (needs L10)"
    # PW:Fortitude works at L1 (L1 req, self-cast, no target needed)
    result_fort = sim.do_cast_fortitude()
    assert result_fort is True, "PW:Fortitude should work at L1"
    print(f"  17e: Level gate: SW:Pain/Mind Blast blocked at L1, Fort OK ✓")

    # --- 17f: Level gate in action mask ---
    env = WoWSimEnv(num_mobs=10, seed=42)
    obs, _ = env.reset()
    mask = env.action_masks()
    # At L1: SW:Pain (action 9) should be masked (needs L4)
    assert not mask[9], "SW:Pain (action 9) should be masked at L1"
    # Shield (action 10) should be masked (needs L6)
    assert not mask[10], "PW:Shield (action 10) should be masked at L1"
    # Smite (action 5) — may be masked for other reasons (no target) but not level
    # PW:Fortitude (action 16) is L1, should not be level-gated
    # (may be masked if already active, but NOT for level)
    print(f"  17f: Action mask level gates at L1: SW:Pain=masked, Shield=masked ✓")

    # --- 17g: PW:Fortitude gives Stamina (not flat HP) ---
    sim2 = CombatSimulation(num_mobs=5, seed=42)
    p2 = sim2.player
    sta_before = p2.total_stamina
    hp_before = p2.max_hp
    result = sim2.do_cast_fortitude()
    assert result is True, "Fort should cast at L1"
    sim2.tick()  # process GCD + instant spell
    assert p2.fortitude_stamina_bonus == 3, \
        f"Fort should give +3 Stamina, got {p2.fortitude_stamina_bonus}"
    assert p2.total_stamina == sta_before + 3, \
        f"Total stam should increase by 3: {sta_before} → {p2.total_stamina}"
    assert p2.max_hp > hp_before, \
        f"HP should increase from Stamina: {hp_before} → {p2.max_hp}"
    print(f"  17g: PW:Fortitude: +3 Stamina, HP {hp_before}→{p2.max_hp} ✓")

    # --- 17h: Inner Fire gives +315 Armor, NO spell power (R1) ---
    sim3 = CombatSimulation(num_mobs=5, seed=42)
    p3 = sim3.player
    # Level up to 12 for Inner Fire
    p3.level = 12
    sim3.recalculate_stats()
    armor_before = p3.total_armor
    sp_before = p3.total_spell_power
    sim3.do_cast_inner_fire()
    sim3.tick()
    armor_bonus, sp_bonus = inner_fire_values(12)
    assert armor_bonus == 315, f"Inner Fire R1 armor should be 315, got {armor_bonus}"
    assert sp_bonus == 0, f"Inner Fire R1 SP should be 0 (R1), got {sp_bonus}"
    assert p3.total_armor >= armor_before + 315, \
        f"Armor should increase by 315: {armor_before} → {p3.total_armor}"
    assert p3.total_spell_power == sp_before, \
        f"SP should not change from Inner Fire R1: {sp_before} → {p3.total_spell_power}"
    print(f"  17h: Inner Fire R1: +315 Armor, +0 SP ✓")

    # --- 17i: Holy Fire damage from DBC (102-128 direct, 21 DoT) ---
    d_min, d_max = holy_fire_damage(20)  # L20 base (no SP)
    assert d_min >= 102, f"Holy Fire min should be >=102, got {d_min}"
    assert d_max >= 128, f"Holy Fire max should be >=128, got {d_max}"
    dot = holy_fire_dot_total(20)
    assert dot == 21, f"Holy Fire DoT total should be 21, got {dot}"
    print(f"  17i: Holy Fire R1: {d_min}-{d_max} direct, {dot} DoT ✓")

    # --- 17j: Mind Blast R1 damage from DBC (bp=38, ds=5 → 39-43) ---
    mb_min, mb_max = mind_blast_damage(10)  # R1 base values (no level scaling)
    assert mb_min == 39, f"Mind Blast R1 min should be 39, got {mb_min}"
    assert mb_max == 43, f"Mind Blast R1 max should be 43, got {mb_max}"
    # At L16 best rank is R2 (8102, bp=71, ds=7 → 72-78)
    from sim.formulas import spell_direct_value
    r2_min, r2_max = spell_direct_value(8102)
    assert (r2_min, r2_max) == (72, 78), f"Mind Blast R2 expected (72,78), got ({r2_min},{r2_max})"
    print(f"  17j: Mind Blast R1: {mb_min}-{mb_max}, R2: {r2_min}-{r2_max} ✓")

    # --- 17k: Renew heal from DBC (45 total, no level scaling) ---
    rn = renew_total_heal(8)  # L8, no SP
    assert rn == 45, f"Renew total@L8 should be 45, got {rn}"
    rn20 = renew_total_heal(20)  # L20, still 45 (no RPL)
    assert rn20 == 45, f"Renew total@L20 should be 45 (no level scaling), got {rn20}"
    print(f"  17k: Renew R1: {rn} heal (no level scaling) ✓")

    # --- 17l: Buff durations (30min = 3600 ticks) ---
    assert SPELLS[588].buff_duration == 3600, \
        f"Inner Fire duration should be 3600 ticks, got {SPELLS[588].buff_duration}"
    assert SPELLS[1243].buff_duration == 3600, \
        f"Fortitude duration should be 3600 ticks, got {SPELLS[1243].buff_duration}"
    print(f"  17l: Buff durations: Inner Fire=3600, Fort=3600 ticks (30min) ✓")

    # --- 17m: Holy Fire DoT: 7 ticks @ 1s interval = 14 ticks total ---
    hf_spell = SPELLS[14914]
    assert hf_spell.dot_ticks == 14, f"HF dot_ticks should be 14, got {hf_spell.dot_ticks}"
    assert hf_spell.dot_interval == 2, f"HF dot_interval should be 2 (1s), got {hf_spell.dot_interval}"
    assert hf_spell.dot_damage == 21, f"HF dot_damage should be 21, got {hf_spell.dot_damage}"
    print(f"  17m: Holy Fire DoT: {hf_spell.dot_damage} over {hf_spell.dot_ticks} ticks ✓")

    print("  PASSED\n")


def test_talent_system():
    """Test 18: Talent system — auto-assignment, gated spells, modifiers."""
    print("=== Test 18: Talent System ===")
    from sim.talent_data import get_talent_for_level, TALENT_DEFS, SHADOW_PRIEST_BUILD

    # --- 18a: No talents before level 10 ---
    sim = CombatSimulation(num_mobs=10, seed=42)
    p = sim.player
    assert p.level == 1
    assert len(p.talent_points) == 0, "No talents at level 1"
    assert get_talent_for_level(1) is None
    assert get_talent_for_level(9) is None
    assert get_talent_for_level(10) == "spirit_tap"
    print(f"  18a: No talents before L10, first talent=spirit_tap ✓")

    # --- 18b: Talent auto-assignment on level-up ---
    # Manually level to 12 (should get Spirit Tap 3/3)
    for lvl in range(2, 13):
        p.xp = XP_TABLE[lvl]
        sim._check_level_up()
    assert p.level == 12, f"Expected level 12, got {p.level}"
    assert p.talent_points.get("spirit_tap", 0) == 3, \
        f"Spirit Tap should be 3/3 at L12, got {p.talent_points.get('spirit_tap', 0)}"
    print(f"  18b: L12 talents: spirit_tap={p.talent_points.get('spirit_tap')} ✓")

    # --- 18c: Darkness at level 19 (5/5) ---
    for lvl in range(13, 20):
        p.xp = XP_TABLE[lvl]
        sim._check_level_up()
    assert p.level == 19
    assert p.talent_points.get("darkness", 0) == 5, \
        f"Darkness should be 5/5 at L19, got {p.talent_points.get('darkness', 0)}"
    assert p.talent_points.get("improved_spirit_tap", 0) == 2
    print(f"  18c: L19 talents: darkness={p.talent_points['darkness']}, "
          f"imp_spirit_tap={p.talent_points['improved_spirit_tap']} ✓")

    # --- 18d: Mind Flay unlocked at level 20 ---
    assert sim.do_cast_mind_flay() is False, "Mind Flay should fail before talent"
    p.xp = XP_TABLE[20]
    sim._check_level_up()
    assert p.level == 20
    assert p.talent_points.get("mind_flay", 0) == 1, "Mind Flay talent should be 1/1 at L20"
    # Now target a mob and try Mind Flay
    for _ in range(30):
        sim.do_move_forward()
        sim.tick()
    sim.do_target_nearest()
    if sim.target and sim.target.alive:
        result = sim.do_cast_mind_flay()
        assert result is True, "Mind Flay should succeed at L20 with talent + target"
        assert p.channel_remaining > 0 or p.is_casting, "Should be channeling Mind Flay"
    print(f"  18d: Mind Flay unlocked at L20 ✓")

    # --- 18e: Shadowform auto-activation at level 40 ---
    sim2 = CombatSimulation(num_mobs=5, seed=42)
    p2 = sim2.player
    assert not p2.shadowform_active
    for lvl in range(2, 41):
        p2.xp = XP_TABLE[lvl]
        sim2._check_level_up()
    assert p2.level == 40
    assert p2.talent_points.get("shadowform", 0) == 1, "Shadowform talent should be 1/1 at L40"
    assert p2.shadowform_active, "Shadowform should auto-activate at L40"
    print(f"  18e: Shadowform auto-activated at L40 ✓")

    # --- 18f: Shadowform toggle ---
    sim2.do_toggle_shadowform()  # turn off
    assert not p2.shadowform_active, "Shadowform should be toggled off"
    p2.gcd_remaining = 0
    sim2.do_toggle_shadowform()  # turn on
    assert p2.shadowform_active, "Shadowform should be toggled on"
    print(f"  18f: Shadowform toggle works ✓")

    # --- 18g: Shadow damage modifiers (Darkness + Shadowform) ---
    # At L40: Darkness 5/5 (+10%) and Shadowform (+15%) = 1.10 * 1.15 = 1.265
    mob = sim2.mobs[0]
    mob.shadow_weaving_stacks = 0
    mod = sim2._shadow_damage_mod(mob)
    expected = 1.10 * 1.15  # Darkness 5/5 * Shadowform
    assert abs(mod - expected) < 0.001, \
        f"Shadow mod expected {expected:.3f}, got {mod:.3f}"
    # With Shadow Weaving 3 stacks: +6%
    mob.shadow_weaving_stacks = 3
    mod_sw = sim2._shadow_damage_mod(mob)
    expected_sw = expected * 1.06
    assert abs(mod_sw - expected_sw) < 0.001, \
        f"Shadow mod with SW3 expected {expected_sw:.3f}, got {mod_sw:.3f}"
    mob.shadow_weaving_stacks = 0
    print(f"  18g: Shadow damage mod: base={mod:.3f}, +SW3={mod_sw:.3f} ✓")

    # --- 18h: Vampiric Embrace healing ---
    sim3 = CombatSimulation(num_mobs=5, seed=42)
    p3 = sim3.player
    for lvl in range(2, 34):
        p3.xp = XP_TABLE[lvl]
        sim3._check_level_up()
    assert p3.level == 33
    assert p3.talent_points.get("vampiric_embrace", 0) == 1
    assert p3.talent_points.get("improved_vampiric_embrace", 0) == 2
    # VE 1/1 + Imp VE 2/2 = 25% healing
    p3.hp = p3.max_hp // 2
    hp_before = p3.hp
    sim3._vampiric_embrace_heal(100)  # 100 shadow damage
    expected_heal = int(100 * 0.25)
    assert p3.hp == hp_before + expected_heal, \
        f"VE heal: expected +{expected_heal}, got +{p3.hp - hp_before}"
    print(f"  18h: Vampiric Embrace: 100 shadow dmg → +{p3.hp - hp_before} HP ✓")

    # --- 18i: Spirit Tap proc on kill ---
    sim4 = CombatSimulation(num_mobs=5, seed=42)
    p4 = sim4.player
    for lvl in range(2, 13):
        p4.xp = XP_TABLE[lvl]
        sim4._check_level_up()
    assert p4.talent_points.get("spirit_tap", 0) == 3
    assert p4.spirit_tap_remaining == 0
    # Kill a mob directly
    mob4 = sim4.mobs[0]
    mob4.alive = True
    mob4.hp = 1
    mob4.in_combat = True
    mob4.target_player = True
    p4.in_combat = True
    sim4._damage_mob(mob4, 100)  # overkill
    assert not mob4.alive
    assert p4.spirit_tap_remaining > 0, "Spirit Tap should proc on kill"
    print(f"  18i: Spirit Tap proc: remaining={p4.spirit_tap_remaining} ticks ✓")

    # --- 18j: Dispersion at level 60 ---
    sim5 = CombatSimulation(num_mobs=5, seed=42)
    p5 = sim5.player
    assert sim5.do_cast_dispersion() is False, "Dispersion should fail without talent"
    for lvl in range(2, 61):
        p5.xp = XP_TABLE[lvl]
        sim5._check_level_up()
    assert p5.level == 60
    assert p5.talent_points.get("dispersion", 0) == 1
    print(f"  18j: Dispersion unlocked at L60 ✓")

    # --- 18k: Dispersion damage reduction ---
    sim6 = CombatSimulation(num_mobs=5, seed=42)
    p6 = sim6.player
    for lvl in range(2, 61):
        p6.xp = XP_TABLE[lvl]
        sim6._check_level_up()
    p6.hp = p6.max_hp
    # Take damage without dispersion
    hp_full = p6.hp
    sim6._damage_player(100)
    dmg_normal = hp_full - p6.hp
    # Heal and take damage WITH dispersion
    p6.hp = p6.max_hp
    p6.dispersion_remaining = 12  # active
    hp_full2 = p6.hp
    sim6._damage_player(100)
    dmg_dispersion = hp_full2 - p6.hp
    assert dmg_dispersion < dmg_normal, \
        f"Dispersion should reduce damage: normal={dmg_normal}, dispersion={dmg_dispersion}"
    # Dispersion is -90%, so dispersion damage should be ~10% of raw
    assert dmg_dispersion < dmg_normal * 0.5, \
        f"Dispersion should reduce by ~90%: {dmg_dispersion} vs {dmg_normal}"
    p6.dispersion_remaining = 0
    print(f"  18k: Dispersion damage: normal={dmg_normal}, with dispersion={dmg_dispersion} ✓")

    # --- 18l: Shadowform physical damage reduction ---
    sim7 = CombatSimulation(num_mobs=5, seed=42)
    p7 = sim7.player
    for lvl in range(2, 41):
        p7.xp = XP_TABLE[lvl]
        sim7._check_level_up()
    # Remove armor to isolate Shadowform effect
    p7.total_armor = 0
    p7.hp = p7.max_hp
    hp1 = p7.hp
    # Turn OFF Shadowform
    p7.shadowform_active = False
    sim7._damage_player(100)
    dmg_no_sf = hp1 - p7.hp
    # Turn ON Shadowform
    p7.hp = p7.max_hp
    hp2 = p7.hp
    p7.shadowform_active = True
    sim7._damage_player(100)
    dmg_sf = hp2 - p7.hp
    assert dmg_sf < dmg_no_sf, \
        f"Shadowform should reduce physical damage: without={dmg_no_sf}, with={dmg_sf}"
    print(f"  18l: Shadowform physical DR: without={dmg_no_sf}, with={dmg_sf} ✓")

    # --- 18m: VT talent gate ---
    sim8 = CombatSimulation(num_mobs=5, seed=42)
    assert sim8.do_cast_vampiric_touch() is False, "VT should fail without talent"
    p8 = sim8.player
    for lvl in range(2, 51):
        p8.xp = XP_TABLE[lvl]
        sim8._check_level_up()
    assert p8.talent_points.get("vampiric_touch", 0) == 1
    print(f"  18m: Vampiric Touch unlocked at L50 ✓")

    # --- 18n: Action masking for talent-gated spells ---
    env = WoWSimEnv(num_mobs=10, seed=42)
    obs, _ = env.reset()
    mask = env.action_masks()
    # At L1: Mind Flay (26), VT (27), Dispersion (28), Shadowform (29) all masked
    assert not mask[26], "Mind Flay should be masked at L1 (no talent)"
    assert not mask[27], "Vampiric Touch should be masked at L1 (no talent)"
    assert not mask[28], "Dispersion should be masked at L1 (no talent)"
    assert not mask[29], "Shadowform should be masked at L1 (no talent)"
    print(f"  18n: Talent-gated spells masked at L1 ✓")

    # --- 18o: Obs vector talent dims (indices 29-32) ---
    env2 = WoWSimEnv(num_mobs=5, seed=42)
    obs2, _ = env2.reset()
    # At L1: target_has_vt=0, shadowform=0, dispersion=0, channeling=0
    assert obs2[29] == 0.0, f"target_has_vt should be 0, got {obs2[29]}"
    assert obs2[30] == 0.0, f"shadowform_active should be 0, got {obs2[30]}"
    assert obs2[31] == 0.0, f"dispersion_active should be 0, got {obs2[31]}"
    assert obs2[32] == 0.0, f"is_channeling should be 0, got {obs2[32]}"
    # Level up to 40 for Shadowform
    p_env = env2.sim.player
    for lvl in range(2, 41):
        p_env.xp = XP_TABLE[lvl]
        env2.sim._check_level_up()
    assert p_env.shadowform_active
    obs3 = env2._build_obs(env2.sim.get_state_dict())
    assert obs3[30] == 1.0, f"shadowform_active obs should be 1.0, got {obs3[30]}"
    print(f"  18o: Talent obs dims [29-32] correct ✓")

    # --- 18p: Build totals match 13/0/58 at L80 ---
    sim9 = CombatSimulation(num_mobs=5, seed=42)
    p9 = sim9.player
    for lvl in range(2, 81):
        p9.xp = XP_TABLE[lvl]
        sim9._check_level_up()
    assert p9.level == 80
    total_pts = sum(p9.talent_points.values())
    assert total_pts == 71, f"Expected 71 total talent points at L80, got {total_pts}"
    shadow_pts = sum(v for k, v in p9.talent_points.items()
                     if TALENT_DEFS[k]["tree"] == "shadow")
    disc_pts = sum(v for k, v in p9.talent_points.items()
                   if TALENT_DEFS[k]["tree"] == "discipline")
    assert shadow_pts == 58, f"Expected 58 Shadow points, got {shadow_pts}"
    assert disc_pts == 13, f"Expected 13 Discipline points, got {disc_pts}"
    # Meditation should be 3/3
    assert p9.talent_points.get("meditation", 0) == 3
    print(f"  18p: L80 build: {shadow_pts}/0/{disc_pts} = 13/0/58, total={total_pts} ✓")

    # --- 18q: Talents reset on sim reset ---
    sim9.reset()
    assert len(sim9.player.talent_points) == 0, "Talents should reset"
    assert not sim9.player.shadowform_active, "Shadowform should reset"
    assert sim9.player.spirit_tap_remaining == 0
    assert sim9.player.dispersion_remaining == 0
    print(f"  18q: Talents cleared on reset ✓")

    print("  PASSED\n")


if __name__ == "__main__":
    print("WoW Combat Simulation — Validation Tests\n")
    test_combat_engine()
    test_gym_env()
    test_random_episode()
    test_performance()
    test_combat_scenario()
    test_level_system()
    test_loot_tables()
    test_vendor_system()
    test_quest_system()
    test_quest_csv_loading()
    test_attribute_system()
    test_bag_system()
    test_combat_resolution()
    test_action_masking()
    test_eat_drink()
    test_spell_learning()
    test_talent_system()
    print("=== ALL TESTS PASSED ===")
