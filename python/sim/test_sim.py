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
                             hit_chance_spell, expertise_pct, armor_penetration_pct)
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
    assert env.observation_space.shape == (38,), f"Obs shape: {env.observation_space.shape}"
    assert env.action_space.n == 17, f"Action space: {env.action_space.n}"
    print(f"  Obs space: {env.observation_space.shape}, dtype={env.observation_space.dtype}")
    print(f"  Action space: Discrete({env.action_space.n})")

    # Reset
    obs, info = env.reset()
    assert obs.shape == (38,), f"Obs shape after reset: {obs.shape}"
    assert obs.dtype == np.float32
    print(f"  Reset obs: shape={obs.shape}, range=[{obs.min():.3f}, {obs.max():.3f}]")

    # Step with each action
    for action in range(17):
        obs, reward, done, trunc, info = env.step(action)
        assert obs.shape == (38,)
        if done:
            obs, info = env.reset()

    print(f"  All 17 actions executed successfully")
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

    # --- Stat scaling (WotLK: includes base stamina/intellect contribution) ---
    # player_max_hp(1) = 72 + stam_hp(priest_base_stam(20,1)=20) = 72 + 20 = 92
    assert player_max_hp(1) == 92, f"HP(1) expected 92, got {player_max_hp(1)}"
    # player_max_hp(2) = 122 + stam_hp(21) = 122 + 20 + 10 = 152
    assert player_max_hp(2) == 152, f"HP(2) expected 152, got {player_max_hp(2)}"
    # player_max_hp(10) = 522 + stam_hp(29) = 522 + 20 + 90 = 632
    assert player_max_hp(10) == 632, f"HP(10) expected 632, got {player_max_hp(10)}"
    # player_max_mana(1) = 123 + int_mana(priest_base_int(22,1)=22) = 123 + 20 + 30 = 173
    assert player_max_mana(1) == 173, f"Mana(1) expected 173, got {player_max_mana(1)}"
    # player_max_mana(2) = 128 + int_mana(23) = 128 + 20 + 45 = 193
    assert player_max_mana(2) == 193, f"Mana(2) expected 193, got {player_max_mana(2)}"
    # Smite/Heal base damage unchanged (no SP)
    min_d, max_d = smite_damage(1)
    assert (min_d, max_d) == (13, 17)
    min_d, max_d = smite_damage(2)
    assert (min_d, max_d) == (23, 27)
    min_h, max_h = heal_amount(1)
    assert (min_h, max_h) == (46, 56)
    min_h, max_h = heal_amount(2)
    assert (min_h, max_h) == (51, 61)
    # Spell functions with SP scaling
    min_d_sp, max_d_sp = smite_damage(1, spell_power=100)
    sp_bonus = int(100 * SP_COEFF_SMITE)  # 71
    assert (min_d_sp, max_d_sp) == (13 + sp_bonus, 17 + sp_bonus), \
        f"Smite+SP expected ({13+sp_bonus},{17+sp_bonus}), got ({min_d_sp},{max_d_sp})"
    print(f"  Stat scaling: HP, Mana, Smite, Heal, SP scaling ✓")

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
    # Pre-equip a weak chest item
    p.equipped_scores[20] = 10.0  # InventoryType 20 (robe slot)

    # Create a mock loot scenario: Dirty Leather Vest (score=35) should be upgrade
    vest = loot_db.get_item(2000)
    # Score: (1*10) + 6 + 15 + 0 + (2*2) = 35
    assert vest is not None
    assert vest.score == 35.0, f"Vest score expected 35.0, got {vest.score}"
    assert vest.score > 10.0, "Vest should be better than equipped"

    # Verify the upgrade logic works
    if vest.inventory_type > 0:
        current = p.equipped_scores.get(vest.inventory_type, 0.0)
        if vest.score > current:
            p.equipped_scores[vest.inventory_type] = vest.score
            p.equipped_upgrade = True
    assert p.equipped_upgrade, "Vest should be detected as upgrade"
    assert p.equipped_scores[20] == 35.0, "Equipped score should update"
    print(f"  7e: Upgrade detection: 10.0 → {p.equipped_scores[20]} ✓")

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

    # --- Test 7g: Inventory capacity (30 slots) ---
    assert INVENTORY_SLOTS == 30, f"Expected 30 inventory slots, got {INVENTORY_SLOTS}"
    sim_inv = CombatSimulation(num_mobs=50, seed=123, loot_db=loot_db)
    assert sim_inv.player.free_slots == INVENTORY_SLOTS, \
        f"Player should start with {INVENTORY_SLOTS} free slots"

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

    # --- Test 8l: AI-driven sell via WoWSimEnv (action 8 triggers vendor nav) ---
    env = WoWSimEnv(num_mobs=5, seed=42)
    obs, _ = env.reset()
    # Fill inventory with items
    for i in range(20):
        env.sim.player.inventory.append(InventoryItem(
            entry=100+i, name=f"Junk_{i}", quality=0,
            sell_price=10, score=3.0, inventory_type=0))
    env.sim.player.free_slots = INVENTORY_SLOTS - 20

    # Action 8 should activate vendor navigation
    assert not env._vendor_nav_active, "Should start inactive"
    obs, reward, done, trunc, info = env.step(8)
    assert env._vendor_nav_active or env.sim.player.free_slots == INVENTORY_SLOTS, \
        "Action 8 should activate vendor nav (or sell if already at vendor)"

    # Run steps until vendor nav completes (max 500 steps)
    for _ in range(500):
        obs, reward, done, trunc, info = env.step(0)  # noop while override navigates
        if done or trunc:
            break
        if env.sim.player.free_slots == INVENTORY_SLOTS:
            break  # sold!

    assert env.sim.player.free_slots == INVENTORY_SLOTS, \
        f"Bot should have sold all items, but free_slots={env.sim.player.free_slots}"
    assert not env._vendor_nav_active, "Vendor nav should be deactivated after sell"
    assert env.sim.player.copper > 0, "Should have earned copper from selling"
    print(f"  8l: AI-driven sell: copper={env.sim.player.copper}, "
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
    assert obs.shape == (38,), f"Expected obs(28,), got {obs.shape}"
    assert env.action_space.n == 17, f"Expected 17 actions, got {env.action_space.n}"
    # Quest dims should be non-zero (quest NPCs are visible)
    assert obs[28] > 0 or obs[26] == 0.0, "Quest NPC obs should reflect nearby NPCs"
    print(f"  9k: WoWSimEnv(enable_quests=True): obs={obs.shape}, "
          f"quest_dims={obs[26:32]} ✓")

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

    # --- Test 11a: Base stat formulas (WotLK — all classes) ---
    # stat_index: 0=str, 1=agi, 2=stam, 3=int, 4=spi
    # Priest base stats at level 1: (15, 17, 20, 22, 23)
    assert class_base_stat(CLASS_PRIEST, 2, 1) == 20   # stamina
    assert class_base_stat(CLASS_PRIEST, 2, 10) == 29   # stam L10 = 20 + 9
    assert class_base_stat(CLASS_PRIEST, 3, 1) == 22    # intellect
    assert class_base_stat(CLASS_PRIEST, 3, 10) == 31   # int L10 = 22 + 9
    assert class_base_stat(CLASS_PRIEST, 4, 1) == 23    # spirit
    # Warrior base stats at level 1: (23, 20, 22, 17, 19)
    assert class_base_stat(CLASS_WARRIOR, 0, 1) == 23   # strength
    assert class_base_stat(CLASS_WARRIOR, 0, 10) == 32   # str L10 = 23 + 9
    assert class_base_stat(CLASS_WARRIOR, 2, 1) == 22   # stamina
    # Rogue base stats at level 1: (18, 24, 20, 17, 19)
    assert class_base_stat(CLASS_ROGUE, 1, 1) == 24     # agility
    assert class_base_stat(CLASS_MAGE, 3, 1) == 24      # mage intellect
    print(f"  11a: Base stats: Priest(stam=20,int=22,spi=23), Warrior(str=23), "
          f"Rogue(agi=24), Mage(int=24) ✓")

    # --- Test 11b: HP with bonus stamina (WotLK: first 20=1HP, above 20=10HP) ---
    hp_base = player_max_hp(1)  # Priest L1: 72 + stam_hp(20)
    hp_bonus = player_max_hp(1, bonus_stamina=5)  # +5 stam -> total 25 -> +50 HP more
    # total_stam = 20+5 = 25, stam_hp = 20 + 5*10 = 70 -> 72+70 = 142
    assert hp_bonus == 142, f"HP(1,+5stam) expected 142, got {hp_bonus}"
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
    mana_base = player_max_mana(1)  # Priest: 123 + int_mana(22) = 173
    mana_bonus = player_max_mana(1, bonus_intellect=10)  # total_int=32 -> 20+12*15=200 -> 323
    assert mana_bonus == 323, f"Mana(1,+10int) expected 323, got {mana_bonus}"
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

    # --- Test 11e: Spell haste from rating ---
    assert spell_haste_pct(1, 0) == 0.0, "No haste without rating"
    haste = spell_haste_pct(1, bonus_haste_rating=50)
    assert haste > 0.0, "Haste rating should give haste %"
    # Higher level needs more rating for same %
    haste_l80 = spell_haste_pct(80, bonus_haste_rating=50)
    assert haste_l80 < haste, "Haste should be harder to get at higher levels"
    print(f"  11e: Haste: L1+50rat={haste:.2f}%, L80+50rat={haste_l80:.2f}% ✓")

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

    # --- Test 11g: Spell power scaling ---
    # Smite: base (13,17) + SP*0.7143
    d_min, d_max = smite_damage(1, spell_power=140)  # 140 * 0.7143 = 100
    assert d_min == 113, f"Smite(1,SP=140) min expected 113, got {d_min}"
    assert d_max == 117, f"Smite(1,SP=140) max expected 117, got {d_max}"
    # Heal: base (46,56) + SP*0.8571
    h_min, h_max = heal_amount(1, spell_power=140)  # 140*0.8571 = 119
    assert h_min == 165, f"Heal(1,SP=140) min expected 165, got {h_min}"
    # SW:Pain: base 30 + SP*0.1833*6
    swp = sw_pain_total(1, spell_power=100)  # 30 + 100*1.1 = 140
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
    p.equipment[20] = chest
    p.equipped_scores[20] = 50.0
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
    p.equipment[26] = sp_wand
    p.equipped_scores[26] = 40.0
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
    assert p2.equipment[1].entry == 5001, "Item should be in head slot"
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
    assert p2.equipment[1].entry == 5001, "Original item should still be equipped"
    print(f"  11j: try_equip_item: upgrade=True, downgrade=False ✓")

    # --- Test 11k: Armor mitigation (WotLK formula) ---
    sim3 = CombatSimulation(num_mobs=5, seed=42)
    p3 = sim3.player
    # Give player some armor gear
    p3.equipment[5] = EquippedItem(
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
    # Stat obs at indices 22-31 (10 dims: SP, spell_crit, spell_haste, armor,
    # AP, melee_crit, dodge, hit, expertise, ArP)
    assert obs[22] == 0.0, f"SP obs should be 0 with no gear, got {obs[22]}"
    assert obs[23] >= 0.0, f"Spell crit obs should be >= 0, got {obs[23]}"
    assert obs[24] == 0.0, f"Spell haste obs should be 0 with no gear, got {obs[24]}"
    assert obs[25] >= 0.0, f"Armor obs should be >= 0, got {obs[25]}"  # agi*2 gives base armor
    assert obs[26] >= 0.0, f"AP obs should be >= 0, got {obs[26]}"
    assert obs[27] >= 0.0, f"Melee crit obs should be >= 0, got {obs[27]}"
    assert obs[28] >= 0.0, f"Dodge obs should be >= 0, got {obs[28]}"
    print(f"  11m: Stat obs [22:32]={obs[22:32]} ✓")

    # --- Test 11n: Equipment persists across ticks ---
    sim4 = CombatSimulation(num_mobs=5, seed=42)
    sim4.player.equipment[5] = EquippedItem(
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
    assert war_ap == int(1 * 3.0 + 23 * 2.0 - 20.0), f"Warrior AP expected 29, got {war_ap}"
    # Rogue: level*2 + str + agi - 20
    rog_str = class_base_stat(CLASS_ROGUE, 0, 1)  # 18
    rog_agi = class_base_stat(CLASS_ROGUE, 1, 1)  # 24
    rog_ap = melee_attack_power(1, rog_str, rog_agi, CLASS_ROGUE)
    assert rog_ap == int(1 * 2.0 + 18 + 24 - 20.0), f"Rogue AP expected 24, got {rog_ap}"
    # Priest: str - 10
    pri_str = class_base_stat(CLASS_PRIEST, 0, 1)  # 15
    pri_agi = class_base_stat(CLASS_PRIEST, 1, 1)
    pri_ap = melee_attack_power(1, pri_str, pri_agi, CLASS_PRIEST)
    assert pri_ap == int(15 - 10.0), f"Priest AP expected 5, got {pri_ap}"
    print(f"  11p: Attack Power: Warrior={war_ap}, Rogue={rog_ap}, Priest={pri_ap} ✓")

    # --- Test 11q: Melee crit from Agility ---
    # Warrior at L1 (base crit 0.0, simpler)
    war_crit = melee_crit_chance(1, 20, class_id=CLASS_WARRIOR)
    assert war_crit > 0.0, f"Warrior melee crit should be > 0, got {war_crit}"
    war_crit_high = melee_crit_chance(1, 100, class_id=CLASS_WARRIOR)
    assert war_crit_high > war_crit, "Higher agi should increase melee crit"
    # Rogue has negative base crit — needs high agi to overcome
    rog_crit_low = melee_crit_chance(1, 24, class_id=CLASS_ROGUE)
    assert rog_crit_low == 0.0, "Rogue L1 base agi clamps to 0% crit"
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
    p5.equipment[21] = melee_gear
    p5.equipped_scores[21] = 80.0
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
    print("=== ALL TESTS PASSED ===")
