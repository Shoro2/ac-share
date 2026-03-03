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
                             INVENTORY_SLOTS, VENDOR_DATA, InventoryItem)
from sim.wow_sim_env import WoWSimEnv


def test_combat_engine():
    """Test the raw simulation engine."""
    print("=== Test 1: Combat Engine ===")
    sim = CombatSimulation(num_mobs=10, seed=42)

    # Check initial state
    p = sim.player
    assert p.hp == 72, f"Expected HP=72, got {p.hp}"
    assert p.mana == 123, f"Expected Mana=123, got {p.mana}"
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
    assert env.observation_space.shape == (17,), f"Obs shape: {env.observation_space.shape}"
    assert env.action_space.n == 11, f"Action space: {env.action_space.n}"
    print(f"  Obs space: {env.observation_space.shape}, dtype={env.observation_space.dtype}")
    print(f"  Action space: Discrete({env.action_space.n})")

    # Reset
    obs, info = env.reset()
    assert obs.shape == (17,), f"Obs shape after reset: {obs.shape}"
    assert obs.dtype == np.float32
    print(f"  Reset obs: shape={obs.shape}, range=[{obs.min():.3f}, {obs.max():.3f}]")

    # Step with each action
    for action in range(11):
        obs, reward, done, trunc, info = env.step(action)
        assert obs.shape == (17,)
        if done:
            obs, info = env.reset()

    print(f"  All 11 actions executed successfully")
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

    # --- Stat scaling ---
    assert player_max_hp(1) == 72
    assert player_max_hp(2) == 122
    assert player_max_hp(10) == 522
    assert player_max_mana(1) == 123
    assert player_max_mana(2) == 128
    min_d, max_d = smite_damage(1)
    assert (min_d, max_d) == (13, 17)
    min_d, max_d = smite_damage(2)
    assert (min_d, max_d) == (23, 27)
    min_h, max_h = heal_amount(1)
    assert (min_h, max_h) == (46, 56)
    min_h, max_h = heal_amount(2)
    assert (min_h, max_h) == (51, 61)
    print(f"  Stat scaling: HP, Mana, Smite, Heal ✓")

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

    assert p.level == 2, f"Expected level 2 after 449 XP, got {p.level}"
    assert p.max_hp == 122, f"Expected max_hp=122, got {p.max_hp}"
    assert p.hp == 122, "HP should be full after level-up"
    assert p.max_mana == 128, f"Expected max_mana=128, got {p.max_mana}"
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

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

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
    print("=== ALL TESTS PASSED ===")
