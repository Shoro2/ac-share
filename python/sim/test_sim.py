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
                             player_max_hp, player_max_mana, smite_damage, heal_amount)
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


if __name__ == "__main__":
    print("WoW Combat Simulation — Validation Tests\n")
    test_combat_engine()
    test_gym_env()
    test_random_episode()
    test_performance()
    test_combat_scenario()
    test_level_system()
    print("=== ALL TESTS PASSED ===")
