# Training Analysis — 1M Steps Run (2026-03-03)

## Run Configuration

| Parameter | Value |
|---|---|
| Total Timesteps | 1,000,960 |
| Bots | 5 (SubprocVecEnv) |
| n_steps | 256 |
| batch_size | 128 |
| learning_rate | 3e-4 |
| ent_coef | 0.1 |
| gamma | 0.99 |
| FPS | ~681-686 |
| Wall Time | ~1458s (~24 min) |
| Quests | Enabled |
| Creature DB | Enabled (data/) |

## Key Metrics Summary

| Metric | Value | Assessment |
|---|---|---|
| Total Kills | ~35 | Extremely low for 1M steps |
| Total Episodes | ~32 | Very few (episodes are too long) |
| Kills/Episode | 0-2 | Almost no combat |
| Episode Length | ~30,000 | Always hits stall detector limit |
| Deaths | ~1 total | Barely dies (because it barely fights) |
| Final Level | 1 (never leveled) | Zero level-ups |
| ep_reward | -1500 to -1750 | Massively negative |
| ep_damage_dealt | 0-80, mostly 0 | Almost no damage output |
| Quest XP | 40-250 (sporadic) | Only XP source (likely EXPLORE quests) |
| Quests Completed | 0-2 per episode | Some quest interaction |
| Kill/Death Ratio | ~35 | Misleading — 35 kills / 1 death over entire run |
| Areas Explored | 1 per episode | Bot stays in starting area |

## Diagnosis: "Learned Passivity"

The bot has converged to a **passive policy** — it barely fights and runs out the stall detector (30k steps) each episode.

### Root Cause: Step-Penalty Dominates All Positive Rewards

The fundamental reward imbalance:

```
Kill reward:     +35.0  (10.0 + 50 XP * 0.5)
Step penalty:    -0.01/step * 30,000 steps = -300.0
Idle penalty:    -0.05/noop * ~20,000 noops = -1000.0
Total per ep:    ~ -1500 to -1750
```

A single kill (+35) is invisible against accumulated penalties (-1300+). The bot would need ~50 kills per episode just to break even — but the action sequence to complete a kill is 10-30+ steps long.

### Why Combat Doesn't Happen

The kill sequence requires a **long chain of correct actions**:
1. Target mob (Action 4)
2. Walk toward mob (Action 1, many steps)
3. Cast Smite (Action 5) — wait 2s cast time
4. Repeat casting 3-4 times
5. Loot corpse (Action 7)

With gamma=0.99, the +35 kill reward discounted back 30 steps is ~24.0. Meanwhile, 30 steps of penalty = -0.3 to -1.5 (certain). The expected value of attempting combat is barely positive — and the bot can't distinguish "approaching mob to fight" from "wandering randomly."

### Value Function Collapse

Training metrics confirm the policy is frozen:

| Metric | Typical Value | Meaning |
|---|---|---|
| value_loss | 1e-7 to 1e-5 | Value fn predicts constant negative return |
| clip_fraction | 0 to 0.005 | Policy barely changes |
| approx_kl | 0.001-0.003 | Almost no policy updates |
| explained_variance | 0.964 | Suspiciously high — predicts same negative value |
| entropy_loss | -2.45 to -2.46 | Stable but not improving |

### Value Loss Spike at ~985k Steps

At iteration 769, `value_loss` spikes to **74.6** and `explained_variance` drops to **-0.143**. This happens when a bot accidentally gets a kill — the value function was so heavily calibrated to "always -1500" that a sudden positive reward causes a massive prediction error. The system recovers within 2-3 iterations (back to value_loss < 0.01).

## Recommendations

### Priority 1: Reduce Step & Idle Penalties

The penalties accumulate too fast relative to combat rewards.

**Option A (conservative):**
- Step penalty: -0.01 → -0.001 (10x reduction)
- Idle penalty: -0.05 → -0.005 (10x reduction)
- 30k steps now = -30 penalty instead of -300+

**Option B (remove idle penalty):**
- Step penalty: -0.01 → -0.002
- Idle penalty: remove entirely (casting already blocks noop penalty)

### Priority 2: Shorten Stall Detector

30,000 steps without XP is far too long — the bot wastes 99% of training time wandering.

- Reduce from 30,000 → 5,000-8,000 steps
- More episodes = more learning signal
- At 5k limit: ~200 episodes instead of 32

### Priority 3: Add Approach Shaping

The bot needs a gradient toward mobs. The live env (`wow_env.py`) has this:
- Approach reward: `clip(delta_dist * 0.05, -0.2, +0.3)` — reward for getting closer to target
- Target reward: small bonus for having a valid target
- These signals help bridge the gap between "no target" and "kill"

### Priority 4: Gamma Reduction

- gamma=0.99 with 30k-step episodes means the effective horizon is extremely long
- Reduce to 0.95-0.97 for faster credit assignment
- Kill reward (+35) at 30 steps with gamma=0.95: ~7.0 vs gamma=0.99: ~24.0
  (but with shorter episodes, 0.97 is fine)

### Expected Impact

With reduced penalties + shorter stall + approach shaping:
- Episodes should be 1,000-5,000 steps instead of 30,000
- More episodes per 1M steps (~200-1000 instead of 32)
- Bot should discover kill→reward connection within first 50k steps
- Kill rate should rise to 2-10 per episode
- Leveling should begin within 500k steps
