"""
WoW Sim Map Visualization — shows bot routes, kills, level-ups on the Northshire map.

Usage:
    # Full route visualization (runs one episode, saves PNG)
    python -m sim.visualize

    # Live mode: refreshes every N steps
    python -m sim.visualize --live --refresh 50

    # Use a trained model instead of random policy
    python -m sim.visualize --model models/PPO/wow_bot_sim_v1.zip

    # Multiple bots
    python -m sim.visualize --bots 3

    # Custom steps and output
    python -m sim.visualize --steps 2000 --output my_run.png
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass, field

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from sim.combat_sim import SPAWN_POSITIONS, MOB_TEMPLATES
from sim.wow_sim_env import WoWSimEnv


# ─── Color Palette ───────────────────────────────────────────────────────

MOB_COLORS = {
    299: "#8B4513",   # Diseased Young Wolf — brown
    6:   "#6A5ACD",   # Kobold Vermin — slate blue
    69:  "#B22222",   # Diseased Timber Wolf — firebrick
    257: "#4B0082",   # Kobold Worker — indigo
}

MOB_MARKERS = {
    299: "v",   # wolf — triangle down
    6:   "s",   # kobold — square
    69:  "v",   # wolf — triangle down
    257: "D",   # kobold worker — diamond
}

BOT_TRAIL_CMAPS = [
    "cool", "autumn", "spring", "winter", "summer",
    "Wistia", "copper", "YlGn",
]


# ─── Data Recording ─────────────────────────────────────────────────────

@dataclass
class TrailPoint:
    step: int
    x: float
    y: float
    hp_pct: float
    level: int
    in_combat: bool
    orientation: float = 0.0


@dataclass
class MapEvent:
    step: int
    x: float
    y: float
    kind: str       # "kill", "death", "levelup", "loot"
    label: str = ""


@dataclass
class BotRecording:
    name: str
    trail: list = field(default_factory=list)
    events: list = field(default_factory=list)
    final_level: int = 1
    total_kills: int = 0
    total_xp: int = 0
    total_deaths: int = 0


# ─── Run Simulation & Record ────────────────────────────────────────────

def run_episode(env: WoWSimEnv, model=None, max_steps: int = 4000,
                record_interval: int = 1) -> BotRecording:
    """Run one episode, recording trail and events."""
    rec = BotRecording(name=env.bot_name)
    obs, _ = env.reset()
    sim = env.sim
    p = sim.player

    rec.trail.append(TrailPoint(
        step=0, x=p.x, y=p.y,
        hp_pct=p.hp / max(1, p.max_hp),
        level=p.level, in_combat=p.in_combat,
        orientation=p.orientation,
    ))

    prev_kills = 0
    prev_level = p.level

    for step in range(1, max_steps + 1):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        p = sim.player

        # Record trail point
        if step % record_interval == 0:
            rec.trail.append(TrailPoint(
                step=step, x=p.x, y=p.y,
                hp_pct=p.hp / max(1, p.max_hp),
                level=p.level, in_combat=p.in_combat,
                orientation=p.orientation,
            ))

        # Detect kills
        if sim.kills > prev_kills:
            for _ in range(sim.kills - prev_kills):
                rec.events.append(MapEvent(
                    step=step, x=p.x, y=p.y, kind="kill",
                ))
            rec.total_kills += sim.kills - prev_kills
            prev_kills = sim.kills

        # Detect level-ups
        if p.level > prev_level:
            rec.events.append(MapEvent(
                step=step, x=p.x, y=p.y, kind="levelup",
                label=f"Lv{p.level}",
            ))
            prev_level = p.level

        # Detect death
        if terminated and p.hp <= 0:
            rec.events.append(MapEvent(
                step=step, x=p.x, y=p.y, kind="death",
            ))
            rec.total_deaths += 1

        if terminated or truncated:
            break

    rec.final_level = p.level
    rec.total_xp = p.xp
    return rec


# ─── Plotting ────────────────────────────────────────────────────────────

def _get_map_bounds():
    """Compute map boundaries from all spawn positions + player spawn."""
    all_x = [-8921.09]
    all_y = [-119.135]
    for positions in SPAWN_POSITIONS.values():
        for (x, y) in positions:
            all_x.append(x)
            all_y.append(y)
    margin = 30.0
    return (min(all_x) - margin, max(all_x) + margin,
            min(all_y) - margin, max(all_y) + margin)


def plot_map(recordings: list, title: str = "WoW Sim — Bot Routes",
             output: str = None, show: bool = True):
    """Plot the full Northshire map with bot trails and events."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    x_min, x_max, y_min, y_max = _get_map_bounds()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (West → East)", color="white", fontsize=10)
    ax.set_ylabel("Y (South → North)", color="white", fontsize=10)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444")

    # Grid
    ax.grid(True, alpha=0.15, color="white", linewidth=0.5)

    # ─── Mob Spawns ──────────────────────────────────────────────────
    for entry, positions in SPAWN_POSITIONS.items():
        tmpl = MOB_TEMPLATES[entry]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        ax.scatter(xs, ys,
                   c=MOB_COLORS[entry],
                   marker=MOB_MARKERS[entry],
                   s=40, alpha=0.6, edgecolors="white", linewidths=0.3,
                   label=f"{tmpl.name} (L{tmpl.min_level}-{tmpl.max_level})",
                   zorder=2)

    # ─── Player Spawn ────────────────────────────────────────────────
    ax.scatter([-8921.09], [-119.135],
               c="#FFD700", marker="*", s=200, edgecolors="white",
               linewidths=1, zorder=10, label="Spawn Point")

    # ─── Bot Trails ──────────────────────────────────────────────────
    for i, rec in enumerate(recordings):
        if len(rec.trail) < 2:
            continue

        cmap_name = BOT_TRAIL_CMAPS[i % len(BOT_TRAIL_CMAPS)]
        cmap = plt.get_cmap(cmap_name)

        xs = [pt.x for pt in rec.trail]
        ys = [pt.y for pt in rec.trail]
        steps = [pt.step for pt in rec.trail]

        # Draw trail as colored line segments (color = time progression)
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = Normalize(vmin=0, vmax=max(steps) if steps else 1)
        lc = LineCollection(segments, cmap=cmap, norm=norm,
                            linewidths=1.8, alpha=0.85, zorder=5)
        lc.set_array(np.array(steps[:-1]))
        ax.add_collection(lc)

        # Start marker
        ax.scatter([xs[0]], [ys[0]], c="lime", marker="o", s=60,
                   edgecolors="white", linewidths=1, zorder=8)

        # End marker with orientation arrow
        last_pt = rec.trail[-1]
        end_color = "red" if rec.total_deaths > 0 else "cyan"
        ax.scatter([xs[-1]], [ys[-1]], c=end_color,
                   marker="o", s=60, edgecolors="white", linewidths=1, zorder=8)
        # Direction arrow
        arrow_len = 8.0
        ax.annotate("", xy=(xs[-1] + math.cos(last_pt.orientation) * arrow_len,
                            ys[-1] + math.sin(last_pt.orientation) * arrow_len),
                     xytext=(xs[-1], ys[-1]),
                     arrowprops=dict(arrowstyle="->", color=end_color,
                                     lw=2.0, mutation_scale=15),
                     zorder=9)

        # Step markers along trail (every ~500 steps)
        marker_interval = max(1, len(rec.trail) // 8)
        for j in range(marker_interval, len(rec.trail), marker_interval):
            pt = rec.trail[j]
            ax.plot(pt.x, pt.y, ".", color="white", markersize=4, zorder=7)
            ax.annotate(str(pt.step), (pt.x, pt.y),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=6, color="white", alpha=0.7, zorder=7)

        # ─── Events ─────────────────────────────────────────────────
        for ev in rec.events:
            if ev.kind == "kill":
                ax.scatter([ev.x], [ev.y], c="#FF4444", marker="x", s=35,
                           linewidths=1.5, zorder=7, alpha=0.8)
            elif ev.kind == "levelup":
                ax.scatter([ev.x], [ev.y], c="#FFD700", marker="*", s=150,
                           edgecolors="black", linewidths=0.8, zorder=9)
                ax.annotate(ev.label, (ev.x, ev.y),
                            textcoords="offset points", xytext=(6, 6),
                            fontsize=9, fontweight="bold", color="#FFD700",
                            zorder=9)
            elif ev.kind == "death":
                ax.scatter([ev.x], [ev.y], c="red", marker="X", s=120,
                           edgecolors="white", linewidths=1, zorder=9)
                ax.annotate("DEATH", (ev.x, ev.y),
                            textcoords="offset points", xytext=(6, -10),
                            fontsize=8, fontweight="bold", color="red",
                            zorder=9)

    # ─── Legend ──────────────────────────────────────────────────────
    # Build custom legend
    handles, labels = ax.get_legend_handles_labels()

    # Add event markers to legend
    handles.append(plt.Line2D([0], [0], marker="x", color="#FF4444",
                              linestyle="None", markersize=8, label="Kill"))
    handles.append(plt.Line2D([0], [0], marker="*", color="#FFD700",
                              linestyle="None", markersize=12, label="Level-Up"))
    handles.append(plt.Line2D([0], [0], marker="X", color="red",
                              linestyle="None", markersize=10, label="Death"))

    # Stats text box
    stats_lines = []
    for rec in recordings:
        line = (f"{rec.name}: {rec.total_kills} kills, "
                f"Lv{rec.final_level}, {rec.total_xp} XP")
        if rec.total_deaths > 0:
            line += f", {rec.total_deaths} deaths"
        stats_lines.append(line)

    legend = ax.legend(handles=handles, loc="upper left",
                       fontsize=8, facecolor="#1a1a2e", edgecolor="#444",
                       labelcolor="white", framealpha=0.9)
    legend.set_zorder(20)

    # Stats box in lower-right
    if stats_lines:
        stats_text = "\n".join(stats_lines)
        trail_len = len(recordings[0].trail) if recordings else 0
        stats_text += f"\n{trail_len} steps recorded"
        ax.text(0.98, 0.02, stats_text,
                transform=ax.transAxes, fontsize=8,
                verticalalignment="bottom", horizontalalignment="right",
                color="white", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e",
                          edgecolor="#444", alpha=0.9),
                zorder=20)

    plt.tight_layout()
    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Map saved: {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ─── Live Mode ───────────────────────────────────────────────────────────

def run_live(env: WoWSimEnv, model=None, max_steps: int = 4000,
             refresh_every: int = 50, output_dir: str = None):
    """Run simulation with periodic map snapshots saved to disk."""
    if output_dir is None:
        output_dir = os.path.join(PARENT_DIR, "sim_frames")
    os.makedirs(output_dir, exist_ok=True)

    rec = BotRecording(name=env.bot_name)
    obs, _ = env.reset()
    sim = env.sim
    p = sim.player

    rec.trail.append(TrailPoint(
        step=0, x=p.x, y=p.y,
        hp_pct=p.hp / max(1, p.max_hp),
        level=p.level, in_combat=p.in_combat,
        orientation=p.orientation,
    ))

    prev_kills = 0
    prev_level = p.level
    frame_num = 0

    for step in range(1, max_steps + 1):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        p = sim.player

        rec.trail.append(TrailPoint(
            step=step, x=p.x, y=p.y,
            hp_pct=p.hp / max(1, p.max_hp),
            level=p.level, in_combat=p.in_combat,
            orientation=p.orientation,
        ))

        if sim.kills > prev_kills:
            for _ in range(sim.kills - prev_kills):
                rec.events.append(MapEvent(step=step, x=p.x, y=p.y, kind="kill"))
            rec.total_kills += sim.kills - prev_kills
            prev_kills = sim.kills

        if p.level > prev_level:
            rec.events.append(MapEvent(
                step=step, x=p.x, y=p.y, kind="levelup",
                label=f"Lv{p.level}",
            ))
            prev_level = p.level

        if terminated and p.hp <= 0:
            rec.events.append(MapEvent(step=step, x=p.x, y=p.y, kind="death"))
            rec.total_deaths += 1

        # Save frame
        if step % refresh_every == 0 or terminated or truncated:
            rec.final_level = p.level
            rec.total_xp = p.xp
            frame_path = os.path.join(output_dir, f"frame_{frame_num:04d}.png")
            plot_map(
                [rec],
                title=f"Step {step} — Lv{p.level} | {rec.total_kills} kills | HP {p.hp}/{p.max_hp}",
                output=frame_path,
                show=False,
            )
            frame_num += 1
            print(f"  Frame {frame_num}: step={step}, kills={rec.total_kills}, "
                  f"level={p.level}")

        if terminated or truncated:
            break

    print(f"\nDone! {frame_num} frames saved to {output_dir}/")
    print(f"Final: Lv{rec.final_level}, {rec.total_kills} kills, "
          f"{rec.total_xp} XP, {rec.total_deaths} deaths")
    return rec


# ─── Multi-Episode Overlay ───────────────────────────────────────────────

def run_multi_episodes(n_episodes: int = 5, max_steps: int = 4000,
                       model=None, seed: int = 42,
                       output: str = None):
    """Run multiple episodes and overlay all routes on one map."""
    recordings = []
    for i in range(n_episodes):
        env = WoWSimEnv(bot_name=f"Bot{i}", seed=seed + i * 1000)
        rec = run_episode(env, model=model, max_steps=max_steps)
        recordings.append(rec)
        print(f"  Episode {i}: Lv{rec.final_level}, {rec.total_kills} kills, "
              f"{rec.total_xp} XP, {rec.total_deaths} deaths, "
              f"{len(rec.trail)} steps")
        env.close()

    output = output or os.path.join(PARENT_DIR, "sim_map.png")
    plot_map(recordings,
             title=f"WoW Sim — {n_episodes} Episodes Overlay",
             output=output, show=False)
    return recordings


# ─── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize WoW Sim bot routes on the Northshire map")
    parser.add_argument("--live", action="store_true",
                        help="Live mode: save frame snapshots every --refresh steps")
    parser.add_argument("--refresh", type=int, default=50,
                        help="Steps between frame saves in live mode (default: 50)")
    parser.add_argument("--steps", type=int, default=4000,
                        help="Max steps per episode (default: 4000)")
    parser.add_argument("--bots", type=int, default=5,
                        help="Number of episodes/bots to overlay (default: 5)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (.zip) for policy")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: python/sim_map.png)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--frames-dir", type=str, default=None,
                        help="Output directory for live mode frames")
    args = parser.parse_args()

    # Load model if specified
    model = None
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
        print(f"Loaded model: {args.model}")

    if args.live:
        env = WoWSimEnv(bot_name="LiveBot", seed=args.seed)
        run_live(env, model=model, max_steps=args.steps,
                 refresh_every=args.refresh, output_dir=args.frames_dir)
        env.close()
    else:
        run_multi_episodes(
            n_episodes=args.bots,
            max_steps=args.steps,
            model=model,
            seed=args.seed,
            output=args.output,
        )


if __name__ == "__main__":
    main()
