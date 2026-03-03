"""
WoW Sim Interactive Map Viewer — episode browser with zoom, filters, and log.

Opens an interactive window at startup.  Use the episode slider to browse
through episodes, checkboxes to filter bots, scroll-wheel to zoom, and
right-click-drag to pan.  A separate log window can be toggled on/off.

Usage:
    # Interactive viewer from training logs (primary mode)
    python -m sim.visualize --log-dir logs/sim_episodes/

    # Filter by bot / limit episodes
    python -m sim.visualize --log-dir logs/sim_episodes/ --bot SimBot0 --last 3

    # Fallback: run simulation directly (quick ad-hoc test)
    python -m sim.visualize --run --steps 2000 --bots 3

    # With trained model (--run mode)
    python -m sim.visualize --run --model models/PPO/wow_bot_sim_v1.zip

    # Static PNG export (no window)
    python -m sim.visualize --log-dir logs/sim_episodes/ --output my_map.png
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
from matplotlib.widgets import Slider, CheckButtons, Button

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from sim.combat_sim import SPAWN_POSITIONS, MOB_TEMPLATES


# ─── Color Palette ───────────────────────────────────────────────────────

# Known mob colors (Northshire hardcoded entries)
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

# Fallback palette for unknown mob entries (creature_db mode)
_EXTRA_COLORS = [
    "#2E8B57", "#DC143C", "#1E90FF", "#FF8C00", "#9932CC",
    "#00CED1", "#FF69B4", "#32CD32", "#BA55D3", "#FF6347",
]
_EXTRA_MARKERS = ["o", "s", "v", "D", "^", "p", "h", "*"]

BOT_TRAIL_CMAPS = [
    "cool", "autumn", "spring", "winter", "summer",
    "Wistia", "copper", "YlGn",
]

# Spawn point (WoW coordinates)
SPAWN_X, SPAWN_Y = -8921.09, -119.135


# ─── Coordinate Rotation ────────────────────────────────────────────────
# Rotate 90° counter-clockwise: (x, y) → (-y, x)

def _rot_xy(x, y):
    """Rotate a single WoW (x, y) coordinate 90° CCW for display."""
    return (-y, x)


def _rot_xs_ys(xs, ys):
    """Rotate lists of WoW coordinates 90° CCW for display."""
    return ([-y for y in ys], list(xs))


# ─── Data structures ────────────────────────────────────────────────────

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
class MobSnapshot:
    entry: int
    name: str
    x: float
    y: float
    level: int
    alive: bool = True


@dataclass
class BotRecording:
    name: str
    episode: int = 0
    trail: list = field(default_factory=list)
    events: list = field(default_factory=list)
    mob_snapshots: list = field(default_factory=list)
    final_level: int = 1
    total_kills: int = 0
    total_xp: int = 0
    total_deaths: int = 0


# ─── Load from log files ────────────────────────────────────────────────

def load_recordings_from_logs(log_dir: str, bot_name: str = None,
                              last_n: int = None) -> list:
    """Load BotRecording objects from JSONL episode logs."""
    from sim.sim_logger import load_episodes
    episodes = load_episodes(log_dir, bot_name=bot_name, last_n=last_n)

    recordings = []
    for ep in episodes:
        rec = BotRecording(name=ep.get("bot", "?"))
        rec.episode = ep.get("episode", 0)
        rec.final_level = ep.get("final_level", 1)
        rec.total_kills = ep.get("kills", 0)
        rec.total_xp = ep.get("xp", 0)
        rec.total_deaths = ep.get("deaths", 0)

        # Trail: each entry is [step, x, y, hp_pct, level, in_combat, orientation]
        for pt in ep.get("trail", []):
            rec.trail.append(TrailPoint(
                step=int(pt[0]),
                x=float(pt[1]),
                y=float(pt[2]),
                hp_pct=float(pt[3]),
                level=int(pt[4]),
                in_combat=bool(pt[5]),
                orientation=float(pt[6]),
            ))

        # Events
        for ev in ep.get("events", []):
            rec.events.append(MapEvent(
                step=ev["s"],
                x=ev["x"],
                y=ev["y"],
                kind=ev["k"],
                label=ev.get("l", ""),
            ))

        # Mobs
        for m in ep.get("mobs", []):
            rec.mob_snapshots.append(MobSnapshot(
                entry=m["e"],
                name=m["n"],
                x=m["x"],
                y=m["y"],
                level=m["lv"],
            ))

        recordings.append(rec)

    return recordings


# ─── Snapshot mobs from live sim ─────────────────────────────────────────

def _snapshot_mobs(sim) -> list:
    """Take a snapshot of all mobs currently in the simulation."""
    snaps = []
    for mob in sim.mobs:
        snaps.append(MobSnapshot(
            entry=mob.template.entry,
            name=mob.template.name,
            x=mob.spawn_x,
            y=mob.spawn_y,
            level=mob.level,
            alive=mob.alive,
        ))
    return snaps


# ─── Run Simulation & Record (--run fallback mode) ───────────────────────

def run_episode(env, model=None, max_steps: int = 4000,
                record_interval: int = 1,
                episode_num: int = 0) -> BotRecording:
    """Run one episode, recording trail and events."""
    rec = BotRecording(name=env.bot_name, episode=episode_num)
    obs, _ = env.reset()
    sim = env.sim
    p = sim.player

    rec.mob_snapshots = _snapshot_mobs(sim)

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

        if step % record_interval == 0:
            rec.trail.append(TrailPoint(
                step=step, x=p.x, y=p.y,
                hp_pct=p.hp / max(1, p.max_hp),
                level=p.level, in_combat=p.in_combat,
                orientation=p.orientation,
            ))

        if sim.creature_db and step % 200 == 0:
            existing_positions = {(s.x, s.y) for s in rec.mob_snapshots}
            for snap in _snapshot_mobs(sim):
                if (snap.x, snap.y) not in existing_positions:
                    rec.mob_snapshots.append(snap)
                    existing_positions.add((snap.x, snap.y))

        if sim.kills > prev_kills:
            for _ in range(sim.kills - prev_kills):
                rec.events.append(MapEvent(
                    step=step, x=p.x, y=p.y, kind="kill",
                ))
            rec.total_kills += sim.kills - prev_kills
            prev_kills = sim.kills

        if p.level > prev_level:
            rec.events.append(MapEvent(
                step=step, x=p.x, y=p.y, kind="levelup",
                label=f"Lv{p.level}",
            ))
            prev_level = p.level

        if terminated and p.hp <= 0:
            rec.events.append(MapEvent(
                step=step, x=p.x, y=p.y, kind="death",
            ))
            rec.total_deaths += 1

        if terminated or truncated:
            break

    if sim.creature_db:
        existing_positions = {(s.x, s.y) for s in rec.mob_snapshots}
        for snap in _snapshot_mobs(sim):
            if (snap.x, snap.y) not in existing_positions:
                rec.mob_snapshots.append(snap)

    rec.final_level = p.level
    rec.total_xp = p.xp
    return rec


# ─── Plotting Helpers ────────────────────────────────────────────────────

def _get_map_bounds(recordings: list):
    """Compute map boundaries from trails, mob snapshots, and hardcoded spawns.

    Returns bounds in ROTATED coordinates (display space).
    """
    sx, sy = _rot_xy(SPAWN_X, SPAWN_Y)
    all_x = [sx]
    all_y = [sy]

    for rec in recordings:
        for pt in rec.trail:
            rx, ry = _rot_xy(pt.x, pt.y)
            all_x.append(rx)
            all_y.append(ry)
        for snap in rec.mob_snapshots:
            rx, ry = _rot_xy(snap.x, snap.y)
            all_x.append(rx)
            all_y.append(ry)

    has_snapshots = any(len(rec.mob_snapshots) > 0 for rec in recordings)
    if not has_snapshots:
        for positions in SPAWN_POSITIONS.values():
            for (x, y) in positions:
                rx, ry = _rot_xy(x, y)
                all_x.append(rx)
                all_y.append(ry)

    margin = 30.0
    return (min(all_x) - margin, max(all_x) + margin,
            min(all_y) - margin, max(all_y) + margin)


def _plot_mobs_from_snapshots(ax, all_snapshots: list):
    """Plot mob positions from snapshots, grouping by entry for legend."""
    by_entry = {}
    for snap in all_snapshots:
        by_entry.setdefault(snap.entry, []).append(snap)

    extra_idx = 0
    plotted_entries = set()

    for entry, snaps in sorted(by_entry.items()):
        if entry in plotted_entries:
            continue
        plotted_entries.add(entry)

        # Rotate mob coordinates
        plot_xs, plot_ys = _rot_xs_ys(
            [s.x for s in snaps], [s.y for s in snaps])
        name = snaps[0].name
        levels = sorted(set(s.level for s in snaps))
        lvl_str = f"L{levels[0]}" if len(levels) == 1 else f"L{levels[0]}-{levels[-1]}"

        color = MOB_COLORS.get(entry, _EXTRA_COLORS[extra_idx % len(_EXTRA_COLORS)])
        marker = MOB_MARKERS.get(entry, _EXTRA_MARKERS[extra_idx % len(_EXTRA_MARKERS)])
        if entry not in MOB_COLORS:
            extra_idx += 1

        ax.scatter(plot_xs, plot_ys, c=color, marker=marker,
                   s=40, alpha=0.6, edgecolors="white", linewidths=0.3,
                   label=f"{name} ({lvl_str})",
                   zorder=2)


def _plot_mobs_from_hardcoded(ax):
    """Plot mob positions from hardcoded SPAWN_POSITIONS."""
    for entry, positions in SPAWN_POSITIONS.items():
        tmpl = MOB_TEMPLATES[entry]
        plot_xs, plot_ys = _rot_xs_ys(
            [p[0] for p in positions], [p[1] for p in positions])
        ax.scatter(plot_xs, plot_ys,
                   c=MOB_COLORS[entry],
                   marker=MOB_MARKERS[entry],
                   s=40, alpha=0.6, edgecolors="white", linewidths=0.3,
                   label=f"{tmpl.name} (L{tmpl.min_level}-{tmpl.max_level})",
                   zorder=2)


def _draw_trail(ax, rec, color_idx):
    """Draw a single bot's trail and events onto *ax*."""
    if len(rec.trail) < 2:
        return

    cmap_name = BOT_TRAIL_CMAPS[color_idx % len(BOT_TRAIL_CMAPS)]
    cmap = plt.get_cmap(cmap_name)

    plot_xs, plot_ys = _rot_xs_ys(
        [pt.x for pt in rec.trail], [pt.y for pt in rec.trail])
    steps = [pt.step for pt in rec.trail]

    # Line segments coloured by time progression
    points = np.array([plot_xs, plot_ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=0, vmax=max(steps) if steps else 1)
    lc = LineCollection(segments, cmap=cmap, norm=norm,
                        linewidths=1.8, alpha=0.85, zorder=5)
    lc.set_array(np.array(steps[:-1]))
    ax.add_collection(lc)

    # Start marker
    ax.scatter([plot_xs[0]], [plot_ys[0]], c="lime", marker="o", s=60,
               edgecolors="white", linewidths=1, zorder=8)

    # End marker with orientation arrow
    last_pt = rec.trail[-1]
    end_color = "red" if rec.total_deaths > 0 else "cyan"
    ax.scatter([plot_xs[-1]], [plot_ys[-1]], c=end_color,
               marker="o", s=60, edgecolors="white", linewidths=1, zorder=8)
    arrow_len = 8.0
    rot_angle = last_pt.orientation + math.pi / 2
    ax.annotate("", xy=(plot_xs[-1] + math.cos(rot_angle) * arrow_len,
                        plot_ys[-1] + math.sin(rot_angle) * arrow_len),
                 xytext=(plot_xs[-1], plot_ys[-1]),
                 arrowprops=dict(arrowstyle="->", color=end_color,
                                 lw=2.0, mutation_scale=15),
                 zorder=9)

    # Bot name label at start
    ax.annotate(rec.name, (plot_xs[0], plot_ys[0]),
                textcoords="offset points", xytext=(6, 6),
                fontsize=8, fontweight="bold", color="lime", zorder=9)

    # Step markers along trail (every ~500 steps)
    marker_interval = max(1, len(rec.trail) // 8)
    for j in range(marker_interval, len(rec.trail), marker_interval):
        pt = rec.trail[j]
        rx, ry = _rot_xy(pt.x, pt.y)
        ax.plot(rx, ry, ".", color="white", markersize=4, zorder=7)
        ax.annotate(str(pt.step), (rx, ry),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=6, color="white", alpha=0.7, zorder=7)

    # Events
    for ev in rec.events:
        ex, ey = _rot_xy(ev.x, ev.y)
        if ev.kind == "kill":
            ax.scatter([ex], [ey], c="#FF4444", marker="x", s=35,
                       linewidths=1.5, zorder=7, alpha=0.8)
        elif ev.kind == "levelup":
            ax.scatter([ex], [ey], c="#FFD700", marker="*", s=150,
                       edgecolors="black", linewidths=0.8, zorder=9)
            ax.annotate(ev.label, (ex, ey),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=9, fontweight="bold", color="#FFD700",
                        zorder=9)
        elif ev.kind == "death":
            ax.scatter([ex], [ey], c="red", marker="X", s=120,
                       edgecolors="white", linewidths=1, zorder=9)
            ax.annotate("DEATH", (ex, ey),
                        textcoords="offset points", xytext=(6, -10),
                        fontsize=8, fontweight="bold", color="red",
                        zorder=9)


# ─── Static plot (PNG export) ────────────────────────────────────────────

def plot_map(recordings: list, title: str = "WoW Sim — Bot Routes",
             output: str = None, show: bool = True):
    """Plot the full map with bot trails and events (rotated 90° CCW)."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    x_min, x_max, y_min, y_max = _get_map_bounds(recordings)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("Y (East \u2192 West)", color="white", fontsize=10)
    ax.set_ylabel("X (South \u2192 North)", color="white", fontsize=10)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444")

    ax.grid(True, alpha=0.15, color="white", linewidth=0.5)

    # Mob spawns
    all_snapshots = []
    for rec in recordings:
        all_snapshots.extend(rec.mob_snapshots)

    if all_snapshots:
        seen = set()
        unique = []
        for snap in all_snapshots:
            key = (snap.entry, round(snap.x, 1), round(snap.y, 1))
            if key not in seen:
                seen.add(key)
                unique.append(snap)
        _plot_mobs_from_snapshots(ax, unique)
    else:
        _plot_mobs_from_hardcoded(ax)

    # Player spawn
    sp_rx, sp_ry = _rot_xy(SPAWN_X, SPAWN_Y)
    ax.scatter([sp_rx], [sp_ry],
               c="#FFD700", marker="*", s=200, edgecolors="white",
               linewidths=1, zorder=10, label="Spawn Point")

    # Bot trails
    for i, rec in enumerate(recordings):
        _draw_trail(ax, rec, i)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], marker="x", color="#FF4444",
                              linestyle="None", markersize=8, label="Kill"))
    handles.append(plt.Line2D([0], [0], marker="*", color="#FFD700",
                              linestyle="None", markersize=12, label="Level-Up"))
    handles.append(plt.Line2D([0], [0], marker="X", color="red",
                              linestyle="None", markersize=10, label="Death"))

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


# ─── Interactive Viewer ──────────────────────────────────────────────────

class InteractiveViewer:
    """Interactive map viewer with episode slider, bot filters, zoom, and
    a toggleable log window."""

    def __init__(self, recordings, title="WoW Sim \u2014 Interactive Map"):
        self.all_recordings = recordings
        self.title = title

        # Extract unique bots and group by episode
        self.bot_names = sorted(set(r.name for r in recordings))
        self._group_by_episode()

        # View state
        self.current_ep_idx = 0
        self.visible_bots = set(self.bot_names)
        self._first_draw = True

        # Log window (separate figure)
        self._log_fig = None

        # Pan state
        self._panning = False
        self._pan_start_px = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None

        # Batch-update flag (suppresses redraws during All/None)
        self._batch_update = False

        # Compute global bounds (across ALL episodes for stable zoom ref)
        self._global_bounds = _get_map_bounds(recordings)

        # Build UI
        self._setup_figure()
        self._draw_map()

    # ── Data grouping ────────────────────────────────────────────────

    def _group_by_episode(self):
        by_ep = {}
        for rec in self.all_recordings:
            by_ep.setdefault(rec.episode, []).append(rec)
        self.episode_keys = sorted(by_ep.keys())
        self.episodes = {k: by_ep[k] for k in self.episode_keys}
        self.num_episodes = max(len(self.episode_keys), 1)

    # ── Figure setup ─────────────────────────────────────────────────

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.set_facecolor("#1a1a2e")
        self.fig.canvas.manager.set_window_title(self.title)

        # ── Main map axes ────────────────────────────────────────────
        self.ax_map = self.fig.add_axes([0.05, 0.08, 0.72, 0.82])
        self.ax_map.set_facecolor("#16213e")

        # ── Episode slider (top) ─────────────────────────────────────
        ax_ep = self.fig.add_axes([0.15, 0.94, 0.55, 0.03])
        ax_ep.set_facecolor("#2a2a4e")
        max_ep = max(self.num_episodes - 1, 0)
        self.ep_slider = Slider(
            ax_ep, "Episode", 0, max(max_ep, 1),
            valinit=0, valstep=1, color="#4a9eff",
        )
        self.ep_slider.label.set_color("white")
        self.ep_slider.valtext.set_color("white")
        self.ep_slider.on_changed(self._on_episode_changed)

        # ── Zoom slider (bottom) ─────────────────────────────────────
        ax_zoom = self.fig.add_axes([0.15, 0.02, 0.55, 0.03])
        ax_zoom.set_facecolor("#2a2a4e")
        self.zoom_slider = Slider(
            ax_zoom, "Zoom", 0.1, 10.0,
            valinit=1.0, color="#4aff9e",
        )
        self.zoom_slider.label.set_color("white")
        self.zoom_slider.valtext.set_color("white")
        self.zoom_slider.on_changed(self._on_zoom_changed)

        # ── Bot filter checkboxes (right side) ───────────────────────
        n_bots = len(self.bot_names)
        check_h = min(max(0.045 * n_bots, 0.10), 0.60)
        check_top = 0.84
        ax_check = self.fig.add_axes(
            [0.80, check_top - check_h, 0.18, check_h])
        ax_check.set_facecolor("#1a1a2e")
        for spine in ax_check.spines.values():
            spine.set_color("#444")
        self.check_buttons = CheckButtons(
            ax_check, self.bot_names, [True] * n_bots)
        for lbl in self.check_buttons.labels:
            lbl.set_color("white")
            lbl.set_fontsize(9)
        self.check_buttons.on_clicked(self._on_bot_toggled)

        # ── All / None buttons ───────────────────────────────────────
        btn_y = check_top + 0.01
        ax_all = self.fig.add_axes([0.80, btn_y, 0.08, 0.03])
        self.btn_all = Button(ax_all, "All",
                              color="#2a2a4e", hovercolor="#3a3a5e")
        self.btn_all.label.set_color("white")
        self.btn_all.label.set_fontsize(8)
        self.btn_all.on_clicked(self._select_all_bots)

        ax_none = self.fig.add_axes([0.89, btn_y, 0.08, 0.03])
        self.btn_none = Button(ax_none, "None",
                               color="#2a2a4e", hovercolor="#3a3a5e")
        self.btn_none.label.set_color("white")
        self.btn_none.label.set_fontsize(8)
        self.btn_none.on_clicked(self._select_no_bots)

        # ── Log toggle + Reset view buttons ──────────────────────────
        btn_bottom = check_top - check_h - 0.06
        ax_log = self.fig.add_axes([0.80, btn_bottom, 0.18, 0.04])
        self.btn_log = Button(ax_log, "Show Log",
                              color="#2a2a4e", hovercolor="#3a3a5e")
        self.btn_log.label.set_color("white")
        self.btn_log.on_clicked(self._toggle_log)

        ax_reset = self.fig.add_axes([0.80, btn_bottom - 0.05, 0.18, 0.04])
        self.btn_reset = Button(ax_reset, "Reset View",
                                color="#2a2a4e", hovercolor="#3a3a5e")
        self.btn_reset.label.set_color("white")
        self.btn_reset.on_clicked(self._reset_view)

        # ── Mouse / keyboard events ──────────────────────────────────
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ── Callbacks ────────────────────────────────────────────────────

    def _on_episode_changed(self, val):
        self.current_ep_idx = int(val)
        self._draw_map()

    def _on_zoom_changed(self, val):
        """Zoom slider: scale view around current view centre."""
        xlim = self.ax_map.get_xlim()
        ylim = self.ax_map.get_ylim()
        cx = (xlim[0] + xlim[1]) / 2
        cy = (ylim[0] + ylim[1]) / 2

        gb = self._global_bounds
        hw = (gb[1] - gb[0]) / 2 * val
        hh = (gb[3] - gb[2]) / 2 * val
        self.ax_map.set_xlim(cx - hw, cx + hw)
        self.ax_map.set_ylim(cy - hh, cy + hh)
        self.fig.canvas.draw_idle()

    def _on_bot_toggled(self, label):
        if label in self.visible_bots:
            self.visible_bots.remove(label)
        else:
            self.visible_bots.add(label)
        if not self._batch_update:
            self._draw_map()

    def _select_all_bots(self, _event):
        self._batch_update = True
        for i, name in enumerate(self.bot_names):
            if name not in self.visible_bots:
                self.check_buttons.set_active(i)
        self._batch_update = False
        self._draw_map()

    def _select_no_bots(self, _event):
        self._batch_update = True
        for i, name in enumerate(self.bot_names):
            if name in self.visible_bots:
                self.check_buttons.set_active(i)
        self._batch_update = False
        self._draw_map()

    def _reset_view(self, _event):
        """Reset zoom/pan to fit current episode data."""
        self.zoom_slider.set_val(1.0)
        ep_recs = self._current_visible_recordings()
        bounds = _get_map_bounds(ep_recs if ep_recs else self.all_recordings)
        self.ax_map.set_xlim(bounds[0], bounds[1])
        self.ax_map.set_ylim(bounds[2], bounds[3])
        self.fig.canvas.draw_idle()

    # ── Scroll-wheel zoom ────────────────────────────────────────────

    def _on_scroll(self, event):
        if event.inaxes != self.ax_map:
            return
        if event.xdata is None or event.ydata is None:
            return
        factor = 0.8 if event.button == "up" else 1.25
        xlim = self.ax_map.get_xlim()
        ylim = self.ax_map.get_ylim()
        mx, my = event.xdata, event.ydata
        self.ax_map.set_xlim(
            mx + (xlim[0] - mx) * factor,
            mx + (xlim[1] - mx) * factor)
        self.ax_map.set_ylim(
            my + (ylim[0] - my) * factor,
            my + (ylim[1] - my) * factor)
        self.fig.canvas.draw_idle()

    # ── Right-click pan ──────────────────────────────────────────────

    def _on_mouse_press(self, event):
        if event.inaxes != self.ax_map or event.button != 3:
            return
        self._panning = True
        self._pan_start_px = (event.x, event.y)
        self._pan_start_xlim = self.ax_map.get_xlim()
        self._pan_start_ylim = self.ax_map.get_ylim()

    def _on_mouse_release(self, event):
        if event.button == 3:
            self._panning = False

    def _on_mouse_move(self, event):
        if not self._panning or event.x is None:
            return
        dx_px = event.x - self._pan_start_px[0]
        dy_px = event.y - self._pan_start_px[1]
        bbox = self.ax_map.get_window_extent()
        xlim = self._pan_start_xlim
        ylim = self._pan_start_ylim
        dx = -(xlim[1] - xlim[0]) * dx_px / bbox.width
        dy = -(ylim[1] - ylim[0]) * dy_px / bbox.height
        self.ax_map.set_xlim(xlim[0] + dx, xlim[1] + dx)
        self.ax_map.set_ylim(ylim[0] + dy, ylim[1] + dy)
        self.fig.canvas.draw_idle()

    # ── Keyboard shortcuts ───────────────────────────────────────────

    def _on_key(self, event):
        if event.key == "right":
            new = min(self.current_ep_idx + 1, self.num_episodes - 1)
            self.ep_slider.set_val(new)
        elif event.key == "left":
            new = max(self.current_ep_idx - 1, 0)
            self.ep_slider.set_val(new)
        elif event.key == "r":
            self._reset_view(None)

    # ── Log window ───────────────────────────────────────────────────

    def _toggle_log(self, _event):
        if self._log_fig is not None and plt.fignum_exists(self._log_fig.number):
            plt.close(self._log_fig)
            self._log_fig = None
            self.btn_log.label.set_text("Show Log")
            self.fig.canvas.draw_idle()
        else:
            self._open_log_window()

    def _open_log_window(self):
        self._log_fig = plt.figure(figsize=(7, 10))
        self._log_fig.set_facecolor("#1a1a2e")
        self._log_fig.canvas.manager.set_window_title("Episode Log")
        self._refresh_log_content()
        self.btn_log.label.set_text("Hide Log")
        self._log_fig.show()
        self.fig.canvas.draw_idle()

    def _refresh_log_content(self):
        if self._log_fig is None:
            return
        self._log_fig.clear()
        ax = self._log_fig.add_axes([0.04, 0.02, 0.92, 0.96])
        ax.set_facecolor("#16213e")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#444")
        text = self._build_log_text()
        ax.text(0.02, 0.98, text,
                transform=ax.transAxes, fontsize=9,
                fontfamily="monospace", color="white",
                verticalalignment="top", linespacing=1.4)
        self._log_fig.canvas.draw_idle()

    def _build_log_text(self):
        if not self.episode_keys:
            return "No episodes loaded."
        if self.current_ep_idx >= len(self.episode_keys):
            return "No episode selected."

        ep_key = self.episode_keys[self.current_ep_idx]
        recs = self.episodes.get(ep_key, [])
        visible = [r for r in recs if r.name in self.visible_bots]

        lines = [f"\u2550\u2550\u2550 Episode {ep_key} \u2550\u2550\u2550", ""]
        for rec in visible:
            lines.append(f"\u2500\u2500 {rec.name} \u2500\u2500")
            lines.append(
                f"  Level: {rec.final_level}  |  Kills: {rec.total_kills}"
                f"  |  XP: {rec.total_xp}  |  Deaths: {rec.total_deaths}")
            steps = rec.trail[-1].step if rec.trail else 0
            lines.append(
                f"  Trail points: {len(rec.trail)}  |  Steps: {steps}")
            lines.append("")

            for ev in rec.events:
                tag = ev.kind.upper()
                extra = f" \u2192 {ev.label}" if ev.label else ""
                lines.append(
                    f"  [Step {ev.step:>5}] {tag}{extra}"
                    f"  ({ev.x:.0f}, {ev.y:.0f})")
            lines.append("")

        if not visible:
            lines.append("  (no visible bots for this episode)")
        return "\n".join(lines)

    # ── Helpers ──────────────────────────────────────────────────────

    def _current_visible_recordings(self):
        if not self.episode_keys:
            return []
        if self.current_ep_idx >= len(self.episode_keys):
            return []
        ep_key = self.episode_keys[self.current_ep_idx]
        return [r for r in self.episodes.get(ep_key, [])
                if r.name in self.visible_bots]

    # ── Main draw ────────────────────────────────────────────────────

    def _draw_map(self):
        """Redraw the map for the current episode / bot selection."""
        # Preserve zoom/pan across redraws
        if self._first_draw:
            saved_xlim = saved_ylim = None
        else:
            saved_xlim = self.ax_map.get_xlim()
            saved_ylim = self.ax_map.get_ylim()

        self.ax_map.clear()
        self.ax_map.set_facecolor("#16213e")

        visible = self._current_visible_recordings()

        # Axis limits
        if saved_xlim is not None:
            self.ax_map.set_xlim(saved_xlim)
            self.ax_map.set_ylim(saved_ylim)
        else:
            bounds = _get_map_bounds(
                visible if visible else self.all_recordings)
            self.ax_map.set_xlim(bounds[0], bounds[1])
            self.ax_map.set_ylim(bounds[2], bounds[3])
            self._first_draw = False

        self.ax_map.set_aspect("equal")

        # Title
        if self.episode_keys:
            ep_key = self.episode_keys[self.current_ep_idx]
            ep_label = f"Episode {ep_key}"
        else:
            ep_label = "No Data"
        n_vis = len(visible)
        self.ax_map.set_title(
            f"{self.title}  \u2014  {ep_label}  ({n_vis} bot"
            f"{'s' if n_vis != 1 else ''})",
            color="white", fontsize=13, fontweight="bold", pad=8)
        self.ax_map.set_xlabel(
            "Y (East \u2192 West)", color="white", fontsize=10)
        self.ax_map.set_ylabel(
            "X (South \u2192 North)", color="white", fontsize=10)
        self.ax_map.tick_params(colors="white", labelsize=8)
        for spine in self.ax_map.spines.values():
            spine.set_color("#444")
        self.ax_map.grid(True, alpha=0.15, color="white", linewidth=0.5)

        # ── Mob spawns ───────────────────────────────────────────────
        all_snaps = []
        for rec in visible:
            all_snaps.extend(rec.mob_snapshots)
        if all_snaps:
            seen = set()
            unique = []
            for snap in all_snaps:
                key = (snap.entry, round(snap.x, 1), round(snap.y, 1))
                if key not in seen:
                    seen.add(key)
                    unique.append(snap)
            _plot_mobs_from_snapshots(self.ax_map, unique)
        else:
            _plot_mobs_from_hardcoded(self.ax_map)

        # ── Player spawn marker ──────────────────────────────────────
        sp_rx, sp_ry = _rot_xy(SPAWN_X, SPAWN_Y)
        self.ax_map.scatter(
            [sp_rx], [sp_ry], c="#FFD700", marker="*", s=200,
            edgecolors="white", linewidths=1, zorder=10,
            label="Spawn Point")

        # ── Bot trails ───────────────────────────────────────────────
        for i, rec in enumerate(visible):
            _draw_trail(self.ax_map, rec, i)

        # ── Legend ───────────────────────────────────────────────────
        handles, _ = self.ax_map.get_legend_handles_labels()
        handles.append(plt.Line2D(
            [0], [0], marker="x", color="#FF4444",
            linestyle="None", markersize=8, label="Kill"))
        handles.append(plt.Line2D(
            [0], [0], marker="*", color="#FFD700",
            linestyle="None", markersize=12, label="Level-Up"))
        handles.append(plt.Line2D(
            [0], [0], marker="X", color="red",
            linestyle="None", markersize=10, label="Death"))
        legend = self.ax_map.legend(
            handles=handles, loc="upper left", fontsize=8,
            facecolor="#1a1a2e", edgecolor="#444",
            labelcolor="white", framealpha=0.9)
        legend.set_zorder(20)

        # ── Update log window if open ────────────────────────────────
        if (self._log_fig is not None
                and plt.fignum_exists(self._log_fig.number)):
            self._refresh_log_content()

        self.fig.canvas.draw_idle()

    # ── Public API ───────────────────────────────────────────────────

    def show(self):
        """Display the interactive viewer (blocking)."""
        plt.show()


# ─── Multi-Episode run (--run fallback) ──────────────────────────────────

def run_multi_episodes(n_episodes: int = 5, max_steps: int = 4000,
                       model=None, seed: int = 42,
                       output: str = None,
                       data_root: str = None,
                       creature_csv_dir: str = None,
                       show: bool = True):
    """Run multiple episodes and overlay all routes on one map."""
    from sim.wow_sim_env import WoWSimEnv

    output = output or os.path.join(PARENT_DIR, "sim_map.png")
    print(f">>> Running {n_episodes} episodes, {max_steps} steps each <<<")

    recordings = []
    for i in range(n_episodes):
        print(f"  Episode {i}/{n_episodes-1}...", end=" ", flush=True)
        env = WoWSimEnv(bot_name=f"Bot{i}", seed=seed + i * 1000,
                        data_root=data_root, creature_csv_dir=creature_csv_dir)
        rec = run_episode(env, model=model, max_steps=max_steps,
                          episode_num=i)
        recordings.append(rec)
        print(f"Lv{rec.final_level}, {rec.total_kills} kills, "
              f"{rec.total_xp} XP, {rec.total_deaths} deaths, "
              f"{len(rec.trail)} steps")
        env.close()

    print(f">>> All episodes done. <<<")

    if output and not show:
        # Static PNG only
        plot_map(recordings,
                 title=f"WoW Sim \u2014 {n_episodes} Episodes Overlay",
                 output=output, show=False)
    else:
        # Interactive viewer
        if output:
            plot_map(recordings,
                     title=f"WoW Sim \u2014 {n_episodes} Episodes Overlay",
                     output=output, show=False)
            print(f"Map saved: {output}")
        viewer = InteractiveViewer(
            recordings,
            title=f"WoW Sim \u2014 {n_episodes} Episodes")
        viewer.show()

    return recordings


# ─── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive WoW Sim map viewer.\n"
                    "Primary mode: reads from log files (--log-dir).\n"
                    "Fallback mode: runs sim directly (--run).")

    # Primary mode: read from logs
    parser.add_argument("--log-dir", type=str,
                        default=r"C:\wowstuff\WoWKI_serv\python\sim_episodes",
                        help="Path to episode log directory (JSONL files)")
    parser.add_argument("--bot", type=str, default=None,
                        help="Show only this bot's episodes (default: all)")
    parser.add_argument("--last", type=int, default=None,
                        help="Show only the last N episodes per bot")

    # Fallback mode: run sim directly
    parser.add_argument("--run", action="store_true",
                        help="Run simulation directly instead of reading logs")
    parser.add_argument("--steps", type=int, default=4000,
                        help="Max steps per episode (--run mode, default: 4000)")
    parser.add_argument("--bots", type=int, default=5,
                        help="Number of episodes/bots (--run mode, default: 5)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (.zip)")
    parser.add_argument("--data-root", type=str,
                        default=r"C:\wowstuff\WoWKI_serv\Data",
                        help="Path to WoW Data/ directory for 3D terrain")
    parser.add_argument("--creature-data", type=str,
                        default=r"C:\wowstuff\WoWKI_serv\python\dbexport",
                        help="Path to creature CSV directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Shared options
    parser.add_argument("--output", type=str, default=None,
                        help="Also save a static PNG to this path")
    parser.add_argument("--no-show", action="store_true",
                        help="Only save PNG, do not open interactive window")
    args = parser.parse_args()

    # Load model if specified
    model = None
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
        print(f"Loaded model: {args.model}")

    if args.run:
        # ─── Fallback mode: run simulation directly ─────────────────
        if args.data_root:
            print(f"3D terrain enabled: {args.data_root}")
        if args.creature_data:
            print(f"Full-world creatures enabled: {args.creature_data}")

        run_multi_episodes(
            n_episodes=args.bots,
            max_steps=args.steps,
            model=model,
            seed=args.seed,
            output=args.output,
            data_root=args.data_root,
            creature_csv_dir=args.creature_data,
            show=not args.no_show,
        )

    else:
        # ─── Primary mode: read from log files ──────────────────────
        print(f">>> Loading episodes from: {args.log_dir} <<<")
        if args.bot:
            print(f">>> Filtering bot: {args.bot} <<<")
        if args.last:
            print(f">>> Last {args.last} episodes per bot <<<")

        recordings = load_recordings_from_logs(
            args.log_dir, bot_name=args.bot, last_n=args.last)

        if not recordings:
            print("No episodes found! Make sure training was run with "
                  "--log-dir.")
            print(f"Expected JSONL files in: {args.log_dir}")
            sys.exit(1)

        print(f">>> Loaded {len(recordings)} episodes <<<")
        for rec in recordings:
            print(f"  {rec.name} ep{rec.episode}: {rec.total_kills} kills, "
                  f"Lv{rec.final_level}, {rec.total_xp} XP, "
                  f"{len(rec.trail)} trail points")

        # Static PNG export if requested
        if args.output:
            n_eps = len(recordings)
            plot_map(recordings,
                     title=f"WoW Sim \u2014 {n_eps} Episodes from Logs",
                     output=args.output, show=False)

        # Interactive viewer (default)
        if not args.no_show:
            viewer = InteractiveViewer(recordings)
            viewer.show()
        elif not args.output:
            print("Nothing to do: --no-show without --output.")


if __name__ == "__main__":
    main()
