"""
WoW Sim Interactive Map Viewer — episode browser with zoom, filters, and log.

Opens an interactive window at startup.  Use the episode slider to browse
through episodes, checkboxes to filter bots, scroll-wheel to zoom, and
right-click-drag to pan.  A separate log window can be toggled on/off.

By default only the top 5 episodes (by kills) are shown.  If the log
directory contains subdirectories with .jsonl files, each subdirectory is
treated as a separate training run and a run selector is shown.

Usage:
    # Interactive viewer from training logs (primary mode)
    python -m sim.visualize --log-dir logs/sim_episodes/

    # Show top 10 episodes by kills instead of default 5
    python -m sim.visualize --log-dir logs/sim_episodes/ --top-kills 10

    # Show all episodes (no filter)
    python -m sim.visualize --log-dir logs/sim_episodes/ --top-kills 0

    # Multi-run: point to parent dir with run subdirs
    python -m sim.visualize --log-dir logs/all_runs/

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
from matplotlib.widgets import Slider, CheckButtons, Button, RadioButtons
try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 300_000_000
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)



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


# ─── Map Background ─────────────────────────────────────────────────────

# Default bounds for the Eastern Kingdoms continent map.
# These are the WoW world-coordinates (x_min, y_min, x_max, y_max) that the
# map image covers.  The values below correspond to the full EK continent map
# as exported by common WoW map tools.  Override with --map-bounds if your
# image uses different extents.
EK_DEFAULT_BOUNDS = (-11700.0, -12000.0, 4000.0, 4000.0)  # x_min, y_min, x_max, y_max


def _load_map_background(image_path, wow_bounds=None):
    """Load a map background image and prepare it for display.

    Args:
        image_path: Path to the map image file.
        wow_bounds: Tuple (x_min, y_min, x_max, y_max) in WoW world coords
                    describing the area covered by the image.  If None, uses
                    EK_DEFAULT_BOUNDS.

    Returns:
        (rotated_image_array, display_extent) where display_extent is
        (disp_x_min, disp_x_max, disp_y_min, disp_y_max) in rotated coords.
    """
    if wow_bounds is None:
        wow_bounds = EK_DEFAULT_BOUNDS

    x_min, y_min, x_max, y_max = wow_bounds

    if not _HAS_PIL:
        raise ImportError("Pillow is required for --map-image. "
                          "Install with: pip install Pillow")
    img = Image.open(image_path)
    img_arr = np.asarray(img)

    # After coordinate rotation: display_x = -wow_y, display_y = wow_x
    disp_x_min = -y_max
    disp_x_max = -y_min
    disp_y_min = x_min
    disp_y_max = x_max

    return img_arr, (disp_x_min, disp_x_max, disp_y_min, disp_y_max)


def _draw_map_background(ax, map_image_data):
    """Draw the map background image onto the axes.

    Args:
        ax: matplotlib Axes.
        map_image_data: Tuple (image_array, extent) from _load_map_background.
    """
    if map_image_data is None:
        return
    img_arr, extent = map_image_data
    ax.imshow(img_arr, extent=extent, aspect="equal", zorder=0, alpha=0.7,
              interpolation="bilinear")


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
    total_quests: int = 0


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
        rec.total_quests = ep.get("quests_completed", 0)

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


# ─── Run Discovery ────────────────────────────────────────────────────────

def discover_runs(log_dir: str) -> dict:
    """Discover training runs from a log directory.

    If log_dir contains subdirectories with .jsonl files, each subdir is a
    run.  If log_dir itself contains .jsonl files, it is treated as a single
    run (named after the directory).

    Returns:
        Dict mapping run_name -> directory_path, sorted alphabetically.
    """
    runs = {}

    # Check subdirectories first
    if os.path.isdir(log_dir):
        for entry in sorted(os.listdir(log_dir)):
            sub = os.path.join(log_dir, entry)
            if os.path.isdir(sub):
                if any(f.endswith(".jsonl") for f in os.listdir(sub)):
                    runs[entry] = sub

    # If no subdirs with logs, treat the dir itself as a single run
    if not runs and os.path.isdir(log_dir):
        if any(f.endswith(".jsonl") for f in os.listdir(log_dir)):
            name = os.path.basename(os.path.normpath(log_dir))
            runs[name] = log_dir

    return runs


def _top_kills_recordings(recordings: list, top_n: int = 5) -> list:
    """Return only the top N recordings sorted by total_kills descending."""
    if top_n <= 0 or len(recordings) <= top_n:
        return recordings
    sorted_recs = sorted(recordings, key=lambda r: r.total_kills, reverse=True)
    return sorted_recs[:top_n]


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
    prev_quests = 0

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

        if sim.quests_completed > prev_quests:
            new_q = sim.quests_completed - prev_quests
            for _ in range(new_q):
                rec.events.append(MapEvent(
                    step=step, x=p.x, y=p.y, kind="quest",
                    label=f"Quest #{sim.quests_completed}",
                ))
            rec.total_quests += new_q
            prev_quests = sim.quests_completed

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
        elif ev.kind == "quest":
            ax.scatter([ex], [ey], c="#00FF88", marker="D", s=80,
                       edgecolors="white", linewidths=0.8, zorder=9)
            label = ev.label if ev.label else "Quest"
            ax.annotate(label, (ex, ey),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=7, fontweight="bold", color="#00FF88",
                        zorder=9)


# ─── Static plot (PNG export) ────────────────────────────────────────────

def plot_map(recordings: list, title: str = "WoW Sim — Bot Routes",
             output: str = None, show: bool = True,
             map_image_data=None):
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

    # Map background
    _draw_map_background(ax, map_image_data)

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
    handles.append(plt.Line2D([0], [0], marker="D", color="#00FF88",
                              linestyle="None", markersize=8, label="Quest"))

    stats_lines = []
    for rec in recordings:
        line = (f"{rec.name}: {rec.total_kills} kills, "
                f"Lv{rec.final_level}, {rec.total_xp} XP")
        if rec.total_deaths > 0:
            line += f", {rec.total_deaths} deaths"
        if rec.total_quests > 0:
            line += f", {rec.total_quests} quests"
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
    """Interactive map viewer with episode slider, bot filters, zoom, run
    selector, and a toggleable log window."""

    def __init__(self, recordings, title="WoW Sim \u2014 Interactive Map",
                 runs=None, top_kills=5, bot_name=None,
                 map_image_data=None):
        """
        Args:
            recordings: List of BotRecording for the initially selected run.
            title: Window title.
            runs: Dict {run_name: dir_path} for run selection. None = no selector.
            top_kills: Show only top N episodes by kills (0 = all).
            bot_name: Bot name filter for loading (passed to load_recordings_from_logs).
            map_image_data: Tuple (image_array, extent) from _load_map_background.
        """
        self._runs = runs or {}
        self._top_kills = top_kills
        self._bot_name_filter = bot_name
        self._map_image_data = map_image_data
        self._run_names = sorted(self._runs.keys()) if self._runs else []
        self._current_run_idx = 0

        # Apply top-kills filter
        if top_kills > 0:
            recordings = _top_kills_recordings(recordings, top_kills)

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
        # Sort by total kills descending (sum across bots in same episode)
        self.episode_keys = sorted(
            by_ep.keys(),
            key=lambda k: sum(r.total_kills for r in by_ep[k]),
            reverse=True)
        self.episodes = {k: by_ep[k] for k in self.episode_keys}
        self.num_episodes = max(len(self.episode_keys), 1)
        # Build episode labels for the dropdown
        self._ep_labels = []
        for k in self.episode_keys:
            recs = by_ep[k]
            kills = sum(r.total_kills for r in recs)
            lvl = max(r.final_level for r in recs)
            bots = ", ".join(r.name for r in recs)
            self._ep_labels.append(
                f"Ep{k}: {kills}k Lv{lvl}")

    # ── Figure setup ─────────────────────────────────────────────────

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.set_facecolor("#1a1a2e")
        self.fig.canvas.manager.set_window_title(self.title)

        # ── Layout constants ──────────────────────────────────────────
        # Left sidebar for episodes, main map in center, right panel
        left_w = 0.13
        right_w = 0.15
        map_left = left_w + 0.01
        map_w = 1.0 - map_left - right_w - 0.02
        map_bottom = 0.06
        map_h = 0.88
        right_x = 1.0 - right_w - 0.01

        # ── Episode selector (left sidebar) ───────────────────────────
        self.fig.text(left_w / 2, 0.96, "Episodes",
                      color="white", fontsize=10, fontweight="bold",
                      ha="center")
        ep_labels = self._ep_labels if self._ep_labels else ["(none)"]
        self._ax_ep = self.fig.add_axes(
            [0.00, map_bottom, left_w, map_h])
        self._ax_ep.set_facecolor("#1a1a2e")
        for spine in self._ax_ep.spines.values():
            spine.set_color("#444")
        self._ep_radio = RadioButtons(
            self._ax_ep, ep_labels, active=0)
        for lbl in self._ep_radio.labels:
            lbl.set_color("white")
            lbl.set_fontsize(8)
            lbl.set_fontfamily("monospace")
        self._ep_radio.on_clicked(self._on_episode_selected)

        # ── Main map axes ────────────────────────────────────────────
        self.ax_map = self.fig.add_axes(
            [map_left, map_bottom, map_w, map_h])
        self.ax_map.set_facecolor("#16213e")

        # ── Zoom slider (bottom, under map) ───────────────────────────
        ax_zoom = self.fig.add_axes(
            [map_left + 0.05, 0.01, map_w - 0.10, 0.03])
        ax_zoom.set_facecolor("#2a2a4e")
        self.zoom_slider = Slider(
            ax_zoom, "Zoom", 0.1, 10.0,
            valinit=1.0, color="#4aff9e",
        )
        self.zoom_slider.label.set_color("white")
        self.zoom_slider.valtext.set_color("white")
        self.zoom_slider.on_changed(self._on_zoom_changed)

        # ── Run selector (right side, top) ───────────────────────────
        self._run_radio = None
        right_top = 0.92
        if len(self._run_names) > 1:
            n_runs = len(self._run_names)
            radio_h = min(max(0.04 * n_runs, 0.08), 0.30)
            self.fig.text(
                right_x + right_w / 2, right_top + 0.01, "Run:",
                color="white", fontsize=10, fontweight="bold",
                ha="center")
            ax_run = self.fig.add_axes(
                [right_x, right_top - radio_h, right_w, radio_h])
            ax_run.set_facecolor("#1a1a2e")
            for spine in ax_run.spines.values():
                spine.set_color("#444")
            self._run_radio = RadioButtons(
                ax_run, self._run_names, active=self._current_run_idx)
            for lbl in self._run_radio.labels:
                lbl.set_color("white")
                lbl.set_fontsize(8)
            self._run_radio.on_clicked(self._on_run_changed)
            right_top = right_top - radio_h - 0.03

        # ── Bot filter checkboxes (right side) ───────────────────────
        n_bots = len(self.bot_names)
        check_h = min(max(0.045 * n_bots, 0.10), 0.40)
        check_top = right_top
        self._ax_check = self.fig.add_axes(
            [right_x, check_top - check_h, right_w, check_h])
        self._ax_check.set_facecolor("#1a1a2e")
        for spine in self._ax_check.spines.values():
            spine.set_color("#444")
        self.check_buttons = CheckButtons(
            self._ax_check, self.bot_names, [True] * n_bots)
        for lbl in self.check_buttons.labels:
            lbl.set_color("white")
            lbl.set_fontsize(9)
        self.check_buttons.on_clicked(self._on_bot_toggled)

        # ── All / None buttons ───────────────────────────────────────
        btn_y = check_top + 0.01
        ax_all = self.fig.add_axes(
            [right_x, btn_y, right_w * 0.48, 0.03])
        self.btn_all = Button(ax_all, "All",
                              color="#2a2a4e", hovercolor="#3a3a5e")
        self.btn_all.label.set_color("white")
        self.btn_all.label.set_fontsize(8)
        self.btn_all.on_clicked(self._select_all_bots)

        ax_none = self.fig.add_axes(
            [right_x + right_w * 0.52, btn_y, right_w * 0.48, 0.03])
        self.btn_none = Button(ax_none, "None",
                               color="#2a2a4e", hovercolor="#3a3a5e")
        self.btn_none.label.set_color("white")
        self.btn_none.label.set_fontsize(8)
        self.btn_none.on_clicked(self._select_no_bots)

        # ── Log toggle + Reset view buttons ──────────────────────────
        btn_bottom = check_top - check_h - 0.06
        ax_log = self.fig.add_axes(
            [right_x, btn_bottom, right_w, 0.04])
        self.btn_log = Button(ax_log, "Show Log",
                              color="#2a2a4e", hovercolor="#3a3a5e")
        self.btn_log.label.set_color("white")
        self.btn_log.on_clicked(self._toggle_log)

        ax_reset = self.fig.add_axes(
            [right_x, btn_bottom - 0.05, right_w, 0.04])
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

    def _on_run_changed(self, label):
        """Switch to a different training run."""
        if label not in self._runs:
            return
        idx = self._run_names.index(label)
        if idx == self._current_run_idx:
            return
        self._current_run_idx = idx
        run_dir = self._runs[label]

        # Load new recordings
        recordings = load_recordings_from_logs(
            run_dir, bot_name=self._bot_name_filter)
        if self._top_kills > 0:
            recordings = _top_kills_recordings(recordings, self._top_kills)

        self.all_recordings = recordings
        self.bot_names = sorted(set(r.name for r in recordings))
        self._group_by_episode()
        self.visible_bots = set(self.bot_names)
        self.current_ep_idx = 0
        self._global_bounds = _get_map_bounds(recordings)
        self._first_draw = True

        # Rebuild episode radio buttons
        self._rebuild_episode_selector()

        # Rebuild bot checkboxes
        self._ax_check.clear()
        self._ax_check.set_facecolor("#1a1a2e")
        for spine in self._ax_check.spines.values():
            spine.set_color("#444")
        n_bots = len(self.bot_names)
        self.check_buttons = CheckButtons(
            self._ax_check, self.bot_names, [True] * n_bots)
        for lbl in self.check_buttons.labels:
            lbl.set_color("white")
            lbl.set_fontsize(9)
        self.check_buttons.on_clicked(self._on_bot_toggled)

        self._draw_map()

    def _on_episode_selected(self, label):
        """Episode radio button clicked."""
        if label in self._ep_labels:
            self.current_ep_idx = self._ep_labels.index(label)
        self._draw_map()

    def _rebuild_episode_selector(self):
        """Rebuild episode radio buttons after run change."""
        self._ax_ep.clear()
        self._ax_ep.set_facecolor("#1a1a2e")
        for spine in self._ax_ep.spines.values():
            spine.set_color("#444")
        ep_labels = self._ep_labels if self._ep_labels else ["(no episodes)"]
        self._ep_radio = RadioButtons(
            self._ax_ep, ep_labels, active=0)
        for lbl in self._ep_radio.labels:
            lbl.set_color("white")
            lbl.set_fontsize(8)
            lbl.set_fontfamily("monospace")
        self._ep_radio.on_clicked(self._on_episode_selected)

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
            if new != self.current_ep_idx and new < len(self._ep_labels):
                self._ep_radio.set_active(new)
        elif event.key == "left":
            new = max(self.current_ep_idx - 1, 0)
            if new != self.current_ep_idx:
                self._ep_radio.set_active(new)
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
                f"  |  XP: {rec.total_xp}  |  Deaths: {rec.total_deaths}"
                f"  |  Quests: {rec.total_quests}")
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
        run_label = ""
        if self._run_names:
            run_label = f"  [{self._run_names[self._current_run_idx]}]"
        top_label = ""
        if self._top_kills > 0:
            top_label = f"  (Top {self._top_kills} by kills)"
        self.ax_map.set_title(
            f"{self.title}{run_label}  \u2014  {ep_label}{top_label}  "
            f"({n_vis} bot{'s' if n_vis != 1 else ''})",
            color="white", fontsize=13, fontweight="bold", pad=8)
        self.ax_map.set_xlabel(
            "Y (East \u2192 West)", color="white", fontsize=10)
        self.ax_map.set_ylabel(
            "X (South \u2192 North)", color="white", fontsize=10)
        self.ax_map.tick_params(colors="white", labelsize=8)
        for spine in self.ax_map.spines.values():
            spine.set_color("#444")
        self.ax_map.grid(True, alpha=0.15, color="white", linewidth=0.5)

        # ── Map background ────────────────────────────────────────────
        _draw_map_background(self.ax_map, self._map_image_data)

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
        handles.append(plt.Line2D(
            [0], [0], marker="D", color="#00FF88",
            linestyle="None", markersize=8, label="Quest"))
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
                       show: bool = True,
                       map_image_data=None):
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
              f"{rec.total_quests} quests, {len(rec.trail)} steps")
        env.close()

    print(f">>> All episodes done. <<<")

    if output and not show:
        # Static PNG only
        plot_map(recordings,
                 title=f"WoW Sim \u2014 {n_episodes} Episodes Overlay",
                 output=output, show=False,
                 map_image_data=map_image_data)
    else:
        # Interactive viewer
        if output:
            plot_map(recordings,
                     title=f"WoW Sim \u2014 {n_episodes} Episodes Overlay",
                     output=output, show=False,
                     map_image_data=map_image_data)
            print(f"Map saved: {output}")
        viewer = InteractiveViewer(
            recordings,
            title=f"WoW Sim \u2014 {n_episodes} Episodes",
            map_image_data=map_image_data)
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
                        help="Path to episode log directory (JSONL files). "
                             "If it contains subdirs with .jsonl files, each "
                             "subdir is treated as a separate run.")
    parser.add_argument("--bot", type=str, default=None,
                        help="Show only this bot's episodes (default: all)")
    parser.add_argument("--last", type=int, default=None,
                        help="Show only the last N episodes per bot")
    parser.add_argument("--top-kills", type=int, default=5,
                        help="Show only top N episodes by kills (default: 5, "
                             "0 = show all)")

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

    # Map background
    parser.add_argument("--map-image", type=str, default=None,
                        help="Path to a map background image (e.g. Eastern "
                             "Kingdoms continent map PNG/JPG)")
    parser.add_argument("--map-bounds", type=str, default=None,
                        help="WoW world-coordinate bounds of the map image "
                             "as 'x_min,y_min,x_max,y_max' (default: EK "
                             "continent bounds -11700,-12000,4000,4000)")

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

    # Load map background — auto-detect from Data dir if not specified
    map_image_data = None
    if not args.map_image:
        auto_path = os.path.join(r"C:\wowstuff\WoWKI_serv\Data", "world_map.png")
        if os.path.exists(auto_path):
            args.map_image = auto_path
    if args.map_image:
        wow_bounds = None
        if args.map_bounds:
            parts = [float(v.strip()) for v in args.map_bounds.split(",")]
            if len(parts) == 4:
                wow_bounds = tuple(parts)
            else:
                print("WARNING: --map-bounds needs exactly 4 values "
                      "(x_min,y_min,x_max,y_max), using defaults")
        else:
            # Auto-detect JSON sidecar file with bounds
            sidecar = args.map_image.rsplit(".", 1)[0] + ".json"
            if os.path.exists(sidecar):
                try:
                    import json
                    with open(sidecar) as f:
                        bd = json.load(f)
                    wow_bounds = (bd["x_min"], bd["y_min"],
                                  bd["x_max"], bd["y_max"])
                    print(f"  Auto-loaded bounds from: {sidecar}")
                except Exception:
                    pass
        map_image_data = _load_map_background(args.map_image, wow_bounds)
        bounds_used = wow_bounds or EK_DEFAULT_BOUNDS
        print(f"Map background: {args.map_image}")
        print(f"  WoW bounds: x=[{bounds_used[0]}, {bounds_used[2]}] "
              f"y=[{bounds_used[1]}, {bounds_used[3]}]")

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
            map_image_data=map_image_data,
        )

    else:
        # ─── Primary mode: read from log files ──────────────────────
        print(f">>> Loading episodes from: {args.log_dir} <<<")
        if args.bot:
            print(f">>> Filtering bot: {args.bot} <<<")
        if args.last:
            print(f">>> Last {args.last} episodes per bot <<<")
        top_n = args.top_kills
        if top_n > 0:
            print(f">>> Showing top {top_n} episodes by kills <<<")

        # Discover runs (subdirs with .jsonl files)
        runs = discover_runs(args.log_dir)
        if not runs:
            print("No episodes found! Make sure training was run with "
                  "--log-dir.")
            print(f"Expected JSONL files in: {args.log_dir}")
            sys.exit(1)

        run_names = sorted(runs.keys())
        print(f">>> Found {len(runs)} run(s): {', '.join(run_names)} <<<")

        # Load the first run initially
        first_run_dir = runs[run_names[0]]
        recordings = load_recordings_from_logs(
            first_run_dir, bot_name=args.bot, last_n=args.last)

        if not recordings:
            print(f"No episodes in run '{run_names[0]}'!")
            sys.exit(1)

        # Apply top-kills filter for printing
        filtered = _top_kills_recordings(recordings, top_n) if top_n > 0 else recordings

        print(f">>> Loaded {len(recordings)} episodes, showing "
              f"{len(filtered)} (top by kills) <<<")
        for rec in filtered:
            print(f"  {rec.name} ep{rec.episode}: {rec.total_kills} kills, "
                  f"Lv{rec.final_level}, {rec.total_xp} XP, "
                  f"{rec.total_quests} quests, {len(rec.trail)} trail points")

        # Static PNG export if requested
        if args.output:
            n_eps = len(filtered)
            plot_map(filtered,
                     title=f"WoW Sim \u2014 Top {n_eps} Episodes by Kills",
                     output=args.output, show=False,
                     map_image_data=map_image_data)

        # Interactive viewer (default)
        if not args.no_show:
            viewer = InteractiveViewer(
                recordings, runs=runs, top_kills=top_n,
                bot_name=args.bot,
                map_image_data=map_image_data)
            viewer.show()
        elif not args.output:
            print("Nothing to do: --no-show without --output.")


if __name__ == "__main__":
    main()
