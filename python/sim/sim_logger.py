"""
Lightweight episode logger for WoW Sim visualization.

Buffers trail/event data in memory during an episode, writes a single
compact JSONL line at episode end.  Zero I/O during the hot simulation
loop — performance impact is effectively zero.

Log format (one .jsonl file per bot):
    Each line = one JSON object representing a complete episode:
    {
        "bot": "SimBot0",
        "episode": 3,
        "steps": 3812,
        "final_level": 2,
        "kills": 7,
        "xp": 420,
        "deaths": 0,
        "damage_dealt": 312,
        "reward": 23.5,
        "trail": [[step, x, y, hp_pct, level, in_combat, orientation], ...],
        "events": [{"s": step, "x": x, "y": y, "k": "kill", "l": ""}, ...],
        "mobs": [{"e": entry, "n": name, "x": x, "y": y, "lv": level}, ...]
    }
"""

import json
import os


class SimEpisodeLogger:
    """Buffers episode data in memory, flushes to JSONL at episode end."""

    def __init__(self, log_dir: str, bot_name: str, record_interval: int = 1):
        self.log_dir = log_dir
        self.bot_name = bot_name
        self.record_interval = max(1, record_interval)
        self._episode_num = 0

        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, f"{bot_name}.jsonl")

        # In-memory buffers (reset each episode)
        self._trail = []
        self._events = []
        self._mobs = []

    def reset(self):
        """Clear buffers for a new episode."""
        self._trail.clear()
        self._events.clear()
        self._mobs.clear()

    def record_step(self, step: int, x: float, y: float, hp_pct: float,
                    level: int, in_combat: bool, orientation: float):
        """Buffer a trail point (only every record_interval-th step)."""
        if step % self.record_interval != 0:
            return
        self._trail.append([
            step,
            round(x, 2),
            round(y, 2),
            round(hp_pct, 3),
            level,
            1 if in_combat else 0,
            round(orientation, 3),
        ])

    def record_event(self, step: int, x: float, y: float,
                     kind: str, label: str = ""):
        """Buffer a gameplay event (kill, death, levelup, loot)."""
        ev = {"s": step, "x": round(x, 2), "y": round(y, 2), "k": kind}
        if label:
            ev["l"] = label
        self._events.append(ev)

    def record_mobs(self, mobs: list):
        """Record mob snapshot list (once per episode, after reset).

        Each mob: dict with entry, name, x, y, level.
        """
        self._mobs = [
            {
                "e": m["entry"],
                "n": m["name"],
                "x": round(m["x"], 2),
                "y": round(m["y"], 2),
                "lv": m["level"],
            }
            for m in mobs
        ]

    def flush_episode(self, stats: dict):
        """Write buffered episode data to the JSONL file.

        Called once at episode end.  stats should contain:
            reward, length, kills, xp, loot, damage_dealt, death,
            final_level, etc.
        """
        record = {
            "bot": self.bot_name,
            "episode": self._episode_num,
            "steps": stats.get("length", len(self._trail)),
            "final_level": stats.get("final_level", 1),
            "kills": stats.get("kills", 0),
            "xp": stats.get("xp", 0),
            "deaths": stats.get("death", 0),
            "damage_dealt": stats.get("damage_dealt", 0),
            "quests_completed": stats.get("quests_completed", 0),
            "reward": round(stats.get("reward", 0.0), 2),
            "trail": self._trail,
            "events": self._events,
            "mobs": self._mobs,
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

        self._episode_num += 1
        self.reset()


def load_episodes(log_dir: str, bot_name: str = None,
                  last_n: int = None) -> list:
    """Load episodes from JSONL log files.

    Args:
        log_dir: Directory containing .jsonl files.
        bot_name: If set, load only this bot's file. Otherwise load all.
        last_n: If set, return only the last N episodes per bot.

    Returns:
        List of episode dicts (sorted by bot, then episode number).
    """
    episodes = []
    if bot_name:
        files = [os.path.join(log_dir, f"{bot_name}.jsonl")]
    else:
        files = sorted(
            os.path.join(log_dir, f)
            for f in os.listdir(log_dir)
            if f.endswith(".jsonl")
        )

    for path in files:
        if not os.path.isfile(path):
            continue
        bot_eps = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                bot_eps.append(json.loads(line))
        if last_n is not None and last_n > 0:
            bot_eps = bot_eps[-last_n:]
        episodes.extend(bot_eps)

    return episodes
