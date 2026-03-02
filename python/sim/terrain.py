"""
Terrain wrapper for the combat simulation.

Provides optional 3D terrain height, LOS, and walkability checks
using the WoW3DEnvironment from test_3d_env.py.

Usage:
    terrain = SimTerrain("/path/to/Data")   # loads map tiles, vmaps
    z = terrain.get_height(-8921, -120)      # terrain height
    los = terrain.check_los(x1,y1,z1, x2,y2,z2)  # line of sight
    ok = terrain.check_walkable(x1,y1,z1, x2,y2,z2)  # walkability
"""

import os
import sys

# Ensure parent dir is on path for test_3d_env import
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from test_3d_env import WoW3DEnvironment, Vec3, INVALID_HEIGHT


class SimTerrain:
    """
    Wraps WoW3DEnvironment for use in CombatSimulation.
    Loads terrain + VMAP data for the Northshire training area.
    """

    MAP_ID = 0              # Eastern Kingdoms
    SPAWN_X = -8921.037
    SPAWN_Y = -120.485
    SPAWN_Z = 82.025
    TILE_RADIUS = 1         # 3x3 grid of tiles around spawn

    def __init__(self, data_root: str, quiet: bool = False):
        self.env = WoW3DEnvironment(data_root)
        self._loaded = False
        self._quiet = quiet
        self._load()

    def _load(self):
        """Load terrain + VMAP data for the Northshire area."""
        # Terrain heights
        loaded = self.env.load_map_area(
            self.MAP_ID, self.SPAWN_X, self.SPAWN_Y, self.TILE_RADIUS
        )
        if loaded == 0:
            if not self._quiet:
                print("  [TERRAIN] WARNING: No map tiles loaded — using flat terrain")
            return

        # VMAP for LOS checks
        self.env.load_vmaps_area(
            self.MAP_ID, self.SPAWN_X, self.SPAWN_Y, self.TILE_RADIUS
        )

        # Build terrain path checker for walkability
        self.env.build_terrain_checker(self.MAP_ID)

        self._loaded = True
        if not self._quiet:
            h = self.get_height(self.SPAWN_X, self.SPAWN_Y)
            print(f"  [TERRAIN] Loaded. Height at spawn: {h:.3f} (expected ~{self.SPAWN_Z:.3f})")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_height(self, x: float, y: float) -> float:
        """Get terrain height at world coordinates (x, y)."""
        if not self._loaded:
            return self.SPAWN_Z
        h = self.env.get_height(self.MAP_ID, x, y)
        if h <= INVALID_HEIGHT + 1:
            return self.SPAWN_Z  # fallback for missing data
        return h

    def check_los(self, x1: float, y1: float, z1: float,
                  x2: float, y2: float, z2: float) -> bool:
        """Check line of sight between two world points. Returns True if clear."""
        if not self._loaded:
            return True
        p1 = Vec3(x1, y1, z1 + 1.7)  # eye height
        p2 = Vec3(x2, y2, z2 + 1.7)
        has_los, _, _ = self.env.check_los(p1, p2)
        return has_los

    def check_walkable(self, x1: float, y1: float, z1: float,
                       x2: float, y2: float, z2: float) -> bool:
        """Check if the terrain path between two points is walkable."""
        if not self._loaded or self.env.terrain_checker is None:
            return True
        walkable, _, _ = self.env.check_path(Vec3(x1, y1, z1), Vec3(x2, y2, z2))
        return walkable
