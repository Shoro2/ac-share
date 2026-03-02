"""
Terrain wrapper for the combat simulation.

Provides optional 3D terrain height, LOS, and walkability checks
using the WoW3DEnvironment from test_3d_env.py.

Tiles are loaded on demand as the player moves across Map 0.
The vmtree (BIH spatial index) is loaded once at init, map tiles
and vmtiles are loaded when the player enters a new tile region.

Usage:
    terrain = SimTerrain("/path/to/Data")   # loads vmtree + spawn tiles
    terrain.ensure_loaded(x, y)             # load tiles around position
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

from test_3d_env import (
    WoW3DEnvironment, Vec3, INVALID_HEIGHT,
    world_to_grid, vmtile_filename, parse_vmtile, parse_vmtree,
    vmtree_filename, MAX_NUMBER_OF_GRIDS,
)


class SimTerrain:
    """
    Wraps WoW3DEnvironment for use in CombatSimulation.
    Loads terrain + VMAP data on demand as the player moves across Map 0.
    """

    MAP_ID = 0              # Eastern Kingdoms
    SPAWN_X = -8921.037
    SPAWN_Y = -120.485
    SPAWN_Z = 82.025
    TILE_RADIUS = 1         # 3x3 grid of tiles around current position

    def __init__(self, data_root: str, quiet: bool = False):
        self.env = WoW3DEnvironment(data_root)
        self._loaded = False
        self._quiet = quiet
        # Track which tiles we've attempted to load (avoids re-reading missing files)
        self._attempted_tiles: set = set()
        # Track which tile center the player is on (skip work when unchanged)
        self._current_center: tuple | None = None
        self._load_initial()

    def _load_initial(self):
        """Load vmtree once and initial tiles around spawn."""
        # Load vmtree once for the whole map (BIH spatial index for LOS)
        vtree_path = os.path.join(self.env.vmaps_dir, vmtree_filename(self.MAP_ID))
        vtree = parse_vmtree(vtree_path, self.MAP_ID)
        if vtree:
            self.env.loaded_vmtrees[self.MAP_ID] = vtree
            if not self._quiet:
                print(f"  [TERRAIN] vmtree: {len(vtree.tree_nodes)} BIH nodes")

        # Load initial tiles around spawn
        self.ensure_loaded(self.SPAWN_X, self.SPAWN_Y)

        if self._loaded and not self._quiet:
            h = self.get_height(self.SPAWN_X, self.SPAWN_Y)
            print(f"  [TERRAIN] Loaded. Height at spawn: {h:.3f} (expected ~{self.SPAWN_Z:.3f})")

    def ensure_loaded(self, x: float, y: float):
        """Ensure map tiles + vmtiles around (x, y) are loaded.

        Call this after the player moves. Cheap no-op when the player
        stays on the same tile (one tuple comparison).
        """
        gx, gy = world_to_grid(x, y)
        if (gx, gy) == self._current_center:
            return  # still on the same tile — nothing to do
        self._current_center = (gx, gy)

        new_map_tiles = False
        vtree = self.env.loaded_vmtrees.get(self.MAP_ID)

        for dx in range(-self.TILE_RADIUS, self.TILE_RADIUS + 1):
            for dy in range(-self.TILE_RADIUS, self.TILE_RADIUS + 1):
                tx, ty = gx + dx, gy + dy
                if not (0 <= tx < MAX_NUMBER_OF_GRIDS and 0 <= ty < MAX_NUMBER_OF_GRIDS):
                    continue
                if (tx, ty) in self._attempted_tiles:
                    continue
                self._attempted_tiles.add((tx, ty))

                # Map tile (height data + area map)
                tile = self.env.load_map_tile(self.MAP_ID, tx, ty)
                if tile and tile.v9 is not None:
                    new_map_tiles = True

                # Vmtile (VMAP spawns for LOS)
                if vtree:
                    vtile_path = os.path.join(
                        self.env.vmaps_dir, vmtile_filename(self.MAP_ID, tx, ty)
                    )
                    spawns = parse_vmtile(vtile_path)
                    if spawns:
                        vtree.spawns.extend(spawns)
                        self.env.los_checker.add_spawns(spawns)

        if new_map_tiles:
            # Rebuild terrain checker with all loaded tiles
            self.env.build_terrain_checker(self.MAP_ID)
            if not self._loaded:
                self._loaded = True

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
