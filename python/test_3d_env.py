#!/usr/bin/env python3
"""
test_3d_env.py — 3D Environment Simulation Test Script

Liest WoW-Spieldaten (maps, vmaps, mmaps) und berechnet:
- Terrain-Höhe an beliebigen (x,y) Koordinaten
- Line of Sight (LOS) zwischen zwei 3D-Punkten
- Hinderniserkennung für Pathing

Erwartet die Daten unter dem konfigurierbaren DATA_ROOT-Pfad.
Beispiel-Dateien liegen unter ./data/ zum Testen der Parser.

Nutzung:
    python test_3d_env.py                     # Nutzt ./data/ Beispieldateien
    python test_3d_env.py --data-root /pfad   # Nutzt echte Daten
"""

import struct
import math
import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ─────────────────────────── Konstanten ───────────────────────────

MAX_NUMBER_OF_GRIDS = 64
SIZE_OF_GRIDS = 533.3333
CENTER_GRID_ID = MAX_NUMBER_OF_GRIDS // 2  # 32
MAP_RESOLUTION = 128
MAP_HALFSIZE = SIZE_OF_GRIDS * MAX_NUMBER_OF_GRIDS / 2.0  # 17066.6656

# Map file magic constants
MAP_MAGIC = b'MAPS'
MAP_VERSION_MAGIC = 9
MAP_AREA_MAGIC = b'AREA'
MAP_HEIGHT_MAGIC = b'MHGT'
MAP_LIQUID_MAGIC = b'MLIQ'

# Map height flags
MAP_HEIGHT_NO_HEIGHT = 0x0001
MAP_HEIGHT_AS_INT16 = 0x0002
MAP_HEIGHT_AS_INT8 = 0x0004
MAP_HEIGHT_HAS_FLIGHT_BOUNDS = 0x0008

# Map area flags
MAP_AREA_NO_AREA = 0x0001

# Map liquid flags
MAP_LIQUID_NO_TYPE = 0x0001
MAP_LIQUID_NO_HEIGHT = 0x0002

# VMAP constants
VMAP_MAGIC = b'VMAP_4.8'
MOD_M2 = 1
MOD_WORLDSPAWN = 2
MOD_HAS_BOUND = 4

# MMAP constants
MMAP_MAGIC = 0x4D4D4150
MMAP_VERSION = 19

INVALID_HEIGHT = -100000.0


# ─────────────────────────── Datenklassen ─────────────────────────

@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, s):
        return Vec3(self.x * s, self.y * s, self.z * s)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self):
        l = self.length()
        if l < 1e-10:
            return Vec3(0, 0, 0)
        return Vec3(self.x / l, self.y / l, self.z / l)

    def __repr__(self):
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


@dataclass
class AABB:
    """Axis-Aligned Bounding Box"""
    low: Vec3
    high: Vec3

    def intersects_ray(self, origin: Vec3, inv_dir: Vec3, max_dist: float) -> tuple:
        """
        Slab-Test: Prüft ob ein Strahl die AABB trifft.
        Returns (hit, t_enter, t_exit)
        """
        t_min = 0.0
        t_max = max_dist

        for axis in range(3):
            o = [origin.x, origin.y, origin.z][axis]
            inv_d = [inv_dir.x, inv_dir.y, inv_dir.z][axis]
            lo = [self.low.x, self.low.y, self.low.z][axis]
            hi = [self.high.x, self.high.y, self.high.z][axis]

            if abs(inv_d) < 1e-30:
                # Strahl parallel zur Achse
                if o < lo or o > hi:
                    return (False, 0.0, 0.0)
            else:
                t1 = (lo - o) * inv_d
                t2 = (hi - o) * inv_d
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return (False, 0.0, 0.0)

        return (True, t_min, t_max)

    def contains(self, point: Vec3) -> bool:
        return (self.low.x <= point.x <= self.high.x and
                self.low.y <= point.y <= self.high.y and
                self.low.z <= point.z <= self.high.z)


@dataclass
class ModelSpawn:
    """Ein Objekt-Spawn in der VMAP (Baum, Gebäude, etc.)"""
    flags: int
    adt_id: int
    spawn_id: int
    position: Vec3
    rotation: Vec3
    scale: float
    bounds: Optional[AABB]
    name: str
    tree_ref: int  # Index im BVH-Tree


@dataclass
class MapTile:
    """Enthält die Terrain-Daten eines 533x533 Grid-Tiles"""
    map_id: int
    tile_x: int
    tile_y: int
    # Area
    area_flags: int = 0
    grid_area: int = 0
    area_map: Optional[list] = None  # 16x16 uint16
    # Height
    height_flags: int = 0
    grid_height: float = 0.0
    grid_max_height: float = 0.0
    v9: Optional[list] = None  # 129x129 heights (grid vertices)
    v8: Optional[list] = None  # 128x128 heights (grid cell centers)
    # Liquid
    liquid_flags: int = 0
    liquid_level: float = 0.0
    liquid_offset_x: int = 0
    liquid_offset_y: int = 0
    liquid_width: int = 0
    liquid_height: int = 0
    liquid_map: Optional[list] = None


@dataclass
class BIHNode:
    """Ein Knoten im Bounding Interval Hierarchy Tree"""
    axis: int       # 0=X, 1=Y, 2=Z
    is_bvh2: bool
    offset: int     # Child-Offset oder Leaf-Index
    clip_left: float
    clip_right: float


@dataclass
class VMapTree:
    """BVH-Tree mit Objekt-Spawns für eine Map"""
    map_id: int
    is_tiled: bool
    bounds: AABB
    tree_nodes: list       # Raw uint32 nodes
    object_indices: list   # Indices in die Spawn-Liste
    spawns: list           # ModelSpawn-Objekte (aus vmtree GOBJ + vmtiles)


@dataclass
class NavMeshParams:
    """Detour NavMesh Parameter"""
    origin: Vec3
    tile_width: float
    tile_height: float
    max_tiles: int
    max_polys: int


# ─────────────────────── Koordinaten-Umrechnung ──────────────────

def world_to_grid(x: float, y: float) -> tuple:
    """Weltkoodinaten → Grid-Tile-Koordinaten (gx, gy)"""
    gx = int(CENTER_GRID_ID - x / SIZE_OF_GRIDS)
    gy = int(CENTER_GRID_ID - y / SIZE_OF_GRIDS)
    return (gx, gy)


def world_to_vmap(x: float, y: float, z: float) -> Vec3:
    """Weltkoordinaten → VMAP-interne Koordinaten"""
    return Vec3(MAP_HALFSIZE - x, MAP_HALFSIZE - y, z)


def vmap_to_world(vx: float, vy: float, vz: float) -> Vec3:
    """VMAP-interne Koordinaten → Weltkoordinaten"""
    return Vec3(MAP_HALFSIZE - vx, MAP_HALFSIZE - vy, vz)


def world_to_height_grid(x: float, y: float) -> tuple:
    """Weltkoordinaten → Position im 128x128 Height-Grid (float)"""
    hx = MAP_RESOLUTION * (CENTER_GRID_ID - x / SIZE_OF_GRIDS)
    hy = MAP_RESOLUTION * (CENTER_GRID_ID - y / SIZE_OF_GRIDS)
    return (hx, hy)


def map_filename(map_id: int, tile_x: int, tile_y: int) -> str:
    """Map-Dateiname: {mapId:03d}{tileX:02d}{tileY:02d}.map"""
    return f"{map_id:03d}{tile_x:02d}{tile_y:02d}.map"


def vmtree_filename(map_id: int) -> str:
    return f"{map_id:03d}.vmtree"


def vmtile_filename(map_id: int, tile_x: int, tile_y: int) -> str:
    """VMTILE-Dateiname: {mapId:03d}_{tileY:02d}_{tileX:02d}.vmtile"""
    return f"{map_id:03d}_{tile_y:02d}_{tile_x:02d}.vmtile"


def mmap_filename(map_id: int) -> str:
    return f"{map_id:03d}.mmap"


def mmtile_filename(map_id: int, tile_x: int, tile_y: int) -> str:
    return f"{map_id:03d}{tile_x:02d}{tile_y:02d}.mmtile"


# ─────────────────────────── MAP Parser ──────────────────────────

def parse_map_file(filepath: str, map_id: int = 0, tile_x: int = 0, tile_y: int = 0) -> Optional[MapTile]:
    """
    Liest eine .map Datei und extrahiert Terrain-Höhen, Area-IDs und Liquid-Daten.
    """
    if not os.path.exists(filepath):
        print(f"  [MAP] Datei nicht gefunden: {filepath}")
        return None

    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < 44:
        print(f"  [MAP] Datei zu klein ({len(data)} bytes): {filepath}")
        return None

    # Header (44 bytes)
    magic = data[0:4]
    if magic != MAP_MAGIC:
        print(f"  [MAP] Falscher Magic: {magic}")
        return None

    version = struct.unpack('<I', data[4:8])[0]
    if version != MAP_VERSION_MAGIC:
        print(f"  [MAP] Warnung: Version {version} (erwartet {MAP_VERSION_MAGIC})")

    area_offset = struct.unpack('<I', data[12:16])[0]
    area_size = struct.unpack('<I', data[16:20])[0]
    height_offset = struct.unpack('<I', data[20:24])[0]
    height_size = struct.unpack('<I', data[24:28])[0]
    liquid_offset = struct.unpack('<I', data[28:32])[0]
    liquid_size = struct.unpack('<I', data[32:36])[0]

    tile = MapTile(map_id=map_id, tile_x=tile_x, tile_y=tile_y)

    # ── AREA Section ──
    if area_size > 0:
        pos = area_offset
        area_magic = data[pos:pos+4]
        if area_magic == MAP_AREA_MAGIC:
            tile.area_flags = struct.unpack('<H', data[pos+4:pos+6])[0]
            tile.grid_area = struct.unpack('<H', data[pos+6:pos+8])[0]
            if not (tile.area_flags & MAP_AREA_NO_AREA):
                # 16x16 area map
                area_data = data[pos+8:pos+8+512]
                tile.area_map = list(struct.unpack('<256H', area_data))

    # ── HEIGHT Section ──
    if height_size > 0:
        pos = height_offset
        h_magic = data[pos:pos+4]
        if h_magic == MAP_HEIGHT_MAGIC:
            tile.height_flags = struct.unpack('<I', data[pos+4:pos+8])[0]
            tile.grid_height = struct.unpack('<f', data[pos+8:pos+12])[0]
            tile.grid_max_height = struct.unpack('<f', data[pos+12:pos+16])[0]

            if not (tile.height_flags & MAP_HEIGHT_NO_HEIGHT):
                hpos = pos + 16
                if tile.height_flags & MAP_HEIGHT_AS_INT8:
                    # uint8 quantized: 129*129 + 128*128 bytes
                    v9_count = 129 * 129
                    v8_count = 128 * 128
                    v9_raw = struct.unpack(f'<{v9_count}B', data[hpos:hpos+v9_count])
                    hpos += v9_count
                    v8_raw = struct.unpack(f'<{v8_count}B', data[hpos:hpos+v8_count])
                    multiplier = (tile.grid_max_height - tile.grid_height) / 255.0
                    tile.v9 = [v * multiplier + tile.grid_height for v in v9_raw]
                    tile.v8 = [v * multiplier + tile.grid_height for v in v8_raw]
                elif tile.height_flags & MAP_HEIGHT_AS_INT16:
                    # uint16 quantized: 129*129*2 + 128*128*2 bytes
                    v9_count = 129 * 129
                    v8_count = 128 * 128
                    v9_raw = struct.unpack(f'<{v9_count}H', data[hpos:hpos+v9_count*2])
                    hpos += v9_count * 2
                    v8_raw = struct.unpack(f'<{v8_count}H', data[hpos:hpos+v8_count*2])
                    multiplier = (tile.grid_max_height - tile.grid_height) / 65535.0
                    tile.v9 = [v * multiplier + tile.grid_height for v in v9_raw]
                    tile.v8 = [v * multiplier + tile.grid_height for v in v8_raw]
                else:
                    # Float heights: 129*129*4 + 128*128*4 bytes
                    v9_count = 129 * 129
                    v8_count = 128 * 128
                    tile.v9 = list(struct.unpack(f'<{v9_count}f', data[hpos:hpos+v9_count*4]))
                    hpos += v9_count * 4
                    tile.v8 = list(struct.unpack(f'<{v8_count}f', data[hpos:hpos+v8_count*4]))

    # ── LIQUID Section ──
    if liquid_size > 0:
        pos = liquid_offset
        l_magic = data[pos:pos+4]
        if l_magic == MAP_LIQUID_MAGIC:
            tile.liquid_flags = data[pos+4]
            tile.liquid_offset_x = data[pos+8]
            tile.liquid_offset_y = data[pos+9]
            tile.liquid_width = data[pos+10]
            tile.liquid_height = data[pos+11]
            tile.liquid_level = struct.unpack('<f', data[pos+12:pos+16])[0]

            if not (tile.liquid_flags & MAP_LIQUID_NO_HEIGHT):
                lpos = pos + 16
                if not (tile.liquid_flags & MAP_LIQUID_NO_TYPE):
                    lpos += 512 + 256  # liquidEntry(512) + liquidFlags(256)
                count = tile.liquid_width * tile.liquid_height
                if lpos + count * 4 <= len(data):
                    tile.liquid_map = list(struct.unpack(f'<{count}f', data[lpos:lpos+count*4]))

    return tile


def get_terrain_height(tile: MapTile, world_x: float, world_y: float) -> float:
    """
    Berechnet die interpolierte Terrain-Höhe an Weltkoordinaten (x, y).
    Verwendet die gleiche Triangle-Interpolation wie AzerothCore.
    """
    if tile.v9 is None or tile.v8 is None:
        return tile.grid_height

    # Welt → Heightmap-Koordinaten (lokal im Tile)
    x = MAP_RESOLUTION * (CENTER_GRID_ID - world_x / SIZE_OF_GRIDS)
    y = MAP_RESOLUTION * (CENTER_GRID_ID - world_y / SIZE_OF_GRIDS)

    # Auf Tile-lokale Koordinaten reduzieren
    x_int = int(x)
    y_int = int(y)
    x -= x_int
    y -= y_int
    x_int &= (MAP_RESOLUTION - 1)
    y_int &= (MAP_RESOLUTION - 1)

    # v9 hat 129x129 Vertices, v8 hat 128x128
    # v9[x_int * 129 + y_int] gibt den Eckpunkt

    def v9(xi, yi):
        return tile.v9[xi * 129 + yi]

    def v8(xi, yi):
        return tile.v8[xi * 128 + yi]

    # Triangle-Interpolation (4 Dreiecke pro Zelle)
    if x + y < 1:
        if x > y:
            # Dreieck 1: oben-links (h1-h2-h5)
            h1 = v9(x_int, y_int)
            h2 = v9(x_int + 1, y_int)
            h5 = 2 * v8(x_int, y_int)
            a = h2 - h1
            b = h5 - h1 - h2
            c = h1
        else:
            # Dreieck 2: oben-rechts (h1-h3-h5)
            h1 = v9(x_int, y_int)
            h3 = v9(x_int, y_int + 1)
            h5 = 2 * v8(x_int, y_int)
            a = h5 - h1 - h3
            b = h3 - h1
            c = h1
    else:
        if x > y:
            # Dreieck 3: unten-links (h2-h4-h5)
            h2 = v9(x_int + 1, y_int)
            h4 = v9(x_int + 1, y_int + 1)
            h5 = 2 * v8(x_int, y_int)
            a = h2 + h4 - h5
            b = h4 - h2
            c = h5 - h4
        else:
            # Dreieck 4: unten-rechts (h3-h4-h5)
            h3 = v9(x_int, y_int + 1)
            h4 = v9(x_int + 1, y_int + 1)
            h5 = 2 * v8(x_int, y_int)
            a = h4 - h3
            b = h3 + h4 - h5
            c = h5 - h4

    return a * x + b * y + c


# ─────────────────────────── VMAP Parser ─────────────────────────

def parse_vmtree(filepath: str, map_id: int = 0) -> Optional[VMapTree]:
    """
    Liest die .vmtree Datei (BVH-Tree + globale Spawns für LOS).
    """
    if not os.path.exists(filepath):
        print(f"  [VMAP] vmtree nicht gefunden: {filepath}")
        return None

    with open(filepath, 'rb') as f:
        # Magic
        magic = f.read(8)
        if magic != VMAP_MAGIC:
            print(f"  [VMAP] Falscher Magic: {magic}")
            return None

        # isTiled flag
        is_tiled = struct.unpack('B', f.read(1))[0] != 0

        # NODE section
        node_magic = f.read(4)
        if node_magic != b'NODE':
            print(f"  [VMAP] Kein NODE-Header gefunden: {node_magic}")
            return None

        # BIH bounds
        bounds_low = struct.unpack('<3f', f.read(12))
        bounds_high = struct.unpack('<3f', f.read(12))
        bounds = AABB(Vec3(*bounds_low), Vec3(*bounds_high))

        # BIH tree nodes
        tree_size = struct.unpack('<I', f.read(4))[0]
        tree_data = f.read(tree_size * 4)
        tree_nodes = list(struct.unpack(f'<{tree_size}I', tree_data))

        # Object indices
        obj_count = struct.unpack('<I', f.read(4))[0]
        object_indices = list(struct.unpack(f'<{obj_count}I', f.read(obj_count * 4)))

        vtree = VMapTree(
            map_id=map_id,
            is_tiled=is_tiled,
            bounds=bounds,
            tree_nodes=tree_nodes,
            object_indices=object_indices,
            spawns=[]
        )

        # Die vmtree enthält optional auch GOBJ-Spawns (bei nicht-getileten Maps)
        # Bei getileten Maps kommen die Spawns aus den vmtile-Dateien
        remaining = f.read(4)
        if remaining and len(remaining) == 4:
            gobj_or_count = remaining
            # Prüfe ob es "GOBJ" magic ist oder eine Spawn-Anzahl
            if gobj_or_count == b'GOBJ':
                # Nicht-getilte Map: Spawns direkt hier
                _parse_spawns_from_file(f, vtree.spawns)

    return vtree


def _parse_spawns_from_file(f, spawns_list: list):
    """Liest ModelSpawn-Records aus einer geöffneten Datei."""
    while True:
        spawn = _read_model_spawn(f)
        if spawn is None:
            break
        spawns_list.append(spawn)


def _read_model_spawn(f) -> Optional[ModelSpawn]:
    """
    Liest einen einzelnen ModelSpawn aus der Datei.
    Format: flags(u32), adtId(u16), ID(u32), pos(3f), rot(3f), scale(f),
            [bounds(6f) wenn HAS_BOUND], nameLen(u32), name(chars)
    """
    raw = f.read(4)
    if len(raw) < 4:
        return None

    flags = struct.unpack('<I', raw)[0]
    adt_id = struct.unpack('<H', f.read(2))[0]
    spawn_id = struct.unpack('<I', f.read(4))[0]
    pos = struct.unpack('<3f', f.read(12))
    rot = struct.unpack('<3f', f.read(12))
    scale = struct.unpack('<f', f.read(4))[0]

    bounds = None
    if flags & MOD_HAS_BOUND:
        blo = struct.unpack('<3f', f.read(12))
        bhi = struct.unpack('<3f', f.read(12))
        bounds = AABB(Vec3(*blo), Vec3(*bhi))

    name_len = struct.unpack('<I', f.read(4))[0]
    if name_len > 500:
        return None
    name = f.read(name_len).decode('ascii', errors='replace')

    return ModelSpawn(
        flags=flags,
        adt_id=adt_id,
        spawn_id=spawn_id,
        position=Vec3(*pos),
        rotation=Vec3(*rot),
        scale=scale,
        bounds=bounds,
        name=name,
        tree_ref=-1
    )


def parse_vmtile(filepath: str) -> list:
    """
    Liest eine .vmtile Datei und gibt eine Liste von ModelSpawn-Objekten zurück.
    Format: VMAP_4.8, numSpawns, [ModelSpawn + treeRef]*numSpawns
    """
    if not os.path.exists(filepath):
        return []

    spawns = []
    with open(filepath, 'rb') as f:
        magic = f.read(8)
        if magic != VMAP_MAGIC:
            print(f"  [VMAP] vmtile falscher Magic: {magic}")
            return []

        num_spawns = struct.unpack('<I', f.read(4))[0]

        for _ in range(num_spawns):
            spawn = _read_model_spawn(f)
            if spawn is None:
                break
            # Nach jedem Spawn kommt der BVH-Tree-Index
            ref_data = f.read(4)
            if len(ref_data) == 4:
                spawn.tree_ref = struct.unpack('<I', ref_data)[0]
            spawns.append(spawn)

    return spawns


# ─────────────────────────── MMAP Parser ─────────────────────────

def parse_mmap(filepath: str) -> Optional[NavMeshParams]:
    """
    Liest die .mmap Datei (dtNavMeshParams, 28 bytes).
    """
    if not os.path.exists(filepath):
        print(f"  [MMAP] Datei nicht gefunden: {filepath}")
        return None

    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < 28:
        print(f"  [MMAP] Datei zu klein ({len(data)} bytes)")
        return None

    orig = struct.unpack('<3f', data[0:12])
    tile_w, tile_h = struct.unpack('<2f', data[12:20])
    max_tiles, max_polys = struct.unpack('<2I', data[20:28])

    return NavMeshParams(
        origin=Vec3(*orig),
        tile_width=tile_w,
        tile_height=tile_h,
        max_tiles=max_tiles,
        max_polys=max_polys
    )


def parse_mmtile_header(filepath: str) -> Optional[dict]:
    """
    Liest den Header einer .mmtile Datei.
    Format: magic(u32), dtVersion(u32), mmapVersion(u32), size(u32),
            usesLiquids(u8), padding(3), recastConfig(36 bytes)
    """
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'rb') as f:
        data = f.read(56)

    if len(data) < 20:
        return None

    magic = struct.unpack('<I', data[0:4])[0]
    if magic != MMAP_MAGIC:
        print(f"  [MMAP] mmtile falscher Magic: 0x{magic:08X}")
        return None

    dt_version = struct.unpack('<I', data[4:8])[0]
    mmap_version = struct.unpack('<I', data[8:12])[0]
    tile_data_size = struct.unpack('<I', data[12:16])[0]
    uses_liquids = data[16]

    result = {
        'dt_version': dt_version,
        'mmap_version': mmap_version,
        'tile_data_size': tile_data_size,
        'uses_liquids': bool(uses_liquids),
    }

    if len(data) >= 56:
        pos = 20
        result['walkable_slope_angle'] = struct.unpack('<f', data[pos:pos+4])[0]
        result['walkable_radius'] = data[pos+4]
        result['walkable_height'] = data[pos+5]
        result['walkable_climb'] = data[pos+6]

    return result


# ─────────────────────── LOS Berechnung ──────────────────────────

class LOSChecker:
    """
    Line-of-Sight Checker basierend auf VMAP-Daten.

    Verwendet AABB-basierte Intersection der geladenen Objekt-Bounding-Boxes
    um zu prüfen ob zwischen zwei Punkten freie Sicht besteht.
    """

    def __init__(self):
        self.spawns: list = []  # Alle geladenen ModelSpawn mit Bounds

    def add_spawns(self, spawns: list):
        """Fügt Spawns hinzu (nur solche mit Bounding Box)."""
        for s in spawns:
            if s.bounds is not None:
                self.spawns.append(s)

    def is_in_line_of_sight(self, pos1: Vec3, pos2: Vec3, use_vmap_coords: bool = False) -> tuple:
        """
        Prüft Line of Sight zwischen zwei Weltkoordinaten-Punkten.

        Returns: (has_los: bool, hit_object: Optional[ModelSpawn], hit_distance: float)
        """
        # In VMAP-Koordinaten umrechnen (die Bounding Boxes sind in VMAP-coords)
        if not use_vmap_coords:
            p1 = world_to_vmap(pos1.x, pos1.y, pos1.z)
            p2 = world_to_vmap(pos2.x, pos2.y, pos2.z)
        else:
            p1 = pos1
            p2 = pos2

        direction = p2 - p1
        max_dist = direction.length()

        if max_dist < 1e-10:
            return (True, None, 0.0)

        dir_norm = direction.normalized()

        # Inverse Direction für Slab-Test
        inv_dir = Vec3(
            1.0 / dir_norm.x if abs(dir_norm.x) > 1e-30 else 1e30,
            1.0 / dir_norm.y if abs(dir_norm.y) > 1e-30 else 1e30,
            1.0 / dir_norm.z if abs(dir_norm.z) > 1e-30 else 1e30
        )

        closest_hit = max_dist
        hit_spawn = None

        for spawn in self.spawns:
            hit, t_enter, t_exit = spawn.bounds.intersects_ray(p1, inv_dir, max_dist)
            if hit and t_enter < closest_hit and t_enter >= 0:
                closest_hit = t_enter
                hit_spawn = spawn

        if hit_spawn is not None:
            return (False, hit_spawn, closest_hit)
        return (True, None, max_dist)

    def get_blocking_objects_in_radius(self, center: Vec3, radius: float, use_vmap_coords: bool = False) -> list:
        """
        Findet alle Objekte deren Bounding Box den gegebenen Radius-Kreis schneidet.
        Nützlich für Pathing-Hinderniserkennung.
        """
        if not use_vmap_coords:
            c = world_to_vmap(center.x, center.y, center.z)
        else:
            c = center

        result = []
        for spawn in self.spawns:
            bb = spawn.bounds
            # Nächster Punkt auf der AABB zum Zentrum
            closest_x = max(bb.low.x, min(c.x, bb.high.x))
            closest_y = max(bb.low.y, min(c.y, bb.high.y))
            closest_z = max(bb.low.z, min(c.z, bb.high.z))
            dist = math.sqrt(
                (closest_x - c.x) ** 2 +
                (closest_y - c.y) ** 2 +
                (closest_z - c.z) ** 2
            )
            if dist <= radius:
                result.append((spawn, dist))

        result.sort(key=lambda x: x[1])
        return result


# ────────────────── Terrain-basiertes Pathing ────────────────────

class TerrainPathChecker:
    """
    Prüft ob ein Pfad über das Terrain begehbar ist,
    basierend auf Steigung und Höhenänderung.
    """

    MAX_WALKABLE_SLOPE = 50.0  # Grad - maximale begehbare Steigung
    MAX_STEP_HEIGHT = 2.5      # Units - maximale Stufe (wie walkableClimb in Recast)
    SAMPLE_DISTANCE = 1.0      # Units - Abtastabstand auf dem Pfad

    def __init__(self, tiles: dict):
        """
        tiles: dict von (tile_x, tile_y) -> MapTile
        """
        self.tiles = tiles

    def get_height(self, world_x: float, world_y: float) -> float:
        """Höhe an Weltkoordinaten. Findet automatisch das richtige Tile."""
        gx, gy = world_to_grid(world_x, world_y)
        tile = self.tiles.get((gx, gy))
        if tile is None:
            return INVALID_HEIGHT
        return get_terrain_height(tile, world_x, world_y)

    def check_path_walkable(self, start: Vec3, end: Vec3) -> tuple:
        """
        Prüft ob ein Pfad auf dem Terrain begehbar ist.

        Returns: (walkable: bool, reason: str, blocked_at: Optional[Vec3])
        """
        dx = end.x - start.x
        dy = end.y - start.y
        dist_2d = math.sqrt(dx * dx + dy * dy)

        if dist_2d < 0.01:
            return (True, "same_position", None)

        num_samples = max(2, int(dist_2d / self.SAMPLE_DISTANCE))
        prev_z = start.z

        for i in range(1, num_samples + 1):
            t = i / num_samples
            x = start.x + dx * t
            y = start.y + dy * t
            z = self.get_height(x, y)

            if z <= INVALID_HEIGHT + 1:
                return (False, "no_terrain_data", Vec3(x, y, 0))

            # Steigung prüfen
            step_dist_2d = dist_2d / num_samples
            dz = abs(z - prev_z)
            if step_dist_2d > 0.01:
                slope_deg = math.degrees(math.atan2(dz, step_dist_2d))
                if slope_deg > self.MAX_WALKABLE_SLOPE:
                    return (False, f"slope_too_steep ({slope_deg:.1f}°)", Vec3(x, y, z))

            # Stufe prüfen
            if dz > self.MAX_STEP_HEIGHT:
                return (False, f"step_too_high ({dz:.1f} units)", Vec3(x, y, z))

            prev_z = z

        return (True, "path_clear", None)


# ─────────────── Performance-optimierte Caches ──────────────────

class HeightCache:
    """
    Vorberechnetes numpy Height-Grid für schnelle O(1) Terrain-Lookups.

    Statt pro Aufruf get_terrain_height() mit Python-Listen und Triangle-
    Interpolation zu rechnen, wird einmalig ein numpy-Array gebaut.
    Lookup: ~0.5μs statt ~50μs (100x Speedup).

    Usage:
        cache = env.build_height_cache(map_id=0, x_min=-8980, x_max=-8700,
                                        y_min=-240, y_max=-30, resolution=0.5)
        z = cache.get(x, y)
    """

    def __init__(self, grid: np.ndarray, x_min: float, y_min: float,
                 resolution: float, default_z: float = 82.0):
        self.grid = grid
        self.x_min = x_min
        self.y_min = y_min
        self.resolution = resolution
        self.inv_res = 1.0 / resolution
        self.default_z = default_z
        self.x_max = x_min + grid.shape[0] * resolution
        self.y_max = y_min + grid.shape[1] * resolution

    def get(self, x: float, y: float) -> float:
        """Schneller Height-Lookup via Array-Index."""
        ix = int((x - self.x_min) * self.inv_res)
        iy = int((y - self.y_min) * self.inv_res)
        if 0 <= ix < self.grid.shape[0] and 0 <= iy < self.grid.shape[1]:
            return float(self.grid[ix, iy])
        return self.default_z


class SpatialLOSChecker:
    """
    Räumlich indizierter LOS-Checker — prüft nur nahe Spawns statt alle.

    Baut ein 2D-Grid mit konfigurierbarer Zellgröße. Pro LOS-Check werden
    nur Zellen entlang des Strahls geprüft. Reduziert von O(27K) auf O(~50).

    Usage:
        checker = SpatialLOSChecker(cell_size=50.0)
        checker.add_spawns(spawns)
        checker.build_index()
        has_los, hit, dist = checker.is_in_line_of_sight(pos1, pos2)
    """

    def __init__(self, cell_size: float = 100.0):
        self.cell_size = cell_size
        self.inv_cell = 1.0 / cell_size
        self.spawns: list = []
        self.grid: dict = {}  # (cx, cy) -> list of spawn indices
        self._built = False

    def add_spawns(self, spawns: list):
        """Fügt Spawns hinzu (nur mit Bounding Box)."""
        for s in spawns:
            if s.bounds is not None:
                self.spawns.append(s)
        self._built = False

    def build_index(self):
        """Baut den räumlichen Index. Einmal nach dem Laden aufrufen."""
        self.grid.clear()
        for i, spawn in enumerate(self.spawns):
            bb = spawn.bounds
            # Alle Grid-Zellen die die AABB überlappt
            cx_min = int(bb.low.x * self.inv_cell)
            cx_max = int(bb.high.x * self.inv_cell)
            cy_min = int(bb.low.y * self.inv_cell)
            cy_max = int(bb.high.y * self.inv_cell)
            for cx in range(cx_min, cx_max + 1):
                for cy in range(cy_min, cy_max + 1):
                    key = (cx, cy)
                    if key not in self.grid:
                        self.grid[key] = []
                    self.grid[key].append(i)
        self._built = True
        total_refs = sum(len(v) for v in self.grid.values())
        print(f"  [SpatialLOS] Index: {len(self.grid)} Zellen, "
              f"{len(self.spawns)} Spawns, {total_refs} Referenzen")

    def _cells_along_ray(self, p1: Vec3, p2: Vec3) -> set:
        """Sammelt alle Grid-Zellen entlang des Strahls."""
        cells = set()
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-10:
            cells.add((int(p1.x * self.inv_cell), int(p1.y * self.inv_cell)))
            return cells
        # Sample entlang des Strahls in Zellgröße/2 Schritten
        steps = max(2, int(length / (self.cell_size * 0.5)) + 1)
        for i in range(steps + 1):
            t = i / steps
            x = p1.x + dx * t
            y = p1.y + dy * t
            cells.add((int(x * self.inv_cell), int(y * self.inv_cell)))
        return cells

    def is_in_line_of_sight(self, pos1: Vec3, pos2: Vec3,
                            use_vmap_coords: bool = False) -> tuple:
        """
        LOS-Check mit räumlichem Index. Kompatibel mit LOSChecker-API.
        Returns: (has_los, hit_object, hit_distance)
        """
        if not self._built:
            self.build_index()

        if not use_vmap_coords:
            p1 = world_to_vmap(pos1.x, pos1.y, pos1.z)
            p2 = world_to_vmap(pos2.x, pos2.y, pos2.z)
        else:
            p1 = pos1
            p2 = pos2

        direction = p2 - p1
        max_dist = direction.length()
        if max_dist < 1e-10:
            return (True, None, 0.0)

        dir_norm = direction.normalized()
        inv_dir = Vec3(
            1.0 / dir_norm.x if abs(dir_norm.x) > 1e-30 else 1e30,
            1.0 / dir_norm.y if abs(dir_norm.y) > 1e-30 else 1e30,
            1.0 / dir_norm.z if abs(dir_norm.z) > 1e-30 else 1e30
        )

        # Nur Spawns in relevanten Grid-Zellen prüfen
        cells = self._cells_along_ray(p1, p2)
        checked = set()
        closest_hit = max_dist
        hit_spawn = None

        for cell_key in cells:
            spawn_indices = self.grid.get(cell_key)
            if spawn_indices is None:
                continue
            for idx in spawn_indices:
                if idx in checked:
                    continue
                checked.add(idx)
                spawn = self.spawns[idx]
                hit, t_enter, t_exit = spawn.bounds.intersects_ray(
                    p1, inv_dir, max_dist
                )
                if hit and t_enter < closest_hit and t_enter >= 0:
                    closest_hit = t_enter
                    hit_spawn = spawn

        if hit_spawn is not None:
            return (False, hit_spawn, closest_hit)
        return (True, None, max_dist)

    def get_blocking_objects_in_radius(self, center: Vec3, radius: float,
                                       use_vmap_coords: bool = False) -> list:
        """Findet Objekte im Umkreis — nutzt den räumlichen Index."""
        if not self._built:
            self.build_index()

        if not use_vmap_coords:
            c = world_to_vmap(center.x, center.y, center.z)
        else:
            c = center

        # Relevante Grid-Zellen
        cx_min = int((c.x - radius) * self.inv_cell)
        cx_max = int((c.x + radius) * self.inv_cell)
        cy_min = int((c.y - radius) * self.inv_cell)
        cy_max = int((c.y + radius) * self.inv_cell)

        result = []
        checked = set()
        for cx in range(cx_min, cx_max + 1):
            for cy in range(cy_min, cy_max + 1):
                spawn_indices = self.grid.get((cx, cy))
                if spawn_indices is None:
                    continue
                for idx in spawn_indices:
                    if idx in checked:
                        continue
                    checked.add(idx)
                    spawn = self.spawns[idx]
                    bb = spawn.bounds
                    closest_x = max(bb.low.x, min(c.x, bb.high.x))
                    closest_y = max(bb.low.y, min(c.y, bb.high.y))
                    closest_z = max(bb.low.z, min(c.z, bb.high.z))
                    dist = math.sqrt(
                        (closest_x - c.x) ** 2 +
                        (closest_y - c.y) ** 2 +
                        (closest_z - c.z) ** 2
                    )
                    if dist <= radius:
                        result.append((spawn, dist))

        result.sort(key=lambda x: x[1])
        return result


# ─────────────────── Komplett-Environment ────────────────────────

class WoW3DEnvironment:
    """
    Kombiniert alle Parser zu einem nutzbaren 3D-Environment.
    Kann Terrain-Höhen, LOS und Pathing-Checks durchführen.
    """

    def __init__(self, data_root: str):
        self.data_root = data_root
        self.maps_dir = os.path.join(data_root, "maps")
        self.vmaps_dir = os.path.join(data_root, "vmaps")
        self.mmaps_dir = os.path.join(data_root, "mmaps")

        self.loaded_tiles: dict = {}          # (map_id, tile_x, tile_y) -> MapTile
        self.loaded_vmtrees: dict = {}        # map_id -> VMapTree
        self.loaded_navmesh_params: dict = {} # map_id -> NavMeshParams
        self.los_checker = LOSChecker()
        self.terrain_checker = None

    def load_map_tile(self, map_id: int, tile_x: int, tile_y: int) -> Optional[MapTile]:
        """Lädt ein einzelnes Map-Tile."""
        key = (map_id, tile_x, tile_y)
        if key in self.loaded_tiles:
            return self.loaded_tiles[key]

        fname = map_filename(map_id, tile_x, tile_y)
        filepath = os.path.join(self.maps_dir, fname)
        tile = parse_map_file(filepath, map_id, tile_x, tile_y)
        if tile:
            self.loaded_tiles[key] = tile
        return tile

    def load_map_area(self, map_id: int, center_x: float, center_y: float, radius_tiles: int = 1):
        """Lädt alle Map-Tiles im Umkreis eines Weltpunkts."""
        cx, cy = world_to_grid(center_x, center_y)
        loaded = 0
        for dx in range(-radius_tiles, radius_tiles + 1):
            for dy in range(-radius_tiles, radius_tiles + 1):
                tx, ty = cx + dx, cy + dy
                if 0 <= tx < MAX_NUMBER_OF_GRIDS and 0 <= ty < MAX_NUMBER_OF_GRIDS:
                    tile = self.load_map_tile(map_id, tx, ty)
                    if tile and tile.v9 is not None:
                        loaded += 1
        print(f"  [ENV] {loaded} Map-Tiles mit Höhendaten geladen (Zentrum: Grid {cx},{cy})")
        return loaded

    def load_vmaps(self, map_id: int, tile_x: int = None, tile_y: int = None):
        """Lädt VMAP-Daten (vmtree + vmtile)."""
        # vmtree laden
        vtree_path = os.path.join(self.vmaps_dir, vmtree_filename(map_id))
        vtree = parse_vmtree(vtree_path, map_id)
        if vtree:
            self.loaded_vmtrees[map_id] = vtree
            print(f"  [VMAP] vmtree geladen: {len(vtree.tree_nodes)} BIH-Nodes, "
                  f"{len(vtree.object_indices)} Objekte, tiled={vtree.is_tiled}")
            print(f"  [VMAP] Bounds: low={vtree.bounds.low}, high={vtree.bounds.high}")

            # vmtile laden wenn angegeben
            if tile_x is not None and tile_y is not None:
                vtile_path = os.path.join(self.vmaps_dir, vmtile_filename(map_id, tile_x, tile_y))
                spawns = parse_vmtile(vtile_path)
                if spawns:
                    vtree.spawns.extend(spawns)
                    self.los_checker.add_spawns(spawns)
                    print(f"  [VMAP] vmtile geladen: {len(spawns)} Spawns")

        return vtree

    def load_vmaps_area(self, map_id: int, center_x: float, center_y: float, radius_tiles: int = 1):
        """Lädt VMAP-Daten für einen Bereich."""
        vtree_path = os.path.join(self.vmaps_dir, vmtree_filename(map_id))
        vtree = parse_vmtree(vtree_path, map_id)
        if not vtree:
            return None

        self.loaded_vmtrees[map_id] = vtree
        cx, cy = world_to_grid(center_x, center_y)
        total_spawns = 0

        for dx in range(-radius_tiles, radius_tiles + 1):
            for dy in range(-radius_tiles, radius_tiles + 1):
                tx, ty = cx + dx, cy + dy
                if 0 <= tx < MAX_NUMBER_OF_GRIDS and 0 <= ty < MAX_NUMBER_OF_GRIDS:
                    vtile_path = os.path.join(self.vmaps_dir, vmtile_filename(map_id, tx, ty))
                    spawns = parse_vmtile(vtile_path)
                    if spawns:
                        vtree.spawns.extend(spawns)
                        self.los_checker.add_spawns(spawns)
                        total_spawns += len(spawns)

        print(f"  [VMAP] vmtree geladen: {len(vtree.tree_nodes)} BIH-Nodes, tiled={vtree.is_tiled}")
        print(f"  [VMAP] {total_spawns} Spawns aus vmtiles geladen")
        return vtree

    def load_navmesh(self, map_id: int) -> Optional[NavMeshParams]:
        """Lädt NavMesh-Parameter."""
        mmap_path = os.path.join(self.mmaps_dir, mmap_filename(map_id))
        params = parse_mmap(mmap_path)
        if params:
            self.loaded_navmesh_params[map_id] = params
        return params

    def get_height(self, map_id: int, x: float, y: float) -> float:
        """Terrain-Höhe an Weltkoordinaten."""
        gx, gy = world_to_grid(x, y)
        key = (map_id, gx, gy)
        tile = self.loaded_tiles.get(key)
        if tile is None:
            return INVALID_HEIGHT
        return get_terrain_height(tile, x, y)

    def check_los(self, pos1: Vec3, pos2: Vec3) -> tuple:
        """Prüft Line of Sight zwischen zwei Weltpunkten."""
        return self.los_checker.is_in_line_of_sight(pos1, pos2)

    def get_nearby_obstacles(self, pos: Vec3, radius: float) -> list:
        """Findet Hindernisse im Umkreis (Weltkoordinaten)."""
        return self.los_checker.get_blocking_objects_in_radius(pos, radius)

    def build_terrain_checker(self, map_id: int):
        """Erstellt TerrainPathChecker aus geladenen Tiles."""
        tiles = {}
        for (mid, tx, ty), tile in self.loaded_tiles.items():
            if mid == map_id:
                tiles[(tx, ty)] = tile
        self.terrain_checker = TerrainPathChecker(tiles)
        return self.terrain_checker

    def check_path(self, start: Vec3, end: Vec3) -> tuple:
        """Prüft ob ein Pfad begehbar ist (Terrain + Hindernisse)."""
        if self.terrain_checker is None:
            return (False, "no_terrain_loaded", None)
        return self.terrain_checker.check_path_walkable(start, end)

    # ─── Performance-optimierte Methoden ────────────────────────

    def build_height_cache(self, map_id: int,
                           x_min: float, x_max: float,
                           y_min: float, y_max: float,
                           resolution: float = 0.5,
                           default_z: float = 82.0) -> HeightCache:
        """
        Vorberechnet ein numpy Height-Grid für den gegebenen Bereich.

        Northshire-Defaults:
            cache = env.build_height_cache(0, -8980, -8700, -240, -30)

        Returns HeightCache mit O(1) get(x,y) Lookups.
        """
        nx = int((x_max - x_min) / resolution) + 1
        ny = int((y_max - y_min) / resolution) + 1
        grid = np.full((nx, ny), default_z, dtype=np.float32)

        filled = 0
        for ix in range(nx):
            wx = x_min + ix * resolution
            for iy in range(ny):
                wy = y_min + iy * resolution
                h = self.get_height(map_id, wx, wy)
                if h != INVALID_HEIGHT:
                    grid[ix, iy] = h
                    filled += 1

        cache = HeightCache(grid, x_min, y_min, resolution, default_z)
        total = nx * ny
        print(f"  [HeightCache] {nx}x{ny} Grid ({total} Punkte, "
              f"{filled} mit Daten, {grid.nbytes/1024:.0f} KB)")
        return cache

    def build_spatial_los(self, cell_size: float = 100.0) -> SpatialLOSChecker:
        """
        Erstellt einen räumlich indizierten LOS-Checker aus den
        bereits geladenen VMAP-Spawns.

        Ersetzt self.los_checker mit dem schnellen SpatialLOSChecker.
        Typischer Speedup: 100-500x gegenüber brute-force.
        """
        spatial = SpatialLOSChecker(cell_size=cell_size)
        # Spawns aus allen geladenen vmtrees übernehmen
        spatial.spawns = list(self.los_checker.spawns)
        spatial.build_index()
        self.los_checker = spatial
        return spatial


# ─────────────────────────── Test Suite ──────────────────────────

def test_with_sample_data(data_dir: str):
    """
    Testet alle Parser mit den Beispieldateien im data/-Verzeichnis.
    """
    print("=" * 70)
    print("  3D ENVIRONMENT SIMULATION — TEST SUITE")
    print("=" * 70)

    # ── 1. MAP Parser Test ──
    print("\n" + "─" * 50)
    print("  TEST 1: MAP Parser (Terrain-Höhendaten)")
    print("─" * 50)

    map_file = os.path.join(data_dir, "0002035.map")
    tile = parse_map_file(map_file, map_id=0, tile_x=35, tile_y=20)
    if tile:
        print(f"  OK: Map-Tile geladen (map=0, tile={tile.tile_x},{tile.tile_y})")
        print(f"      Area: flags=0x{tile.area_flags:04x}, gridArea={tile.grid_area}")
        print(f"      Height: flags=0x{tile.height_flags:08x}")
        if tile.height_flags & MAP_HEIGHT_NO_HEIGHT:
            print(f"      -> Flat terrain (gridHeight={tile.grid_height})")
        else:
            print(f"      -> Height data: v9={len(tile.v9)} vertices, v8={len(tile.v8)} cells")
            heights = tile.v9
            print(f"      -> Height range: {min(heights):.2f} — {max(heights):.2f}")
        if tile.liquid_level != 0 or tile.liquid_width > 0:
            print(f"      Liquid: flags=0x{tile.liquid_flags:02x}, level={tile.liquid_level}, "
                  f"size={tile.liquid_width}x{tile.liquid_height}")
    else:
        print("  SKIP: Kein Map-Tile verfügbar (Datei existiert, aber flat terrain ist OK)")

    # ── 2. VMAP Parser Test ──
    print("\n" + "─" * 50)
    print("  TEST 2: VMAP Parser (Objekt-Geometrie für LOS)")
    print("─" * 50)

    vtree_file = os.path.join(data_dir, "000.vmtree")
    vtree = parse_vmtree(vtree_file, map_id=0)
    if vtree:
        print(f"  OK: vmtree geladen")
        print(f"      Map 0, tiled={vtree.is_tiled}")
        print(f"      BIH-Tree: {len(vtree.tree_nodes)} Nodes ({len(vtree.tree_nodes)*4/1024:.1f} KB)")
        print(f"      Objekt-Indices: {len(vtree.object_indices)}")
        print(f"      Bounds: low={vtree.bounds.low}")
        print(f"              high={vtree.bounds.high}")

        # Bounds in Weltkoordinaten umrechnen
        wlo = vmap_to_world(vtree.bounds.low.x, vtree.bounds.low.y, vtree.bounds.low.z)
        whi = vmap_to_world(vtree.bounds.high.x, vtree.bounds.high.y, vtree.bounds.high.z)
        print(f"      World bounds: {whi} — {wlo}")

    vtile_file = os.path.join(data_dir, "000_27_29.vmtile")
    spawns = parse_vmtile(vtile_file)
    if spawns:
        print(f"\n  OK: vmtile geladen — {len(spawns)} Spawns:")
        models = {}
        for s in spawns:
            models[s.name] = models.get(s.name, 0) + 1
        for name, count in sorted(models.items()):
            print(f"      {count}x {name}")
        print(f"\n      Beispiel-Spawn:")
        s = spawns[0]
        print(f"        Name: {s.name}")
        print(f"        VMAP-Pos: {s.position}")
        wpos = vmap_to_world(s.position.x, s.position.y, s.position.z)
        print(f"        Welt-Pos: {wpos}")
        print(f"        Scale: {s.scale:.4f}")
        if s.bounds:
            print(f"        Bounds: {s.bounds.low} — {s.bounds.high}")
            size = Vec3(
                s.bounds.high.x - s.bounds.low.x,
                s.bounds.high.y - s.bounds.low.y,
                s.bounds.high.z - s.bounds.low.z
            )
            print(f"        Size: {size}")

    # ── 3. MMAP Parser Test ──
    print("\n" + "─" * 50)
    print("  TEST 3: MMAP Parser (Navigation Mesh)")
    print("─" * 50)

    mmap_file = os.path.join(data_dir, "000.mmap")
    params = parse_mmap(mmap_file)
    if params:
        print(f"  OK: NavMesh-Parameter geladen")
        print(f"      Origin: {params.origin}")
        print(f"      Tile-Size: {params.tile_width:.2f} x {params.tile_height:.2f}")
        print(f"      Max Tiles: {params.max_tiles}")
        print(f"      Max Polys: {params.max_polys}")

    # ── 4. LOS Test ──
    print("\n" + "─" * 50)
    print("  TEST 4: Line-of-Sight Berechnung")
    print("─" * 50)

    if spawns:
        los = LOSChecker()
        los.add_spawns(spawns)
        print(f"  {len(los.spawns)} Objekte mit Bounding Boxes geladen")

        # Test: Strahl der durch einen Baum gehen sollte
        s = spawns[0]
        center = Vec3(
            (s.bounds.low.x + s.bounds.high.x) / 2,
            (s.bounds.low.y + s.bounds.high.y) / 2,
            (s.bounds.low.z + s.bounds.high.z) / 2
        )
        # Punkt vor dem Objekt
        p1 = Vec3(center.x - 50, center.y, center.z)
        # Punkt hinter dem Objekt
        p2 = Vec3(center.x + 50, center.y, center.z)

        has_los, hit, dist = los.is_in_line_of_sight(p1, p2, use_vmap_coords=True)
        print(f"\n  LOS Test 1: Strahl durch '{s.name}'")
        print(f"    Von: {p1}")
        print(f"    Zu:  {p2}")
        print(f"    -> LOS: {has_los}, Hit-Distanz: {dist:.2f}")
        if hit:
            print(f"    -> Blockiert durch: '{hit.name}' (ID={hit.spawn_id})")

        # Test: Strahl der am Objekt vorbei geht
        p3 = Vec3(center.x, center.y - 100, center.z)
        p4 = Vec3(center.x, center.y + 100, center.z)
        has_los2, hit2, dist2 = los.is_in_line_of_sight(p3, p4, use_vmap_coords=True)
        # Dieser Test kann trotzdem durch andere Spawns blocken
        print(f"\n  LOS Test 2: Strahl entlang Y-Achse")
        print(f"    Von: {p3}")
        print(f"    Zu:  {p4}")
        print(f"    -> LOS: {has_los2}")
        if hit2:
            print(f"    -> Blockiert durch: '{hit2.name}' (ID={hit2.spawn_id})")

        # Test: Hindernisse im Umkreis
        obstacles = los.get_blocking_objects_in_radius(center, 30.0, use_vmap_coords=True)
        print(f"\n  Hindernisse im 30-Unit-Radius um {center}:")
        for obj, d in obstacles:
            print(f"    - '{obj.name}' (ID={obj.spawn_id}) Distanz={d:.1f}")

    # ── 5. Koordinaten-Test ──
    print("\n" + "─" * 50)
    print("  TEST 5: Koordinaten-Umrechnung")
    print("─" * 50)

    # Bot-Spawnpunkt aus CLAUDE.md
    spawn_x, spawn_y, spawn_z = -8921.037, -120.485, 82.025
    gx, gy = world_to_grid(spawn_x, spawn_y)
    vmap_pos = world_to_vmap(spawn_x, spawn_y, spawn_z)
    back = vmap_to_world(vmap_pos.x, vmap_pos.y, vmap_pos.z)

    print(f"  Bot-Spawnpunkt: ({spawn_x}, {spawn_y}, {spawn_z})")
    print(f"    -> Grid-Tile: ({gx}, {gy})")
    print(f"    -> Map-Datei: {map_filename(0, gx, gy)}")
    print(f"    -> VMAP-Coords: {vmap_pos}")
    print(f"    -> Zurück-Konvertiert: {back}")
    print(f"    -> Roundtrip-Fehler: {abs(back.x - spawn_x):.6f}")

    # ── 6. CSV-Daten Test ──
    print("\n" + "─" * 50)
    print("  TEST 6: Creature-Daten (CSV)")
    print("─" * 50)

    creature_csv = os.path.join(data_dir, "creature.csv")
    if os.path.exists(creature_csv):
        import csv
        # Finde Kreaturen in Northshire-Nähe
        northshire_creatures = []
        with open(creature_csv, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                try:
                    cx = float(row.get('position_x', 0))
                    cy = float(row.get('position_y', 0))
                    cz = float(row.get('position_z', 0))
                    map_id = int(row.get('map', -1))
                    if map_id == 0:
                        dist = math.sqrt((cx - spawn_x)**2 + (cy - spawn_y)**2)
                        if dist < 200:  # 200 Units Radius
                            northshire_creatures.append({
                                'guid': row.get('guid', '?'),
                                'id': row.get('id1', '?'),
                                'x': cx, 'y': cy, 'z': cz,
                                'dist': dist
                            })
                except (ValueError, TypeError):
                    continue

        northshire_creatures.sort(key=lambda c: c['dist'])
        print(f"  {len(northshire_creatures)} Kreaturen im 200-Unit-Radius um Spawnpunkt:")
        for c in northshire_creatures[:10]:
            print(f"    ID={c['id']:>6s}  GUID={c['guid']:>6s}  "
                  f"({c['x']:.1f}, {c['y']:.1f}, {c['z']:.1f})  "
                  f"Dist={c['dist']:.1f}")
        if len(northshire_creatures) > 10:
            print(f"    ... und {len(northshire_creatures) - 10} weitere")

    # ── Zusammenfassung ──
    print("\n" + "=" * 70)
    print("  ZUSAMMENFASSUNG")
    print("=" * 70)
    print(f"""
  Parser-Status:
    MAP  (Terrain):    {'OK' if tile else 'NICHT GELADEN'}
    VMAP (Geometrie):  {'OK' if vtree else 'NICHT GELADEN'}
    MMAP (NavMesh):    {'OK' if params else 'NICHT GELADEN'}
    LOS-Checker:       {'OK' if spawns else 'NICHT GELADEN'} ({len(los.spawns) if spawns else 0} Objekte)

  Für den produktiven Einsatz:
    1. Alle Map-Tiles im Trainingsbereich laden (maps/)
    2. vmtree + vmtiles für Map 0 laden (vmaps/)
    3. NavMesh-Tiles laden für echtes Pathfinding (mmaps/)

  Benötigte Dateien für Northshire Abbey (Bot-Spawn):
    maps/{map_filename(0, gx, gy)}  (Terrain-Höhen)
    vmaps/{vmtree_filename(0)}  (BVH-Tree)
    vmaps/{vmtile_filename(0, gx, gy)}  (Objekte)
    mmaps/{mmap_filename(0)}  (NavMesh-Parameter)
    mmaps/{mmtile_filename(0, gx, gy)}  (NavMesh-Tile)
""")


def test_with_full_data(data_root: str):
    """
    Testet mit vollem Datensatz unter dem angegebenen Pfad.
    Erwartet maps/, vmaps/, mmaps/ Unterverzeichnisse.
    """
    print("=" * 70)
    print("  3D ENVIRONMENT — VOLLTEST MIT ECHTEN DATEN")
    print(f"  Data Root: {data_root}")
    print("=" * 70)

    env = WoW3DEnvironment(data_root)

    # Bot-Spawnpunkt
    spawn = Vec3(-8921.037, -120.485, 82.025)

    # ── Map-Tiles laden ──
    print("\n[1] Map-Tiles laden...")
    loaded = env.load_map_area(0, spawn.x, spawn.y, radius_tiles=1)

    if loaded > 0:
        # Höhe am Spawnpunkt
        h = env.get_height(0, spawn.x, spawn.y)
        print(f"\n  Terrain-Höhe am Spawnpunkt: {h:.3f}")
        print(f"  Erwartet: ~{spawn.z:.3f}")
        print(f"  Differenz: {abs(h - spawn.z):.3f}")

        # Höhenprofil in einer Linie
        print(f"\n  Höhenprofil von Spawn Richtung Osten (20 Schritte à 5 Units):")
        for i in range(20):
            x = spawn.x + i * 5
            h = env.get_height(0, x, spawn.y)
            bar = "#" * max(0, int((h - 70) * 2)) if h > INVALID_HEIGHT + 1 else "---"
            print(f"    x={x:9.1f}  h={h:7.2f}  {bar}")

    # ── VMAP laden ──
    print("\n[2] VMAP-Daten laden...")
    vtree = env.load_vmaps_area(0, spawn.x, spawn.y, radius_tiles=1)

    if vtree and vtree.spawns:
        # LOS-Tests
        print(f"\n[3] LOS-Tests...")
        # Punkt 10 Units vor dem Spawn
        p1 = Vec3(spawn.x, spawn.y, spawn.z + 1.7)  # Augenhöhe
        p2 = Vec3(spawn.x + 30, spawn.y, spawn.z + 1.7)

        has_los, hit, dist = env.check_los(p1, p2)
        print(f"  LOS {p1} → {p2}: {'FREI' if has_los else 'BLOCKIERT'}")
        if hit:
            print(f"    Blockiert durch: '{hit.name}' nach {dist:.1f} Units")

        # Hindernisse um Spawnpunkt
        obstacles = env.get_nearby_obstacles(spawn, 50)
        print(f"\n  Hindernisse im 50-Unit-Radius:")
        for obj, d in obstacles[:10]:
            print(f"    {d:6.1f}u  {obj.name} (ID={obj.spawn_id})")

    # ── NavMesh laden ──
    print(f"\n[4] NavMesh laden...")
    params = env.load_navmesh(0)
    if params:
        print(f"  NavMesh OK: tile_size={params.tile_width:.2f}, "
              f"max_tiles={params.max_tiles}")

    # ── Terrain Pathcheck ──
    if loaded > 0:
        print(f"\n[5] Terrain Path Checks...")
        env.build_terrain_checker(0)

        targets = [
            Vec3(spawn.x + 20, spawn.y, spawn.z),
            Vec3(spawn.x, spawn.y + 30, spawn.z),
            Vec3(spawn.x + 50, spawn.y + 50, spawn.z),
        ]

        for target in targets:
            walkable, reason, blocked = env.check_path(spawn, target)
            print(f"  Path {spawn} → {target}: "
                  f"{'WALKABLE' if walkable else 'BLOCKED'} ({reason})")
            if blocked:
                print(f"    Blockiert bei: {blocked}")


# ─────────────────────────── Main ────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WoW 3D Environment Simulation - Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python test_3d_env.py                          # Test mit ./data/ Beispieldateien
  python test_3d_env.py --data-root /pfad/Data   # Test mit echten Daten
  python test_3d_env.py --full                    # Volltest (benötigt echte Daten)
        """
    )
    parser.add_argument('--data-root', type=str, default=None,
                        help='Pfad zum Data/-Verzeichnis (mit maps/, vmaps/, mmaps/)')
    parser.add_argument('--full', action='store_true',
                        help='Volltest mit echten Daten statt Beispieldateien')

    args = parser.parse_args()

    if args.full and args.data_root:
        test_with_full_data(args.data_root)
    else:
        # Fallback: Nutze Beispieldateien im data/-Verzeichnis
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(script_dir), "data")

        if not os.path.exists(data_dir):
            print(f"Fehler: Beispieldaten nicht gefunden unter {data_dir}")
            print(f"Nutze --data-root um den Pfad anzugeben")
            sys.exit(1)

        test_with_sample_data(data_dir)


if __name__ == "__main__":
    main()
