"""Build a stitched world-map PNG from WoW WorldMap BLP tiles.

Reads WorldMapArea.dbc for coordinate mapping and stitches zone tiles from
a ZIP archive (or extracted directory) of Interface/WorldMap/ BLP files.

Outputs:
  - A single PNG image of the composited map
  - A JSON sidecar file with WoW coordinate bounds (for --map-bounds)

Usage:
    # Build Eastern Kingdoms map from ZIP
    python -m sim.build_world_map \
        --zip data/1659008088-atlasworldmap_wotlk.zip \
        --dbc data/dbc/WorldMapArea.dbc \
        --map-id 0 \
        --output data/eastern_kingdoms.png

    # Build just Elwynn Forest + surrounding zones
    python -m sim.build_world_map \
        --zip data/1659008088-atlasworldmap_wotlk.zip \
        --dbc data/dbc/WorldMapArea.dbc \
        --map-id 0 --zones Elwynn,Westfall,Redridge,Duskwood,DunMorogh,Stormwind \
        --output data/elwynn_area.png

    # Build just the continent overview
    python -m sim.build_world_map \
        --zip data/1659008088-atlasworldmap_wotlk.zip \
        --dbc data/dbc/WorldMapArea.dbc \
        --map-id 0 --continent-only \
        --output data/ek_continent.png

    # Use the output with the visualizer:
    python -m sim.visualize --log-dir logs/sim_episodes/ \
        --map-image data/eastern_kingdoms.png
    # (bounds are auto-loaded from the .json sidecar)
"""

import os
import sys
import json
import struct
import argparse
import zipfile
from io import BytesIO
from dataclasses import dataclass

# Default paths on the dev machine
_DEFAULT_DATA_ROOT = r"C:\wowstuff\WoWKI_serv\Data"
_DEFAULT_ZIP = os.path.join(_DEFAULT_DATA_ROOT, "1659008088-atlasworldmap_wotlk.zip")
_DEFAULT_DBC = os.path.join(_DEFAULT_DATA_ROOT, "dbc", "WorldMapArea.dbc")
_DEFAULT_OUTPUT = os.path.join(_DEFAULT_DATA_ROOT, "world_map.png")

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ─── WorldMapArea DBC Parser ────────────────────────────────────────────

@dataclass
class WorldMapAreaEntry:
    id: int
    map_id: int
    area_id: int
    internal_name: str
    y1: float  # left bound (WoW Y)
    y2: float  # right bound (WoW Y)
    x1: float  # top bound (WoW X)
    x2: float  # bottom bound (WoW X)
    virtual_map_id: int


def parse_world_map_area_dbc(filepath: str) -> list:
    """Parse WorldMapArea.dbc and return list of WorldMapAreaEntry.

    DBC format: xinxffffixx (11 uint32 fields per record, 44 bytes).
    """
    data = open(filepath, "rb").read()
    magic = data[0:4]
    if magic != b"WDBC":
        print(f"ERROR: Not a WDBC file: {filepath}")
        return []

    record_count, field_count, record_size, string_size = struct.unpack(
        "<4I", data[4:20])
    records_start = 20
    string_table_start = records_start + record_count * record_size

    def get_string(offset):
        if offset == 0 or string_table_start + offset >= len(data):
            return ""
        end = data.index(b"\0", string_table_start + offset)
        return data[string_table_start + offset:end].decode(
            "utf-8", errors="replace")

    entries = []
    for i in range(record_count):
        pos = records_start + i * record_size
        n_fields = record_size // 4
        fields = struct.unpack(f"<{n_fields}I", data[pos:pos + record_size])

        # Format: xinxffffixx
        rec_id = fields[0]
        map_id = fields[1]
        area_id = fields[2]
        internal_name = get_string(fields[3])
        y1 = struct.unpack("<f", struct.pack("<I", fields[4]))[0]
        y2 = struct.unpack("<f", struct.pack("<I", fields[5]))[0]
        x1 = struct.unpack("<f", struct.pack("<I", fields[6]))[0]
        x2 = struct.unpack("<f", struct.pack("<I", fields[7]))[0]
        virtual_map_id = struct.unpack("<i", struct.pack("<I", fields[8]))[0]

        entries.append(WorldMapAreaEntry(
            id=rec_id, map_id=map_id, area_id=area_id,
            internal_name=internal_name,
            y1=y1, y2=y2, x1=x1, x2=x2,
            virtual_map_id=virtual_map_id))

    return entries


# ─── Tile Stitcher ──────────────────────────────────────────────────────

def _read_blp_fallback(data: bytes) -> Image.Image:
    """Fallback BLP2 reader for encodings Pillow can't handle (e.g. encoding 3).

    BLP2 header (148 bytes):
      magic(4) type(4) encoding(1) alphaDepth(1) alphaEncoding(1) hasMips(1)
      width(4) height(4) mipOffsets(16*4) mipSizes(16*4)
    """
    if data[:4] != b"BLP2":
        raise ValueError("Not a BLP2 file")

    _type, encoding, alpha_depth, alpha_enc, has_mips = struct.unpack_from(
        "<IBBBB", data, 4)
    width, height = struct.unpack_from("<II", data, 12)
    mip_offsets = struct.unpack_from("<16I", data, 20)
    mip_sizes = struct.unpack_from("<16I", data, 84)

    offset = mip_offsets[0]
    size = mip_sizes[0]

    if encoding == 3:
        # Uncompressed BGRA
        pixels = data[offset:offset + size]
        if len(pixels) < width * height * 4:
            raise ValueError(f"BLP encoding 3: not enough pixel data "
                             f"({len(pixels)} < {width * height * 4})")
        img = Image.frombytes("RGBA", (width, height), pixels, "raw", "BGRA")
        return img
    elif encoding == 1:
        # Palette-based: 256 BGRA palette + indexed pixels
        palette_data = data[offset:offset + 256 * 4]
        idx_offset = offset + 256 * 4
        idx_data = data[idx_offset:idx_offset + width * height]
        # Build RGB image from palette
        palette_rgb = []
        for i in range(256):
            b, g, r, a = palette_data[i * 4:(i + 1) * 4]
            palette_rgb.extend([r, g, b])
        img = Image.new("P", (width, height))
        img.putpalette(palette_rgb)
        img.putdata(list(idx_data))
        return img.convert("RGBA")

    raise ValueError(f"Unsupported BLP encoding {encoding}")


def _open_image_with_blp_fallback(data: bytes) -> Image.Image:
    """Try Pillow first, fall back to custom BLP reader."""
    try:
        img = Image.open(BytesIO(data))
        img.load()  # force decode to catch BLP errors early
        return img
    except Exception:
        # Try custom BLP reader
        return _read_blp_fallback(data)


TILE_W, TILE_H = 256, 256
GRID_COLS, GRID_ROWS = 4, 3
ZONE_W = TILE_W * GRID_COLS   # 1024
ZONE_H = TILE_H * GRID_ROWS   # 768


def stitch_zone_tiles(tile_loader, zone_name: str) -> Image.Image:
    """Stitch 12 tiles (4x3 grid) into a single 1024x768 image.

    tile_loader: callable(zone_name, tile_num) -> PIL.Image or None
    """
    canvas = Image.new("RGBA", (ZONE_W, ZONE_H), (0, 0, 0, 0))
    for n in range(1, 13):
        tile = tile_loader(zone_name, n)
        if tile is None:
            continue
        # Tile numbering: 1-4 = top row, 5-8 = middle, 9-12 = bottom
        col = (n - 1) % GRID_COLS
        row = (n - 1) // GRID_COLS
        tile = tile.convert("RGBA")
        if tile.size != (TILE_W, TILE_H):
            tile = tile.resize((TILE_W, TILE_H), Image.LANCZOS)
        canvas.paste(tile, (col * TILE_W, row * TILE_H))
    return canvas


def make_zip_tile_loader(zf: zipfile.ZipFile):
    """Create a tile loader that reads from a ZIP archive.

    Only uses BLP files. Auto-detects any path prefix inside the ZIP
    (e.g. ``1659008088-atlasworldmap_wotlk/Interface/WorldMap/...``).
    """
    # Build a lookup: normalised suffix -> actual zip path (BLP only)
    blp_lookup = {}
    for name in zf.namelist():
        if not name.lower().endswith(".blp"):
            continue
        # Find "Interface/WorldMap/" part and key on everything after it
        idx = name.find("Interface/WorldMap/")
        if idx == -1:
            # try case-insensitive
            idx = name.lower().find("interface/worldmap/")
        if idx != -1:
            suffix = name[idx:]  # e.g. Interface/WorldMap/Elwynn/Elwynn1.blp
            blp_lookup[suffix] = name

    def loader(zone_name, tile_num):
        key = f"Interface/WorldMap/{zone_name}/{zone_name}{tile_num}.blp"
        actual = blp_lookup.get(key)
        if actual:
            try:
                img_data = zf.read(actual)
                return _open_image_with_blp_fallback(img_data)
            except Exception:
                pass
        return None

    return loader


def make_dir_tile_loader(base_dir: str):
    """Create a tile loader that reads from an extracted directory."""
    def loader(zone_name, tile_num):
        # Only use BLP files (PNG versions may be incorrect)
        path = os.path.join(base_dir, "Interface", "WorldMap",
                            zone_name, f"{zone_name}{tile_num}.blp")
        if os.path.exists(path):
            try:
                img_data = open(path, "rb").read()
                return _open_image_with_blp_fallback(img_data)
            except Exception:
                pass
        return None

    return loader


# ─── Map Compositor ─────────────────────────────────────────────────────

def composite_world_map(entries: list, tile_loader,
                        pixels_per_unit: float = 0.5,
                        zone_filter: set = None,
                        continent_only: bool = False) -> tuple:
    """Composite zone maps onto a single world canvas.

    Args:
        entries: List of WorldMapAreaEntry for the target map.
        tile_loader: callable(zone_name, tile_num) -> PIL.Image
        pixels_per_unit: Resolution (pixels per WoW world unit).
        zone_filter: If set, only include these zone internal_names.
        continent_only: If True, only render the continent (area_id=0) entry.

    Returns:
        (PIL.Image, wow_bounds) where wow_bounds is (x_min, y_min, x_max, y_max).
    """
    # Filter entries
    if continent_only:
        entries = [e for e in entries if e.area_id == 0]
    elif zone_filter:
        zone_filter_lower = {z.lower() for z in zone_filter}
        entries = [e for e in entries
                   if e.internal_name.lower() in zone_filter_lower
                   or e.area_id == 0]  # always include continent as bg

    if not entries:
        print("ERROR: No matching zones found!")
        return None, None

    # Determine overall world bounds from ZONE entries only
    # (continent entry has huge bounds — use zone bounds for sizing)
    # WoW coords: x1 = top (north), x2 = bottom (south)
    #             y1 = left (west), y2 = right (east)
    zone_entries = [e for e in entries if e.area_id != 0]
    bounds_entries = zone_entries if zone_entries else entries
    all_x_min = min(e.x2 for e in bounds_entries)  # southernmost
    all_x_max = max(e.x1 for e in bounds_entries)  # northernmost
    all_y_min = min(e.y2 for e in bounds_entries)  # easternmost
    all_y_max = max(e.y1 for e in bounds_entries)  # westernmost

    wow_bounds = (all_x_min, all_y_min, all_x_max, all_y_max)

    # Canvas size
    world_w = all_y_max - all_y_min  # Y range
    world_h = all_x_max - all_x_min  # X range
    canvas_w = int(world_w * pixels_per_unit)
    canvas_h = int(world_h * pixels_per_unit)

    print(f"World bounds: x=[{all_x_min:.0f}, {all_x_max:.0f}] "
          f"y=[{all_y_min:.0f}, {all_y_max:.0f}]")
    print(f"Canvas size: {canvas_w}x{canvas_h} px "
          f"({pixels_per_unit} px/unit)")

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # Sort: continent first (background), then zones on top
    entries_sorted = sorted(entries, key=lambda e: (0 if e.area_id == 0 else 1))

    for entry in entries_sorted:
        name = entry.internal_name
        print(f"  Stitching {name}...", end=" ", flush=True)

        zone_img = stitch_zone_tiles(tile_loader, name)

        # Check if zone has any content
        if zone_img.getbbox() is None:
            print("(no tiles found, skipping)")
            continue

        # Calculate zone dimensions in pixels
        zone_world_w = entry.y1 - entry.y2  # Y range
        zone_world_h = entry.x1 - entry.x2  # X range
        zone_px_w = int(zone_world_w * pixels_per_unit)
        zone_px_h = int(zone_world_h * pixels_per_unit)

        if zone_px_w <= 0 or zone_px_h <= 0:
            print("(invalid bounds, skipping)")
            continue

        # Resize zone image to match world scale
        zone_img = zone_img.resize((zone_px_w, zone_px_h), Image.LANCZOS)

        # Calculate paste position
        # Image y-axis is flipped (top=0): x_max maps to top, x_min to bottom
        # Image x-axis: y_min (east) maps to left=0
        paste_x = int((entry.y2 - all_y_min) * pixels_per_unit)
        paste_y = int((all_x_max - entry.x1) * pixels_per_unit)

        # Alpha-composite to layer zones properly
        canvas.paste(zone_img, (paste_x, paste_y), zone_img)
        print(f"{zone_px_w}x{zone_px_h} px")

    return canvas, wow_bounds


# ─── CLI ────────────────────────────────────────────────────────────────

def main():
    if not _HAS_PIL:
        print("ERROR: Pillow is required. Install with: pip install Pillow")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Build a stitched world-map PNG from WoW WorldMap tiles.")
    parser.add_argument("--zip", type=str, default=_DEFAULT_ZIP,
                        help="Path to ZIP archive with Interface/WorldMap/ tiles")
    parser.add_argument("--dir", type=str, default=None,
                        help="Path to extracted directory with Interface/WorldMap/")
    parser.add_argument("--dbc", type=str, default=_DEFAULT_DBC,
                        help="Path to WorldMapArea.dbc")
    parser.add_argument("--map-id", type=int, default=0,
                        help="Map ID to render (0=Eastern Kingdoms, 1=Kalimdor)")
    parser.add_argument("--zones", type=str, default=None,
                        help="Comma-separated zone names to include "
                             "(e.g. Elwynn,Westfall,Redridge)")
    parser.add_argument("--continent-only", action="store_true",
                        help="Only render the continent overview (no zone detail)")
    parser.add_argument("--ppu", type=float, default=0.5,
                        help="Pixels per world unit (default: 0.5)")
    parser.add_argument("--output", type=str, default=_DEFAULT_OUTPUT,
                        help="Output PNG path (default: Data/world_map.png)")
    parser.add_argument("--list-zones", action="store_true",
                        help="List available zones and exit")
    args = parser.parse_args()

    dbc_path = args.dbc
    if not os.path.exists(dbc_path):
        print(f"ERROR: WorldMapArea.dbc not found: {dbc_path}")
        sys.exit(1)

    zip_path = args.zip
    if not zip_path and not args.dir:
        print("ERROR: No tile source specified. Use --zip or --dir")
        sys.exit(1)

    # Parse DBC
    print(f"Parsing WorldMapArea.dbc: {dbc_path}")
    all_entries = parse_world_map_area_dbc(dbc_path)
    print(f"  {len(all_entries)} entries total")

    # Filter to target map
    map_entries = [e for e in all_entries if e.map_id == args.map_id]
    print(f"  {len(map_entries)} entries for map_id={args.map_id}")

    if args.list_zones:
        print(f"\nZones for map_id={args.map_id}:")
        for e in sorted(map_entries, key=lambda e: e.internal_name):
            marker = " [continent]" if e.area_id == 0 else ""
            print(f"  {e.internal_name:25s} area={e.area_id}{marker}")
        return

    # Create tile loader
    if zip_path:
        print(f"Opening tile archive: {zip_path}")
        zf = zipfile.ZipFile(zip_path, "r")
        tile_loader = make_zip_tile_loader(zf)
    else:
        print(f"Reading tiles from: {args.dir}")
        tile_loader = make_dir_tile_loader(args.dir)

    # Zone filter
    zone_filter = None
    if args.zones:
        zone_filter = set(args.zones.split(","))
        print(f"Zone filter: {zone_filter}")

    # Build
    print(f"\nCompositing world map...")
    canvas, wow_bounds = composite_world_map(
        map_entries, tile_loader,
        pixels_per_unit=args.ppu,
        zone_filter=zone_filter,
        continent_only=args.continent_only)

    if canvas is None:
        sys.exit(1)

    # Save PNG
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas_rgb = Image.new("RGB", canvas.size, (22, 33, 62))  # #16213e bg
    canvas_rgb.paste(canvas, mask=canvas)
    canvas_rgb.save(output_path, "PNG")
    print(f"\nSaved: {output_path} ({canvas.size[0]}x{canvas.size[1]})")

    # Save JSON sidecar with bounds
    x_min, y_min, x_max, y_max = wow_bounds
    bounds_path = output_path.rsplit(".", 1)[0] + ".json"
    bounds_data = {
        "x_min": x_min, "y_min": y_min,
        "x_max": x_max, "y_max": y_max,
        "map_id": args.map_id,
        "description": f"WoW world coordinate bounds for {output_path}",
        "usage": f"--map-image {output_path} "
                 f"--map-bounds \"{x_min},{y_min},{x_max},{y_max}\""
    }
    with open(bounds_path, "w") as f:
        json.dump(bounds_data, f, indent=2)
    print(f"Bounds: {bounds_path}")
    print(f"\nUsage with visualizer:")
    print(f"  python -m sim.visualize --log-dir <logs> "
          f"--map-image {output_path}")
    print(f"  (bounds auto-loaded from {bounds_path})")

    if zip_path:
        zf.close()


if __name__ == "__main__":
    main()
