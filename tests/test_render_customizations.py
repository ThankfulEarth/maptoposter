"""End-to-end render tests for create_poster.

These tests drive the *real* matplotlib render + savefig pipeline (no mocking of
drawing code) and assert on the actual output pixels. Only the OSM data-source
functions are monkeypatched (fetch_roads_offline, fetch_features,
_get_land_polygons) so the render runs against small synthetic fixture
geometries instead of hitting the network or a real PBF extract.

This closes the gap left by the flag-emission/parsing unit tests: it proves
that theme colors, layer toggles, and buildings visibly change the rendered
poster, not just that the CLI plumbing accepts the options.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from PIL import Image
from shapely.geometry import LineString, Polygon

# Add parent directory to path for imports (matches the rest of the suite)
sys.path.insert(0, str(Path(__file__).parent.parent))

import create_map_poster

# Tiny fixture bbox (west, south, east, north) — no real-world meaning, just a
# small patch of the globe far from any UTM zone boundary.
BBOX = (10.000, 50.000, 10.020, 50.020)
WEST, SOUTH, EAST, NORTH = BBOX
POINT = ((SOUTH + NORTH) / 2, (WEST + EAST) / 2)

# One unique, mutually-distinguishable sentinel color per theme key so each
# customization's effect on pixels can be checked independently.
SENTINEL_THEME = {
    "name": "sentinel-test",
    "bg": "#FF00FF",
    "text": "#101010",
    "gradient_color": "#8A2BE2",
    "water": "#00FFFF",
    "parks": "#FFFF00",
    "building": "#00FF00",
    "road_motorway": "#FF7F00",
    "road_primary": "#7F00FF",
    "road_secondary": "#0044FF",
    "road_tertiary": "#804000",
    "road_residential": "#FF0080",
    "road_default": "#555555",
}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _color_present(image_path: Path, hex_color: str, tolerance: int = 12) -> bool:
    """Whether any pixel in the PNG is within `tolerance` of `hex_color` per channel."""
    target = np.array(_hex_to_rgb(hex_color), dtype=int)
    with Image.open(image_path) as img:
        arr = np.array(img.convert("RGB"), dtype=int)
    diff = np.abs(arr - target)
    return bool(np.any(np.all(diff <= tolerance, axis=-1)))


def _fixture_roads() -> gpd.GeoDataFrame:
    """Two diagonals across the full bbox, tagged so get_road_color resolves distinct colors."""
    return gpd.GeoDataFrame(
        {"highway": ["primary", "residential"]},
        geometry=[
            LineString([(WEST + 0.002, SOUTH + 0.002), (EAST - 0.002, NORTH - 0.002)]),
            LineString([(WEST + 0.002, NORTH - 0.002), (EAST - 0.002, SOUTH + 0.002)]),
        ],
        crs="EPSG:4326",
    )


# Water/parks/buildings polygons are placed in the middle 50% of the bbox's
# latitude span so they aren't touched by the top/bottom gradient fade (which
# only tints the outer quarters), keeping their sentinel colors pure for pixel
# matching.
def _fixture_water() -> gpd.GeoDataFrame:
    poly = Polygon(
        [
            (WEST + 0.002, SOUTH + 0.006),
            (WEST + 0.008, SOUTH + 0.006),
            (WEST + 0.008, SOUTH + 0.009),
            (WEST + 0.002, SOUTH + 0.009),
        ]
    )
    return gpd.GeoDataFrame({"natural": ["water"]}, geometry=[poly], crs="EPSG:4326")


def _fixture_parks() -> gpd.GeoDataFrame:
    poly = Polygon(
        [
            (EAST - 0.008, SOUTH + 0.006),
            (EAST - 0.002, SOUTH + 0.006),
            (EAST - 0.002, SOUTH + 0.009),
            (EAST - 0.008, SOUTH + 0.009),
        ]
    )
    return gpd.GeoDataFrame({"leisure": ["park"]}, geometry=[poly], crs="EPSG:4326")


def _fixture_buildings() -> gpd.GeoDataFrame:
    poly = Polygon(
        [
            (WEST + 0.006, SOUTH + 0.011),
            (WEST + 0.014, SOUTH + 0.011),
            (WEST + 0.014, SOUTH + 0.014),
            (WEST + 0.006, SOUTH + 0.014),
        ]
    )
    return gpd.GeoDataFrame({"building": [True]}, geometry=[poly], crs="EPSG:4326")


def _fixture_land() -> gpd.GeoDataFrame:
    """Huge polygon guaranteed to cover the render crop once clipped to fetch_bbox.

    create_poster only paints THEME['bg'] where land polygons exist (there's no
    bundled land shapefile in this checkout), so without this fixture the 'bg'
    sentinel would never reach a pixel and the "theme colors applied" assertion
    would be untestable.
    """
    poly = Polygon([(-170, -80), (170, -80), (170, 80), (-170, 80)])
    return gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326")


@pytest.fixture
def fixture_data_sources(monkeypatch):
    """Patch OSM data-source functions with small synthetic fixtures; no network/PBF."""
    monkeypatch.setattr(create_map_poster, "USE_OFFLINE_OSM", True)
    monkeypatch.setattr(create_map_poster, "THEME", dict(SENTINEL_THEME))

    def fake_fetch_roads_offline(bbox, road_detail="high"):
        return _fixture_roads()

    def fake_fetch_features(point, dist, tags, name, bbox=None):
        if name == "water":
            return _fixture_water()
        if name == "parks":
            return _fixture_parks()
        if name == "buildings":
            return _fixture_buildings()
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    def fake_get_land_polygons(clip_bbox=None):
        return _fixture_land()

    monkeypatch.setattr(create_map_poster, "fetch_roads_offline", fake_fetch_roads_offline)
    monkeypatch.setattr(create_map_poster, "fetch_features", fake_fetch_features)
    monkeypatch.setattr(create_map_poster, "_get_land_polygons", fake_get_land_polygons)


def _render(tmp_path: Path, name: str, **overrides) -> Path:
    output = tmp_path / f"{name}.png"
    kwargs = dict(
        city="Sentinel City",
        country="Testland",
        point=POINT,
        dist=1000,
        output_file=str(output),
        output_format="png",
        width=4,
        height=4,
        dpi=200,
        bbox=BBOX,
        draw_water=True,
        draw_parks=True,
        draw_roads=True,
        draw_buildings=True,
        title_font=None,
    )
    kwargs.update(overrides)
    create_map_poster.create_poster(**kwargs)
    return output


@pytest.mark.slow
def test_theme_colors_and_road_hierarchy_render(tmp_path, fixture_data_sources):
    """Full render: bg/water/parks/building/road-hierarchy sentinel colors all appear."""
    output = _render(tmp_path, "full")

    assert output.exists()
    assert output.stat().st_size > 1000

    assert _color_present(output, SENTINEL_THEME["bg"]), "bg (land) color missing — theme not applied"
    assert _color_present(output, SENTINEL_THEME["water"]), "water color missing"
    assert _color_present(output, SENTINEL_THEME["parks"]), "parks color missing"
    assert _color_present(output, SENTINEL_THEME["building"]), "building color missing"
    assert _color_present(output, SENTINEL_THEME["road_primary"]), "road_primary color missing"
    assert _color_present(output, SENTINEL_THEME["road_residential"]), "road_residential color missing"


@pytest.mark.slow
def test_water_layer_toggle_changes_pixels(tmp_path, fixture_data_sources):
    """draw_water=True paints the water sentinel; draw_water=False must not."""
    on = _render(tmp_path, "water_on", draw_water=True)
    off = _render(tmp_path, "water_off", draw_water=False)

    assert _color_present(on, SENTINEL_THEME["water"])
    assert not _color_present(off, SENTINEL_THEME["water"])


@pytest.mark.slow
def test_buildings_layer_toggle_changes_pixels(tmp_path, fixture_data_sources):
    """draw_buildings=True paints the building sentinel; draw_buildings=False must not."""
    on = _render(tmp_path, "buildings_on", draw_buildings=True)
    off = _render(tmp_path, "buildings_off", draw_buildings=False)

    assert _color_present(on, SENTINEL_THEME["building"])
    assert not _color_present(off, SENTINEL_THEME["building"])


@pytest.mark.slow
def test_render_with_default_title_font_produces_file(tmp_path, fixture_data_sources):
    """title_font=None falls back to bundled Roboto and still renders a real poster."""
    output = _render(tmp_path, "default_font", title_font=None)

    assert output.exists()
    assert output.stat().st_size > 1000
