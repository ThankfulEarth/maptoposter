"""Pytest fixtures for maptoposter tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon


@pytest.fixture
def tmp_cache_dir() -> Generator[Path, None, None]:
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_gdf() -> gpd.GeoDataFrame:
    """Create a sample GeoDataFrame for testing."""
    from shapely.wkt import loads as wkt_loads

    # Create geometries using shapely.wkt.loads (older API for compatibility)
    geometries = [
        wkt_loads("POINT (12.55 55.68)"),  # Copenhagen area
        wkt_loads("POINT (12.58 55.70)"),
        wkt_loads("LINESTRING (12.50 55.65, 12.60 55.70)"),
        wkt_loads("POLYGON ((12.52 55.66, 12.56 55.66, 12.56 55.69, 12.52 55.69, 12.52 55.66))"),
    ]

    return gpd.GeoDataFrame(
        {
            "name": ["point1", "point2", "line1", "polygon1"],
            "highway": ["crossing", None, "primary", None],
            "natural": [None, None, None, "water"],
        },
        geometry=geometries,
        crs="EPSG:4326",
    )


@pytest.fixture
def mock_pbf_path(tmp_cache_dir: Path) -> Path:
    """Create a mock PBF file path (empty file for testing cache logic)."""
    pbf_path = tmp_cache_dir / "mock-planet.osm.pbf"
    pbf_path.write_bytes(b"mock pbf data")
    return pbf_path
