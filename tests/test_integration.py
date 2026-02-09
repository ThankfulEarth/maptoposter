"""
Integration tests for OSM offline caching.

These tests require actual OSM data and are marked with @pytest.mark.integration.
Run with: pytest -m integration --slow

Environment variables:
- OSM_PLANET_PATH: Path to planet or regional PBF file
- OSM_TEST_CACHE: Path to test cache directory (will be created)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import geopandas as gpd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from osm_cache import OsmExtractCache, snap_bbox_to_grid
from osm_reader import OsmOfflineReader

# Copenhagen bbox for testing
COPENHAGEN_BBOX = (12.45, 55.60, 12.70, 55.75)


@pytest.fixture
def planet_path() -> Path:
    """Get planet file path from environment."""
    path = os.environ.get("OSM_PLANET_PATH")
    if not path:
        pytest.skip("OSM_PLANET_PATH not set")
    planet = Path(path)
    if not planet.exists():
        pytest.skip(f"Planet file not found: {planet}")
    return planet


@pytest.fixture
def test_cache_dir(tmp_path: Path) -> Path:
    """Get test cache directory."""
    cache_dir = Path(os.environ.get("OSM_TEST_CACHE", str(tmp_path / "osm_cache")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.mark.integration
@pytest.mark.slow
class TestExtractFromPlanet:
    """Integration tests for extracting from planet file."""

    def test_extract_copenhagen(self, planet_path: Path, test_cache_dir: Path):
        """Extract Copenhagen bbox from planet file."""
        cache = OsmExtractCache(planet_path, test_cache_dir)

        result = cache.get_extract(COPENHAGEN_BBOX)

        assert result.exists()
        assert result.stat().st_size > 1_000_000  # Should be >1MB

    def test_extract_caches_result(self, planet_path: Path, test_cache_dir: Path):
        """Second request should use cache."""
        cache = OsmExtractCache(planet_path, test_cache_dir)

        # First request
        result1 = cache.get_extract(COPENHAGEN_BBOX)
        mtime1 = result1.stat().st_mtime

        # Second request (should be cached)
        result2 = cache.get_extract(COPENHAGEN_BBOX)
        mtime2 = result2.stat().st_mtime

        assert result1 == result2
        assert mtime1 == mtime2  # File shouldn't have been modified


@pytest.mark.integration
@pytest.mark.slow
class TestOfflineReader:
    """Integration tests for offline reader."""

    def test_get_roads(self, planet_path: Path, test_cache_dir: Path):
        """Get road network from offline extract."""
        cache = OsmExtractCache(planet_path, test_cache_dir)
        reader = OsmOfflineReader(cache)

        roads = reader.get_roads(COPENHAGEN_BBOX)

        assert isinstance(roads, gpd.GeoDataFrame)
        assert len(roads) > 0
        assert "highway" in roads.columns or roads.empty

    def test_get_water(self, planet_path: Path, test_cache_dir: Path):
        """Get water features from offline extract."""
        cache = OsmExtractCache(planet_path, test_cache_dir)
        reader = OsmOfflineReader(cache)

        water = reader.get_water(COPENHAGEN_BBOX)

        assert isinstance(water, gpd.GeoDataFrame)
        # Copenhagen has water, but empty is also valid

    def test_get_parks(self, planet_path: Path, test_cache_dir: Path):
        """Get park features from offline extract."""
        cache = OsmExtractCache(planet_path, test_cache_dir)
        reader = OsmOfflineReader(cache)

        parks = reader.get_parks(COPENHAGEN_BBOX)

        assert isinstance(parks, gpd.GeoDataFrame)


@pytest.mark.integration
class TestOfflineReaderProperties:
    """Test invariants and properties of offline reader output."""

    def test_roads_have_valid_geometries(self, planet_path: Path, test_cache_dir: Path):
        """All road geometries should be valid."""
        from shapely import LineString, MultiLineString

        cache = OsmExtractCache(planet_path, test_cache_dir)
        reader = OsmOfflineReader(cache)
        roads = reader.get_roads(COPENHAGEN_BBOX)

        assert len(roads) > 0, "Expected roads in Copenhagen"

        invalid_count = 0
        for geom in roads.geometry:
            if not geom.is_valid:
                invalid_count += 1

        assert invalid_count == 0, f"Found {invalid_count} invalid geometries"

        # Most roads should be linear (LineStrings for rendering as paths)
        linear_count = sum(
            1 for g in roads.geometry if isinstance(g, (LineString, MultiLineString))
        )
        assert linear_count > len(roads) * 0.3, "Expected significant portion of roads to be linear"

    def test_roads_have_highway_tag(self, planet_path: Path, test_cache_dir: Path):
        """All roads should have a highway tag."""
        cache = OsmExtractCache(planet_path, test_cache_dir)
        reader = OsmOfflineReader(cache)
        roads = reader.get_roads(COPENHAGEN_BBOX)

        assert "highway" in roads.columns
        assert roads["highway"].notna().all(), "Some roads missing highway tag"

    def test_water_has_valid_geometries(self, planet_path: Path, test_cache_dir: Path):
        """Water features should be valid geometries."""
        cache = OsmExtractCache(planet_path, test_cache_dir)
        reader = OsmOfflineReader(cache)
        water = reader.get_water(COPENHAGEN_BBOX)

        if len(water) > 0:
            invalid_count = sum(1 for g in water.geometry if not g.is_valid)
            assert invalid_count == 0, f"Found {invalid_count} invalid water geometries"

    def test_output_has_correct_crs(self, planet_path: Path, test_cache_dir: Path):
        """All outputs should be in WGS84 (EPSG:4326)."""
        cache = OsmExtractCache(planet_path, test_cache_dir)
        reader = OsmOfflineReader(cache)

        roads = reader.get_roads(COPENHAGEN_BBOX)
        water = reader.get_water(COPENHAGEN_BBOX)
        parks = reader.get_parks(COPENHAGEN_BBOX)

        for gdf, name in [(roads, "roads"), (water, "water"), (parks, "parks")]:
            if len(gdf) > 0:
                assert gdf.crs is not None, f"{name} missing CRS"
                assert gdf.crs.to_epsg() == 4326, f"{name} has wrong CRS: {gdf.crs}"

    def test_geometries_within_bbox(self, planet_path: Path, test_cache_dir: Path):
        """All geometries should intersect the requested bbox."""
        from shapely.geometry import box

        cache = OsmExtractCache(planet_path, test_cache_dir)
        reader = OsmOfflineReader(cache)
        roads = reader.get_roads(COPENHAGEN_BBOX)

        # Use snapped bbox since that's what the cache uses
        snapped = snap_bbox_to_grid(COPENHAGEN_BBOX)
        bbox_geom = box(*snapped)

        for geom in roads.geometry:
            assert geom.intersects(
                bbox_geom
            ), f"Geometry outside bbox: {geom.bounds} vs {snapped}"
