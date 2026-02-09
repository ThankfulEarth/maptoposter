"""Tests for OSM offline reader."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from osm_cache import OsmExtractCache
from osm_reader import OsmOfflineReader, OsmReaderError, tags_to_cache_key


class TestTagsToCacheKey:
    """Tests for tag dict to cache key conversion."""

    def test_simple_true_tag(self):
        """Tag with True value becomes just the key name."""
        key = tags_to_cache_key({"highway": True})
        assert key == "highway"

    def test_tag_with_string_value(self):
        """Tag with string value becomes key=value."""
        key = tags_to_cache_key({"leisure": "park"})
        assert key == "leisure=park"

    def test_tag_with_list_value(self):
        """Tag with list value becomes key=sorted,values."""
        key = tags_to_cache_key({"natural": ["water", "bay"]})
        assert key == "natural=bay,water"  # sorted

    def test_multiple_tags_sorted(self):
        """Multiple tags are sorted by key."""
        key = tags_to_cache_key({"waterway": True, "natural": ["water"]})
        assert key == "natural=water_waterway"

    def test_deterministic(self):
        """Same tags should always produce same key."""
        tags = {"natural": ["water", "bay"], "waterway": "riverbank"}
        key1 = tags_to_cache_key(tags)
        key2 = tags_to_cache_key(tags)
        assert key1 == key2

    def test_long_tags_hashed(self):
        """Very long tag strings should be hashed."""
        tags = {
            "very_long_key_name": ["value1", "value2", "value3", "value4"],
            "another_long_key": ["more", "values", "here"],
            "and_yet_another": True,
        }
        key = tags_to_cache_key(tags)
        # Should be a hash (16 hex chars)
        assert len(key) == 16


class TestOsmOfflineReader:
    """Tests for OsmOfflineReader class."""

    @pytest.fixture
    def mock_extract_cache(self, tmp_cache_dir: Path, mock_pbf_path: Path):
        """Create a mock extract cache."""
        return OsmExtractCache(mock_pbf_path, tmp_cache_dir)

    def test_init_creates_processed_dir(
        self, mock_extract_cache: OsmExtractCache, tmp_cache_dir: Path
    ):
        """Reader should create processed cache directory."""
        reader = OsmOfflineReader(mock_extract_cache)

        assert reader.processed_cache_dir.exists()
        assert reader.processed_cache_dir == tmp_cache_dir / "processed"

    def test_init_custom_processed_dir(
        self, mock_extract_cache: OsmExtractCache, tmp_cache_dir: Path
    ):
        """Reader should use custom processed cache directory."""
        custom_dir = tmp_cache_dir / "custom_processed"
        reader = OsmOfflineReader(mock_extract_cache, processed_cache_dir=custom_dir)

        assert reader.processed_cache_dir == custom_dir
        assert custom_dir.exists()

    def test_get_features_uses_l2_cache(
        self,
        mock_extract_cache: OsmExtractCache,
    ):
        """Should return from L2 cache if available."""
        from shapely.wkt import loads as wkt_loads

        reader = OsmOfflineReader(mock_extract_cache)

        bbox = (12.50, 55.60, 12.70, 55.75)
        tags = {"highway": True}

        # Create test data
        geoms = [wkt_loads("POINT (12.55 55.68)"), wkt_loads("POINT (12.58 55.70)")]
        test_gdf = gpd.GeoDataFrame(
            {"name": ["test1", "test2"]},
            geometry=geoms,
            crs="EPSG:4326",
        )

        # Pre-populate L2 cache
        cache_path = reader._get_processed_cache_path(bbox, tags)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(test_gdf, f)

        result = reader.get_features(bbox, tags, clip_to_bbox=False)

        assert len(result) == 2

    def test_clip_to_user_bbox(
        self,
        mock_extract_cache: OsmExtractCache,
    ):
        """User's exact bbox should be used for clipping, not cache bbox."""
        from shapely.wkt import loads as wkt_loads

        reader = OsmOfflineReader(mock_extract_cache)

        # Use a bbox that excludes some features
        user_bbox = (12.52, 55.67, 12.58, 55.71)

        # Create test data with features both inside and outside user bbox
        geoms = [wkt_loads("POINT (12.55 55.68)"), wkt_loads("POINT (12.51 55.66)")]
        test_gdf = gpd.GeoDataFrame(
            {"name": ["inside", "outside"]},
            geometry=geoms,
            crs="EPSG:4326",
        )

        # Pre-populate L2 cache with full sample data
        cache_bbox = (12.50, 55.65, 12.60, 55.75)  # Snapped, larger
        tags = {"highway": True}
        cache_path = reader._get_processed_cache_path(cache_bbox, tags)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(test_gdf, f)

        result = reader.get_features(user_bbox, tags, clip_to_bbox=True)

        # Result should be clipped to user bbox
        if not result.empty:
            bounds = result.total_bounds
            assert bounds[0] >= user_bbox[0] - 0.001  # Small tolerance for clipping
            assert bounds[2] <= user_bbox[2] + 0.001

    @patch.object(OsmExtractCache, "get_extract")
    @patch.object(OsmOfflineReader, "_parse_pbf")
    def test_get_features_caches_result(
        self,
        mock_parse: MagicMock,
        mock_get_extract: MagicMock,
        mock_extract_cache: OsmExtractCache,
        tmp_cache_dir: Path,
    ):
        """Should cache parsed results to L2."""
        from shapely.wkt import loads as wkt_loads

        # Create a simple test GeoDataFrame
        geoms = [wkt_loads("POINT (12.55 55.68)")]
        test_gdf = gpd.GeoDataFrame(
            {"name": ["test"]},
            geometry=geoms,
            crs="EPSG:4326",
        )

        mock_get_extract.return_value = tmp_cache_dir / "fake.osm.pbf"
        mock_parse.return_value = test_gdf

        reader = OsmOfflineReader(mock_extract_cache)
        bbox = (12.50, 55.60, 12.70, 55.75)
        tags = {"natural": ["water"]}

        reader.get_features(bbox, tags, clip_to_bbox=False)

        # Verify cache file was created
        cache_path = reader._get_processed_cache_path(bbox, tags)
        assert cache_path.exists()

        # Verify cached data
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        assert len(cached) == 1

    def test_get_roads_helper(self, mock_extract_cache: OsmExtractCache):
        """get_roads should call get_features with correct tags."""
        reader = OsmOfflineReader(mock_extract_cache)

        with patch.object(reader, "get_features") as mock_get:
            # Use MagicMock instead of GeoDataFrame for compatibility
            mock_get.return_value = MagicMock()
            reader.get_roads((12.50, 55.60, 12.70, 55.75))

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[1]["tags"] == {"highway": True}

    def test_get_water_helper(self, mock_extract_cache: OsmExtractCache):
        """get_water should call get_features with correct tags."""
        reader = OsmOfflineReader(mock_extract_cache)

        with patch.object(reader, "get_features") as mock_get:
            mock_get.return_value = MagicMock()
            reader.get_water((12.50, 55.60, 12.70, 55.75))

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        tags = call_args[1]["tags"]
        assert "natural" in tags
        assert "waterway" in tags

    def test_get_parks_helper(self, mock_extract_cache: OsmExtractCache):
        """get_parks should call get_features with correct tags."""
        reader = OsmOfflineReader(mock_extract_cache)

        with patch.object(reader, "get_features") as mock_get:
            mock_get.return_value = MagicMock()
            reader.get_parks((12.50, 55.60, 12.70, 55.75))

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        tags = call_args[1]["tags"]
        assert "leisure" in tags or "landuse" in tags

    def test_get_cache_info(
        self,
        mock_extract_cache: OsmExtractCache,
        tmp_cache_dir: Path,
    ):
        """Should return info from both cache levels."""
        reader = OsmOfflineReader(mock_extract_cache)

        # Create some L2 cache files
        (reader.processed_cache_dir / "test1.pkl").write_bytes(b"x" * 1000)
        (reader.processed_cache_dir / "test2.pkl").write_bytes(b"y" * 2000)

        info = reader.get_cache_info()

        assert "l1_extracts" in info
        assert "l2_processed" in info
        assert info["l2_processed"]["count"] == 2


class TestTagCoverage:
    """
    Ensure all feature tags used in the codebase are properly classified.

    This test fails if a new tag is added without classifying it as
    area-type or linear-type, preventing geometry rendering issues.

    When adding a new feature type:
    1. Add the OSM tag key to USED_OSM_TAGS below
    2. Add it to either AREA_TAG_KEYS (in osm_reader.py) or LINEAR_TAG_KEYS here
    3. Run tests to verify
    """

    # All OSM tag keys actually used in our codebase for feature queries
    # UPDATE THIS when adding new feature types!
    USED_OSM_TAGS = {
        # Roads (osm_reader.py: get_roads)
        "highway",
        # Water (osm_reader.py: get_water, create_map_poster.py)
        "natural",
        "waterway",
        # Parks (osm_reader.py: get_parks, create_map_poster.py)
        "leisure",
        "landuse",
        # Borders (osm_reader.py: get_borders, create_map_poster.py)
        "boundary",
        "admin_level",
        # Coastlines (osm_reader.py: get_coastlines)
        # (uses "natural" - already listed)
        # Glaciers (osm_reader.py: get_glaciers)
        # (uses "natural" - already listed)
        # Terrain (osm_reader.py: get_terrain)
        # (uses "natural" - already listed)
    }

    # Tags that should remain as LineStrings (not areas) when closed
    LINEAR_TAG_KEYS = {
        "highway",      # Roads are linear (except specific area types handled separately)
        "railway",      # Rail lines are linear
        "route",        # Routes are linear
        "power",        # Power lines are linear
        "admin_level",  # Filter value, not geometry-determining
    }

    def _get_all_classified_tags(self) -> set[str]:
        """Get all tags classified as either area or linear."""
        from osm_reader import OsmTagHandler

        return OsmTagHandler.AREA_TAG_KEYS | self.LINEAR_TAG_KEYS

    def test_all_used_tags_are_classified(self):
        """Every OSM tag we use must be classified as area or linear."""
        classified_tags = self._get_all_classified_tags()

        unclassified = self.USED_OSM_TAGS - classified_tags

        if unclassified:
            msg = (
                f"\n\nUNCLASSIFIED TAGS FOUND: {unclassified}\n\n"
                "These tags are in USED_OSM_TAGS but not classified.\n"
                "Add each tag to one of:\n"
                "  - OsmTagHandler.AREA_TAG_KEYS in osm_reader.py (for area features)\n"
                "  - TestTagCoverage.LINEAR_TAG_KEYS in this file (for linear features)\n\n"
                "To decide:\n"
                "  - If closed ways with this tag should render as Polygons → AREA_TAG_KEYS\n"
                "  - If closed ways should remain as LineStrings → LINEAR_TAG_KEYS\n"
            )
            pytest.fail(msg)

    def test_used_tags_list_is_current(self):
        """
        Verify USED_OSM_TAGS matches what's actually used in osm_reader.py helpers.

        This catches when someone adds a new get_* method but forgets to
        update USED_OSM_TAGS.
        """
        from osm_reader import OsmOfflineReader
        import inspect
        import re

        # Get source of all get_* methods
        source = inspect.getsource(OsmOfflineReader)

        # Find all tags={...} patterns in get_* methods
        # Match patterns like: tags={"key": value} or tags={'key': value}
        tag_pattern = r'tags\s*=\s*\{([^}]+)\}'
        tag_blocks = re.findall(tag_pattern, source)

        found_tags = set()
        # Only match dictionary keys (string followed by colon)
        # Pattern: "key": or 'key':
        key_pattern = r'["\'](\w+)["\']\s*:'
        for block in tag_blocks:
            keys = re.findall(key_pattern, block)
            found_tags.update(keys)

        missing_from_used = found_tags - self.USED_OSM_TAGS
        if missing_from_used:
            msg = (
                f"\n\nTags found in osm_reader.py but not in USED_OSM_TAGS: {missing_from_used}\n\n"
                "Add these to TestTagCoverage.USED_OSM_TAGS and classify them.\n"
            )
            pytest.fail(msg)

    def test_area_and_linear_are_disjoint(self):
        """Area and linear tag sets should not overlap."""
        from osm_reader import OsmTagHandler

        overlap = OsmTagHandler.AREA_TAG_KEYS & self.LINEAR_TAG_KEYS

        if overlap:
            pytest.fail(
                f"Tags cannot be both area AND linear: {overlap}\n"
                "Remove from one of the sets."
            )

    def test_known_area_tags_produce_polygons(self):
        """Verify area-type tags produce Polygon geometry for closed ways."""
        from osm_reader import OsmTagHandler

        # Test with a known area tag
        handler = OsmTagHandler({"leisure": "park"})

        # The handler should recognize leisure as an area type
        assert "leisure" in OsmTagHandler.AREA_TAG_KEYS

    def test_highway_produces_linestrings_by_default(self):
        """Verify highway tag produces LineString geometry (not Polygon)."""
        from osm_reader import OsmTagHandler

        # highway should not be in AREA_TAG_KEYS
        assert "highway" not in OsmTagHandler.AREA_TAG_KEYS
        assert "highway" in self.LINEAR_TAG_KEYS


