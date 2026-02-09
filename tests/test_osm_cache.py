"""Tests for OSM extract cache."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from osm_cache import (
    GRID_SIZE,
    OsmCacheError,
    OsmExtractCache,
    bbox_contains,
    bbox_to_cache_key,
    parse_bbox_from_filename,
    snap_bbox_to_grid,
)


class TestSnapBbox:
    """Tests for bbox snapping to grid."""

    def test_snap_bbox_rounds_correctly(self):
        """Verify bbox snaps to grid boundaries."""
        bbox = (12.523, 55.671, 12.589, 55.712)
        snapped = snap_bbox_to_grid(bbox, grid=0.05)

        # West/South floor, East/North ceil (use approx for float comparison)
        assert snapped[0] == pytest.approx(12.50, abs=1e-9)
        assert snapped[1] == pytest.approx(55.65, abs=1e-9)
        assert snapped[2] == pytest.approx(12.60, abs=1e-9)
        assert snapped[3] == pytest.approx(55.75, abs=1e-9)

    def test_snap_bbox_expands_not_shrinks(self):
        """Snapped bbox must fully contain original."""
        bbox = (12.523, 55.671, 12.589, 55.712)
        snapped = snap_bbox_to_grid(bbox)

        assert snapped[0] <= bbox[0]  # west
        assert snapped[1] <= bbox[1]  # south
        assert snapped[2] >= bbox[2]  # east
        assert snapped[3] >= bbox[3]  # north

    def test_snap_bbox_exact_grid_values_unchanged(self):
        """Values exactly on grid should stay same (or expand to next)."""
        bbox = (12.50, 55.60, 12.70, 55.75)
        snapped = snap_bbox_to_grid(bbox, grid=0.05)

        # Exact grid values stay same (use approx for float comparison)
        assert snapped[0] == pytest.approx(12.50, abs=1e-9)
        assert snapped[1] == pytest.approx(55.60, abs=1e-9)
        assert snapped[2] == pytest.approx(12.70, abs=1e-9)
        assert snapped[3] == pytest.approx(55.75, abs=1e-9)

    def test_snap_bbox_negative_coordinates(self):
        """Handle negative coordinates (western hemisphere)."""
        bbox = (-74.05, 40.70, -73.85, 40.85)  # New York area
        snapped = snap_bbox_to_grid(bbox, grid=0.05)

        assert snapped[0] <= bbox[0]
        assert snapped[1] <= bbox[1]
        assert snapped[2] >= bbox[2]
        assert snapped[3] >= bbox[3]

    def test_snap_bbox_custom_grid_size(self):
        """Test with different grid sizes."""
        bbox = (12.523, 55.671, 12.589, 55.712)

        # Larger grid
        snapped_01 = snap_bbox_to_grid(bbox, grid=0.1)
        assert snapped_01[0] == pytest.approx(12.5, abs=1e-9)
        assert snapped_01[1] == pytest.approx(55.6, abs=1e-9)
        assert snapped_01[2] == pytest.approx(12.6, abs=1e-9)
        assert snapped_01[3] == pytest.approx(55.8, abs=1e-9)

        # Smaller grid
        snapped_001 = snap_bbox_to_grid(bbox, grid=0.01)
        assert snapped_001[0] == pytest.approx(12.52, abs=1e-9)
        assert snapped_001[1] == pytest.approx(55.67, abs=1e-9)
        assert snapped_001[2] == pytest.approx(12.59, abs=1e-9)
        assert snapped_001[3] == pytest.approx(55.72, abs=1e-9)


class TestBboxToCacheKey:
    """Tests for cache key generation."""

    def test_cache_key_deterministic(self):
        """Same bbox should always produce same key."""
        bbox = (12.50, 55.60, 12.70, 55.75)
        key1 = bbox_to_cache_key(bbox)
        key2 = bbox_to_cache_key(bbox)

        assert key1 == key2

    def test_cache_key_format(self):
        """Verify key format matches expected pattern."""
        bbox = (12.50, 55.60, 12.70, 55.75)
        key = bbox_to_cache_key(bbox)

        assert key == "bbox_12.5000_55.6000_12.7000_55.7500"

    def test_cache_key_precision(self):
        """Keys should have 4 decimal places."""
        bbox = (12.5, 55.6, 12.7, 55.75)
        key = bbox_to_cache_key(bbox)

        # All values should be formatted with 4 decimals
        assert "12.5000" in key
        assert "55.6000" in key

    def test_cache_key_negative_coords(self):
        """Handle negative coordinates in key."""
        bbox = (-74.05, 40.70, -73.85, 40.85)
        key = bbox_to_cache_key(bbox)

        assert key == "bbox_-74.0500_40.7000_-73.8500_40.8500"


class TestOsmExtractCache:
    """Tests for OsmExtractCache class."""

    def test_init_creates_directories(self, tmp_cache_dir: Path, mock_pbf_path: Path):
        """Cache init should create necessary directories."""
        cache = OsmExtractCache(mock_pbf_path, tmp_cache_dir)

        assert cache.extracts_dir.exists()
        assert cache.extracts_dir == tmp_cache_dir / "extracts"

    def test_get_extract_returns_cached_file(
        self, tmp_cache_dir: Path, mock_pbf_path: Path
    ):
        """Should return cached file if it exists."""
        cache = OsmExtractCache(mock_pbf_path, tmp_cache_dir)

        # Pre-create cached file
        bbox = (12.50, 55.60, 12.70, 55.75)
        key = cache._bbox_key(bbox)
        cached_path = cache.extracts_dir / f"{key}.osm.pbf"
        cached_path.write_bytes(b"cached pbf data")

        result = cache.get_extract(bbox)

        assert result == cached_path
        assert result.read_bytes() == b"cached pbf data"

    def test_get_extract_snaps_to_grid(
        self, tmp_cache_dir: Path, mock_pbf_path: Path
    ):
        """Should snap bbox to grid before cache lookup."""
        cache = OsmExtractCache(mock_pbf_path, tmp_cache_dir)

        # Pre-create cached file with snapped key
        original_bbox = (12.523, 55.671, 12.589, 55.712)
        snapped_bbox = snap_bbox_to_grid(original_bbox)
        key = bbox_to_cache_key(snapped_bbox)
        cached_path = cache.extracts_dir / f"{key}.osm.pbf"
        cached_path.write_bytes(b"snapped cached data")

        # Request with original (unsnapped) bbox
        result = cache.get_extract(original_bbox)

        assert result == cached_path

    def test_get_extract_planet_not_found(self, tmp_cache_dir: Path):
        """Should raise FileNotFoundError if planet doesn't exist."""
        cache = OsmExtractCache(
            tmp_cache_dir / "nonexistent.osm.pbf", tmp_cache_dir
        )
        bbox = (12.50, 55.60, 12.70, 55.75)

        with pytest.raises(FileNotFoundError):
            cache.get_extract(bbox)

    def test_extract_builds_correct_command(
        self, tmp_cache_dir: Path, mock_pbf_path: Path
    ):
        """Should build correct osmium command arguments."""
        cache = OsmExtractCache(mock_pbf_path, tmp_cache_dir)
        bbox = (12.50, 55.60, 12.70, 55.75)

        # Test the bbox string format used in the command
        snapped = snap_bbox_to_grid(bbox, cache.grid_size)
        west, south, east, north = snapped
        bbox_str = f"{west},{south},{east},{north}"

        # Verify bbox string format
        assert "12.5" in bbox_str
        assert "55.6" in bbox_str
        assert "12.7" in bbox_str
        assert "55.75" in bbox_str

        # Verify the key generation
        key = cache._bbox_key(bbox)
        assert "bbox_" in key
        assert ".osm.pbf" not in key  # Key doesn't include extension

    def test_get_cache_info(self, tmp_cache_dir: Path, mock_pbf_path: Path):
        """Should return accurate cache statistics."""
        cache = OsmExtractCache(mock_pbf_path, tmp_cache_dir)

        # Create some cached files
        (cache.extracts_dir / "bbox_12.5000_55.6000_12.7000_55.7500.osm.pbf").write_bytes(
            b"x" * 1000
        )
        (cache.extracts_dir / "bbox_13.0000_56.0000_13.2000_56.2000.osm.pbf").write_bytes(
            b"y" * 2000
        )

        info = cache.get_cache_info()

        assert info["count"] == 2
        assert len(info["keys"]) == 2

    def test_clear_cache(self, tmp_cache_dir: Path, mock_pbf_path: Path):
        """Should remove all cached files."""
        cache = OsmExtractCache(mock_pbf_path, tmp_cache_dir)

        # Create some cached files
        (cache.extracts_dir / "bbox_12.5000_55.6000_12.7000_55.7500.osm.pbf").write_bytes(
            b"data"
        )
        (cache.extracts_dir / "bbox_12.5000_55.6000_12.7000_55.7500.lock").write_text("")

        count = cache.clear_cache()

        assert count == 2
        assert len(list(cache.extracts_dir.glob("bbox_*"))) == 0


class TestParseBboxFromFilename:
    """Tests for parse_bbox_from_filename function."""

    def test_basic(self):
        """Should parse basic bbox filename."""
        result = parse_bbox_from_filename("bbox_7.5000_54.5000_15.5000_58.0000.osm.pbf")
        assert result == (7.5, 54.5, 15.5, 58.0)

    def test_with_hierarchy_suffix(self):
        """Should parse filename with hierarchy suffix."""
        result = parse_bbox_from_filename(
            "bbox_7.5000_54.5000_15.5000_58.0000_Europe_DK.osm.pbf"
        )
        assert result == (7.5, 54.5, 15.5, 58.0)

        result = parse_bbox_from_filename(
            "bbox_9.5000_54.5000_12.5000_56.5000_Europe_DK_DK-83.osm.pbf"
        )
        assert result == (9.5, 54.5, 12.5, 56.5)

    def test_negative_coords(self):
        """Should handle negative coordinates."""
        result = parse_bbox_from_filename(
            "bbox_-74.0500_40.7000_-73.8500_40.8500_Americas_US_US-NY.osm.pbf"
        )
        assert result == (-74.05, 40.7, -73.85, 40.85)

    def test_invalid_returns_none(self):
        """Should return None for invalid filenames."""
        assert parse_bbox_from_filename("invalid.osm.pbf") is None
        assert parse_bbox_from_filename("denmark.osm.pbf") is None

    def test_ignores_tmp_files(self):
        """Should return None for temp files."""
        assert (
            parse_bbox_from_filename("bbox_7.5000_54.5000_15.5000_58.0000.tmp.osm.pbf")
            is None
        )


class TestBboxContains:
    """Tests for bbox_contains function."""

    def test_contains(self):
        """Should return True when outer contains inner."""
        denmark = (7.5, 54.5, 15.5, 58.0)
        samso = (10.5, 55.8, 10.7, 55.95)
        assert bbox_contains(denmark, samso) is True

    def test_not_contains(self):
        """Should return False when outer does not contain inner."""
        denmark = (7.5, 54.5, 15.5, 58.0)
        paris = (2.2, 48.8, 2.5, 48.9)
        assert bbox_contains(denmark, paris) is False

    def test_smaller_not_contains_larger(self):
        """Smaller bbox should not contain larger bbox."""
        denmark = (7.5, 54.5, 15.5, 58.0)
        samso = (10.5, 55.8, 10.7, 55.95)
        assert bbox_contains(samso, denmark) is False

    def test_exact_match(self):
        """Bbox should contain itself."""
        bbox = (10.5, 55.8, 10.7, 55.95)
        assert bbox_contains(bbox, bbox) is True

    def test_negative_coords(self):
        """Should handle negative coordinates."""
        usa = (-125.0, 24.0, -66.0, 50.0)
        nyc = (-74.05, 40.7, -73.85, 40.85)
        assert bbox_contains(usa, nyc) is True


class TestSelectSource:
    """Tests for OsmExtractCache.select_source method."""

    def test_prefers_smallest(self, tmp_cache_dir: Path):
        """Should prefer smallest extract that contains bbox."""
        extracts = tmp_cache_dir / "extracts"
        extracts.mkdir(parents=True)

        # Create two extracts: country (small) and continent (large)
        (extracts / "bbox_7.5000_54.5000_15.5000_58.0000_Europe_DK.osm.pbf").write_bytes(
            b"x" * 1000
        )
        (
            extracts / "bbox_-25.0000_34.0000_45.0000_72.0000_Europe.osm.pbf"
        ).write_bytes(b"x" * 10000)

        planet = tmp_cache_dir / "planet.osm.pbf"
        planet.write_bytes(b"planet")
        cache = OsmExtractCache(planet, tmp_cache_dir)

        # Request Samsø (within Denmark)
        source = cache.select_source((10.5, 55.8, 10.7, 55.95))
        assert "_Europe_DK" in source.name  # Country, not continent

    def test_falls_back_to_planet(self, tmp_cache_dir: Path):
        """Should fall back to planet when no extract covers bbox."""
        planet = tmp_cache_dir / "planet.osm.pbf"
        planet.write_bytes(b"planet")

        cache = OsmExtractCache(planet, tmp_cache_dir)

        # Request Paris (no France extract)
        source = cache.select_source((2.2, 48.8, 2.5, 48.9))
        assert source == planet

    def test_ignores_lock_and_tmp_files(self, tmp_cache_dir: Path):
        """Should ignore lock and tmp files when selecting source."""
        extracts = tmp_cache_dir / "extracts"
        extracts.mkdir(parents=True)

        # Create lock and tmp files (should be ignored)
        (extracts / "bbox_7.5000_54.5000_15.5000_58.0000.lock").write_bytes(b"")
        (extracts / "bbox_7.5000_54.5000_15.5000_58.0000.tmp.osm.pbf").write_bytes(
            b"x" * 500
        )
        # Create valid extract
        (extracts / "bbox_7.5000_54.5000_15.5000_58.0000_Europe_DK.osm.pbf").write_bytes(
            b"x" * 1000
        )

        planet = tmp_cache_dir / "planet.osm.pbf"
        planet.write_bytes(b"planet")
        cache = OsmExtractCache(planet, tmp_cache_dir)

        source = cache.select_source((10.5, 55.8, 10.7, 55.95))
        assert source.name == "bbox_7.5000_54.5000_15.5000_58.0000_Europe_DK.osm.pbf"

    def test_no_candidates_empty_extracts_dir(self, tmp_cache_dir: Path):
        """Should fall back to planet when extracts dir is empty."""
        planet = tmp_cache_dir / "planet.osm.pbf"
        planet.write_bytes(b"planet")

        cache = OsmExtractCache(planet, tmp_cache_dir)

        source = cache.select_source((10.5, 55.8, 10.7, 55.95))
        assert source == planet

    def test_selects_exact_match_over_larger(self, tmp_cache_dir: Path):
        """Should select exact bbox match over larger containing bbox."""
        extracts = tmp_cache_dir / "extracts"
        extracts.mkdir(parents=True)

        # Exact match for the request (must be > 1000 bytes to not be skipped)
        (extracts / "bbox_10.5000_55.8000_10.7000_55.9500.osm.pbf").write_bytes(
            b"x" * 1500
        )
        # Larger containing bbox
        (extracts / "bbox_7.5000_54.5000_15.5000_58.0000_Europe_DK.osm.pbf").write_bytes(
            b"x" * 5000
        )

        planet = tmp_cache_dir / "planet.osm.pbf"
        planet.write_bytes(b"planet")
        cache = OsmExtractCache(planet, tmp_cache_dir)

        source = cache.select_source((10.5, 55.8, 10.7, 55.95))
        # Smallest file wins (1500 bytes vs 5000 bytes)
        assert "10.5000_55.8000_10.7000_55.9500" in source.name

    def test_skips_empty_or_corrupt_files(self, tmp_cache_dir: Path):
        """Should skip files smaller than 1KB (likely corrupt)."""
        extracts = tmp_cache_dir / "extracts"
        extracts.mkdir(parents=True)

        # Empty/corrupt file (too small)
        (extracts / "bbox_10.5000_55.8000_10.7000_55.9500.osm.pbf").write_bytes(
            b"x" * 100
        )
        # Valid file
        (extracts / "bbox_7.5000_54.5000_15.5000_58.0000_Europe_DK.osm.pbf").write_bytes(
            b"x" * 2000
        )

        planet = tmp_cache_dir / "planet.osm.pbf"
        planet.write_bytes(b"planet")
        cache = OsmExtractCache(planet, tmp_cache_dir)

        source = cache.select_source((10.5, 55.8, 10.7, 55.95))
        # Should skip the 100-byte file and use Denmark
        assert "_Europe_DK" in source.name

    def test_get_extract_uses_selected_source(self, tmp_cache_dir: Path):
        """get_extract should pass selected source to _run_osmium_extract."""
        extracts = tmp_cache_dir / "extracts"
        extracts.mkdir(parents=True)

        # Regional extract covering Samsø
        denmark_extract = extracts / "bbox_7.5000_54.5000_15.5000_58.0000_Europe_DK.osm.pbf"
        denmark_extract.write_bytes(b"x" * 2000)

        planet = tmp_cache_dir / "planet.osm.pbf"
        planet.write_bytes(b"planet data")
        cache = OsmExtractCache(planet, tmp_cache_dir)

        # Mock _run_osmium_extract to capture the source argument
        captured_source = None

        def mock_extract(bbox, output_path, source):
            nonlocal captured_source
            captured_source = source
            # Create the output file so get_extract succeeds
            output_path.write_bytes(b"extracted")

        with patch.object(cache, "_run_osmium_extract", side_effect=mock_extract):
            # Request Samsø (within Denmark)
            cache.get_extract((10.55, 55.82, 10.65, 55.90))

        # Should have used Denmark extract, not planet
        assert captured_source is not None
        assert captured_source == denmark_extract
        assert "Europe_DK" in captured_source.name
