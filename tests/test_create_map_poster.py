"""Tests for create_map_poster pure functions and utilities."""

from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from create_map_poster import (
    _cache_path,
    cache_get,
    cache_set,
    classify_highway,
    generate_output_filename,
    get_road_color,
    get_road_width,
    point_dist_to_bbox,
    resolve_road_detail,
    ROAD_DETAIL_LEVELS,
)


class TestPointDistToBbox:
    """Tests for point_dist_to_bbox coordinate math."""

    def test_returns_four_element_tuple(self):
        bbox = point_dist_to_bbox((55.68, 12.57), 5000)
        assert len(bbox) == 4

    def test_bbox_order_is_west_south_east_north(self):
        west, south, east, north = point_dist_to_bbox((55.68, 12.57), 5000)
        assert west < east
        assert south < north

    def test_center_is_within_bbox(self):
        lat, lon = 55.68, 12.57
        west, south, east, north = point_dist_to_bbox((lat, lon), 5000)
        assert west < lon < east
        assert south < lat < north

    def test_symmetric_around_center(self):
        lat, lon = 55.68, 12.57
        west, south, east, north = point_dist_to_bbox((lat, lon), 5000)
        assert pytest.approx(lat - south, rel=1e-6) == pytest.approx(north - lat, rel=1e-6)
        assert pytest.approx(lon - west, rel=1e-6) == pytest.approx(east - lon, rel=1e-6)

    def test_larger_distance_gives_larger_bbox(self):
        small = point_dist_to_bbox((55.68, 12.57), 1000)
        large = point_dist_to_bbox((55.68, 12.57), 10000)
        assert large[0] < small[0]  # west more negative
        assert large[1] < small[1]  # south more negative
        assert large[2] > small[2]  # east larger
        assert large[3] > small[3]  # north larger

    def test_equator_vs_high_latitude(self):
        """Longitude delta should be larger at equator (cos(0)=1) than at high lat."""
        equator = point_dist_to_bbox((0.0, 12.0), 5000)
        arctic = point_dist_to_bbox((70.0, 12.0), 5000)
        equator_lon_delta = equator[2] - equator[0]
        arctic_lon_delta = arctic[2] - arctic[0]
        assert arctic_lon_delta > equator_lon_delta

    def test_latitude_delta_independent_of_longitude(self):
        """Latitude delta should be the same regardless of longitude position."""
        bbox1 = point_dist_to_bbox((45.0, 0.0), 5000)
        bbox2 = point_dist_to_bbox((45.0, 90.0), 5000)
        lat_delta1 = bbox1[3] - bbox1[1]
        lat_delta2 = bbox2[3] - bbox2[1]
        assert pytest.approx(lat_delta1, rel=1e-10) == lat_delta2

    def test_approximate_distance_accuracy(self):
        """5000m should give roughly 0.045 degrees latitude delta."""
        west, south, east, north = point_dist_to_bbox((45.0, 10.0), 5000)
        lat_delta = north - 45.0
        # 5000m / 111000 m/deg ≈ 0.045
        assert pytest.approx(lat_delta, abs=0.001) == 5000 / 111000

    def test_zero_distance(self):
        west, south, east, north = point_dist_to_bbox((55.0, 12.0), 0)
        assert pytest.approx(west, abs=1e-10) == 12.0
        assert pytest.approx(east, abs=1e-10) == 12.0
        assert pytest.approx(south, abs=1e-10) == 55.0
        assert pytest.approx(north, abs=1e-10) == 55.0


class TestClassifyHighway:
    """Tests for classify_highway normalization."""

    def test_string_passthrough(self):
        assert classify_highway("motorway") == "motorway"
        assert classify_highway("residential") == "residential"

    def test_list_takes_first(self):
        assert classify_highway(["primary", "secondary"]) == "primary"

    def test_empty_list_returns_unclassified(self):
        assert classify_highway([]) == "unclassified"

    def test_non_string_returns_unclassified(self):
        assert classify_highway(42) == "unclassified"
        assert classify_highway(None) == "unclassified"


class TestGetRoadColor:
    """Tests for get_road_color theme lookups."""

    @pytest.fixture(autouse=True)
    def setup_theme(self):
        """Set up the global THEME dict for road color tests."""
        import create_map_poster
        create_map_poster.THEME = {
            "road_motorway": "#FF0000",
            "road_primary": "#00FF00",
            "road_secondary": "#0000FF",
            "road_tertiary": "#FFFF00",
            "road_residential": "#FF00FF",
            "road_default": "#888888",
        }

    def test_motorway(self):
        assert get_road_color("motorway") == "#FF0000"
        assert get_road_color("motorway_link") == "#FF0000"

    def test_primary(self):
        assert get_road_color("trunk") == "#00FF00"
        assert get_road_color("primary") == "#00FF00"
        assert get_road_color("primary_link") == "#00FF00"

    def test_secondary(self):
        assert get_road_color("secondary") == "#0000FF"
        assert get_road_color("secondary_link") == "#0000FF"

    def test_tertiary(self):
        assert get_road_color("tertiary") == "#FFFF00"
        assert get_road_color("tertiary_link") == "#FFFF00"

    def test_residential(self):
        assert get_road_color("residential") == "#FF00FF"
        assert get_road_color("living_street") == "#FF00FF"
        assert get_road_color("unclassified") == "#FF00FF"

    def test_unknown_highway_returns_default(self):
        assert get_road_color("cycleway") == "#888888"
        assert get_road_color("path") == "#888888"

    def test_handles_list_input(self):
        assert get_road_color(["motorway", "primary"]) == "#FF0000"


class TestGetRoadWidth:
    """Tests for get_road_width line widths."""

    def test_motorway_widest(self):
        assert get_road_width("motorway") == 1.2

    def test_primary(self):
        assert get_road_width("trunk") == 1.0
        assert get_road_width("primary") == 1.0

    def test_secondary(self):
        assert get_road_width("secondary") == 0.8

    def test_tertiary(self):
        assert get_road_width("tertiary") == 0.6

    def test_default_narrowest(self):
        assert get_road_width("residential") == 0.4
        assert get_road_width("service") == 0.4
        assert get_road_width("unknown") == 0.4

    def test_width_hierarchy(self):
        """Road widths should decrease with road importance."""
        widths = [
            get_road_width("motorway"),
            get_road_width("primary"),
            get_road_width("secondary"),
            get_road_width("tertiary"),
            get_road_width("residential"),
        ]
        assert widths == sorted(widths, reverse=True)

    def test_handles_list_input(self):
        assert get_road_width(["secondary", "tertiary"]) == 0.8


class TestResolveRoadDetail:
    """Tests for resolve_road_detail auto-detection."""

    def test_explicit_levels_pass_through(self):
        assert resolve_road_detail("low") == "low"
        assert resolve_road_detail("medium") == "medium"
        assert resolve_road_detail("high") == "high"

    def test_auto_without_bbox_returns_high(self):
        assert resolve_road_detail("auto", bbox=None) == "high"

    def test_auto_large_bbox_returns_low(self):
        # Large country-sized bbox > 500,000 km²
        result = resolve_road_detail("auto", bbox=(-10, 35, 30, 70))
        assert result == "low"

    def test_auto_medium_bbox_returns_medium(self):
        # Medium region-sized bbox ~ 5,000 km²
        result = resolve_road_detail("auto", bbox=(12.0, 55.0, 13.0, 56.0))
        assert result == "medium"

    def test_auto_small_bbox_returns_high(self):
        # Small city-sized bbox < 500 km²
        result = resolve_road_detail("auto", bbox=(12.5, 55.6, 12.7, 55.8))
        assert result == "high"


class TestRoadDetailLevels:
    """Tests for ROAD_DETAIL_LEVELS constant."""

    def test_all_levels_defined(self):
        assert "low" in ROAD_DETAIL_LEVELS
        assert "medium" in ROAD_DETAIL_LEVELS
        assert "high" in ROAD_DETAIL_LEVELS

    def test_low_is_subset_of_medium(self):
        assert set(ROAD_DETAIL_LEVELS["low"]).issubset(set(ROAD_DETAIL_LEVELS["medium"]))

    def test_medium_is_subset_of_high(self):
        assert set(ROAD_DETAIL_LEVELS["medium"]).issubset(set(ROAD_DETAIL_LEVELS["high"]))

    def test_motorway_in_all_levels(self):
        for level in ROAD_DETAIL_LEVELS.values():
            assert "motorway" in level


class TestGenerateOutputFilename:
    """Tests for generate_output_filename."""

    def test_includes_city_and_theme(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("create_map_poster.POSTERS_DIR", os.path.join(tmpdir, "posters")):
                result = generate_output_filename("Tokyo", "noir", "png")
                assert "tokyo" in result
                assert "noir" in result
                assert result.endswith(".png")

    def test_spaces_replaced_with_underscores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("create_map_poster.POSTERS_DIR", os.path.join(tmpdir, "posters")):
                result = generate_output_filename("New York", "blueprint", "svg")
                assert "new_york" in result
                assert result.endswith(".svg")

    def test_creates_posters_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            posters_dir = os.path.join(tmpdir, "posters")
            with patch("create_map_poster.POSTERS_DIR", posters_dir):
                generate_output_filename("Tokyo", "noir", "png")
                assert os.path.exists(posters_dir)


class TestCachePath:
    """Tests for _cache_path."""

    def test_returns_pkl_extension(self):
        result = _cache_path("test_key")
        assert result.endswith(".pkl")

    def test_sanitizes_path_separators(self):
        result = _cache_path(f"a{os.sep}b{os.sep}c")
        basename = os.path.basename(result)
        assert os.sep not in basename or basename.count(os.sep) == 0


class TestCacheGetSet:
    """Tests for cache_get and cache_set."""

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("create_map_poster.CACHE_DIR", tmpdir):
                cache_set("test_roundtrip", {"foo": 42})
                result = cache_get("test_roundtrip")
                assert result == {"foo": 42}

    def test_get_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("create_map_poster.CACHE_DIR", tmpdir):
                result = cache_get("nonexistent_key")
                assert result is None

    def test_stores_complex_objects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("create_map_poster.CACHE_DIR", tmpdir):
                data = {"coords": (55.68, 12.57), "names": ["a", "b"]}
                cache_set("complex_data", data)
                result = cache_get("complex_data")
                assert result == data

    def test_creates_cache_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "new_cache")
            with patch("create_map_poster.CACHE_DIR", cache_path):
                cache_set("test_mkdir", "value")
                assert os.path.exists(cache_path)


class TestLoadTheme:
    """Tests for load_theme."""

    def test_loads_valid_theme_file(self):
        from create_map_poster import load_theme

        with tempfile.TemporaryDirectory() as tmpdir:
            theme_data = {
                "name": "Test Theme",
                "bg": "#FFFFFF",
                "text": "#000000",
                "water": "#0000FF",
            }
            theme_path = os.path.join(tmpdir, "test_theme.json")
            with open(theme_path, "w") as f:
                json.dump(theme_data, f)

            with patch("create_map_poster.THEMES_DIR", tmpdir):
                result = load_theme("test_theme")
                assert result["name"] == "Test Theme"
                assert result["bg"] == "#FFFFFF"

    def test_missing_theme_returns_default(self):
        from create_map_poster import load_theme

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("create_map_poster.THEMES_DIR", tmpdir):
                result = load_theme("nonexistent_theme")
                # Should return the embedded fallback theme
                assert "bg" in result
                assert "text" in result
                assert result["name"] == "Feature-Based Shading"


class TestGetAvailableThemes:
    """Tests for get_available_themes."""

    def test_finds_json_files(self):
        from create_map_poster import get_available_themes

        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["noir", "ocean", "blueprint"]:
                with open(os.path.join(tmpdir, f"{name}.json"), "w") as f:
                    json.dump({"name": name}, f)
            # Non-json file should be ignored
            with open(os.path.join(tmpdir, "readme.txt"), "w") as f:
                f.write("not a theme")

            with patch("create_map_poster.THEMES_DIR", tmpdir):
                themes = get_available_themes()
                assert sorted(themes) == ["blueprint", "noir", "ocean"]

    def test_empty_directory(self):
        from create_map_poster import get_available_themes

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("create_map_poster.THEMES_DIR", tmpdir):
                themes = get_available_themes()
                assert themes == []

    def test_returns_sorted(self):
        from create_map_poster import get_available_themes

        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["z_theme", "a_theme", "m_theme"]:
                with open(os.path.join(tmpdir, f"{name}.json"), "w") as f:
                    json.dump({}, f)

            with patch("create_map_poster.THEMES_DIR", tmpdir):
                themes = get_available_themes()
                assert themes == sorted(themes)
